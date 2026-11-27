import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from model.model import Dan, DanConfig, config_dict
from sft.chat import encode_chat

SAMPLE_QUESTIONS = [
    "Hi! Who are you?",
    "What is 12 + 7?",
    "Give me three tips to sleep better.",
    "Who was the first person to walk on the moon?",
    "Who won the 1987 Tour de France?",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("init", help="checkpoint pre-entraine, ex. checkpoints/dan/final.pt")
    p.add_argument("--data", default="data/sft")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--resume", action="store_true", help="reprend depuis le dernier step_*.pt de --out")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max_steps", type=int, help="arrete apres N etapes (pour tester)")
    p.add_argument("--batch_rows", type=int, default=32)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_rows", type=int, default=256)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every_min", type=float, default=60)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=sys.platform != "win32")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def to_xy(ids, mask, device):
    ids = torch.from_numpy(ids.astype(np.int64)).to(device, non_blocking=True)
    mask = torch.from_numpy(mask.astype(np.bool_)).to(device, non_blocking=True)
    x = ids[:, :-1]
    y = torch.where(mask[:, 1:], ids[:, 1:], torch.full_like(ids[:, 1:], -1))
    return x, y


def loss_sum(fwd, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = fwd(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.reshape(-1),
                           ignore_index=-1, reduction="sum")


@torch.no_grad()
def evaluate(fwd, ids, mask, args, device):
    fwd.eval()
    total, count = 0.0, 0
    n = min(len(ids), args.eval_rows)
    for i in range(0, n, args.micro_batch):
        x, y = to_xy(ids[i:i + args.micro_batch], mask[i:i + args.micro_batch], device)
        total += loss_sum(fwd, x, y).item()
        count += (y != -1).sum().item()
    fwd.train()
    return total / max(count, 1)


@torch.no_grad()
def samples(model, tok, device):
    model.eval()
    stops = (tok.special["<|endoftext|>"], tok.special["<|im_end|>"])
    out = []
    for q in SAMPLE_QUESTIONS:
        ids, _ = encode_chat(tok, [{"role": "user", "content": q}], generation=True)
        idx = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(idx, max_new_tokens=80, temperature=0.4, top_k=50,
                                 repetition_penalty=1.1, stop_ids=stops)
        if gen and gen[-1] in stops:
            gen = gen[:-1]
        out.append(f"> {q}\n{tok.decode(gen).strip()}")
    model.train()
    return out


def save(path, model, opt, step, args):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": config_dict(model.config), "args": vars(args), "sft": True}, tmp)
    os.replace(tmp, path)
    print(f"  sauvegarde -> {path}", flush=True)


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data = ROOT / args.data
    train_ids = np.load(data / "train_ids.npy", mmap_mode="r")
    train_mask = np.load(data / "train_mask.npy", mmap_mode="r")
    val_ids = np.load(data / "val_ids.npy", mmap_mode="r")
    val_mask = np.load(data / "val_mask.npy", mmap_mode="r")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    start = None
    if args.resume and sorted(out.glob("step_*.pt")):
        start = sorted(out.glob("step_*.pt"))[-1]
    ckpt = torch.load(start or args.init, map_location=device, weights_only=False)
    config = DanConfig(**ckpt["config"])
    assert train_ids.shape[1] - 1 <= config.max_seq_len, \
        f"lignes de {train_ids.shape[1] - 1} tokens, contexte du modele {config.max_seq_len}"

    model = Dan(config).to(device)
    model.load_state_dict(ckpt["model"])
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    step = 0
    if start:
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"Reprise depuis {start} (etape {step})")
    else:
        print(f"Depart depuis {args.init} (etape de pre-entrainement {ckpt['step']})")
    del ckpt

    assert args.batch_rows % args.micro_batch == 0
    accum = args.batch_rows // args.micro_batch
    steps_per_epoch = len(train_ids) // args.batch_rows
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"{len(train_ids)} lignes, {total_steps} etapes de {args.batch_rows} lignes "
          f"({args.epochs} epoques, {train_mask.sum() / 1e6:.1f} M tokens appris par epoque)")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr * max(0.0, (total_steps - s) / max(1, total_steps - args.warmup))

    def order(epoch):
        return np.random.default_rng(args.seed + epoch).permutation(len(train_ids))

    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    fwd = torch.compile(model) if args.compile else model

    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    eval_f = open(out / "eval.csv", "a", newline="")
    eval_log = csv.writer(eval_f)

    model.train()
    last_save = time.time()
    t_log, tok_log = time.time(), 0
    loss_acc, n_acc = 0.0, 0
    epoch, perm = -1, None

    while step < total_steps:
        if step % args.eval_every == 0:
            v = evaluate(fwd, val_ids, val_mask, args, device)
            print(f"[eval] etape {step}  val {v:.3f}", flush=True)
            eval_log.writerow([step, round(v, 4)])
            eval_f.flush()
            with open(out / "samples.txt", "a", encoding="utf-8", newline="") as f:
                f.write(f"===== etape {step} =====\n")
                for s in samples(model, tok, device):
                    f.write(s + "\n---\n")
            t_log, tok_log = time.time(), 0

        e, pos = divmod(step, steps_per_epoch)
        if e != epoch:
            epoch, perm = e, order(e)
        batch = np.sort(perm[pos * args.batch_rows:(pos + 1) * args.batch_rows])

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        micro = [to_xy(train_ids[batch[i:i + args.micro_batch]], train_mask[batch[i:i + args.micro_batch]], device)
                 for i in range(0, len(batch), args.micro_batch)]
        n_tok = sum((y != -1).sum() for _, y in micro).clamp(min=1)

        opt.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for x, y in micro:
            loss = loss_sum(fwd, x, y) / n_tok
            loss.backward()
            step_loss += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        loss_acc += step_loss.item()
        n_acc += 1
        tok_log += args.batch_rows * (train_ids.shape[1] - 1)

        if step % args.log_every == 0:
            dt = time.time() - t_log
            speed = tok_log / dt
            eta = (total_steps - step) * args.batch_rows * (train_ids.shape[1] - 1) / speed / 60
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"etape {step}/{total_steps}  epoque {step / steps_per_epoch:.2f}  "
                  f"loss {loss_acc / n_acc:.3f}  lr {lr:.2e}  grad {grad_norm.item():.2f}  "
                  f"{speed:,.0f} tok/s  mem {mem:.1f} Go  reste {eta:.0f} min", flush=True)
            log.writerow([step, round(loss_acc / n_acc, 4), lr, round(grad_norm.item(), 3), round(speed)])
            log_f.flush()
            loss_acc, n_acc = 0.0, 0
            t_log, tok_log = time.time(), 0

        if time.time() - last_save > args.save_every_min * 60:
            save(out / f"step_{step:06d}.pt", model, opt, step, args)
            for old in sorted(out.glob("step_*.pt"))[:-2]:
                old.unlink()
            last_save = time.time()

    v = evaluate(fwd, val_ids, val_mask, args, device)
    print(f"[eval finale] val {v:.3f}")
    eval_log.writerow([step, round(v, 4)])
    with open(out / "samples.txt", "a", encoding="utf-8", newline="") as f:
        f.write(f"===== final =====\n")
        for s in samples(model, tok, device):
            f.write(s + "\n---\n")
    save(out / "final.pt", model, opt, step, args)
    log_f.close()
    eval_f.close()


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        sys.stdout.flush()
        os._exit(0)

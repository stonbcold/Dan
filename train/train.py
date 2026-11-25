import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from model.model import CONFIGS, Dan, DanConfig, config_dict

TOKENIZED = ROOT / "data" / "tokenized"

PRESETS = {
    "mini": dict(total_tokens=450e6, batch_tokens=65_536, micro_batch=16, warmup=200,
                 lr=3e-3, eval_every=250, out="checkpoints/mini"),
    "dan": dict(total_tokens=10e9, batch_tokens=524_288, micro_batch=8, warmup=500,
                lr=3e-3, eval_every=500, out="checkpoints/dan"),
}

DEFAULT_WEIGHTS = "fineweb=0.62,finemath=0.26,code_translated=0.08,code_textblock=0.04"

SAMPLE_PROMPTS = [
    "The capital of France is",
    "Once upon a time, a little girl",
    "Question: What is 7 + 5?\nAnswer:",
    "def is_even(n):",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(CONFIGS), default="mini")
    p.add_argument("--out")
    p.add_argument("--resume", help="chemin d'un .pt, ou 'auto' pour le plus recent de --out")
    p.add_argument("--total_tokens", type=float)
    p.add_argument("--total_steps", type=int)
    p.add_argument("--batch_tokens", type=int)
    p.add_argument("--micro_batch", type=int)
    p.add_argument("--seq_len", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--warmup", type=int)
    p.add_argument("--decay_frac", type=float, default=0.2)
    p.add_argument("--decay_start", type=int)
    p.add_argument("--stop_before_decay", action="store_true")
    p.add_argument("--max_hours", type=float, help="sauvegarde puis s'arrete apres N heures")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every_min", type=float, default=60)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=sys.platform != "win32")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for k, v in PRESETS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


class Batches:
    def __init__(self, weights, split, seq_len, micro_batch, seed, device):
        self.names = list(weights)
        self.arrays = [np.memmap(TOKENIZED / f"{n}_{split}.bin", dtype=np.uint16, mode="r")
                       for n in self.names]
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.seq_len, self.micro_batch, self.device = seq_len, micro_batch, device
        self.rng = np.random.default_rng(seed)

    def next(self):
        T = self.seq_len
        src = self.rng.choice(len(self.arrays), size=self.micro_batch, p=self.p)
        buf = np.empty((self.micro_batch, T + 1), dtype=np.int64)
        for i, s in enumerate(src):
            a = self.arrays[s]
            j = self.rng.integers(0, len(a) - T - 1)
            buf[i] = a[j:j + T + 1]
        t = torch.from_numpy(buf).pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def lr_at(step, args, total_steps, decay_start):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step < decay_start:
        return args.lr
    return args.lr * max(0.0, (total_steps - step) / max(1, total_steps - decay_start))


@torch.no_grad()
def evaluate(model, weights, args, device):
    model.eval()
    losses = {}
    for k, name in enumerate(weights):
        data = Batches({name: 1.0}, "val", args.seq_len, args.micro_batch, 1234 + k, device)
        total = 0.0
        for _ in range(args.eval_batches):
            x, y = data.next()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / args.eval_batches
    model.train()
    return losses


@torch.no_grad()
def samples(model, tok, device):
    model.eval()
    out = []
    for prompt in SAMPLE_PROMPTS:
        idx = torch.tensor([tok.encode(prompt)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(idx, max_new_tokens=48, temperature=0.7, top_k=50,
                                 repetition_penalty=1.1, stop_ids=(tok.special["<|endoftext|>"],))
        out.append(prompt + tok.decode(ids))
    model.train()
    return out


def save(path, model, opt, step, args, train_data):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "config": config_dict(model.config),
        "args": vars(args),
        "data_rng": train_data.rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)
    print(f"  sauvegarde -> {path}", flush=True)


def prune(out, keep):
    old = sorted(out.glob("step_*.pt"))[:-keep]
    for p in old:
        p.unlink()


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ckpt = None
    if args.resume:
        path = sorted(out.glob("step_*.pt"))[-1] if args.resume == "auto" else Path(args.resume)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = DanConfig(**ckpt["config"])
        print(f"Reprise depuis {path} (etape {ckpt['step']})")
    else:
        config = CONFIGS[args.model]
    if args.seq_len is None:
        args.seq_len = config.max_seq_len

    model = Dan(config).to(device)
    print(f"Dan '{args.model}' : {model.num_params() / 1e6:.1f} M parametres")

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
    train_data = Batches(weights, "train", args.seq_len, args.micro_batch, args.seed, device)

    step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        train_data.rng.bit_generator.state = ckpt["data_rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        del ckpt

    tokens_per_micro = args.micro_batch * args.seq_len
    assert args.batch_tokens % tokens_per_micro == 0, "batch_tokens doit etre un multiple de micro_batch * seq_len"
    accum = args.batch_tokens // tokens_per_micro
    total_steps = args.total_steps or int(args.total_tokens // args.batch_tokens)
    decay_start = args.decay_start if args.decay_start is not None else int(total_steps * (1 - args.decay_frac))
    print(f"{total_steps} etapes de {args.batch_tokens:,} tokens ({accum} x {tokens_per_micro:,}), "
          f"decroissance a partir de l'etape {decay_start}")

    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    fwd = torch.compile(model) if args.compile else model

    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    eval_f = open(out / "eval.csv", "a", newline="")
    eval_log = csv.writer(eval_f)

    model.train()
    t_start = time.time()
    last_save = time.time()
    t_log, tok_log = time.time(), 0
    loss_acc = torch.zeros((), device=device)
    n_acc = 0

    while step < total_steps:
        if args.stop_before_decay and step >= decay_start:
            save(out / "pre_decay.pt", model, opt, step, args, train_data)
            break

        if step % args.eval_every == 0:
            losses = evaluate(fwd, weights, args, device)
            mean = sum(losses[n] * weights[n] for n in weights) / sum(weights.values())
            print(f"[eval] etape {step}  val {mean:.3f}  " +
                  "  ".join(f"{n} {v:.3f}" for n, v in losses.items()), flush=True)
            eval_log.writerow([step, round(mean, 4)] + [round(losses[n], 4) for n in weights])
            eval_f.flush()
            with open(out / "samples.txt", "a", encoding="utf-8", newline="") as f:
                f.write(f"===== etape {step} =====\n")
                for s in samples(model, tok, device):
                    f.write(s + "\n---\n")
            t_log, tok_log = time.time(), 0

        lr = lr_at(step, args, total_steps, decay_start)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = train_data.next()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            (loss / accum).backward()
            loss_acc += loss.detach() / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        n_acc += 1
        tok_log += args.batch_tokens

        if step % args.log_every == 0:
            loss_val = loss_acc.item() / n_acc
            loss_acc.zero_()
            n_acc = 0
            dt = time.time() - t_log
            speed = tok_log / dt
            eta = (total_steps - step) * args.batch_tokens / speed / 3600
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"etape {step}/{total_steps}  loss {loss_val:.3f}  lr {lr:.2e}  "
                  f"grad {grad_norm.item():.2f}  {speed:,.0f} tok/s  mem {mem:.1f} Go  "
                  f"reste {eta:.1f} h", flush=True)
            log.writerow([step, step * args.batch_tokens, round(loss_val, 4), lr,
                          round(grad_norm.item(), 3), round(speed)])
            log_f.flush()
            t_log, tok_log = time.time(), 0

        if time.time() - last_save > args.save_every_min * 60:
            save(out / f"step_{step:06d}.pt", model, opt, step, args, train_data)
            prune(out, args.keep)
            last_save = time.time()

        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print(f"Limite de {args.max_hours} h atteinte a l'etape {step}")
            save(out / f"step_{step:06d}.pt", model, opt, step, args, train_data)
            prune(out, args.keep)
            break

    if step >= total_steps:
        losses = evaluate(fwd, weights, args, device)
        print(f"[eval finale] " + "  ".join(f"{n} {v:.3f}" for n, v in losses.items()))
        save(out / "final.pt", model, opt, step, args, train_data)

    log_f.close()
    eval_f.close()


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        sys.stdout.flush()
        os._exit(0)

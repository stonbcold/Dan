import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from model.model import load_dan

OUT = ROOT / "data" / "sft"

SHOTS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("How many legs does a spider have?", "8"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which planet is known as the Red Planet?", "Mars"),
]
FEWSHOT = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in SHOTS)


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def is_correct(pred, aliases):
    p = normalize(pred)
    if not p:
        return False
    if p in aliases:
        return True
    padded = f" {p} "
    return any(len(a) >= 3 and f" {a} " in padded and len(p.split()) <= len(a.split()) + 3
               for a in aliases)


@torch.no_grad()
def sample(model, prompt_ids, n, temperature, max_tokens, newline_ids):
    B = n + 1
    idx = torch.tensor([prompt_ids] * B, device="cuda")
    cache = [dict() for _ in model.blocks]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(idx, cache=cache)
    pos = idx.shape[1]
    out = torch.empty(B, 0, dtype=torch.long, device="cuda")
    done = torch.zeros(B, dtype=torch.bool, device="cuda")
    for _ in range(max_tokens):
        scores = logits[:, -1, :].float()
        greedy = scores[:1].argmax(dim=-1, keepdim=True)
        probs = F.softmax(scores[1:] / temperature, dim=-1)
        nxt = torch.cat([greedy, torch.multinomial(probs, 1)])
        out = torch.cat([out, nxt], dim=1)
        done |= newline_ids[nxt.squeeze(1)]
        if done.all():
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(nxt, cache=cache, start_pos=pos)
        pos += 1
    return out.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=16)
    p.add_argument("--known", type=int, default=4, help="bonnes reponses minimum pour 'connu'")
    p.add_argument("--unknown", type=int, default=1, help="bonnes reponses maximum pour 'inconnu'")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, ckpt = load_dan(args.ckpt)
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    newline_ids = torch.tensor([b"\n" in tok.vocab[i] for i in range(len(tok.vocab))], device="cuda")
    print(f"Checkpoint charge (etape {ckpt['step']})")

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=args.split)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    order = order[:args.n]

    OUT.mkdir(parents=True, exist_ok=True)
    ck = Path(args.ckpt)
    out_path = OUT / f"idk_{args.split}_{ck.parent.name}_{ck.stem}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done_ids = {json.loads(line)["id"] for line in f}
        print(f"Reprise : {len(done_ids)} questions deja traitees")

    counts = {"connu": 0, "inconnu": 0, "incertain": 0}
    greedy_ok = total = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        for k, i in enumerate(order, 1):
            ex = ds[i]
            if ex["question_id"] in done_ids:
                continue
            question = ex["question"].strip()
            aliases = set(ex["answer"]["normalized_aliases"])
            prompt = tok.encode(FEWSHOT + f"Q: {question}\nA:", allowed_special=False)
            rows = sample(model, prompt, args.samples, args.temperature, args.max_tokens, newline_ids)
            answers = [tok.decode(r).split("\n", 1)[0].strip() for r in rows]
            ok = [is_correct(a, aliases) for a in answers]
            n_ok = sum(ok[1:])
            if ok[0] and n_ok >= args.known:
                label = "connu"
            elif not ok[0] and n_ok <= args.unknown:
                label = "inconnu"
            else:
                label = "incertain"
            counts[label] += 1
            greedy_ok += ok[0]
            total += 1
            f.write(json.dumps({
                "id": ex["question_id"], "question": question, "gold": ex["answer"]["value"],
                "greedy": answers[0], "samples": answers[1:], "greedy_ok": ok[0],
                "n_ok": n_ok, "label": label,
            }, ensure_ascii=False) + "\n")
            if total % 100 == 0:
                f.flush()
                rate = total / (time.time() - t0)
                print(f"{k}/{len(order)} | exactitude {greedy_ok / total:.1%} | {counts} | "
                      f"{rate:.1f} q/s | reste {(len(order) - k) / rate / 60:.0f} min", flush=True)

    if total:
        print(f"\nExactitude (greedy) : {greedy_ok / total:.1%} sur {total} questions")
        print({k: f"{v} ({v / total:.0%})" for k, v in counts.items()})
    print(f"Resultats dans {out_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from eval.ask import prompt_ids
from model.model import load_dan
from sft.idk import normalize

REFUSALS = ["don't know", "do not know", "not sure", "not confident", "rather not guess",
            "don't have reliable", "no reliable"]


def is_refusal(text):
    t = text.lower().replace("’", "'")
    return any(r in t for r in REFUSALS)


def contains_answer(text, aliases, question):
    padded = f" {normalize(text)} "
    q = f" {normalize(question)} "
    return any(f" {a} " in padded for a in aliases if a and f" {a} " not in q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max_tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, ckpt = load_dan(args.ckpt)
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    stops = (tok.special["<|endoftext|>"], tok.special["<|im_end|>"])

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)

    counts = {"juste": 0, "faux": 0, "refus": 0}
    results = []
    for k, i in enumerate(order[:args.n], 1):
        ex = ds[i]
        idx = torch.tensor([prompt_ids(tok, ex["question"].strip(), chat=True)], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(idx, max_new_tokens=args.max_tokens, temperature=0.0,
                                 repetition_penalty=1.0, stop_ids=stops)
        if ids and ids[-1] in stops:
            ids = ids[:-1]
        text = tok.decode(ids).strip()
        if is_refusal(text):
            verdict = "refus"
        elif contains_answer(text, ex["answer"]["normalized_aliases"], ex["question"]):
            verdict = "juste"
        else:
            verdict = "faux"
        counts[verdict] += 1
        results.append({"question": ex["question"], "gold": ex["answer"]["value"],
                        "answer": text, "verdict": verdict})
        print(f"{k}/{args.n}  {counts}", end="\r", flush=True)

    n = len(results)
    answered = counts["juste"] + counts["faux"]
    print(f"\n\nSur {n} questions TriviaQA (validation) :")
    for v in counts:
        print(f"  {v:6s} {counts[v]:5d}  ({counts[v] / n:.1%})")
    if answered:
        print(f"  quand il repond, il a raison {counts['juste'] / answered:.1%} du temps")
    out = ROOT / "eval" / f"honesty_{Path(args.ckpt).parent.name}_{Path(args.ckpt).stem}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Details : {out}")


if __name__ == "__main__":
    main()

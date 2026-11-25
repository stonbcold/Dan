import argparse
import json
import sys
from pathlib import Path

import regex as re
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from eval.ask import build_prompt
from model.model import load_dan

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract(text):
    marked = "####" in text
    if marked:
        text = text.split("####", 1)[1]
    nums = NUMBER.findall(text)
    if not nums:
        return None
    return nums[0 if marked else -1].replace(",", "").rstrip(".")


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=0, help="nombre de questions (0 = toutes)")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()

    model, ckpt = load_dan(args.ckpt)
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    if args.n:
        test = test.select(range(args.n))

    fewshot = "".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
                      for ex in train.select(range(args.shots)))
    stops = (tok.special["<|endoftext|>"], tok.special["<|im_end|>"])

    correct, results = 0, []
    for k, ex in enumerate(test, 1):
        if args.chat:
            prompt = build_prompt(ex["question"], chat=True)
        else:
            prompt = fewshot + f"Question: {ex['question']}\nAnswer:"
        idx = torch.tensor([tok.encode(prompt)], device="cuda")
        text = ""
        ids = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in model.stream(idx, max_new_tokens=args.max_tokens, temperature=0.0,
                                  repetition_penalty=1.0, stop_ids=stops):
                ids.append(t)
                text = tok.decode(ids)
                if "\n\nQuestion:" in text:
                    text = text.split("\n\nQuestion:", 1)[0]
                    break
        pred = extract(text)
        gold = extract(ex["answer"])
        ok = to_float(pred) is not None and to_float(pred) == to_float(gold)
        correct += ok
        results.append({"question": ex["question"], "gold": gold, "pred": pred, "ok": ok, "output": text})
        print(f"{k}/{len(test)}  exactitude {correct / k:.1%}", end="\r", flush=True)

    out = ROOT / "eval" / f"gsm8k_{Path(args.ckpt).stem}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nGSM8K : {correct}/{len(test)} = {correct / len(test):.2%}  (details : {out})")


if __name__ == "__main__":
    main()

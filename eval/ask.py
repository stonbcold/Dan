import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from model.model import load_dan

SYSTEM = "You are Dan, a small helpful assistant. Answer briefly and honestly."


def build_prompt(question, chat):
    if not chat:
        return question
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def answer(model, tok, question, args):
    idx = torch.tensor([tok.encode(build_prompt(question, args.chat))], device="cuda")
    stops = (tok.special["<|endoftext|>"], tok.special["<|im_end|>"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ids = model.generate(idx, max_new_tokens=args.max_tokens, temperature=args.temperature,
                             top_k=50, repetition_penalty=args.rep, stop_ids=stops)
    if ids and ids[-1] in stops:
        ids = ids[:-1]
    return tok.decode(ids).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--file", help="un fichier texte, une question par ligne")
    p.add_argument("--chat", action="store_true", help="format chat (apres le fine-tuning)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--rep", type=float, default=1.1)
    p.add_argument("--max_tokens", type=int, default=200)
    args = p.parse_args()

    model, ckpt = load_dan(args.ckpt)
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    print(f"Checkpoint charge (etape {ckpt['step']})")

    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8-sig").splitlines()
        questions = [q.strip().replace("\\n", "\n") for q in lines if q.strip()]
        out = Path(args.file).with_name(f"reponses_{Path(args.ckpt).stem}.txt")
        with open(out, "w", encoding="utf-8", newline="") as f:
            for k, q in enumerate(questions, 1):
                a = answer(model, tok, q, args)
                f.write(f"### {k}. {q}\n{a}\n\n")
                print(f"{k}/{len(questions)}", end="\r")
        print(f"\nReponses ecrites dans {out}")
        return

    while True:
        try:
            q = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip():
            print(answer(model, tok, q, args))


if __name__ == "__main__":
    main()

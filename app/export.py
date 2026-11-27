import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from model.model import load_dan
from sft.chat import encode_chat

CHECK_TEXTS = [
    "Hello! How are you doing today?",
    "If x + 12 = 2048, then x = 2036.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    "Café, naïve, résumé, Ångström  and   spaces\n\n\ttabs",
    "Emoji \U0001F600\U0001F680 and 日本語, it's we'll they're I'd",
    "<|im_start|>user\nWhat is 7 * 8?<|im_end|>\n<|im_start|>assistant\n",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/sft/final.pt")
    p.add_argument("--out", default="export/dan")
    args = p.parse_args()

    model, ckpt = load_dan(ROOT / args.ckpt, device="cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tensors = {k: v.float().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(tensors, out / "dan.safetensors", metadata={"step": str(ckpt["step"])})
    json.dump(ckpt["config"], open(out / "config.json", "w"), indent=1)
    shutil.copy(ROOT / "dan_tokenizer.json", out / "dan_tokenizer.json")
    print(f"{len(tensors)} tenseurs, {sum(t.numel() for t in tensors.values()) / 1e6:.1f} M parametres -> {out}")

    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    prompt, _ = encode_chat(tok, [{"role": "user", "content": "What is the capital of France?"}], generation=True)
    with torch.no_grad():
        logits, _ = model(torch.tensor([prompt]))
    check = {
        "texts": CHECK_TEXTS,
        "ids": [tok.encode(t) for t in CHECK_TEXTS],
        "ids_no_special": [tok.encode(t, allowed_special=False) for t in CHECK_TEXTS],
        "prompt": prompt,
        "logits_last": logits[0, -1].tolist(),
    }
    json.dump(check, open(out / "check.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Fichier de verification : {out / 'check.json'}")


if __name__ == "__main__":
    main()

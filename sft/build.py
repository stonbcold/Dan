import argparse
import json
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from bpe import BPETokenizer
from common import DECONTAM, iter_ngrams, words
from sft.chat import encode_chat

KNOWN = ["{a}.", "It's {a}.", "The answer is {a}.", "{a}"]
IDK = [
    "I don't know.",
    "I'm not sure, I don't know the answer to that.",
    "I don't know that one, sorry.",
    "I'm not confident about this, so I'd rather not guess.",
    "Sorry, I don't know.",
    "I don't have reliable knowledge about that.",
]

tok = ngrams = first = None


def init():
    global tok, ngrams, first
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    with open(DECONTAM, "rb") as f:
        d = pickle.load(f)
    ngrams, first = d["ngrams"], d["first"]


def contaminated(messages):
    w = words("\n".join(m["content"] for m in messages))
    return any(w[i] in first and g in ngrams for i, g in iter_ngrams(w))


def process(messages):
    if contaminated(messages):
        return None
    ids, mask = encode_chat(tok, messages)
    return np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)


def smoltalk(n):
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    if n:
        ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))
    for ex in ds:
        msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
        if any(m["role"] == "assistant" for m in msgs):
            yield msgs


def nice(a):
    return a.title() if a.isupper() else a


def idk(paths, ratio, known_min, repeat, rng):
    known, unknown = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["greedy_ok"] and r["n_ok"] >= known_min:
                    known.append(r)
                elif r["label"] == "inconnu":
                    unknown.append(r)
    rng.shuffle(unknown)
    n_unknown = min(len(unknown), max(int(ratio * len(known)), 200))
    print(f"I don't know : {len(known)} connues, {n_unknown} inconnues gardees sur {len(unknown)}")
    for _ in range(repeat):
        for r in known:
            yield [{"role": "user", "content": r["question"]},
                   {"role": "assistant", "content": rng.choice(KNOWN).format(a=nice(r["gold"]))}]
        for r in unknown[:n_unknown]:
            yield [{"role": "user", "content": r["question"]},
                   {"role": "assistant", "content": rng.choice(IDK)}]


def extra(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["messages"]


def pack(examples, row_len, pad):
    place = []
    row, used, dropped = 0, 0, 0
    for ids, _ in examples:
        n = len(ids)
        if n > row_len:
            place.append(None)
            dropped += 1
            continue
        if used + n > row_len:
            row, used = row + 1, 0
        place.append((row, used))
        used += n
    n_rows = row + 1 if used else row
    rows = np.full((n_rows, row_len), pad, dtype=np.uint16)
    masks = np.zeros((n_rows, row_len), dtype=np.uint8)
    for (ids, mask), pl in zip(examples, place):
        if pl is not None:
            r, c = pl
            rows[r, c:c + len(ids)] = ids
            masks[r, c:c + len(ids)] = mask
    return rows, masks, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sft")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--smoltalk", type=int, default=0, help="nombre de conversations (0 = toutes)")
    p.add_argument("--idk", nargs="*", default=[], help="fichiers produits par sft/idk.py")
    p.add_argument("--idk_ratio", type=float, default=2.0, help="inconnues gardees par question connue")
    p.add_argument("--idk_repeat", type=int, default=4, help="nombre de repetitions des exemples I don't know")
    p.add_argument("--known", type=int, default=3, help="tirages justes minimum (sur 5) pour 'connu'")
    p.add_argument("--extra", nargs="*", default=[], help="fichiers jsonl avec un champ messages")
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    rng = random.Random(0)
    convs = [(m, "smoltalk") for m in smoltalk(args.smoltalk)]
    convs += [(m, "idk") for m in idk(args.idk, args.idk_ratio, args.known, args.idk_repeat, rng)]
    convs += [(m, "extra") for m in extra(args.extra)]
    rng.shuffle(convs)
    print(f"{len(convs)} conversations a tokeniser")

    with Pool(args.workers, initializer=init) as pool:
        results = pool.map(process, [m for m, _ in convs], chunksize=256)

    stats = {}
    examples = []
    for (_, src), r in zip(convs, results):
        s = stats.setdefault(src, {"gardees": 0, "contaminees": 0, "tokens": 0, "appris": 0})
        if r is None:
            s["contaminees"] += 1
            continue
        s["gardees"] += 1
        s["tokens"] += len(r[0])
        s["appris"] += int(r[1].sum())
        examples.append(r)
    for src, s in stats.items():
        print(f"  {src}: {s}")

    n_val = max(1, int(len(examples) * args.val_frac))
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    pad = tok.special["<|pad|>"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for split, part in [("val", examples[:n_val]), ("train", examples[n_val:])]:
        rows, masks, dropped = pack(part, args.seq_len + 1, pad)
        np.save(out / f"{split}_ids.npy", rows)
        np.save(out / f"{split}_mask.npy", masks)
        fill = (rows != pad).mean()
        print(f"{split}: {len(rows)} lignes de {args.seq_len + 1} tokens, remplissage {fill:.0%}, "
              f"{masks.sum() / 1e6:.1f} M tokens appris, {dropped} trop longues ignorees")


if __name__ == "__main__":
    main()

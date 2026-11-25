import random
import sys

import numpy as np

from common import ROOT, TOKENIZED

sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer

name = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 50

tok = BPETokenizer()
tok.load(ROOT / "dan_tokenizer.json")
eot = tok.special["<|endoftext|>"]

data = np.memmap(TOKENIZED / f"{name}.bin", dtype=np.uint16, mode="r")
print(f"{name} : {len(data):,} tokens")

out = TOKENIZED / f"{name}_extraits.txt"
rng = random.Random(0)
with open(out, "w", encoding="utf-8", newline="") as f:
    for k in range(count):
        pos = rng.randrange(len(data))
        window = np.asarray(data[max(0, pos - 20000):pos + 20000])
        rel = pos - max(0, pos - 20000)
        ends = np.flatnonzero(window == eot)
        start = ends[ends < rel].max() + 1 if (ends < rel).any() else 0
        stop = ends[ends >= rel].min() if (ends >= rel).any() else len(window)
        text = tok.decode(window[start:stop].tolist())
        f.write(f"===== extrait {k + 1} ({stop - start} tokens) =====\n{text[:2000]}\n\n")
print(f"{count} extraits ecrits dans {out}")

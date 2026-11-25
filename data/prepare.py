import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import regex as re

from common import DECONTAM, ROOT, SOURCES, TOKENIZED, iter_ngrams, raw_dir, words

sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer

WORKERS = 12
RG_PER_TASK = 20
VAL_TOKENS = 5_000_000

FILTERS = {
    "fineweb":  dict(min_chars=200, max_sym=0.25, max_dup_lines=0.3, min_lang=0.65),
    "finemath": dict(min_chars=200, max_sym=0.35, max_dup_lines=0.3, min_lang=None),
    "code_translated": dict(min_chars=100, max_sym=0.45, max_dup_lines=None, min_lang=None),
    "code_textblock":  dict(min_chars=100, max_sym=0.45, max_dup_lines=None, min_lang=None),
}

SYM = re.compile(r"[^\w\s]")

tok =eot = ngrams = first = cfg = None


def init(name):
    global tok, eot, ngrams, first, cfg
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    assert max(tok.vocab) < 65536, "ids trop grands pour uint16"
    eot = tok.special["<|endoftext|>"]
    with open(DECONTAM, "rb") as f:
        d = pickle.load(f)
    ngrams, first = d["ngrams"], d["first"]
    cfg = FILTERS[name]


def reject_reason(text, lang):
    if len(text) < cfg["min_chars"]:
        return "trop_court"
    if cfg["min_lang"] is not None and lang is not None and lang < cfg["min_lang"]:
        return "langue"
    if len(SYM.findall(text)) / len(text) > cfg["max_sym"]:
        return "symboles"
    if cfg["max_dup_lines"] is not None:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 10:
            dup = sum(len(l) * (c - 1) for l, c in Counter(lines).items())
            if dup / sum(map(len, lines)) > cfg["max_dup_lines"]:
                return "lignes_repetees"
    w = words(text)
    for i, g in iter_ngrams(w):
        if w[i] in first and g in ngrams:
            return "contamination [" + g + "]"
    return None


def process(task):
    path, rg0, rg1 = task
    pf = pq.ParquetFile(path)
    cols = ["text"]
    if cfg["min_lang"] is not None and "language_score" in pf.schema_arrow.names:
        cols.append("language_score")
    table = pf.read_row_groups(list(range(rg0, rg1)), columns=cols)
    texts = table.column("text").to_pylist()
    langs = table.column("language_score").to_pylist() if len(cols) > 1 else [None] * len(texts)

    stats, examples = Counter(), {}
    hashes, lens, chunks = [], [], []
    for text, lang in zip(texts, langs):
        stats["lus"] += 1
        text = (text or "").strip()
        reason = reject_reason(text, None if lang is None else float(lang))
        if reason:
            key = reason.split(" ", 1)[0]
            stats[key] += 1
            examples.setdefault(key, reason + "\n" + text[:500])
            continue
        ids = tok.encode(text, allowed_special=False)
        ids.append(eot)
        hashes.append(hashlib.md5(text.encode("utf-8")).digest()[:8])
        lens.append(len(ids))
        chunks.append(np.array(ids, dtype=np.uint16))
    arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
    return hashes, lens, arr, stats, examples


def make_tasks(files):
    tasks = []
    for p in files:
        n = pq.ParquetFile(p).metadata.num_row_groups
        tasks += [(str(p), a, min(a + RG_PER_TASK, n)) for a in range(0, n, RG_PER_TASK)]
    return tasks


def prepare(name):
    TOKENIZED.mkdir(exist_ok=True)
    stats, rejected = Counter(), {}
    seen = set()
    totals = {"train": 0, "val": 0}
    parts = SOURCES[name]

    with open(TOKENIZED / f"{name}_train.bin", "wb") as f_train, \
         open(TOKENIZED / f"{name}_val.bin", "wb") as f_val:
        for repo, prefix, target in parts:
            files = sorted((raw_dir(name) / prefix).glob("*.parquet"))
            if not files:
                print(f"[{name}] aucun fichier dans {raw_dir(name) / prefix}, lance download.py")
                continue
            tasks = make_tasks(files)
            val_goal = VAL_TOKENS // len(parts)
            n_val = n_train = 0
            t0 = last = time.time()
            print(f"[{name}] {prefix} : {len(files)} fichiers, {len(tasks)} taches, objectif {target / 1e9:.2f} G tokens")

            with Pool(WORKERS, initializer=init, initargs=(name,)) as pool:
                for hashes, lens, arr, st, ex in pool.imap(process, tasks):
                    stats.update(st)
                    for reason, text in ex.items():
                        rejected.setdefault(reason, [])
                        if len(rejected[reason]) < 20:
                            rejected[reason].append(text)
                    off = 0
                    for h, n in zip(hashes, lens):
                        doc = arr[off:off + n]
                        off += n
                        if h in seen:
                            stats["doublon"] += 1
                            continue
                        seen.add(h)
                        stats["gardes"] += 1
                        if n_val < val_goal:
                            f_val.write(doc.tobytes())
                            n_val += n
                        else:
                            f_train.write(doc.tobytes())
                            n_train += n
                    if time.time() - last > 30:
                        last = time.time()
                        speed = n_train / (last - t0)
                        eta = (target - n_train) / max(speed, 1) / 60
                        print(f"[{name}] {n_train / 1e9:.3f} / {target / 1e9:.2f} G tokens"
                              f"  {speed / 1e6:.2f} M tok/s  reste ~{eta:.0f} min", flush=True)
                    if n_train >= target:
                        break
            if n_train < target:
                print(f"[{name}] ATTENTION : {prefix} epuise a {n_train / 1e9:.3f} G tokens")
            totals["train"] += n_train
            totals["val"] += n_val

    meta = {"dtype": "uint16", "tokens": totals, "stats": dict(stats), "filters": FILTERS[name]}
    with open(TOKENIZED / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(TOKENIZED / f"{name}_rejets.txt", "w", encoding="utf-8", newline="") as f:
        for reason, texts in rejected.items():
            for t in texts:
                f.write(f"===== {reason} =====\n{t}\n\n")
    print(f"[{name}] termine : {totals['train'] / 1e9:.3f} G tokens train, "
          f"{totals['val'] / 1e6:.1f} M val\n  {dict(stats)}")


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        prepare(name)

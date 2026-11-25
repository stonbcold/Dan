import heapq
import json
from collections import Counter

import regex as re


SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<search>",
    "</search>",
    "<|pad|>",
]

VOCAB_SIZE = 49152


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}
        self._cache = {}

    def train(self, text, vocab_size=VOCAB_SIZE, verbose=True):
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

        special_pat = "|".join(re.escape(t) for t in SPECIAL_TOKENS)
        chunks = Counter()
        for part in re.split(special_pat, text):
            chunks.update(re.findall(SPLIT_PATTERN, part))
        if verbose:
            print(f"{sum(chunks.values()):,} morceaux, {len(chunks):,} uniques")

        words = [list(c.encode("utf-8")) for c in chunks]
        freqs = list(chunks.values())

        pair_counts = Counter()
        where = {}
        for idx, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where.setdefault(pair, set()).add(idx)

        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)

        for i in range(n_merges):
            best = None
            while heap:
                neg_c, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_c and -neg_c > 0:
                    best = pair
                    break
            if best is None:
                if verbose:
                    print(f"Plus aucune paire a fusionner apres {i} fusions")
                n_merges = i
                break

            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            changed = set()
            for idx in where.pop(best, ()):
                w, f = words[idx], freqs[idx]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    changed.add(pair)
                w = list(self._merge_word(w, best, new_id))
                words[idx] = w
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    where.setdefault(pair, set()).add(idx)
                    changed.add(pair)
            pair_counts.pop(best, None)
            for pair in changed:
                c = pair_counts.get(pair, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pair))

            if verbose and (i + 1) % 1000 == 0:
                tok = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"fusion {i + 1}/{n_merges} : {tok!r}")

        next_id = 256 + n_merges
        for k, tok in enumerate(SPECIAL_TOKENS):
            self.special[tok] = next_id + k
            self.vocab[next_id + k] = tok.encode("utf-8")

        self._cache = {}

    @staticmethod
    def _count_pairs(words):
        pairs = Counter()
        for ids, count in words.items():
            for j in range(len(ids) - 1):
                pairs[(ids[j], ids[j + 1])] += count
        return pairs

    @staticmethod
    def _merge_word(ids, pair, new_id):
        out = []
        j = 0
        while j < len(ids):
            if j + 1 < len(ids) and (ids[j], ids[j + 1]) == pair:
                out.append(new_id)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        return tuple(out)

    def encode(self, text, allowed_special=True):
        ids = []

        if allowed_special and self.special:
            toks = sorted(self.special, key=len, reverse=True)
            special_pat = "(" + "|".join(re.escape(t) for t in toks) + ")"
            parts = re.split(special_pat, text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if allowed_special and part in self.special:
                ids.append(self.special[part])
                continue
            for chunk in re.findall(SPLIT_PATTERN, part):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        ids = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge_word(ids, pair, self.merges[pair])

        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path="dan_tokenizer.json"):
        merges = sorted(([a, b, new_id] for (a, b), new_id in self.merges.items()),
                        key=lambda m: m[2])
        data = {
            "version": 1,
            "merges": merges,
            "special": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path="dan_tokenizer.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}

        for k, (a, b, new_id) in enumerate(data["merges"]):
            if new_id != 256 + k:
                raise ValueError(f"merge {k} : id {new_id} attendu {256 + k}")
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        self.special = data["special"]
        for tok, i in self.special.items():
            self.vocab[i] = tok.encode("utf-8")

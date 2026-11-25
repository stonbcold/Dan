import time
from pathlib import Path

from bpe import VOCAB_SIZE, BPETokenizer

SAMPLE = Path("data/sample/sample.txt")
OUT = "dan_tokenizer.json"
HOLDOUT = 2 * 1024 * 1024

text = SAMPLE.read_text(encoding="utf-8")
cut = text.rfind("<|endoftext|>", 0, len(text) - HOLDOUT) + len("<|endoftext|>")
train_text, test_text = text[:cut], text[cut:]
print(f"Entrainement : {len(train_text) / 1e6:.1f} M caracteres, test : {len(test_text) / 1e6:.1f} M")

tok = BPETokenizer()
t0 = time.time()
tok.train(train_text, vocab_size=VOCAB_SIZE, verbose=True)
print(f"Entrainement termine en {(time.time() - t0) / 60:.1f} min")

tok.save(OUT)
tok2 = BPETokenizer()
tok2.load(OUT)

assert len(tok2.vocab) == VOCAB_SIZE, len(tok2.vocab)
assert max(tok2.special.values()) == VOCAB_SIZE - 1

t0 = time.time()
ids = tok2.encode(test_text, allowed_special=False)
dt = time.time() - t0
assert tok2.decode(ids) == test_text

n_words = len(test_text.split())
n_bytes = len(test_text.encode("utf-8"))
print(f"\nSur le texte de test (jamais vu) :")
print(f"  {len(ids) / n_words:.2f} tokens par mot")
print(f"  {n_bytes / len(ids):.2f} octets par token")
print(f"  vitesse d'encodage : {len(ids) / dt:,.0f} tokens/s")

examples = [
    "Hello! How are you doing today?",
    "The mitochondria is the powerhouse of the cell.",
    "If x + 12 = 2048, then x = 2036.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    "<|im_start|>user\nWhat is 7 * 8?<|im_end|>\n<|im_start|>assistant\n56<|im_end|>",
]
print("\nExemples de decoupage :")
for s in examples:
    pieces = [tok2.decode([i]) for i in tok2.encode(s)]
    print(f"  {len(pieces):3d} tokens | {pieces}")

longest = sorted(range(256, len(tok2.vocab) - len(tok2.special)),
                 key=lambda i: len(tok2.vocab[i]), reverse=True)[:15]
print("\nTokens les plus longs :")
print("  ", [tok2.decode([i]) for i in longest])

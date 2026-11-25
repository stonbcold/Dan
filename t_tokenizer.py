import os
import tempfile

from bpe import BPETokenizer as T

assert T._merge_word((1, 2, 3, 1, 2), (1, 2), 99) == (99, 3, 99)
assert T._merge_word((1, 1, 1), (1, 1), 99) == (99, 1)
assert T._merge_word((5,), (1, 2), 99) == (5,)
assert T._merge_word((), (1, 2), 99) == ()

c =T._count_pairs({(1, 2, 3): 10, (1, 2): 5})
assert c[(1, 2)] == 15
assert c[(2, 3)] == 10
assert c[(1, 3)] == 0

TRAIN_TEXT = (
    "The cat sat on the mat. The dog sat on the log. " * 20
    + "def add(a, b):\n    return a + b\n" * 10
    + "2 + 2 = 4, 12 * 3 = 36, x^2 + 1 = 0. " * 10
    + "<|endoftext|>"
    + "Hello, how are you? I'm fine, thanks! " * 10
)
tok = T()
tok.train(TRAIN_TEXT, vocab_size=400, verbose=False)

samples = [
    "The cat sat on the mat.",
    "def f(x):\n    return x * 2\n",
    "Solve 3x + 5 = 20, so x = 5.",
    "Café, naïve, résumé, Ångström",
    "Emoji test \U0001F600\U0001F680 and 日本語",
    "multiple   spaces\n\n\ttabs",
    "",
]
for s in samples:
    assert tok.decode(tok.encode(s)) == s, f"aller-retour rate : {s!r}"

ids = tok.encode("2048")
assert len(ids) == 4, ids
assert [tok.decode([i]) for i in ids] == ["2", "0", "4", "8"]

for name, i in tok.special.items():
    assert tok.encode(name) == [i], name
ids = tok.encode("<|im_start|>user\nHi<|im_end|>")
assert ids[0] == tok.special["<|im_start|>"]
assert ids[-1] == tok.special["<|im_end|>"]
assert tok.decode(ids) == "<|im_start|>user\nHi<|im_end|>"

ids = tok.encode("<|im_end|>", allowed_special=False)
assert tok.special["<|im_end|>"] not in ids
assert len(ids) > 1
assert tok.decode(ids) == "<|im_end|>"

path = os.path.join(tempfile.gettempdir(), "dan_tok_test.json")
tok.save(path)
tok2 = T()
tok2.load(path)
os.remove(path)
assert tok2.merges == tok.merges
assert tok2.vocab == tok.vocab
assert tok2.special == tok.special
for s in samples:
    assert tok2.encode(s) == tok.encode(s)

print("OK")

from pathlib import Path

from datasets import load_dataset

OUT = Path("data/sample/sample.txt")
TOTAL = 100 * 1024 * 1024
EOT = "<|endoftext|>"


def fineweb():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for ex in ds.shuffle(seed=0, buffer_size=10_000):
        if ex["int_score"] >= 3:
            yield ex["text"]


def smoltalk():
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    for ex in ds.shuffle(seed=0, buffer_size=10_000):
        yield "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in ex["messages"])


def finemath():
    ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    for ex in ds.shuffle(seed=0, buffer_size=10_000):
        yield ex["text"]


def python_code():
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for ex in ds.shuffle(seed=0, buffer_size=10_000):
        yield ex["content"]


SOURCES = [
    ("fineweb-edu", fineweb, 0.60),
    ("smoltalk", smoltalk, 0.20),
    ("finemath", finemath, 0.10),
    ("python", python_code, 0.10),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as f:
    for name, gen, share in SOURCES:
        budget = int(TOTAL * share)
        written = n = 0
        for text in gen():
            text = text.strip()
            if not text:
                continue
            doc = text + EOT
            f.write(doc)
            written += len(doc.encode("utf-8"))
            n += 1
            if written >= budget:
                break
        print(f"{name}: {n:,} documents, {written / 1e6:.1f} Mo")

print(f"Total : {OUT.stat().st_size / 1e6:.1f} Mo -> {OUT}")

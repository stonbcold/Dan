from pathlib import Path

import regex as re

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
RAW = DATA / "raw"
TOKENIZED = DATA / "tokenized"
DECONTAM = DATA / "decontam_ngrams.pkl"

SOURCES = {
    "fineweb": [("HuggingFaceFW/fineweb-edu", "sample/10BT/", 6.0e9)],
    "finemath": [("HuggingFaceTB/finemath", "finemath-4plus/", 2.5e9)],
    "code_translated": [("LLM360/MegaMath", "megamath-translated-code/", 0.75e9)],
    "code_textblock": [("LLM360/MegaMath", "megamath-text-code-block/", 0.75e9)],
}

RAW_DIRS = {"code_translated": "code", "code_textblock": "code"}


def raw_dir(name):
    return RAW / RAW_DIRS.get(name, name)

N = 13
WORD = re.compile(r"\w+")


def words(text):
    return WORD.findall(text.lower())


def iter_ngrams(w):
    for i in range(len(w) - N + 1):
        yield i, " ".join(w[i:i + N])

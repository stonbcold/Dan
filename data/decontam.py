import pickle

from datasets import load_dataset

from common import DECONTAM, iter_ngrams, words

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]

texts = {}

ds = load_dataset("openai/gsm8k", "main", split="test")
texts["gsm8k"] = [ex["question"] + "\n" + ex["answer"] for ex in ds]

texts["math"] = []
for sub in MATH_SUBJECTS:
    ds = load_dataset("EleutherAI/hendrycks_math", sub, split="test")
    texts["math"] += [ex["problem"] + "\n" + ex["solution"] for ex in ds]

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
texts["math500"] = [ex["problem"] + "\n" + ex["solution"] for ex in ds]

ds = load_dataset("openai/openai_humaneval", split="test")
texts["humaneval"] = [ex["prompt"] + ex["canonical_solution"] + "\n" + ex["test"] for ex in ds]


def informative(g):
    w = g.split(" ")
    return len(set(w)) >= 7 and sum(any(c.isalpha() for c in x) for x in w) >= 4


ngrams = set()
for name, docs in texts.items():
    before = len(ngrams)
    for t in docs:
        ngrams.update(g for _, g in iter_ngrams(words(t)) if informative(g))
    print(f"{name}: {len(docs)} exemples, {len(ngrams) - before:,} nouveaux 13-grammes")

first = {g.split(" ", 1)[0] for g in ngrams}

with open(DECONTAM, "wb") as f:
    pickle.dump({"ngrams": ngrams, "first": first}, f)
print(f"Total : {len(ngrams):,} 13-grammes, {len(first):,} premiers mots -> {DECONTAM}")

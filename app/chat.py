import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bpe import BPETokenizer
from eval.honesty import is_refusal
from model.model import load_dan
from sft.chat import SYSTEM, encode_chat

MEMORY = Path(__file__).resolve().parent / "memory.json"
WIKI = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "DanChat/0.1 (student project)"}


STOPWORDS = set("who what when where which why how is are was were did does do the a an of in on "
                "to for by with whom whose tall old big long many far high deep much".split())
PERSONAL = re.compile(r"\b(my|me|mine|i|i'm|am i|do i)\b", re.I)
FACTUAL = re.compile(r"^(who|when|where|which|in which|in what|what (is|was|are|were) the|"
                     r"how (tall|old|big|long|many|far|high|deep))\b", re.I)
MATH = re.compile(r"\d\s*[-+*/x^=]\s*\d")


def clean_query(q):
    return " ".join(w for w in re.findall(r"[\w'-]+", q) if w.lower() not in STOPWORDS)


def wiki_get(params):
    url = WIKI + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=10) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def search(query, max_chars=600):
    try:
        hits = wiki_get({"action": "query", "list": "search", "srsearch": clean_query(query), "srlimit": 1})
        hits = hits["query"]["search"]
        if not hits:
            return None, None
        title = hits[0]["title"]
        pages = wiki_get({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                          "redirects": 1, "titles": title})["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "").strip()
        return title, text[:max_chars].rsplit(".", 1)[0] + "." if text else None
    except Exception as e:
        print(f"  [recherche impossible : {e}]")
        return None, None


def load_memory():
    try:
        return json.loads(MEMORY.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_memory(facts):
    MEMORY.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")


def reading(text, question, about_me=False):
    what = "facts about me" if about_me else "text"
    label = "Facts" if about_me else "Text"
    return (f"Read the following {what} and answer the question.\n\n{label}: {text}\n\n"
            f"Question: {question}\nAnswer in one sentence.")


def build_ids(tok, history, max_len):
    msgs = list(history)
    while True:
        ids, _ = encode_chat(tok, msgs, generation=True, system=SYSTEM)
        if len(ids) <= max_len or len(msgs) <= 1:
            return ids[-max_len:]
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]


def reply(model, tok, ids, args, show=True, greedy=False):
    stops = (tok.special["<|endoftext|>"], tok.special["<|im_end|>"])
    idx = torch.tensor([ids], device="cuda")
    out, printed = [], ""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in model.stream(idx, max_new_tokens=args.max_tokens, temperature=0.0 if greedy else args.temperature,
                              top_k=50, repetition_penalty=1.0 if greedy else args.rep, stop_ids=stops):
            if t in stops:
                break
            out.append(t)
            text = tok.decode(out)
            if show and not text.endswith("�"):
                print(text[len(printed):], end="", flush=True)
                printed = text
    if show:
        print()
    return tok.decode(out).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/sft/final.pt")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--rep", type=float, default=1.1)
    p.add_argument("--max_tokens", type=int, default=300)
    p.add_argument("--no_search", action="store_true")
    args = p.parse_args()

    model, _ = load_dan(ROOT / args.ckpt)
    tok = BPETokenizer()
    tok.load(ROOT / "dan_tokenizer.json")
    max_len = model.config.max_seq_len - args.max_tokens
    facts = load_memory()
    history = []
    print("Dan est pret. Commandes : remember <fait>, /memory, /forget, /search <requete>, /new, /quit")

    while True:
        try:
            user = input("\nToi > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        low = user.lower()
        if low in ("/quit", "/exit"):
            break
        if low == "/new":
            history = []
            print("(nouvelle conversation)")
            continue
        if low == "/memory":
            print("\n".join(f"- {f}" for f in facts) or "(memoire vide)")
            continue
        if low == "/forget":
            facts = []
            save_memory(facts)
            print("(memoire effacee)")
            continue
        m = re.match(r"^(?:/remember|remember(?: that)?)\s+(.+)$", user, re.I)
        if m:
            facts.append(m.group(1).strip().rstrip("."))
            save_memory(facts)
            print(f"(retenu : {facts[-1]})")
            continue

        question = user[8:].strip() if low.startswith("/search ") else user
        forced = low.startswith("/search ") or (
            not args.no_search and bool(FACTUAL.search(question)) and not MATH.search(question)
            and not PERSONAL.search(question))
        about_me = bool(facts) and bool(PERSONAL.search(question))
        if about_me:
            facts_text = ". ".join(f[0].upper() + f[1:] for f in facts) + "."
        history.append({"role": "user", "content": question})

        if forced:
            answer = None
        elif about_me:
            print("Dan > ", end="", flush=True)
            msgs = [{"role": "user", "content": reading(facts_text, question, about_me=True)}]
            answer = reply(model, tok, build_ids(tok, msgs, max_len), args, greedy=True)
        else:
            print("Dan > ", end="", flush=True)
            answer = reply(model, tok, build_ids(tok, history, max_len), args)

        if forced or (answer is not None and is_refusal(answer) and not args.no_search):
            title, text = search(question)
            if text:
                print(f"  [recherche Wikipedia : {title}]")
                print("Dan > ", end="", flush=True)
                msgs = [{"role": "user", "content": reading(text, question)}]
                answer = reply(model, tok, build_ids(tok, msgs, max_len), args, greedy=True)
            elif answer is None:
                print("Dan > ", end="", flush=True)
                answer = reply(model, tok, build_ids(tok, history, max_len), args)

        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        sys.stdout.flush()
        import os
        os._exit(0)

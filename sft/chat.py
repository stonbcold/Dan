SYSTEM = "You are Dan, a small helpful assistant. Answer briefly and honestly."


def encode_chat(tok, messages, generation=False, system=SYSTEM):
    if system and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": system}] + list(messages)
    start, end = tok.special["<|im_start|>"], tok.special["<|im_end|>"]
    newline = tok.encode("\n", allowed_special=False)
    ids, mask = [], []

    def add(tokens, train):
        ids.extend(tokens)
        mask.extend([train] * len(tokens))

    for m in messages:
        train = int(m["role"] == "assistant")
        add([start] + tok.encode(m["role"] + "\n", allowed_special=False), 0)
        add(tok.encode(m["content"].strip(), allowed_special=False), train)
        add([end], train)
        add(newline, 0)
    if generation:
        add([start] + tok.encode("assistant\n", allowed_special=False), 0)
    return ids, mask

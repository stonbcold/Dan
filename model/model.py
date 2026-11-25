import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DanConfig:
    vocab_size: int = 49152
    d_model: int = 640
    n_layers: int = 30
    n_heads: int = 10
    n_kv_heads: int = 5
    ffn_dim: int = 1728
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True


CONFIGS = {
    "dan": DanConfig(),
    "mini": DanConfig(d_model=192, n_layers=6, n_heads=6, n_kv_heads=3, ffn_dim=512, max_seq_len=512),
}


def rope_cache(head_dim, seq_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads = c.n_heads
        self.n_kv_heads = c.n_kv_heads
        self.head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(c.n_heads * self.head_dim, c.d_model, bias=False)
        if c.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=c.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=c.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.up = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.ffn = SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.ffn(self.ffn_norm(x))


class Dan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        cos, sin = rope_cache(c.d_model // c.n_heads, c.max_seq_len, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * c.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.o.weight", "ffn.down.weight")):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache=None, start_pos=0):
        T = idx.shape[1]
        cos = self.cos[start_pos:start_pos + T]
        sin = self.sin[start_pos:start_pos + T]
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if cache is None else cache[i])
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def stream(self, idx, max_new_tokens=256, temperature=0.4, top_k=50,
               repetition_penalty=1.1, stop_ids=()):
        cache = [dict() for _ in self.blocks]
        idx = idx[:, -self.config.max_seq_len:]
        logits, _ = self(idx, cache=cache)
        pos = idx.shape[1]
        seen = idx
        for _ in range(max_new_tokens):
            scores = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                prev = scores.gather(1, seen)
                prev = torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty)
                scores.scatter_(1, seen, prev)
            if temperature <= 0:
                nxt = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    kth = torch.topk(scores, min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(scores, dim=-1), 1)
            token = nxt.item()
            yield token
            if token in stop_ids or pos >= self.config.max_seq_len:
                return
            seen = torch.cat([seen, nxt], dim=1)
            logits, _ = self(nxt, cache=cache, start_pos=pos)
            pos += 1

    def generate(self, idx, **kwargs):
        return list(self.stream(idx, **kwargs))


def load_dan(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Dan(DanConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def config_dict(config):
    return asdict(config)

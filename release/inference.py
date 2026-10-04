"""
inference.py — miniLLM v2.2-49M : script d'inférence autonome.

Ne dépend que de ce dépôt : config.json, tokenizer.json,
model.safetensors (ou best.pt) et ce fichier. Aucune dépendance au
code d'entraînement.

Usage :
    python inference.py --weights model.safetensors --config config.json \
        --tokenizer tokenizer.json --prompt "Quelle est la capitale du Cameroun ?"

    python inference.py --weights model.safetensors --config config.json \
        --tokenizer tokenizer.json --mode chat
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# ══════════════════════════════════════════════════════════════════════════
#  Config (lue depuis config.json — voir ce fichier pour les valeurs)
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    d_model: int
    n_heads: int
    kv_heads: int
    ffn_hidden: int
    max_seq_len: int
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    qk_norm: bool = True
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        names = {"vocab_size", "n_layers", "d_model", "n_heads", "kv_heads", "ffn_hidden",
                 "max_seq_len", "dropout", "rope_theta", "tie_embeddings", "qk_norm", "norm_eps"}
        return cls(**{k: v for k, v in d.items() if k in names})


# ══════════════════════════════════════════════════════════════════════════
#  Architecture (identique à celle utilisée à l'entraînement)
# ══════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight.to(x.dtype)


def build_rope_tables(head_dim, max_seq_len, theta, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    x32 = x.float()
    half = x.shape[-1] // 2
    rot = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
    out = x32 * cos[None, :, None, :] + rot * sin[None, :, None, :]
    return out.to(x.dtype)


class KVCache:
    def __init__(self, cfg: ModelConfig, batch_size, max_len, device, dtype):
        shape = (cfg.n_layers, batch_size, cfg.kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def layer(self, i):
        return self.k[i], self.v[i]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.kv_heads, self.hd = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.kv_heads
        self.dropout = cfg.dropout
        d = cfg.d_model
        self.wq = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d, self.kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(d, self.kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin, cache=None, start_pos: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd)
        k = self.wk(x).view(B, T, self.kv_heads, self.hd)
        v = self.wv(x).view(B, T, self.kv_heads, self.hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        mask = None
        if cache is not None:
            k_buf, v_buf = cache
            k_buf[:, :, start_pos:start_pos + T] = k
            v_buf[:, :, start_pos:start_pos + T] = v
            S = start_pos + T
            k = k_buf[:, :, :S].to(q.dtype)
            v = v_buf[:, :, :S].to(q.dtype)
            if T > 1 and start_pos > 0:
                i = torch.arange(T, device=x.device)[:, None] + start_pos
                j = torch.arange(S, device=x.device)[None, :]
                mask = j <= i
        is_causal = (mask is None) and (T > 1)

        if self.n_rep > 1:
            S = k.shape[2]
            k = k[:, :, None].expand(B, self.kv_heads, self.n_rep, S, self.hd).reshape(B, self.n_heads, S, self.hd)
            v = v[:, :, None].expand(B, self.kv_heads, self.n_rep, S, self.hd).reshape(B, self.n_heads, S, self.hd)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal,
        )
        return self.wo(out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, start_pos: int = 0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, start_pos)
        return x + self.ffn(self.ffn_norm(x))


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.refresh_rope()

    def refresh_rope(self):
        dev = self.tok_emb.weight.device
        cos, sin = build_rope_tables(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta, dev)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def to_inference(self, device, dtype):
        self.to(device=device, dtype=dtype)
        self.refresh_rope()
        return self.eval()

    def forward(self, idx, cache: Optional[KVCache] = None, start_pos: int = 0, last_only: bool = False):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache.layer(i) if cache is not None else None, start_pos)
        x = self.norm_f(x)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x)


def load_model(weights_path: str, config_path: str, device: str = "auto", dtype: str = "auto"):
    cfg = ModelConfig.from_json(config_path)
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else (device if device != "auto" else "cpu"))
    if dev.type == "cuda" and dtype == "auto":
        dt = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    elif dtype == "auto":
        dt = torch.float32
    else:
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    model = MiniLLM(cfg)
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to_inference(dev, dt)
    return model


# ══════════════════════════════════════════════════════════════════════════
#  Tokenizer (wrapper minimal autour de tokenizer.json)
# ══════════════════════════════════════════════════════════════════════════
EOT, SYSTEM, USER, ASSISTANT, END = "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"


class MiniTokenizer:
    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)
        self.eot_id = self.tok.token_to_id(EOT)
        self.system_id = self.tok.token_to_id(SYSTEM)
        self.user_id = self.tok.token_to_id(USER)
        self.assistant_id = self.tok.token_to_id(ASSISTANT)
        self.end_id = self.tok.token_to_id(END)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @staticmethod
    def sanitize(text: str) -> str:
        return text.replace("<|", "< |")

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(self.sanitize(text), add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def encode_chat(self, messages, add_generation_prompt: bool = False) -> List[int]:
        role_ids = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        ids: List[int] = []
        for m in messages:
            ids.append(role_ids[m["role"]])
            ids += self.encode(m["content"].strip())
            ids.append(self.end_id)
        if add_generation_prompt:
            ids.append(self.assistant_id)
        return ids

    def stop_ids(self) -> Set[int]:
        return {self.end_id, self.eot_id}

    def forbidden_in_answer(self) -> List[int]:
        return [self.system_id, self.user_id, self.assistant_id]


# ══════════════════════════════════════════════════════════════════════════
#  Génération (KV-cache, anti-répétition)
# ══════════════════════════════════════════════════════════════════════════
def _banned_by_ngram(generated: List[int], n: int) -> Set[int]:
    if n < 2 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


def sample_token(logits, generated, *, temperature, top_k, top_p, min_p, repetition_penalty,
                  no_repeat_ngram, forbidden, valid_vocab, rng):
    logits = logits.float().clone()
    if valid_vocab < logits.numel():
        logits[valid_vocab:] = -float("inf")
    if forbidden:
        logits[list(forbidden)] = -float("inf")
    if repetition_penalty != 1.0 and generated:
        ids = torch.tensor(sorted(set(generated[-256:])), device=logits.device)
        score = logits[ids]
        logits[ids] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    if no_repeat_ngram >= 2:
        banned = _banned_by_ngram(generated, no_repeat_ngram)
        if banned:
            logits[list(banned)] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if min_p > 0:
        probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[(cum - sp) > top_p * cum[-1]] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=rng))


@torch.inference_mode()
def stream_tokens(model, prompt_ids, *, max_new_tokens=128, temperature=0.7, top_k=40, top_p=0.9,
                   min_p=0.0, repetition_penalty=1.1, no_repeat_ngram=3, stop_ids=frozenset(),
                   forbidden=(), valid_vocab=None, seed=None) -> Iterator[int]:
    cfg = model.cfg
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(prompt_ids) >= cfg.max_seq_len:
        prompt_ids = prompt_ids[-(cfg.max_seq_len - 1):]
    budget = min(max_new_tokens, cfg.max_seq_len - len(prompt_ids))
    rng = None
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    valid_vocab = valid_vocab or cfg.vocab_size
    cache = KVCache(cfg, 1, cfg.max_seq_len, device, dtype)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = model(idx, cache=cache, start_pos=0, last_only=True)
    pos = len(prompt_ids)
    generated: List[int] = []
    for _ in range(budget):
        tok = sample_token(logits[0, -1], generated, temperature=temperature, top_k=top_k, top_p=top_p,
                            min_p=min_p, repetition_penalty=repetition_penalty, no_repeat_ngram=no_repeat_ngram,
                            forbidden=forbidden, valid_vocab=valid_vocab, rng=rng)
        if tok in stop_ids:
            return
        generated.append(tok)
        yield tok
        if pos >= cfg.max_seq_len:
            return
        logits = model(torch.tensor([[tok]], dtype=torch.long, device=device), cache=cache, start_pos=pos, last_only=True)
        pos += 1


def chat_reply(model, tok: MiniTokenizer, messages: List[dict], *, max_new_tokens=128, **kw) -> str:
    ids = tok.encode_chat(messages, add_generation_prompt=True)
    out = list(stream_tokens(model, ids, max_new_tokens=max_new_tokens, stop_ids=tok.stop_ids(),
                              forbidden=tok.forbidden_in_answer(), valid_vocab=tok.vocab_size, **kw))
    return tok.decode(out, skip_special=True).strip()


def main():
    p = argparse.ArgumentParser(description="miniLLM v2.2-49M — inférence autonome")
    p.add_argument("--weights", required=True, help="model.safetensors ou best.pt")
    p.add_argument("--config", default="config.json")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--mode", choices=["chat", "complete"], default="chat")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rep_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    a = p.parse_args()

    model = load_model(a.weights, a.config, a.device, a.dtype)
    tok = MiniTokenizer(a.tokenizer)
    print(f"miniLLM chargé | device={next(model.parameters()).device} | dtype={next(model.parameters()).dtype}")

    sampling = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                     repetition_penalty=a.rep_penalty, no_repeat_ngram=a.no_repeat_ngram)

    if a.prompt:
        print(chat_reply(model, tok, [{"role": "user", "content": a.prompt}], max_new_tokens=a.max_new, **sampling))
        return

    print("Chat — /quit pour sortir.\n")
    history: List[dict] = []
    while True:
        try:
            user = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("/quit", "/exit"):
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = chat_reply(model, tok, history, max_new_tokens=a.max_new, **sampling)
        print("miniLLM >", reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

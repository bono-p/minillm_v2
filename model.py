"""
model.py — MiniLLM v2 : Transformer décodeur (RMSNorm + RoPE + SwiGLU + GQA + QK-norm).

Changements par rapport à l'ancienne version :
  * RoPE en réel (cos/sin, rotate-half) au lieu de nombres complexes -> compatible torch.compile.
  * KV-cache pré-alloué : la génération passe de O(n²) à O(n) (5 à 20x plus rapide).
  * QK-norm : évite les explosions d'attention en fp16 (T4) et permet un LR plus haut.
  * forward(..., reduction="sum") + ignore_index=-1 : loss masquée réellement utilisée (SFT).
  * GQA sans copie inutile en entraînement quand kv_heads == n_heads.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Briques
# ══════════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight.to(x.dtype)


def build_rope_tables(head_dim: int, max_seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)              # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [T, D]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x : [B, T, H, D] ; cos/sin : [T, D] (déjà découpés aux bonnes positions)."""
    x32 = x.float()
    half = x.shape[-1] // 2
    rot = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
    out = x32 * cos[None, :, None, :] + rot * sin[None, :, None, :]
    return out.to(x.dtype)


class KVCache:
    """Cache K/V pré-alloué : [n_layers, B, kv_heads, max_len, head_dim]."""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype):
        shape = (cfg.n_layers, batch_size, cfg.kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len

    def layer(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B, H, T, D]

        mask = None
        if cache is not None:
            k_buf, v_buf = cache
            k_buf[:, :, start_pos:start_pos + T] = k
            v_buf[:, :, start_pos:start_pos + T] = v
            S = start_pos + T
            k = k_buf[:, :, :S].to(q.dtype)
            v = v_buf[:, :, :S].to(q.dtype)
            if T > 1 and start_pos > 0:            # chunk avec passé : masque causal décalé
                i = torch.arange(T, device=x.device)[:, None] + start_pos
                j = torch.arange(S, device=x.device)[None, :]
                mask = j <= i
        is_causal = (mask is None) and (T > 1)

        if self.n_rep > 1:                          # GQA : chaque KV head sert n_rep têtes de Q
            S = k.shape[2]
            k = k[:, :, None].expand(B, self.kv_heads, self.n_rep, S, self.hd).reshape(B, self.n_heads, S, self.hd)
            v = v[:, :, None].expand(B, self.kv_heads, self.n_rep, S, self.hd).reshape(B, self.n_heads, S, self.hd)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
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


# ══════════════════════════════════════════════════════════════════════════════
#  Modèle
# ══════════════════════════════════════════════════════════════════════════════
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
        self.apply(self._init_weights)
        # init "résiduelle" (GPT-2) : sorties de couches / sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def refresh_rope(self):
        """(Re)construit les tables RoPE en float32 sur le device courant (à appeler après .to(dtype))."""
        dev = self.tok_emb.weight.device
        cos, sin = build_rope_tables(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta, dev)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def to_inference(self, device, dtype):
        self.to(device=device, dtype=dtype)
        self.refresh_rope()          # cos/sin restent en float32
        return self.eval()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, idx, targets=None, *, cache: Optional[KVCache] = None, start_pos: int = 0,
                last_only: bool = False, reduction: str = "mean"):
        """
        idx     : [B, T] ids
        targets : [B, T] ids, -1 = ignoré (prompt / padding). Si fourni -> (logits, loss).
        cache   : KVCache (génération) ; start_pos = nb de tokens déjà dans le cache.
        """
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"Séquence trop longue : {start_pos + T} > max_seq_len={self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache.layer(i) if cache is not None else None, start_pos)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1, reduction=reduction)
            return logits, loss
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x), None

    # ── utilitaires ─────────────────────────────────────────────────────
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())      # les poids partagés ne sont comptés qu'une fois

    def configure_optimizers(self, lr, weight_decay, betas, device_type):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

# model.py — MiniLLM v2
#
# Architecture hybride optimisée :
#   RMSNorm + RoPE + SwiGLU + GQA + Flash Attention
#
# Sources d'inspiration :
#   - Gemma (Google)   : RMSNorm, SwiGLU, RoPE, GQA, no bias
#   - GPT-NeoX (EA)    : RoPE popularisé, Flash Attention
#   - nanochat (Karp.) : initialisation, bfloat16, style d'entraînement
#   - nanoGPT  (Karp.) : structure propre et hackable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelConfig


# ══════════════════════════════════════════════════════════════════════════════
#  RMSNorm
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Plus rapide que LayerNorm : pas de soustraction de la moyenne, pas de biais.
    Utilisé dans LLaMA, Gemma, Mistral, et pratiquement tous les LLMs modernes.

    x_norm = x / RMS(x) * weight    où RMS(x) = sqrt(mean(x²) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))   # γ (gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcul en float32 pour la stabilité numérique, même si x est en bf16
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
#  RoPE — Rotary Positional Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Pré-calcule les fréquences complexes pour RoPE.

    L'idée : encoder la position directement dans les vecteurs Q et K
    via une rotation dans l'espace complexe, sans paramètres appris.
    Permet une bien meilleure extrapolation en longueur que les embeddings absolus.

    Retourne : [max_seq_len, head_dim // 2] (nombres complexes)
    """
    assert head_dim % 2 == 0, "head_dim doit être pair pour RoPE"
    # Fréquences : [head_dim/2]
    freqs = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))
    # Positions : [max_seq_len]
    t = torch.arange(max_seq_len, device=device)
    # Produit externe → [max_seq_len, head_dim/2]
    freqs = torch.outer(t, freqs)
    # Convertir en nombres complexes (cos + i*sin)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    q: torch.Tensor,                    # [B, T, n_heads, head_dim]
    k: torch.Tensor,                    # [B, T, kv_heads, head_dim]
    freqs_cis: torch.Tensor             # [T, head_dim // 2]  (complexes)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applique RoPE à Q et K.

    La rotation dans l'espace complexe encode la position relative
    entre les tokens de façon naturelle (dot product Q·K est invariant
    par rapport à la différence de position, pas à la position absolue).
    """
    # Reshaper pour la multiplication complexe
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Adapter les dimensions : [1, T, 1, head_dim/2]
    freqs = freqs_cis[:q_.shape[1]].unsqueeze(0).unsqueeze(2)

    # Multiplication complexe = rotation
    q_out = torch.view_as_real(q_ * freqs).flatten(-2)
    k_out = torch.view_as_real(k_ * freqs).flatten(-2)

    return q_out.to(q.dtype), k_out.to(k.dtype)


# ══════════════════════════════════════════════════════════════════════════════
#  SwiGLU — Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """
    Feed-Forward Network avec activation SwiGLU.

    SwiGLU(x) = SiLU(x · W_gate) ⊙ (x · W_up)  → x · W_down

    Introduit par Noam Shazeer (2020), adopté par LLaMA, Gemma, Mistral.
    Surpasse empiriquement GELU sur presque tous les benchmarks NLP.

    Note : 3 matrices (gate, up, down) au lieu de 2 pour GELU-FFN,
           donc ffn_hidden ≈ 2/3 * 4 * d_model pour budget comparable.
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        # Pas de biais : standard dans les LLMs modernes (Gemma, LLaMA)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden,  d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating : SiLU (Swish) contrôle le flux d'information
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Grouped Query Attention (GQA)
# ══════════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    Attention multi-têtes avec support optionnel de GQA.

    ┌─────────────────────────────────────────────────────────┐
    │  MHA : kv_heads = n_heads   (standard, 50M)             │
    │  GQA : kv_heads < n_heads   (plus efficace, utile à 1B) │
    │  MQA : kv_heads = 1         (extrême, pour inférence)   │
    └─────────────────────────────────────────────────────────┘

    GQA réduit la taille du KV cache d'un facteur n_heads / kv_heads,
    ce qui permet des séquences plus longues ou des batches plus grands.
    Utilisé dans Gemma 2, LLaMA 3, Mistral, etc.

    Utilise F.scaled_dot_product_attention (Flash Attention fusionné
    depuis PyTorch 2.0 — pas besoin d'installer flash_attn séparément).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads   = cfg.n_heads
        self.kv_heads  = cfg.kv_heads
        self.head_dim  = cfg.head_dim
        self.n_rep     = cfg.n_heads // cfg.kv_heads   # facteur de répétition GQA
        self.dropout   = cfg.dropout

        # Pas de biais sur les projections (standard moderne)
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads  * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model,                  bias=False)

    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        freqs_cis: torch.Tensor,   # [T, head_dim // 2]
    ) -> torch.Tensor:
        B, T, D = x.shape

        # ── Projections Q, K, V ──────────────────────────────────────────
        q = self.wq(x).view(B, T, self.n_heads,  self.head_dim)
        k = self.wk(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.kv_heads, self.head_dim)

        # ── RoPE : encoder la position dans Q et K ───────────────────────
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # ── GQA : répéter K et V pour correspondre à n_heads ────────────
        if self.n_rep > 1:
            # [B, T, kv_heads, head_dim] → [B, T, n_heads, head_dim]
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # ── Format pour l'attention : [B, n_heads, T, head_dim] ─────────
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Flash Attention (causal) ──────────────────────────────────────
        # F.scaled_dot_product_attention utilise automatiquement Flash Attention
        # quand disponible (PyTorch 2.0+), sans installation supplémentaire.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True      # masque causal : token i ne voit pas i+1, i+2...
        )

        # ── Réassembler et projeter ───────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Bloc Transformer
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Bloc transformer avec pré-normalisation (Pre-RMSNorm).

    Architecture :
        x → RMSNorm → Attention → + résiduel
          → RMSNorm → SwiGLU   → + résiduel

    La normalisation AVANT l'attention (Pre-Norm) est plus stable
    que la Post-Norm originale de "Attention is All You Need".
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn      = GroupedQueryAttention(cfg)
        self.norm_ffn  = RMSNorm(cfg.d_model)
        self.ffn       = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Attention avec connexion résiduelle
        x = x + self.attn(self.norm_attn(x), freqs_cis)
        # FFN avec connexion résiduelle
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  MiniLLM — Modèle principal
# ══════════════════════════════════════════════════════════════════════════════

class MiniLLM(nn.Module):
    """
    MiniLLM v2 — Transformer decoder-only optimisé et scalable.

    Architecture moderne (2024) combinant les meilleures innovations :
    ┌────────────────────────────────────────────────────────────────┐
    │ Embedding (vocab_size × d_model)                               │
    │                                                                │
    │ × n_layers :                                                   │
    │   Pre-RMSNorm → GQA + RoPE + Flash Attention → résiduel       │
    │   Pre-RMSNorm → SwiGLU FFN                   → résiduel       │
    │                                                                │
    │ RMSNorm finale                                                 │
    │ Linear (d_model → vocab_size) [poids partagés avec embedding]  │
    └────────────────────────────────────────────────────────────────┘

    Scalable de 15M à 1B sans modifier le code — seule la config change.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding d'entrée ───────────────────────────────────────────
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Blocs transformer ────────────────────────────────────────────
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # ── Normalisation finale ──────────────────────────────────────────
        self.norm_f  = RMSNorm(cfg.d_model)

        # ── Tête de langage (projection → vocabulaire) ───────────────────
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Partager les poids embedding/sortie :
        # réduit les paramètres + améliore la qualité (Press & Wolf 2017)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # ── Fréquences RoPE pré-calculées (buffer, non entraînable) ──────
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False    # ne pas sauvegarder dans le checkpoint
        )

        # ── Initialisation des poids ──────────────────────────────────────
        self.apply(self._init_weights)
        # Initialisation spéciale pour les projections résiduelles (style GPT-2) :
        # diviser par sqrt(2 * n_layers) pour éviter l'explosion des résidus profonds
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module):
        """Initialisation normale standard pour linéaires et embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        idx:     torch.Tensor,                    # [B, T] — indices des tokens
        targets: Optional[torch.Tensor] = None,   # [B, T] — tokens cibles (training)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retourne (logits, loss) :
        - Si targets fourni  : logits [B, T, V] + cross-entropy loss
        - Si targets absent  : logits [B, 1, V] (dernier token uniquement, inférence)
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, (
            f"Séquence trop longue : {T} > max_seq_len={self.cfg.max_seq_len}"
        )

        # Token embeddings : [B, T, D]
        x = self.tok_emb(idx)

        # Fréquences RoPE pour les T premières positions
        freqs_cis = self.freqs_cis[:T]

        # Passer dans les couches transformer
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Normalisation finale
        x = self.norm_f(x)

        if targets is not None:
            # Mode entraînement : calculer la loss sur toute la séquence
            logits = self.lm_head(x)                          # [B, T, V]
            loss   = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1                                # ignorer le padding
            )
            return logits, loss
        else:
            # Mode inférence : seulement le dernier token (plus efficace)
            logits = self.lm_head(x[:, [-1], :])              # [B, 1, V]
            return logits, None

    # ── Utilitaires ───────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        """Nombre réel de paramètres (embeddings comptés une fois si tied)."""
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(
        self,
        lr:           float,
        min_lr:       float,
        weight_decay: float,
        beta1:        float,
        beta2:        float,
        device:       str,
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW avec séparation weight decay / no-decay.

        Règle standard :
        - Matrices de poids (dim >= 2) : weight decay appliqué
        - Biais, gains RMSNorm (dim < 2) : pas de weight decay
        """
        import inspect

        decay_params   = [p for n, p in self.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters()
                          if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # AdamW fusionné (CUDA uniquement) — plus rapide si disponible
        fused_ok  = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and (device == "cuda")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=1e-8,
            fused=use_fused if fused_ok else False,
        )

        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer : {n_decay:,} params avec weight_decay, "
              f"{n_nodecay:,} sans | fused={use_fused}")
        return optimizer

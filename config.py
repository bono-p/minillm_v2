# config.py — MiniLLM v2
# Configuration unique pour tout le projet.
# Un seul fichier à modifier pour changer la taille du modèle.

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    """
    Configuration de l'architecture MiniLLM v2.

    Composants :
    - RoPE      : positional embeddings sans paramètres appris
    - RMSNorm   : normalisation rapide (pas de soustraction de moyenne)
    - SwiGLU    : activation FFN haute performance
    - GQA       : Grouped Query Attention (kv_heads < n_heads réduit le KV cache)
    - Flash Att : via F.scaled_dot_product_attention (PyTorch 2.0+)
    """

    # ── Vocabulaire ────────────────────────────────────────────────────────
    vocab_size: int = 50_257          # cl100k_base (GPT-4) ; adapter si tokenizer custom

    # ── Architecture ───────────────────────────────────────────────────────
    n_layers:   int   = 10            # nombre de blocs transformer
    d_model:    int   = 512           # dimension des embeddings
    n_heads:    int   = 8             # têtes d'attention Query
    kv_heads:   int   = 8            # têtes K/V (= n_heads → MHA ; < n_heads → GQA)
    ffn_hidden: Optional[int] = None  # taille cachée du FFN (auto = 8/3 * d_model)
    max_seq_len: int  = 1024          # longueur de contexte maximale
    dropout:    float = 0.0           # désactivé par défaut (meilleur avec grandes données)

    # ── RoPE ───────────────────────────────────────────────────────────────
    rope_theta: float = 10_000.0      # base des fréquences RoPE (10k standard)

    # ── Misc ───────────────────────────────────────────────────────────────
    tie_embeddings: bool = True       # partager poids embedding entrée/sortie

    def __post_init__(self):
        # Calculer ffn_hidden automatiquement si non fourni
        if self.ffn_hidden is None:
            raw = int(8 / 3 * self.d_model)
            self.ffn_hidden = (raw + 63) // 64 * 64   # multiple de 64

        assert self.d_model % self.n_heads == 0,   "d_model doit être divisible par n_heads"
        assert self.n_heads  % self.kv_heads == 0, "n_heads doit être divisible par kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_params(self) -> int:
        """Estimation du nombre de paramètres (sans compter le head si tied)."""
        emb       = self.vocab_size * self.d_model
        attn      = (self.d_model * self.n_heads  * self.head_dim   # Q
                   + self.d_model * self.kv_heads * self.head_dim   # K
                   + self.d_model * self.kv_heads * self.head_dim   # V
                   + self.d_model * self.d_model)                    # O
        ffn       = (self.d_model * self.ffn_hidden * 2             # gate + up
                   + self.ffn_hidden * self.d_model)                 # down
        norms     = 2 * self.d_model                                # 2 RMSNorm/couche
        per_layer = attn + ffn + norms
        final     = self.d_model                                    # norm finale
        head      = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + final + head

    def __str__(self):
        n = self.count_params()
        unit = "M" if n >= 1_000_000 else "K"
        val  = n / 1_000_000 if n >= 1_000_000 else n / 1_000
        return (f"MiniLLM-{val:.0f}{unit} | "
                f"layers={self.n_layers} d_model={self.d_model} "
                f"heads={self.n_heads} kv={self.kv_heads} "
                f"ffn={self.ffn_hidden} ctx={self.max_seq_len}")


# ─── Presets prêts à l'emploi ─────────────────────────────────────────────────
#
# Tous les modèles partagent la même architecture, seule la taille change.
# Scalable de 15M → 1B sans changer une ligne du code modèle.
#
PRESETS: dict[str, ModelConfig] = {
    "15M":  ModelConfig(n_layers=6,  d_model=384,  n_heads=6,  kv_heads=6),
    "50M":  ModelConfig(n_layers=10, d_model=512,  n_heads=8,  kv_heads=8),
    "125M": ModelConfig(n_layers=12, d_model=768,  n_heads=12, kv_heads=12),
    "350M": ModelConfig(n_layers=24, d_model=1024, n_heads=16, kv_heads=8,  max_seq_len=2048),
    "1B":   ModelConfig(n_layers=20, d_model=2048, n_heads=16, kv_heads=4,  max_seq_len=4096),
}


@dataclass
class TrainConfig:
    """Configuration d'entraînement — tout régler ici."""

    # ── Modèle ─────────────────────────────────────────────────────────────
    model_size:    str   = "50M"          # clé dans PRESETS
    resume_from:   str   = ""             # chemin vers un checkpoint (vide = from scratch)

    # ── Données ────────────────────────────────────────────────────────────
    data_path:     str   = "data/train.bin"
    val_path:      str   = "data/val.bin"
    seq_len:       int   = 1024           # doit correspondre à ModelConfig.max_seq_len

    # ── Entraînement ───────────────────────────────────────────────────────
    max_iters:     int   = 50_000
    batch_size:    int   = 8              # batch par GPU
    grad_accum:    int   = 8             # accumulation → batch effectif = batch_size * grad_accum

    # ── Optimiseur ─────────────────────────────────────────────────────────
    lr:            float = 3e-4
    min_lr:        float = 3e-5           # lr minimale en fin de cosine decay
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    warmup_iters:  int   = 1_000         # itérations de warmup linéaire
    beta1:         float = 0.9
    beta2:         float = 0.95

    # ── Évaluation & logs ──────────────────────────────────────────────────
    eval_every:    int   = 500
    eval_iters:    int   = 50
    log_every:     int   = 10
    save_every:    int   = 2_000
    out_dir:       str   = "checkpoints"

    # ── Système ────────────────────────────────────────────────────────────
    device:  str  = "auto"      # "auto" → cuda si dispo, sinon mps, sinon cpu
    dtype:   str  = "auto"      # "auto" → bfloat16 si cuda, sinon float32
    compile: bool = True        # torch.compile (désactiver si erreur ou Windows)
    seed:    int  = 42


# ─── Ajout rolling checkpoint dans TrainConfig ────────────────────────────────
# (ajouté après la définition de TrainConfig)
TrainConfig.rolling_keep = 2   # nombre de ckpt_XXXXXX.pt gardés simultanément

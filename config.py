# config.py — MiniLLM v2
# Configuration unique pour tout le projet.
# Un seul fichier à modifier pour changer la taille du modèle.

from dataclasses import dataclass
from typing import Optional


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

    vocab_size: int = 100_277
    n_layers: int = 10
    d_model: int = 512
    n_heads: int = 8
    kv_heads: int = 8
    ffn_hidden: Optional[int] = None
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.ffn_hidden is None:
            raw = int(8 / 3 * self.d_model)
            self.ffn_hidden = (raw + 63) // 64 * 64

        assert self.d_model % self.n_heads == 0, "d_model doit être divisible par n_heads"
        assert self.n_heads % self.kv_heads == 0, "n_heads doit être divisible par kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_params(self) -> int:
        emb = self.vocab_size * self.d_model
        attn = (
            self.d_model * self.n_heads * self.head_dim
            + self.d_model * self.kv_heads * self.head_dim
            + self.d_model * self.kv_heads * self.head_dim
            + self.d_model * self.d_model
        )
        ffn = (
            self.d_model * self.ffn_hidden * 2
            + self.ffn_hidden * self.d_model
        )
        norms = 2 * self.d_model
        per_layer = attn + ffn + norms
        final = self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + final + head

    def __str__(self):
        n = self.count_params()
        unit = "M" if n >= 1_000_000 else "K"
        val = n / 1_000_000 if n >= 1_000_000 else n / 1_000
        return (
            f"MiniLLM-{val:.0f}{unit} | "
            f"layers={self.n_layers} d_model={self.d_model} "
            f"heads={self.n_heads} kv={self.kv_heads} "
            f"ffn={self.ffn_hidden} ctx={self.max_seq_len}"
        )


PRESETS: dict[str, ModelConfig] = {
    "15M": ModelConfig(n_layers=6, d_model=384, n_heads=6, kv_heads=6),
    "50M": ModelConfig(n_layers=10, d_model=512, n_heads=8, kv_heads=8),
    "125M": ModelConfig(n_layers=12, d_model=768, n_heads=12, kv_heads=12),
    "350M": ModelConfig(n_layers=24, d_model=1024, n_heads=16, kv_heads=8, max_seq_len=2048),
    "1B": ModelConfig(n_layers=20, d_model=2048, n_heads=16, kv_heads=4, max_seq_len=4096),
}


@dataclass
class TrainConfig:
    """Configuration d'entraînement — tout régler ici."""

    model_size: str = "50M"
    resume_from: str = ""
    resume_last: bool = True
    mode: str = "pretrain"

    data_path: str = "data/train.bin"
    val_path: str = "data/val.bin"
    seq_len: int = 1024

    max_iters: int = 50_000
    batch_size: int = 8
    grad_accum: int = 8

    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 1_000
    beta1: float = 0.9
    beta2: float = 0.95

    eval_every: int = 500
    eval_iters: int = 50
    log_every: int = 10
    save_every: int = 2_000
    out_dir: str = "checkpoints"

    device: str = "auto"
    dtype: str = "auto"
    compile: bool = True
    seed: int = 42
    reset_iter: bool = False
    eval_on_zero: bool = False
    validation_fixed: bool = True
    randomize_train: bool = True

    ddp: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    rolling_keep: int = 2


TrainConfig.rolling_keep = 2

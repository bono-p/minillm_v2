"""
config.py — MiniLLM v2 : configuration unique (modèle + entraînement).

Points clés par rapport à l'ancienne version :
  * Les presets portent leur VRAI nombre de paramètres (calculé, jamais écrit à la main).
  * Vocabulaire par défaut = 32 000 (tokenizer BPE entraîné sur le corpus) au lieu de 100 277.
  * get_preset() renvoie une COPIE : plus de mutation du dictionnaire global.
  * Toute la config d'entraînement est un vrai dataclass (plus de `rolling_keep` greffé après coup).
  * Les checkpoints stockent la config sous forme de dict simple -> torch.load(weights_only=True).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, asdict, fields
from typing import Optional

DEFAULT_VOCAB_SIZE = 32_000


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# ══════════════════════════════════════════════════════════════════════════════
#  Modèle
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_layers: int = 10
    d_model: int = 512
    n_heads: int = 8
    kv_heads: int = 8                 # < n_heads => GQA
    ffn_hidden: Optional[int] = None  # None => ~8/3 * d_model arrondi à 64
    max_seq_len: int = 512
    dropout: float = 0.0              # dropout d'attention (0 pour le pré-entraînement)
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    qk_norm: bool = True              # RMSNorm sur Q et K : stabilise l'entraînement en fp16
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.ffn_hidden is None:
            self.ffn_hidden = round_up(int(8 / 3 * self.d_model), 64)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads")
        if self.n_heads % self.kv_heads != 0:
            raise ValueError("n_heads doit être divisible par kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim doit être pair (RoPE)")

    # ── propriétés dérivées ─────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_params(self) -> int:
        """Nombre exact de paramètres (vérifié par les tests contre le vrai modèle)."""
        d, hd = self.d_model, self.head_dim
        attn = d * (self.n_heads * hd) + 2 * d * (self.kv_heads * hd) + (self.n_heads * hd) * d
        ffn = 3 * d * self.ffn_hidden
        norms = 2 * d
        qk = 2 * hd if self.qk_norm else 0
        per_layer = attn + ffn + norms + qk
        emb = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return emb + head + self.n_layers * per_layer + d

    def count_non_embedding_params(self) -> int:
        return self.count_params() - self.vocab_size * self.d_model

    def size_name(self) -> str:
        n = self.count_params()
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        return f"{round(n / 1e6)}M" if n >= 1e6 else f"{round(n / 1e3)}K"

    # ── (dé)sérialisation ───────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Architectures (l'ancienne "15M/50M/125M/350M/1B", + un modèle minuscule pour les tests).
# Le NOM du preset est calculé à partir du nombre réel de paramètres (vocab 32k par défaut).
_PRESET_SPECS = [
    dict(n_layers=6,  d_model=256,  n_heads=4,  kv_heads=4,  max_seq_len=512),
    dict(n_layers=6,  d_model=384,  n_heads=6,  kv_heads=6,  max_seq_len=512),
    dict(n_layers=10, d_model=512,  n_heads=8,  kv_heads=8,  max_seq_len=512),
    dict(n_layers=12, d_model=768,  n_heads=12, kv_heads=12, max_seq_len=1024),
    dict(n_layers=24, d_model=1024, n_heads=16, kv_heads=8,  max_seq_len=2048),
    dict(n_layers=20, d_model=2048, n_heads=16, kv_heads=4,  max_seq_len=2048),
]


def _build_presets() -> dict:
    presets = {}
    for spec in _PRESET_SPECS:
        cfg = ModelConfig(**spec)
        presets[cfg.size_name()] = cfg
    return presets


PRESETS = _build_presets()
DEFAULT_PRESET = "49M"


def get_preset(name: str, **overrides) -> ModelConfig:
    """Renvoie une COPIE du preset `name` avec d'éventuels overrides (vocab_size, max_seq_len...)."""
    if name not in PRESETS:
        raise KeyError(f"Preset inconnu '{name}'. Disponibles : {list(PRESETS)}")
    return dataclasses.replace(PRESETS[name], **overrides)


# ══════════════════════════════════════════════════════════════════════════════
#  Entraînement (pré-entraînement ET fine-tuning : même dataclass)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    mode: str = "pretrain"          # "pretrain" | "sft"
    model_size: str = DEFAULT_PRESET
    data_dir: str = "data/pretrain"  # dossier avec train.bin/val.bin/meta.json (ou données SFT)
    out_dir: str = "checkpoints/pretrain"

    # reprise / initialisation
    resume: str = "auto"             # "auto" = dernier ckpt de out_dir ; "" = repartir de zéro ; ou un chemin
    init_from: str = ""              # poids seuls (ex. SFT depuis un pré-entraînement) ; optimiseur neuf

    # séquences / batch
    seq_len: int = 512
    batch_size: int = 16             # par GPU et par micro-pas
    grad_accum: int = 8

    # durée
    max_iters: int = 15_000          # pretrain (en SFT : calculé à partir de epochs si epochs > 0)
    epochs: int = 0                  # SFT

    # optimisation
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 500
    schedule: str = "cosine"         # "cosine" | "wsd" (warmup-stable-decay : idéal si la session peut être coupée)
    decay_frac: float = 0.2          # wsd : fraction finale en décroissance

    # évaluation / sauvegarde
    eval_every: int = 500
    eval_iters: int = 50             # nb de batchs de VAL, TOUJOURS les mêmes (val fixe)
    eval_at_start: bool = False
    save_every: int = 1000
    keep: int = 2                    # nb de checkpoints complets conservés
    log_every: int = 25
    patience: int = 0                # early stopping (en nombre d'évals sans amélioration), 0 = off
    max_minutes: float = 0.0         # budget temps : sauvegarde propre puis arrêt (sessions Kaggle/Colab limitées), 0 = off

    # matériel
    device: str = "auto"
    dtype: str = "auto"              # auto : bf16 si GPU Ampere+, sinon fp16 (jamais de bf16 émulé sur T4)
    compile: bool = False
    dropout: float = -1.0            # -1 = garder celui du checkpoint / preset
    seed: int = 1337

    @classmethod
    def for_mode(cls, mode: str) -> "TrainConfig":
        if mode == "pretrain":
            return cls(mode="pretrain")
        if mode == "sft":
            return cls(
                mode="sft", data_dir="data/sft", out_dir="checkpoints/sft",
                init_from="checkpoints/pretrain/best.pt", resume="auto",
                batch_size=16, grad_accum=2, max_iters=0, epochs=3,
                lr=2e-4, min_lr=2e-5, weight_decay=0.01, warmup_iters=50,
                eval_every=250, eval_iters=0, eval_at_start=True, save_every=250,
                log_every=10, patience=5,
            )
        raise ValueError(f"mode inconnu : {mode}")

    def to_dict(self) -> dict:
        return asdict(self)

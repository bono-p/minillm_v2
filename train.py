# train.py — MiniLLM v2
#
# Lancement :
#   python train.py                        # preset 50M, paramètres par défaut
#   python train.py --size 15M             # modèle plus petit
#   python train.py --size 125M --seq 2048 # modèle plus grand, contexte plus long
#   python train.py --resume checkpoints/best.pt   # reprendre un entraînement

import os
import sys
import math
import time
import argparse
import torch

from config import ModelConfig, TrainConfig, PRESETS
from model  import MiniLLM


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(pref: str, device: str) -> torch.dtype:
    if pref != "auto":
        return {"float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16":  torch.float16}[pref]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay avec warmup linéaire."""
    # Phase warmup : montée linéaire
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    # Après le decay : lr minimale
    if it >= cfg.max_iters:
        return cfg.min_lr
    # Cosine decay
    progress = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader minimal — lit un fichier .bin de tokens uint16
# ══════════════════════════════════════════════════════════════════════════════

class BinDataLoader:
    """
    Chargeur de données rapide depuis un fichier binaire de tokens.

    Le fichier .bin contient des tokens uint16 consécutifs (produit par prepare_data.py).
    Utilise numpy.memmap pour ne pas charger tout en RAM.
    """
    def __init__(self, path: str, batch_size: int, seq_len: int, device: str):
        import numpy as np
        assert os.path.exists(path), (
            f"Fichier de données introuvable : {path}\n"
            f"Lance d'abord : python prepare_data.py ton_corpus.txt"
        )
        self.data       = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.device     = device
        self.pos        = 0

        total_tokens = len(self.data)
        total_batches = total_tokens // (batch_size * seq_len)
        print(f"  → {total_tokens:,} tokens | ~{total_batches:,} batches")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np
        B, T = self.batch_size, self.seq_len
        needed = B * T + 1

        # Revenir au début si on a tout parcouru
        if self.pos + needed > len(self.data):
            self.pos = 0

        chunk = torch.from_numpy(
            self.data[self.pos : self.pos + needed].astype(np.int64)
        )
        self.pos += B * T

        x = chunk[:-1].view(B, T).to(self.device)   # entrée
        y = chunk[1:].view(B, T).to(self.device)    # cible (décalée d'un token)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
#  Boucle d'entraînement
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig):
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype  = resolve_dtype(cfg.dtype, device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  MiniLLM v2 — Entraînement")
    print(f"{'═'*60}")
    print(f"  Appareil : {device}  |  Dtype : {dtype}")

    # ── Modèle ─────────────────────────────────────────────────────────────
    mcfg             = PRESETS[cfg.model_size]
    mcfg.max_seq_len = cfg.seq_len

    model = MiniLLM(mcfg).to(device)
    print(f"  Modèle   : {mcfg}")
    print(f"  Params   : {model.n_params / 1e6:.2f}M")
    print(f"  Batch    : {cfg.batch_size} × {cfg.grad_accum} accum = "
          f"{cfg.batch_size * cfg.grad_accum} effectif")
    print(f"  Tokens/it: {cfg.batch_size * cfg.grad_accum * cfg.seq_len:,}")
    print(f"{'─'*60}")

    iter_start = 0

    # ── Reprendre un checkpoint ─────────────────────────────────────────────
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"  Reprise depuis : {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        # Supprimer le préfixe torch.compile si nécessaire
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        iter_start = ckpt.get("iter", 0)
        print(f"  Iteration de reprise : {iter_start}")

    # ── Compilation (accélère ~20-30% sur CUDA) ─────────────────────────────
    if cfg.compile and device == "cuda":
        print("  Compilation torch.compile... (première itération plus lente)")
        model = torch.compile(model)

    # ── Optimiseur ─────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        weight_decay=cfg.weight_decay,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        device=device,
    )

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ckpt = torch.load(cfg.resume_from, map_location=device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    # ── Données ────────────────────────────────────────────────────────────
    print(f"\n  Données train :")
    train_loader = BinDataLoader(cfg.data_path, cfg.batch_size, cfg.seq_len, device)
    print(f"  Données val   :")
    val_loader   = BinDataLoader(cfg.val_path,  cfg.batch_size, cfg.seq_len, device)

    # ── Contexte de précision mixte ────────────────────────────────────────
    if device in ("cuda", "mps"):
        ctx = torch.amp.autocast(device_type=device, dtype=dtype)
    else:
        ctx = torch.autocast(device_type="cpu", dtype=dtype, enabled=False)

    # Scaler uniquement pour float16 (pas nécessaire pour bfloat16)
    use_scaler = (dtype == torch.float16) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── Variables de suivi ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    t0            = time.time()
    tokens_seen   = iter_start * cfg.batch_size * cfg.grad_accum * cfg.seq_len

    print(f"\n{'═'*60}")
    print(f"  Début de l'entraînement — {cfg.max_iters:,} itérations")
    print(f"{'═'*60}\n")

    # ══════════════════════════════════════════════════════════════════════
    #  Boucle principale
    # ══════════════════════════════════════════════════════════════════════
    for it in range(iter_start, cfg.max_iters + 1):

        # ── Learning rate scheduling ────────────────────────────────────
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Évaluation périodique ──────────────────────────────────────
        if it % cfg.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(cfg.eval_iters):
                    xv, yv = val_loader.next_batch()
                    with ctx:
                        _, loss = model(xv, yv)
                    val_losses.append(loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            elapsed  = time.time() - t0
            tps      = (cfg.eval_every * cfg.batch_size
                        * cfg.grad_accum * cfg.seq_len) / max(elapsed, 1)

            print(f"iter {it:6d} | val_loss {val_loss:.4f} | "
                  f"lr {lr:.1e} | {tps/1000:.1f}k tok/s | "
                  f"elapsed {format_time(elapsed)}")

            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, mcfg, it, val_loss,
                                 os.path.join(cfg.out_dir, "best.pt"))
                print(f"  ✓ Nouveau meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

            model.train()
            t0 = time.time()

        if it == cfg.max_iters:
            break

        # ── Sauvegarde périodique ──────────────────────────────────────
        if it > 0 and it % cfg.save_every == 0:
            _save_checkpoint(model, optimizer, mcfg, it, None,
                             os.path.join(cfg.out_dir, f"ckpt_{it:06d}.pt"))

        # ══════════════════════════════════════════════════════════════
        #  Gradient accumulation
        #  Divise le batch effectif en micro-batches pour économiser la VRAM
        # ══════════════════════════════════════════════════════════════
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum):
            x, y = train_loader.next_batch()
            tokens_seen += x.numel()

            with ctx:
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum    # normaliser pour l'accumulation

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping (stabilise l'entraînement)
        if cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        # ── Log rapide ────────────────────────────────────────────────
        if it % cfg.log_every == 0 and it > 0:
            print(f"  it {it:5d} | loss {accum_loss:.4f} | "
                  f"lr {lr:.1e} | tokens {tokens_seen/1e6:.1f}M")

    # ── Fin de l'entraînement ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Entraînement terminé !")
    print(f"  Meilleure val_loss : {best_val_loss:.4f}")
    print(f"  Checkpoint final   : {cfg.out_dir}/best.pt")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Sauvegarde checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg:       ModelConfig,
    it:        int,
    val_loss:  float | None,
    path:      str,
):
    # Supprimer le préfixe torch.compile pour la portabilité
    raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = raw_model.state_dict()

    torch.save({
        "iter":      it,
        "model":     state_dict,
        "optimizer": optimizer.state_dict(),
        "val_loss":  val_loss,
        "config":    cfg,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(description="MiniLLM v2 — Entraînement")

    p.add_argument("--size",    default=cfg.model_size,  choices=list(PRESETS),
                   help="Taille du modèle (défaut: 50M)")
    p.add_argument("--data",    default=cfg.data_path,
                   help="Fichier train.bin")
    p.add_argument("--val",     default=cfg.val_path,
                   help="Fichier val.bin")
    p.add_argument("--seq",     type=int, default=cfg.seq_len,
                   help="Longueur de séquence")
    p.add_argument("--batch",   type=int, default=cfg.batch_size,
                   help="Taille du micro-batch")
    p.add_argument("--accum",   type=int, default=cfg.grad_accum,
                   help="Étapes d'accumulation de gradient")
    p.add_argument("--iters",   type=int, default=cfg.max_iters,
                   help="Nombre d'itérations")
    p.add_argument("--lr",      type=float, default=cfg.lr,
                   help="Learning rate max")
    p.add_argument("--out",     default=cfg.out_dir,
                   help="Dossier de sauvegarde")
    p.add_argument("--resume",  default=cfg.resume_from,
                   help="Chemin checkpoint pour reprendre")
    p.add_argument("--device",  default=cfg.device,
                   help="Appareil : auto | cuda | mps | cpu")
    p.add_argument("--no-compile", action="store_true",
                   help="Désactiver torch.compile")

    args = p.parse_args()

    cfg.model_size  = args.size
    cfg.data_path   = args.data
    cfg.val_path    = args.val
    cfg.seq_len     = args.seq
    cfg.batch_size  = args.batch
    cfg.grad_accum  = args.accum
    cfg.max_iters   = args.iters
    cfg.lr          = args.lr
    cfg.out_dir     = args.out
    cfg.resume_from = args.resume
    cfg.device      = args.device
    cfg.compile     = not args.no_compile

    return cfg


if __name__ == "__main__":
    train(parse_args())

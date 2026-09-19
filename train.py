# train.py — MiniLLM v2
#
# Lancement :
#   python train.py                        # preset 50M, paramètres par défaut
#   python train.py --size 15M             # modèle plus petit
#   python train.py --size 125M --seq 2048 # modèle plus grand
#   python train.py --resume checkpoints/best.pt   # reprendre un entraînement

import argparse
import math
import os
import random
import time
from typing import Any, Dict

import torch

from config import ModelConfig, TrainConfig, PRESETS
from model import MiniLLM


# ════════════════════════════════════════════════════════════════
#  Utilitaires
# ════════════════════════════════════════════════════════

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
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[pref]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay avec warmup linéaire."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def format_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.1f}h"


def strip_compiled_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


# ════════════════════════════════════════════════════════════════
#  DataLoader
#  NOTE : dtype int32 — cl100k_base a 100 277 tokens > 65 535 (max uint16)
# ════════════════════════════════════════════════════════════════


class BinDataLoader:
    """
    Chargeur de données depuis un fichier .bin de tokens int32.
    Utilise numpy.memmap pour ne pas charger tout en RAM.

    - randomize=True : sampling aléatoire par offsets valides
    - randomize=False : validation fixe / déterministe
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        seq_len: int,
        device: str,
        randomize: bool = True,
        seed: int = 42,
        start_offset: int = 0,
    ):
        import numpy as np

        assert os.path.exists(path), (
            f"Fichier de données introuvable : {path}\n"
            f"Lance d'abord la tokenisation (section 4 du notebook)."
        )

        self.data = np.memmap(path, dtype=np.int32, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.randomize = randomize
        self.window = batch_size * seq_len + 1
        self.start_offset = start_offset
        self.pos = start_offset
        self.rng = random.Random(seed)

        max_start = max(0, len(self.data) - self.window)
        self.max_start = max_start
        n_batches = max(1, len(self.data) // (batch_size * seq_len))
        print(f"  → {len(self.data):,} tokens | ~{n_batches:,} batches | sample={'random' if randomize else 'fixed'}")

    def next_batch(self) -> tuple:
        import numpy as np

        B, T = self.batch_size, self.seq_len
        needed = B * T + 1
        n_tokens = len(self.data)
        if needed > n_tokens:
            raise ValueError(
                f"Données insuffisantes : {n_tokens} tokens pour un batch de {needed} tokens "
                f"(batch={B}, seq={T})"
            )

        if self.randomize:
            max_start = max(0, n_tokens - needed)
            offset = self.rng.randint(0, max_start)
        else:
            if self.max_start <= 0:
                offset = 0
            else:
                offset = self.pos % (self.max_start + 1)
                self.pos = (self.pos + B * T) % (self.max_start + 1)

        chunk = torch.from_numpy(self.data[offset : offset + needed].astype(np.int64))
        x = chunk[:-1].view(B, T).to(self.device)
        y = chunk[1:].view(B, T).to(self.device)
        return x, y


# ════════════════════════════════════════════════════════════════
#  Checkpointing
# ════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    it: int,
    val_loss: float | None,
    best_val_loss: float | None,
    best_iter: int | None,
    mode: str,
    path: str,
):
    """Sauvegarde le checkpoint avec la compatibilité minimale nécessaire."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    payload = {
        "iter": it,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_iter": best_iter,
        "config": cfg,
        "mode": mode,
        "rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)


def _load_checkpoint(path: str, device: str):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def _rolling_cleanup(ckpt_dir: str, keep: int = 2):
    """Garde seulement les keep derniers checkpoints numérotés."""
    numbered = sorted(
        [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("ckpt_") and f.endswith(".pt")
        ]
    )
    removed = 0
    while len(numbered) > keep:
        oldest = os.path.join(ckpt_dir, numbered.pop(0))
        try:
            os.remove(oldest)
            print(f"  🗑️  Rolling — supprimé : {os.path.basename(oldest)}")
            removed += 1
        except OSError as e:
            print(f"  ⚠️  Impossible de supprimer {oldest} : {e}")
    return removed


# ════════════════════════════════════════════════════════════════
#  Boucle d'entraînement
# ═════���══════════════════════════════════════════════════════════


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "best.pt")
    last_path = os.path.join(cfg.out_dir, "last.pt")

    print(f"\n{'═'*60}")
    print(f"  MiniLLM v2 — Entraînement")
    print(f"{'═'*60}")
    print(f"  Appareil : {device}  |  Dtype : {dtype}  |  Mode : {cfg.mode}")

    mcfg = PRESETS[cfg.model_size]
    mcfg.max_seq_len = cfg.seq_len
    model = MiniLLM(mcfg).to(device)

    print(f"  Modèle   : {mcfg}")
    print(f"  Params   : {model.n_params/1e6:.2f}M")
    print(f"  Batch    : {cfg.batch_size} × {cfg.grad_accum} accum = {cfg.batch_size * cfg.grad_accum} effectif")
    print(f"  Rolling  : {cfg.rolling_keep} checkpoint(s) numérotés gardés")
    print(f"{'─'*60}")

    iter_start = 0
    best_val_loss = float("inf")
    best_iter = 0

    # ── Reprise propre ──────────────────────────────────────────────────────
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"  Reprise depuis : {cfg.resume_from}")
        ckpt = _load_checkpoint(cfg.resume_from, device)
        if ckpt is None:
            raise FileNotFoundError(f"Checkpoint introuvable : {cfg.resume_from}")
        state = strip_compiled_state(ckpt["model"])
        model.load_state_dict(state)

        raw_iter = ckpt.get("iter", 0)
        iter_start = 0 if getattr(cfg, "reset_iter", False) else raw_iter
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        best_iter = int(ckpt.get("best_iter", raw_iter))

        if ckpt.get("mode"):
            cfg.mode = ckpt["mode"]
        if ckpt.get("rng_state") is not None:
            torch.random.set_rng_state(ckpt["rng_state"])

        if getattr(cfg, "reset_iter", False):
            print(f"  Poids chargés depuis iter={raw_iter} → reset iter=0 (fine-tuning)")
        else:
            print(f"  Reprise depuis iter={iter_start} | best_val_loss={best_val_loss:.4f} | best_iter={best_iter}")

    # ── Compilation ──────────────────────────────────────────────────────────
    if cfg.compile and device == "cuda":
        print("  Compilation torch.compile...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        cfg.lr, cfg.min_lr, cfg.weight_decay, cfg.beta1, cfg.beta2, device
    )

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ckpt = _load_checkpoint(cfg.resume_from, device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    train_loader = BinDataLoader(cfg.data_path, cfg.batch_size, cfg.seq_len, device, randomize=cfg.randomize_train)
    val_loader = BinDataLoader(cfg.val_path, cfg.batch_size, cfg.seq_len, device, randomize=False, seed=cfg.seed + 1)

    if device in ("cuda", "mps"):
        ctx = torch.amp.autocast(device_type=device, dtype=dtype)
    else:
        ctx = torch.autocast(device_type="cpu", dtype=dtype, enabled=False)

    use_scaler = (dtype == torch.float16) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    t0 = time.time()
    tokens_seen = iter_start * cfg.batch_size * cfg.grad_accum * cfg.seq_len

    print(f"\n{'═'*60}")
    print(f"  Début — {cfg.max_iters:,} itérations")
    print(f"{'═'*60}\n")

    for it in range(iter_start, cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        do_eval = (it > 0 and it % cfg.eval_every == 0) or (cfg.eval_on_zero and it == 0)
        if do_eval:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(cfg.eval_iters):
                    xv, yv = val_loader.next_batch()
                    with ctx:
                        _, loss = model(xv, yv)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            elapsed = time.time() - t0
            tps = (cfg.eval_every * cfg.batch_size * cfg.grad_accum * cfg.seq_len) / max(elapsed, 1)
            print(f"iter {it:6d} | val_loss {val_loss:.4f} | lr {lr:.1e} | {tps/1000:.1f}k tok/s | {format_time(elapsed)}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iter = it
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    cfg=mcfg,
                    it=it,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    best_iter=best_iter,
                    mode=cfg.mode,
                    path=best_path,
                )
                print(f"  ✅ Nouveau best : val_loss={val_loss:.4f} → {best_path}")

            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=mcfg,
                it=it,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_iter=best_iter,
                mode=cfg.mode,
                path=last_path,
            )

            model.train()
            t0 = time.time()

        if it == cfg.max_iters:
            break

        if it > 0 and it % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{it:06d}.pt")
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=mcfg,
                it=it,
                val_loss=None,
                best_val_loss=best_val_loss,
                best_iter=best_iter,
                mode=cfg.mode,
                path=ckpt_path,
            )
            print(f"  💾 Checkpoint : {os.path.basename(ckpt_path)}")
            _rolling_cleanup(cfg.out_dir, keep=cfg.rolling_keep)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum):
            x, y = train_loader.next_batch()
            tokens_seen += x.numel()
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if it % cfg.log_every == 0:
            print(f"  it {it:5d} | loss {accum_loss:.4f} | lr {lr:.1e} | tokens {tokens_seen/1e6:.1f}M")

    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=mcfg,
        it=cfg.max_iters,
        val_loss=best_val_loss,
        best_val_loss=best_val_loss,
        best_iter=best_iter,
        mode=cfg.mode,
        path=last_path,
    )

    print(f"\n{'═'*60}")
    print(f"  Terminé ! Meilleure val_loss : {best_val_loss:.4f} | best_iter={best_iter}")
    print(f"  Checkpoint best : {best_path}")
    print(f"  Checkpoint last : {last_path}")
    print(f"{'═'*60}\n")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="MiniLLM v2 — Entraînement")
    p.add_argument("--size", default=cfg.model_size, choices=list(PRESETS))
    p.add_argument("--data", default=cfg.data_path)
    p.add_argument("--val", default=cfg.val_path)
    p.add_argument("--seq", type=int, default=cfg.seq_len)
    p.add_argument("--batch", type=int, default=cfg.batch_size)
    p.add_argument("--accum", type=int, default=cfg.grad_accum)
    p.add_argument("--iters", type=int, default=cfg.max_iters)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--out", default=cfg.out_dir)
    p.add_argument("--resume", default=cfg.resume_from)
    p.add_argument("--keep", type=int, default=cfg.rolling_keep, help="Nombre de checkpoints numérotés à garder (défaut: 2)")
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--mode", default=cfg.mode, choices=["pretrain", "sft"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-eval-zero", action="store_true")
    args = p.parse_args()
    cfg.model_size = args.size
    cfg.data_path = args.data
    cfg.val_path = args.val
    cfg.seq_len = args.seq
    cfg.batch_size = args.batch
    cfg.grad_accum = args.accum
    cfg.max_iters = args.iters
    cfg.lr = args.lr
    cfg.out_dir = args.out
    cfg.resume_from = args.resume
    cfg.rolling_keep = args.keep
    cfg.device = args.device
    cfg.mode = args.mode
    cfg.compile = not args.no_compile
    cfg.eval_on_zero = not args.no_eval_zero
    return cfg


if __name__ == "__main__":
    train(parse_args())

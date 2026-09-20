"""
checkpoint.py — sauvegarde / chargement robustes.

Règles (corrigent les problèmes relevés à l'audit) :
  * Écriture ATOMIQUE (fichier temporaire + os.replace) : une session Colab/Kaggle coupée en plein
    write ne corrompt plus le checkpoint.
  * ckpt_XXXXXXX.pt = état COMPLET (modèle + optimiseur + scaler + iter + best_val_loss + tokens vus).
    La reprise ("auto") prend le plus récent, JAMAIS best.pt.
  * best.pt = poids seuls + config (léger, ~1/3 d'un checkpoint complet). Sidecar best.json avec la
    meilleure val_loss : elle survit aux reprises, donc un run repris ne peut plus écraser un meilleur modèle.
  * final.pt = poids de la dernière itération.
  * Nettoyage rolling trié NUMÉRIQUEMENT (avant : tri alphabétique qui pouvait supprimer les nouveaux).
  * Tout est stocké en types simples -> torch.load(weights_only=True), plus de pickle de classes.
"""
from __future__ import annotations

import glob
import json
import os
import re
import warnings
from typing import Optional

import torch

FORMAT_VERSION = 2
_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def atomic_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def clean_state_dict(sd: dict) -> dict:
    """Retire les préfixes ajoutés par torch.compile ('_orig_mod.') et DDP ('module.')."""
    out = {}
    for k, v in sd.items():
        for prefix in ("_orig_mod.", "module."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def _cpu_state_dict(model) -> dict:
    return {k: v.detach().cpu() for k, v in clean_state_dict(model.state_dict()).items()}


# ── écriture ────────────────────────────────────────────────────────────────
def save_full(out_dir, it, model, optimizer, scaler, model_cfg, train_cfg, best_val_loss, no_improve, tokens_seen, extra=None):
    path = os.path.join(out_dir, f"ckpt_{it:07d}.pt")
    atomic_save({
        "format": FORMAT_VERSION,
        "model": _cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "iter": int(it),
        "best_val_loss": float(best_val_loss),
        "no_improve": int(no_improve),
        "tokens_seen": int(tokens_seen),
        **(extra or {}),
    }, path)
    return path


def save_best(out_dir, it, model, model_cfg, train_cfg, val_loss, tokens_seen, extra=None):
    path = os.path.join(out_dir, "best.pt")
    atomic_save({
        "format": FORMAT_VERSION,
        "model": _cpu_state_dict(model),
        "config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "iter": int(it),
        "val_loss": float(val_loss),
        "tokens_seen": int(tokens_seen),
        **(extra or {}),
    }, path)
    atomic_write_text(json.dumps({"val_loss": float(val_loss), "iter": int(it)}), os.path.join(out_dir, "best.json"))
    return path


def save_final(out_dir, it, model, model_cfg, train_cfg, val_loss, tokens_seen, extra=None):
    path = os.path.join(out_dir, "final.pt")
    atomic_save({
        "format": FORMAT_VERSION,
        "model": _cpu_state_dict(model),
        "config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "iter": int(it),
        "val_loss": None if val_loss is None else float(val_loss),
        "tokens_seen": int(tokens_seen),
        **(extra or {}),
    }, path)
    return path


# ── lecture ─────────────────────────────────────────────────────────────────
def load_checkpoint(path: str, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:                                     # anciens checkpoints (config picklée)
        warnings.warn(f"weights_only=True a échoué ({type(e).__name__}); rechargement non restreint de {path}. "
                      "Ne charge que des fichiers en lesquels tu as confiance.")
        return torch.load(path, map_location=map_location, weights_only=False)


def list_checkpoints(out_dir: str):
    items = []
    for p in glob.glob(os.path.join(out_dir, "ckpt_*.pt")):
        m = _CKPT_RE.search(p)
        if m:
            items.append((int(m.group(1)), p))
    return sorted(items)


def latest_checkpoint(out_dir: str) -> Optional[str]:
    items = list_checkpoints(out_dir)
    return items[-1][1] if items else None


def rolling_cleanup(out_dir: str, keep: int) -> None:
    items = list_checkpoints(out_dir)
    for _, p in items[:-keep] if keep > 0 else []:
        try:
            os.remove(p)
        except OSError:
            pass


def read_best_val(out_dir: str) -> float:
    p = os.path.join(out_dir, "best.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return float(json.load(f)["val_loss"])
        except Exception:
            pass
    return float("inf")

"""
inspect_model.py — inspecte les presets et vérifie qu'un modèle est sain AVANT de lancer un long entraînement.

  python inspect_model.py                 # tableau des presets (vrais nombres de paramètres) + estimation mémoire
  python inspect_model.py --size 49M      # + instancie le modèle, vérifie la loss initiale et mesure la vitesse
  python inspect_model.py --ckpt checkpoints/sft/best.pt      # décrit un checkpoint
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from checkpoint import load_checkpoint
from config import ModelConfig, PRESETS, get_preset
from model import MiniLLM


def table():
    print(f"{'preset':>7} | {'couches':>7} {'d_model':>7} {'têtes':>5} {'kv':>3} {'ffn':>5} {'ctx':>5} | "
          f"{'params totaux':>14} {'dont embeddings':>16} | poids fp32   optim.+grad (entraînement)")
    print("-" * 122)
    for name, c in PRESETS.items():
        n, emb = c.count_params(), c.vocab_size * c.d_model
        print(f"{name:>7} | {c.n_layers:>7} {c.d_model:>7} {c.n_heads:>5} {c.kv_heads:>3} {c.ffn_hidden:>5} {c.max_seq_len:>5} | "
              f"{n:>14,} {emb / n * 100:>14.0f} % | {n * 4 / 1e9:>7.2f} Go   {n * 16 / 1e9:>7.2f} Go")
    print(f"\n(vocabulaire par défaut : {next(iter(PRESETS.values())).vocab_size:,} tokens ; les noms sont calculés, pas écrits à la main)")


def check(size: str, seq: int, device: str):
    cfg = get_preset(size, max_seq_len=max(seq, 64))
    m = MiniLLM(cfg).to(device)
    n = m.num_params()
    assert n == cfg.count_params(), "count_params() ne correspond pas au vrai modèle !"
    print(f"\nModèle {cfg.size_name()} : {n:,} paramètres ({cfg.count_non_embedding_params():,} hors embeddings)")
    x = torch.randint(0, cfg.vocab_size, (2, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, seq), device=device)          # cibles INDÉPENDANTES (l'ancien test utilisait y = x)
    _, loss = m(x, y)
    print(f"Loss initiale : {loss.item():.3f} (attendu ≈ ln(V) = {math.log(cfg.vocab_size):.3f}) "
          f"{'✓' if abs(loss.item() - math.log(cfg.vocab_size)) < 0.5 else '⚠️ init suspecte'}")
    loss.backward()
    gn = math.sqrt(sum((p.grad.float() ** 2).sum().item() for p in m.parameters() if p.grad is not None))
    print(f"Norme du gradient : {gn:.2f} | gradients non finis : {any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)}")
    t = time.perf_counter()
    for _ in range(3):
        m.zero_grad()
        m(x, y)[1].backward()
    print(f"Pas forward+backward (batch 2x{seq}) : {(time.perf_counter() - t) / 3 * 1000:.0f} ms sur {device}")


def describe(path: str):
    ck = load_checkpoint(path)
    cfg = ModelConfig.from_dict(ck["config"])
    print(f"{path}\n  modèle {cfg.size_name()} ({cfg.count_params():,} params) | vocab {cfg.vocab_size} | ctx {cfg.max_seq_len}")
    for k in ("iter", "val_loss", "best_val_loss", "tokens_seen"):
        if k in ck:
            print(f"  {k}: {ck[k]}")
    tc = ck.get("train_config", {})
    if tc:
        print(f"  mode: {tc.get('mode')} | lr {tc.get('lr')} | batch {tc.get('batch_size')}x{tc.get('grad_accum')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", default=None)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    table()
    if a.size:
        check(a.size, a.seq, a.device)
    if a.ckpt:
        describe(a.ckpt)

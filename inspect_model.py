# inspect_model.py — MiniLLM v3

import argparse
import os

import torch

from config import PRESETS
from model import MiniLLM


DISPLAY_NAMES = {
    "15M": "MiniLLM-49M",
    "50M": "MiniLLM-83M",
    "125M": "MiniLLM-162M",
    "350M": "MiniLLM-381M",
    "1B": "MiniLLM-1.09B",
}


def print_separator(char="═", width=70):
    print(char * width)


def format_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1e9:.3f}B ({count:,})"
    return f"{count / 1e6:.3f}M ({count:,})"


def inspect_preset(size: str):
    cfg = PRESETS[size]
    model = MiniLLM(cfg)
    n_total = model.n_params
    n_emb = cfg.vocab_size * cfg.d_model
    n_blocks = sum(p.numel() for n, p in model.named_parameters() if "blocks" in n)
    n_head = sum(p.numel() for n, p in model.named_parameters() if "lm_head" in n and "tok_emb" not in n)

    print_separator()
    print(f"  {DISPLAY_NAMES[size]}  (preset historique : {size})")
    print_separator()
    print(f"  Paramètres réels  : {format_params(n_total)}")
    print(f"  ├── Embedding     : {n_emb:,} ({n_emb / n_total * 100:.1f}%)")
    print(f"  ├── Blocs (×{cfg.n_layers:2d})  : {n_blocks:,} ({n_blocks / n_total * 100:.1f}%)")
    print(f"  └── LM Head       : {'partagé (tied)' if cfg.tie_embeddings else format_params(n_head)}")
    print()
    print("  Architecture :")
    print(f"  ├── n_layers      : {cfg.n_layers}")
    print(f"  ├── d_model       : {cfg.d_model}")
    print(f"  ├── n_heads (Q)   : {cfg.n_heads}")
    print(f"  ├── kv_heads      : {cfg.kv_heads}")
    print(f"  ├── head_dim      : {cfg.head_dim}")
    print(f"  ├── ffn_hidden    : {cfg.ffn_hidden}")
    print(f"  ├── max_seq_len   : {cfg.max_seq_len}")
    print(f"  └── vocab_size    : {cfg.vocab_size:,}")
    print()

    x = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.max_seq_len)))
    with torch.no_grad():
        logits, loss = model(x, x)
    print(f"  ✓ Input  : {list(x.shape)}")
    print(f"  ✓ Logits : {list(logits.shape)}")
    print(f"  ✓ Loss   : {loss.item():.4f} (attendu ≈ {torch.log(torch.tensor(cfg.vocab_size)).item():.2f})")
    print_separator()
    print()


def inspect_all_presets():
    print_separator()
    print("  MiniLLM v3 — nombre réel de paramètres")
    print_separator()
    print(f"  {'Preset':>7} | {'Nom réel':>14} | {'Paramètres':>18} | {'Layers':>6} | {'d_model':>7} | {'Ctx':>5}")
    print("  " + "─" * 70)
    for size, cfg in PRESETS.items():
        n = MiniLLM(cfg).n_params
        print(f"  {size:>7} | {DISPLAY_NAMES[size]:>14} | {n:>18,} | {cfg.n_layers:>6} | {cfg.d_model:>7} | {cfg.max_seq_len:>5}")
    print_separator()
    print()


def inspect_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = MiniLLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)

    print_separator()
    print(f"  Checkpoint       : {path}")
    print(f"  Iteration        : {ckpt.get('iter', '?')}")
    print(f"  Val loss         : {ckpt.get('val_loss', '?')}")
    print(f"  Paramètres réels : {format_params(model.n_params)}")
    print(f"  Mode             : {ckpt.get('mode', 'legacy/pretrain')}")
    print("  ✓ Checkpoint valide")
    print_separator()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLLM v3 — inspection des paramètres réels")
    parser.add_argument("--size", default=None, choices=list(PRESETS))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.checkpoint:
        inspect_checkpoint(args.checkpoint)
    elif args.size:
        inspect_preset(args.size)
    else:
        inspect_all_presets()
        inspect_preset("50M")

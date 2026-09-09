# inspect_model.py — MiniLLM v2
#
# Vérifie et affiche les infos sur le modèle sans l'entraîner.
# Utile pour valider l'architecture avant de lancer un long entraînement.
#
# Usage :
#   python inspect_model.py               # affiche tous les presets
#   python inspect_model.py --size 50M    # détails du modèle 50M
#   python inspect_model.py --checkpoint checkpoints/best.pt

import argparse
import torch
from config import ModelConfig, PRESETS
from model  import MiniLLM


def print_separator(char="═", width=62):
    print(char * width)


def inspect_preset(size: str):
    """Affiche les détails d'un preset."""
    cfg   = PRESETS[size]
    model = MiniLLM(cfg)

    n_total   = model.n_params
    n_emb     = cfg.vocab_size * cfg.d_model
    n_blocks  = sum(p.numel() for n, p in model.named_parameters()
                    if "blocks" in n)
    n_head    = sum(p.numel() for n, p in model.named_parameters()
                    if "lm_head" in n and "tok_emb" not in n)

    print_separator()
    print(f"  MiniLLM-{size}")
    print_separator()
    print(f"  Paramètres total  : {n_total/1e6:.2f}M")
    print(f"  ├── Embedding     : {n_emb/1e6:.2f}M  ({n_emb/n_total*100:.1f}%)")
    print(f"  ├── Blocs (×{cfg.n_layers:2d})  : {n_blocks/1e6:.2f}M  ({n_blocks/n_total*100:.1f}%)")
    print(f"  └── LM Head       : {'partagé (tied)' if cfg.tie_embeddings else f'{n_head/1e6:.2f}M'}")
    print()
    print(f"  Architecture :")
    print(f"  ├── n_layers      : {cfg.n_layers}")
    print(f"  ├── d_model       : {cfg.d_model}")
    print(f"  ├── n_heads (Q)   : {cfg.n_heads}")
    print(f"  ├── kv_heads      : {cfg.kv_heads}  {'(MHA)' if cfg.kv_heads == cfg.n_heads else f'(GQA, ×{cfg.n_heads//cfg.kv_heads} ratio)'}")
    print(f"  ├── head_dim      : {cfg.head_dim}")
    print(f"  ├── ffn_hidden    : {cfg.ffn_hidden}")
    print(f"  ├── max_seq_len   : {cfg.max_seq_len}")
    print(f"  └── vocab_size    : {cfg.vocab_size:,}")
    print()

    # Par couche
    per_layer_attn = (cfg.d_model * cfg.n_heads  * cfg.head_dim
                    + cfg.d_model * cfg.kv_heads * cfg.head_dim * 2
                    + cfg.d_model * cfg.d_model)
    per_layer_ffn  = (cfg.d_model * cfg.ffn_hidden * 2 + cfg.ffn_hidden * cfg.d_model)
    per_layer      = per_layer_attn + per_layer_ffn + 2 * cfg.d_model
    print(f"  Par couche :")
    print(f"  ├── Attention     : {per_layer_attn/1e6:.3f}M")
    print(f"  ├── SwiGLU FFN    : {per_layer_ffn/1e6:.3f}M")
    print(f"  └── Total/couche  : {per_layer/1e6:.3f}M")
    print()

    # Estimation mémoire
    bytes_fp32  = n_total * 4
    bytes_bf16  = n_total * 2
    # Pendant l'entraînement : poids fp32 + gradients fp32 + optimizer states (×2)
    train_mem   = n_total * 4 * 4    # poids + grads + Adam m + Adam v
    print(f"  Mémoire estimée :")
    print(f"  ├── Inférence fp32  : {bytes_fp32/1e9:.2f} GB")
    print(f"  ├── Inférence bf16  : {bytes_bf16/1e9:.2f} GB")
    print(f"  └── Entraînement    : ~{train_mem/1e9:.2f} GB (poids + grads + Adam)")
    print()

    # Test forward rapide
    print(f"  Test forward pass...")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        logits, loss = model(x, x)
    print(f"  ✓ Input  : {list(x.shape)}")
    print(f"  ✓ Logits : {list(logits.shape)}")
    print(f"  ✓ Loss   : {loss.item():.4f}  (attendu ≈ {torch.log(torch.tensor(cfg.vocab_size)).item():.2f})")
    print_separator()
    print()


def inspect_all_presets():
    """Tableau comparatif de tous les presets."""
    print_separator()
    print("  MiniLLM v2 — Comparatif des tailles")
    print_separator()
    print(f"  {'Size':>6} | {'Params':>8} | {'Layers':>6} | {'d_model':>7} | "
          f"{'Heads':>5} | {'KV':>3} | {'FFN':>5} | {'Ctx':>5}")
    print("  " + "─" * 58)

    for size, cfg in PRESETS.items():
        n = cfg.count_params()
        unit = "M" if n >= 1e6 else "K"
        val  = n/1e6 if n >= 1e6 else n/1e3
        print(f"  {size:>6} | {val:>6.1f}{unit} | {cfg.n_layers:>6} | "
              f"{cfg.d_model:>7} | {cfg.n_heads:>5} | {cfg.kv_heads:>3} | "
              f"{cfg.ffn_hidden:>5} | {cfg.max_seq_len:>5}")

    print_separator()
    print()


def inspect_checkpoint(path: str):
    """Affiche les infos d'un checkpoint sauvegardé."""
    import os
    assert os.path.exists(path), f"Checkpoint introuvable : {path}"

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    print_separator()
    print(f"  Checkpoint : {path}")
    print_separator()
    print(f"  Iteration  : {ckpt.get('iter', '?')}")
    print(f"  Val loss   : {ckpt.get('val_loss', '?')}")
    print(f"  Config     : {cfg}")
    print()

    # Charger et vérifier
    model = MiniLLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    print(f"  Paramètres : {model.n_params/1e6:.2f}M")
    print(f"  ✓ Checkpoint valide")
    print_separator()
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MiniLLM v2 — Inspection du modèle")
    p.add_argument("--size",       default=None, choices=list(PRESETS),
                   help="Inspecter un preset spécifique")
    p.add_argument("--checkpoint", default=None,
                   help="Inspecter un fichier checkpoint .pt")
    args = p.parse_args()

    if args.checkpoint:
        inspect_checkpoint(args.checkpoint)
    elif args.size:
        inspect_preset(args.size)
    else:
        inspect_all_presets()
        inspect_preset("50M")

# prepare_data.py — MiniLLM v2
#
# Convertit un corpus texte en fichiers binaires de tokens
# prêts pour l'entraînement.
#
# Usage :
#   python prepare_data.py corpus.txt
#   python prepare_data.py corpus.txt --out data/ --val_ratio 0.02
#   python prepare_data.py dossier/  --out data/   # traite tous les .txt du dossier
#
# Format de sortie :
#   data/train.bin  — tokens uint16 consécutifs
#   data/val.bin    — idem (val_ratio % des tokens)
#   data/meta.json  — métadonnées (nb tokens, tokenizer, etc.)

import os
import sys
import json
import glob
import argparse
import numpy as np
import tiktoken
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Préparation principale
# ══════════════════════════════════════════════════════════════════════════════

def prepare(
    input_path: str,
    output_dir: str   = "data",
    val_ratio:  float = 0.01,
    encoding:   str   = "cl100k_base",
    chunk_size: int   = 1_000_000,    # traiter le texte par morceaux (RAM)
):
    """
    Tokenise un corpus et le sauvegarde en binaire pour l'entraînement.

    Args:
        input_path : fichier .txt ou dossier contenant des .txt
        output_dir : dossier de destination (créé si absent)
        val_ratio  : proportion réservée à la validation (défaut 1%)
        encoding   : tokenizer tiktoken à utiliser
        chunk_size : taille des morceaux en caractères (pour gros corpus)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Collecter les fichiers ─────────────────────────────────────────────
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "**", "*.txt"),
                                 recursive=True))
        if not files:
            print(f"Aucun fichier .txt trouvé dans {input_path}")
            sys.exit(1)
        print(f"  {len(files)} fichier(s) .txt trouvé(s)")
    else:
        assert os.path.exists(input_path), f"Fichier introuvable : {input_path}"
        files = [input_path]

    # ── Tokenizer ─────────────────────────────────────────────────────────
    enc = tiktoken.get_encoding(encoding)
    print(f"  Tokenizer : {encoding} | vocab size : {enc.n_vocab:,}")

    # ── Tokenisation ──────────────────────────────────────────────────────
    all_tokens: list[int] = []
    total_chars = 0

    for file_path in files:
        size_mb = os.path.getsize(file_path) / 1e6
        print(f"  Traitement : {file_path}  ({size_mb:.1f} MB)")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        total_chars += len(text)

        # Tokeniser par morceaux pour les très grands fichiers
        for i in range(0, len(text), chunk_size):
            chunk  = text[i : i + chunk_size]
            tokens = enc.encode_ordinary(chunk)   # sans token spéciaux
            all_tokens.extend(tokens)

            if len(text) > chunk_size:
                pct = min(100, (i + chunk_size) / len(text) * 100)
                print(f"    {pct:.0f}% — {len(all_tokens):,} tokens...", end="\r")

        # Ajouter un séparateur de document entre les fichiers
        if len(files) > 1:
            all_tokens.append(enc.eot_token)    # <|endoftext|>

        print(f"    → {len(all_tokens):,} tokens accumulés        ")

    # ── Statistiques ──────────────────────────────────────────────────────
    total_tokens = len(all_tokens)
    compression  = total_chars / max(total_tokens, 1)

    print(f"\n{'─'*50}")
    print(f"  Total caractères : {total_chars:,}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Compression      : {compression:.2f} chars/token")

    # Estimation de la durée d'entraînement (très approximatif)
    # Règle de Chinchilla : entraîner sur ~20x les paramètres en tokens
    for size, n_params in [("15M", 15e6), ("50M", 50e6), ("125M", 125e6)]:
        optimal = int(20 * n_params)
        epochs  = total_tokens / max(optimal, 1)
        print(f"  → Pour {size} : {optimal/1e6:.0f}M tokens optimaux | "
              f"ce corpus = {epochs:.1f}x")

    # ── Split train / val ──────────────────────────────────────────────────
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    n_val   = max(1000, int(total_tokens * val_ratio))
    n_train = total_tokens - n_val

    train_tokens = tokens_array[:n_train]
    val_tokens   = tokens_array[n_train:]

    print(f"\n  Train : {n_train:,} tokens")
    print(f"  Val   : {n_val:,} tokens")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")
    meta_path  = os.path.join(output_dir, "meta.json")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    meta = {
        "tokenizer":    encoding,
        "vocab_size":   enc.n_vocab,
        "n_train":      int(n_train),
        "n_val":        int(n_val),
        "n_total":      int(total_tokens),
        "files":        [str(f) for f in files],
        "eot_token":    enc.eot_token,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Fichiers créés :")
    print(f"    {train_path}  ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"    {val_path}  ({os.path.getsize(val_path)/1e6:.1f} MB)")
    print(f"    {meta_path}")
    print(f"\n  Prêt ! Lance l'entraînement avec : python train.py")


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires supplémentaires
# ══════════════════════════════════════════════════════════════════════════════

def inspect_data(data_path: str, n_tokens: int = 50):
    """Affiche les premiers tokens d'un fichier .bin pour vérifier."""
    import numpy as np

    assert os.path.exists(data_path), f"Fichier introuvable : {data_path}"

    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    enc    = tiktoken.get_encoding("cl100k_base")

    print(f"Fichier : {data_path}")
    print(f"Tokens  : {len(tokens):,}")
    print(f"\nPremiers {n_tokens} tokens décodés :")
    print("─" * 40)
    sample = tokens[:n_tokens].tolist()
    print(repr(enc.decode(sample)))
    print("─" * 40)


def estimate_training_time(
    n_tokens:   int,
    model_size: str = "50M",
    gpu:        str = "RTX 3090",
):
    """Estimation grossière du temps d'entraînement."""
    # Throughput approximatif (tokens/sec) selon GPU et taille
    throughput = {
        "RTX 3090": {"15M": 150_000, "50M": 80_000,  "125M": 40_000},
        "A100":     {"15M": 400_000, "50M": 200_000, "125M": 100_000},
        "RTX 4090": {"15M": 200_000, "50M": 100_000, "125M": 50_000},
        "CPU":      {"15M": 2_000,   "50M": 800,     "125M": 400},
    }.get(gpu, {"15M": 80_000, "50M": 40_000, "125M": 20_000})

    tps     = throughput.get(model_size, 40_000)
    seconds = n_tokens / tps

    print(f"\nEstimation pour {n_tokens/1e6:.0f}M tokens, modèle {model_size}, {gpu} :")
    print(f"  ~{tps/1000:.0f}k tokens/sec → {seconds/3600:.1f} heures")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MiniLLM v2 — Préparation du corpus d'entraînement"
    )
    p.add_argument("input",       nargs="?", default=None,
                   help="Fichier .txt ou dossier contenant des .txt")
    p.add_argument("--out",       default="data",
                   help="Dossier de sortie (défaut: data/)")
    p.add_argument("--val_ratio", type=float, default=0.01,
                   help="Fraction de tokens pour la validation (défaut: 0.01 = 1%%)")
    p.add_argument("--encoding",  default="cl100k_base",
                   help="Tokenizer tiktoken (défaut: cl100k_base)")
    p.add_argument("--inspect",   default=None,
                   help="Inspecter un fichier .bin existant")

    args = p.parse_args()

    if args.inspect:
        inspect_data(args.inspect)
    elif args.input:
        prepare(
            input_path = args.input,
            output_dir = args.out,
            val_ratio  = args.val_ratio,
            encoding   = args.encoding,
        )
    else:
        p.print_help()
        print("\nExemple :")
        print("  python prepare_data.py mon_corpus.txt")
        print("  python prepare_data.py dossier_textes/ --out data/ --val_ratio 0.02")

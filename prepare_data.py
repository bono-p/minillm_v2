# prepare_data.py — MiniLLM v2
#
# Convertit un corpus texte en fichiers binaires de tokens (int32).
#
# NOTE dtype : cl100k_base a 100 277 tokens → uint16 max = 65 535 → OVERFLOW.
# On utilise int32 (4 octets/token) au lieu de uint16 (2 octets/token).
#
# Usage :
#   python prepare_data.py corpus.txt
#   python prepare_data.py dossier/  --out data/
#   python prepare_data.py data/train.bin --inspect

import os, sys, json, glob, argparse
import numpy as np
import tiktoken
from pathlib import Path


def prepare(
    input_path: str,
    output_dir: str   = "data",
    val_ratio:  float = 0.01,
    encoding:   str   = "cl100k_base",
    chunk_size: int   = 500_000,
):
    os.makedirs(output_dir, exist_ok=True)

    # Collecter les fichiers
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True))
        if not files:
            print(f"Aucun fichier .txt trouvé dans {input_path}")
            sys.exit(1)
        print(f"  {len(files)} fichier(s) .txt")
    else:
        assert os.path.exists(input_path), f"Introuvable : {input_path}"
        files = [input_path]

    enc = tiktoken.get_encoding(encoding)
    print(f"  Tokenizer : {encoding} | vocab : {enc.n_vocab:,}")

    # Tokenisation chunk par chunk (économise la RAM)
    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")
    meta_path  = os.path.join(output_dir, "meta.json")

    total_chars = total_train = total_val = 0
    import random; rng = random.Random(42)

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for fpath in files:
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {fpath}  ({size_mb:.1f} MB)")
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            total_chars += len(text)

            for i in range(0, len(text), chunk_size):
                chunk  = text[i : i + chunk_size]
                tokens = enc.encode_ordinary(chunk)
                arr    = np.array(tokens, dtype=np.int32)   # int32 : compatible cl100k_base
                if rng.random() < val_ratio:
                    f_val.write(arr.tobytes())
                    total_val += len(arr)
                else:
                    f_train.write(arr.tobytes())
                    total_train += len(arr)

            if len(files) > 1:
                eot = np.array([enc.eot_token], dtype=np.int32)
                f_train.write(eot.tobytes())
                total_train += 1

    total = total_train + total_val
    print(f"\n  Total    : {total:,} tokens ({total_chars/1e6:.1f}M chars)")
    print(f"  Train    : {total_train:,}  |  Val : {total_val:,}")
    print(f"  dtype    : int32 (4 octets/token)")

    meta = {
        "tokenizer": encoding, "vocab_size": enc.n_vocab,
        "n_train": int(total_train), "n_val": int(total_val),
        "dtype": "int32", "files": [str(f) for f in files],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Prêt → python train.py")
    return train_path, val_path


def inspect_data(path: str, n: int = 50):
    assert os.path.exists(path), f"Introuvable : {path}"
    data = np.memmap(path, dtype=np.int32, mode="r")
    enc  = tiktoken.get_encoding("cl100k_base")
    print(f"Fichier : {path}")
    print(f"Tokens  : {len(data):,}  |  dtype : {data.dtype}")
    print(f"\nPremiers {n} tokens :")
    print("─" * 40)
    print(repr(enc.decode(data[:n].tolist())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input",     nargs="?", default=None)
    p.add_argument("--out",     default="data")
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--inspect", default=None)
    args = p.parse_args()
    if args.inspect:
        inspect_data(args.inspect)
    elif args.input:
        prepare(args.input, args.out, args.val_ratio)
    else:
        p.print_help()

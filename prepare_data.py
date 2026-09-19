# prepare_data.py — MiniLLM v3
#
# Prépare un corpus de pré-entraînement et/ou de QA en fichiers binaires int32.
# Les sources texte existantes restent supportées ; PIAF peut être ajouté comme
# source QA sans remplacer Wikipédia ni les autres corpus.
#
# Exemples :
#   python prepare_data.py corpus.txt wikipedia.txt --out data/
#   python prepare_data.py data/ --out data/
#   python prepare_data.py piaf-train.json --out data/ --source piaf
#   python prepare_data.py wikipedia.txt piaf-train.json --out data/
#
# PIAF est attendu au format SQuAD/JSON : data[] -> paragraphs[] ->
# context + qas[] -> question + answers[]. Les exemples QA sont convertis
# avec le même template que generate.py et portent un marqueur de provenance.
# Les fichiers d'entrée ne sont pas téléchargés automatiquement : l'utilisateur
# doit vérifier la licence et fournir les fichiers légalement obtenus.

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import tiktoken

DEFAULT_SYSTEM = "Tu es un assistant utile, précis et concis."
END_TAG = "<|end|>"


def conversation_text(question: str, answer: str, context: str | None = None) -> str:
    parts = ["<|system|>", DEFAULT_SYSTEM]
    if context and context.strip():
        parts.extend(["<|context|>", context.strip()])
    parts.extend(["<|user|>", question.strip(), "<|assistant|>", answer.strip(), END_TAG])
    return "\n".join(parts) + "\n"


def _answer_text(answers) -> str:
    if isinstance(answers, dict):
        return str(answers.get("text", "")).strip()
    if isinstance(answers, list) and answers:
        return _answer_text(answers[0])
    return ""


def iter_piaf_examples(obj: dict) -> Iterable[tuple[str, str, str]]:
    """Lit PIAF et les exports compatibles SQuAD sans dépendre d'un schéma unique."""
    articles = obj.get("data", obj if isinstance(obj, list) else [])
    if isinstance(articles, dict):
        articles = [articles]
    for article in articles:
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", "")).strip()
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                answer = _answer_text(qa.get("answers", []))
                if question and answer and context:
                    yield context, question, answer


def load_piaf(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    examples = [conversation_text(q, a, c) for c, q, a in iter_piaf_examples(obj)]
    if not examples:
        raise ValueError(f"Aucun exemple PIAF/SQuAD valide trouvé dans {path}")
    return examples


def collect_inputs(input_paths: list[str]) -> list[str]:
    files = []
    for input_path in input_paths:
        if os.path.isdir(input_path):
            files.extend(glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True))
            files.extend(glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True))
        elif os.path.exists(input_path):
            files.append(input_path)
        else:
            raise FileNotFoundError(f"Introuvable : {input_path}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("Aucun fichier .txt ou .json trouvé")
    return files


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare(
    input_paths: str | list[str],
    output_dir: str = "data",
    val_ratio: float = 0.01,
    encoding: str = "cl100k_base",
    chunk_size: int = 500_000,
    seed: int = 42,
    source: str = "auto",
):
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    files = collect_inputs(input_paths)
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding(encoding)
    print(f"  Tokenizer : {encoding} | vocab : {enc.n_vocab:,}")
    print(f"  Sources   : {len(files)} fichier(s) ; PIAF ajouté sans remplacer les textes")

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.json")
    rng = random.Random(seed)
    total_chars = total_train = total_val = 0
    source_stats = {}

    def write_tokens(handle, tokens):
        arr = np.asarray(tokens, dtype=np.int32)
        handle.write(arr.tobytes())
        return len(arr)

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for fpath in files:
            suffix = Path(fpath).suffix.lower()
            kind = "piaf" if source == "piaf" or (source == "auto" and suffix == ".json") else "text"
            print(f"  {fpath} ({kind})")

            if kind == "piaf":
                examples = load_piaf(fpath)
                source_stats[fpath] = {"kind": "piaf", "examples": len(examples), "train_tokens": 0, "val_tokens": 0}
                for example in examples:
                    tokens = enc.encode(example) + [enc.eot_token]
                    target = f_val if rng.random() < val_ratio else f_train
                    count = write_tokens(target, tokens)
                    if target is f_val:
                        total_val += count
                        source_stats[fpath]["val_tokens"] += count
                    else:
                        total_train += count
                        source_stats[fpath]["train_tokens"] += count
                continue

            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                text = clean_text(fh.read())
            total_chars += len(text)
            source_stats[fpath] = {"kind": "text", "characters": len(text), "train_tokens": 0, "val_tokens": 0}
            for i in range(0, len(text), chunk_size):
                tokens = enc.encode_ordinary(text[i:i + chunk_size]) + [enc.eot_token]
                target = f_val if rng.random() < val_ratio else f_train
                count = write_tokens(target, tokens)
                if target is f_val:
                    total_val += count
                    source_stats[fpath]["val_tokens"] += count
                else:
                    total_train += count
                    source_stats[fpath]["train_tokens"] += count

    total = total_train + total_val
    print(f"\n  Total : {total:,} tokens | Train : {total_train:,} | Val : {total_val:,}")
    print("  dtype : int32 (4 octets/token)")
    meta = {
        "tokenizer": encoding,
        "vocab_size": enc.n_vocab,
        "dtype": "int32",
        "seed": seed,
        "val_ratio": val_ratio,
        "sources": source_stats,
        "piaf_format": "SQuAD/PIAF -> system/context/user/assistant/end",
        "license_note": "Vérifier la licence de chaque source avant utilisation ou redistribution.",
        "n_train": int(total_train),
        "n_val": int(total_val),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Prêt → python train.py --mode pretrain")
    return train_path, val_path


def inspect_data(path: str, n: int = 50):
    assert os.path.exists(path), f"Introuvable : {path}"
    data = np.memmap(path, dtype=np.int32, mode="r")
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Fichier : {path}\nTokens : {len(data):,} | dtype : {data.dtype}")
    print(repr(enc.decode(data[:n].tolist())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Corpus .txt et/ou exports PIAF/SQuAD .json")
    parser.add_argument("--out", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--source", choices=["auto", "text", "piaf"], default="auto")
    parser.add_argument("--inspect", default=None)
    args = parser.parse_args()
    if args.inspect:
        inspect_data(args.inspect)
    else:
        prepare(args.input, args.out, args.val_ratio, args.encoding, seed=args.seed, source=args.source)

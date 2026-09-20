"""
prepare_data.py — construit le corpus de pré-entraînement.

Étapes (chacune utilisable seule, ou `all`) :
  corpus     Wikipédia FR (+ FineWeb-2 FR optionnel, + tes .txt) -> nettoyés -> data/corpus/corpus.jsonl
  tokenizer  entraîne un BPE 32k sur ce corpus                    -> data/tokenizer.json
  tokenize   encode le corpus (avec <|endoftext|> entre documents) -> data/pretrain/{train,val}.bin + meta.json

Corrections par rapport à l'ancien pipeline :
  * split train/val PAR DOCUMENT (hash du texte) : avant, 128 k articles étaient devenus 4,19 M "articles"
    (coupe sur les lignes vides) et val partageait des paragraphes avec train ;
  * <|endoftext|> après chaque document : le modèle apprend enfin à terminer un texte ;
  * nettoyage des phrases trouées de Wikipédia (voir text_cleaning.py) ;
  * streaming : plus besoin de télécharger tout Wikipédia (5 Go) pour en utiliser 5 %.

Exemples :
  python prepare_data.py all --out data --wiki_docs 250000 --web_docs 300000
  python prepare_data.py all --out data --local_txt mes_textes/*.txt --wiki_docs 0     # 100 % hors-ligne
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from typing import Iterable, Iterator, List, Optional

import numpy as np

from mini_tokenizer import MiniTokenizer, save_meta, train_tokenizer
from text_cleaning import clean_web_text, clean_wikipedia_text


# ══════════════════════════════════════════════════════════════════════════════
#  Sources
# ══════════════════════════════════════════════════════════════════════════════
def iter_wikipedia(max_docs: int, seed: int, stats: dict) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    seen = 0
    for ex in ds:
        stats["wiki_raw"] = stats.get("wiki_raw", 0) + 1
        doc = clean_wikipedia_text(ex["text"], stats=stats)
        if doc:
            seen += 1
            yield doc
            if seen >= max_docs:
                break


def iter_fineweb_fr(max_docs: int, seed: int, stats: dict) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    seen = 0
    for ex in ds:
        stats["web_raw"] = stats.get("web_raw", 0) + 1
        doc = clean_web_text(ex["text"])
        if doc:
            seen += 1
            yield doc
            if seen >= max_docs:
                break


def iter_local_txt(patterns: List[str], chunk_chars: int = 4000) -> Iterator[str]:
    """Tes propres textes : chaque fichier est découpé en documents d'environ chunk_chars (à la limite d'un paragraphe)."""
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            buf: List[str] = []
            size = 0
            for para in text.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                buf.append(para)
                size += len(para)
                if size >= chunk_chars:
                    yield "\n\n".join(buf)
                    buf, size = [], 0
            if size >= 200:
                yield "\n\n".join(buf)


# ══════════════════════════════════════════════════════════════════════════════
#  Étapes
# ══════════════════════════════════════════════════════════════════════════════
def build_corpus(out: str, wiki_docs: int = 250_000, web_docs: int = 0, local_txt: Optional[List[str]] = None,
                 seed: int = 42) -> str:
    os.makedirs(os.path.join(out, "corpus"), exist_ok=True)
    path = os.path.join(out, "corpus", "corpus.jsonl")
    stats: dict = {}
    seen_hashes = set()
    n_docs = n_chars = n_dupes = 0

    def sources() -> Iterable:
        if wiki_docs > 0:
            yield "wiki", iter_wikipedia(wiki_docs, seed, stats)
        if web_docs > 0:
            yield "web", iter_fineweb_fr(web_docs, seed, stats)
        if local_txt:
            yield "local", iter_local_txt(local_txt)

    with open(path, "w", encoding="utf-8") as f:
        for name, it in sources():
            for doc in it:
                h = zlib.crc32(doc[:600].encode("utf-8")) ^ (len(doc) << 32)
                if h in seen_hashes:
                    n_dupes += 1
                    continue
                seen_hashes.add(h)
                f.write(json.dumps({"text": doc, "src": name}, ensure_ascii=False) + "\n")
                n_docs += 1
                n_chars += len(doc)
                if n_docs % 20_000 == 0:
                    print(f"  … {n_docs:,} documents ({n_chars / 1e6:.0f} M caractères)")
    print(f"\nCorpus : {n_docs:,} documents, {n_chars / 1e6:.0f} M caractères, {n_dupes} doublons ignorés -> {path}")
    if stats.get("sent_total"):
        print(f"Nettoyage Wikipédia : {stats['sent_dropped'] / stats['sent_total'] * 100:.1f} % des phrases supprimées "
              f"(phrases trouées) ; {stats.get('wiki_raw', 0):,} articles lus")
    if n_docs == 0:
        raise RuntimeError("Corpus vide : vérifie tes sources (--wiki_docs / --web_docs / --local_txt).")
    return path


def _iter_jsonl(path: str, stride: int = 1) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                yield json.loads(line)["text"]


def build_tokenizer(out: str, vocab_size: int = 32_000, max_mb: int = 300) -> MiniTokenizer:
    corpus = os.path.join(out, "corpus", "corpus.jsonl")
    size = os.path.getsize(corpus)
    stride = max(1, int(size / (max_mb * 1e6)))            # sous-échantillonne si le corpus dépasse max_mb
    print(f"Entraînement du tokenizer BPE (vocab={vocab_size}) sur ~{min(size / 1e6, max_mb):.0f} Mo (1 doc sur {stride})…")
    tok = train_tokenizer(_iter_jsonl(corpus, stride), vocab_size=vocab_size, save_path=os.path.join(out, "tokenizer.json"))
    print(f"Tokenizer : {tok.vocab_size} tokens (embedding paddé à {tok.padded_vocab_size})")
    return tok


def tokenize_corpus(out: str, val_permille: int = 10, max_tokens: int = 0, batch_docs: int = 2000) -> dict:
    tok = MiniTokenizer(os.path.join(out, "tokenizer.json"))
    dtype = np.uint16 if tok.padded_vocab_size <= 65_535 else np.uint32
    pre_dir = os.path.join(out, "pretrain")
    os.makedirs(pre_dir, exist_ok=True)
    corpus = os.path.join(out, "corpus", "corpus.jsonl")
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    files = {s: open(os.path.join(pre_dir, f"{s}.bin"), "wb") for s in counts}
    total_chars = 0

    def flush(texts: List[str]):
        nonlocal total_chars
        encoded = tok.encode_batch(texts)
        for text, ids in zip(texts, encoded):
            split = "val" if (zlib.crc32(text[:400].encode("utf-8")) % 1000) < val_permille else "train"
            arr = np.asarray(ids + [tok.eot_id], dtype=dtype)      # <|endoftext|> après chaque document
            files[split].write(arr.tobytes())
            counts[split] += len(arr)
            docs[split] += 1
            total_chars += len(text)

    try:
        batch: List[str] = []
        for text in _iter_jsonl(corpus):
            batch.append(text)
            if len(batch) >= batch_docs:
                flush(batch)
                batch = []
                if max_tokens and counts["train"] + counts["val"] >= max_tokens:
                    break
        if batch:
            flush(batch)
    finally:
        for f in files.values():
            f.close()

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size, "padded_vocab_size": tok.padded_vocab_size,
        "eot_id": tok.eot_id, "n_train_tokens": counts["train"], "n_val_tokens": counts["val"],
        "n_train_docs": docs["train"], "n_val_docs": docs["val"],
        "chars_per_token": round(total_chars / max(1, counts["train"] + counts["val"]), 3),
        "tokenizer": "tokenizer.json",
    }
    save_meta(os.path.join(pre_dir, "meta.json"), meta)
    print(f"\nTokens : train={counts['train']:,} ({docs['train']:,} docs) | val={counts['val']:,} ({docs['val']:,} docs) "
          f"| {meta['chars_per_token']} caractères/token")
    if counts["val"] < 100_000:
        print("⚠️  val très petit : augmente --val_permille (ou utilise plus de documents).")
    # vérification d'intégrité
    arr = np.memmap(os.path.join(pre_dir, "train.bin"), dtype=dtype, mode="r")
    assert int(arr.max()) < tok.padded_vocab_size, "id de token hors vocabulaire !"
    print("Aperçu décodé :", repr(tok.decode(arr[:60].tolist())[:200]))
    return meta


def main():
    p = argparse.ArgumentParser(description="MiniLLM v2 — préparation des données")
    p.add_argument("stage", choices=["corpus", "tokenizer", "tokenize", "all"])
    p.add_argument("--out", default="data")
    p.add_argument("--wiki_docs", type=int, default=250_000, help="nb d'articles Wikipédia FR (0 = aucun)")
    p.add_argument("--web_docs", type=int, default=0, help="nb de documents FineWeb-2 FR (0 = aucun)")
    p.add_argument("--local_txt", nargs="*", default=None, help="tes fichiers .txt (motifs glob acceptés)")
    p.add_argument("--vocab_size", type=int, default=32_000)
    p.add_argument("--tok_mb", type=int, default=300, help="Mo de texte max pour entraîner le tokenizer")
    p.add_argument("--val_permille", type=int, default=10, help="‰ de documents envoyés en validation")
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.stage in ("corpus", "all"):
        build_corpus(a.out, a.wiki_docs, a.web_docs, a.local_txt, a.seed)
    if a.stage in ("tokenizer", "all"):
        build_tokenizer(a.out, a.vocab_size, a.tok_mb)
    if a.stage in ("tokenize", "all"):
        tokenize_corpus(a.out, a.val_permille, a.max_tokens)


if __name__ == "__main__":
    main()

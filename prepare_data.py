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
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from mini_tokenizer import MiniTokenizer, save_meta, train_tokenizer
from text_cleaning import clean_web_text, clean_wikipedia_text, refilter_doc
# Réutilise les loaders SFT existants (mêmes sources, même nettoyage/filtre de personnalité) pour que le
# pré-entraînement et le SFT voient le même format de données, mix conversationnel inclus : Alpaca, OASST-FR,
# Magpie-FR, PIAF, Alpaca-GPT4-FR, Evol-Instruct-FR (single-turn) + ShareGPT-FR (vrais dialogues multi-tours).
from sft_data import (_conv, _piaf_window, _safe, load_alpaca_style, load_french_alpaca, load_jsonl, load_magpie_fr,
                      load_oasst_fr, load_piaf, load_sharegpt_fr, load_sharegpt_style)
from synthetic_qa import build_synthetic_qa

Conv = Dict  # {"messages": [{"role": "user"/"assistant", "content": str}, ...]}


def _hf_login(token: Optional[str]) -> None:
    """Connecte huggingface_hub si un token est fourni (--hf_token ou variable d'env HF_TOKEN /
    HUGGINGFACE_HUB_TOKEN) : utile seulement si l'une des sources devient 'gated'."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✓ Connecté à Hugging Face Hub.")


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


def load_fquad(max_examples: int, seed: int = 0) -> List[Conv]:
    """FQuAD (Illuin) : QA extractive sur Wikipédia FR, même format que PIAF déjà utilisé en SFT.
    Peut être 'gated' selon la révision HF -> passe --hf_token si besoin (voir _hf_login)."""
    from datasets import load_dataset
    ds = load_dataset("illuin/fquad", split="train")
    idx = np.random.default_rng(seed).permutation(len(ds))
    out: List[Conv] = []
    for i in idx:
        row = ds[int(i)]
        texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
        if not texts:
            continue
        ans = texts[0].strip()
        if not ans:
            continue
        ctx = _piaf_window(row["context"], int(starts[0]), len(texts[0]))
        if ans not in ctx:
            continue
        q = f"Réponds à la question à partir du texte.\n\nTexte : {ctx}\n\nQuestion : {row['question'].strip()}"
        a = ans if ans[-1] in ".!?" else ans + "."
        out.append(_conv(q, a))
        if len(out) >= max_examples:
            break
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Étapes
# ══════════════════════════════════════════════════════════════════════════════
def build_corpus(out: str, wiki_docs: int = 250_000, web_docs: int = 0, local_txt: Optional[List[str]] = None,
                 alpaca_conv: int = 0, oasst_conv: int = 0, magpie_conv: int = 0, piaf_conv: int = 0,
                 alpaca_gpt4_conv: int = 0, evol_conv: int = 0, sharegpt_conv: int = 0,
                 french_instruct_conv: int = 0, vonewman_conv: int = 0, fquad_conv: int = 0,
                 synthetic_repeat_pretrain: int = 0, persona: Optional[str] = None, persona_repeat: int = 50,
                 conv_repeat: int = 4, seed: int = 42, hf_token: Optional[str] = None) -> str:
    """Les sources 'text' (wiki/web/local) sont écrites une seule fois, dédupliquées.
    Les sources 'chat' (mix conversationnel réutilisé des loaders SFT + ShareGPT-FR multi-tours + synthétique +
    personnalité) sont écrites conv_repeat fois chacune (par défaut 4), sans dédup entre répétitions — voulu,
    elles sont bien plus petites que wiki/web et doivent peser proportionnellement dans le mélange final."""
    _hf_login(hf_token)
    os.makedirs(os.path.join(out, "corpus"), exist_ok=True)
    path = os.path.join(out, "corpus", "corpus.jsonl")
    stats: dict = {}
    seen_hashes = set()
    n_docs = n_chars = n_dupes = 0
    per_src: Dict[str, int] = {}

    def text_sources() -> Iterable:
        if wiki_docs > 0:
            yield "wiki", iter_wikipedia(wiki_docs, seed, stats)
        if web_docs > 0:
            yield "web", iter_fineweb_fr(web_docs, seed, stats)
        if local_txt:
            yield "local", iter_local_txt(local_txt)

    def chat_sources() -> Iterable:
        # décalage de seed pour que l'échantillon pris ici recoupe le moins possible celui pris plus tard par
        # sft_data.py (seed par défaut 0) : limite (sans l'éliminer) le risque de fuite pré-entraînement -> val SFT.
        cseed = seed + 1000
        if alpaca_conv > 0:
            yield "alpaca_conv", _safe("alpaca_conv", load_french_alpaca, alpaca_conv, seed=cseed)
        if oasst_conv > 0:
            yield "oasst_conv", _safe("oasst_conv", load_oasst_fr, oasst_conv, seed=cseed)
        if magpie_conv > 0:
            yield "magpie_conv", _safe("magpie_conv", load_magpie_fr, magpie_conv, seed=cseed)
        if piaf_conv > 0:
            yield "piaf_conv", _safe("piaf_conv", load_piaf, piaf_conv, seed=cseed)
        if alpaca_gpt4_conv > 0:
            yield "alpaca_gpt4_conv", _safe("alpaca_gpt4_conv", load_alpaca_style,
                                            "FreedomIntelligence/alpaca-gpt4-french", alpaca_gpt4_conv, seed=cseed)
        if evol_conv > 0:
            yield "evol_conv", _safe("evol_conv", load_alpaca_style, "FreedomIntelligence/evol-instruct-french",
                                     evol_conv, max_answer_chars=450, seed=cseed)
        if sharegpt_conv > 0:          # dialogues multi-tours
            yield "sharegpt_conv", _safe("sharegpt_conv", load_sharegpt_fr, sharegpt_conv, seed=cseed)
        if french_instruct_conv > 0:   # dialogues multi-tours (miroir ShareGPT de angeluriot/french_instruct, MIT)
            yield "french_instruct_conv", _safe("french_instruct_conv", load_sharegpt_style,
                                               "MaziyarPanahi/french_instruct_sharegpt", french_instruct_conv, seed=cseed)
        if vonewman_conv > 0:          # dialogues multi-tours (Apache-2.0, ~109k conversations dispo)
            yield "vonewman_conv", _safe("vonewman_conv", load_sharegpt_style,
                                        "vonewman/french-instruction-dataset", vonewman_conv, seed=cseed)
        if fquad_conv > 0:
            yield "fquad_conv", _safe("fquad_conv", load_fquad, fquad_conv, seed=cseed)
        if synthetic_repeat_pretrain > 0:
            yield "synthetic", [c for r in range(synthetic_repeat_pretrain) for c in build_synthetic_qa(seed=cseed + r)]
        if persona:
            yield f"persona(x{persona_repeat})", _safe("persona", load_jsonl, persona) * persona_repeat

    with open(path, "w", encoding="utf-8") as f:
        for name, it in text_sources():
            n_before = n_docs
            for doc in it:
                h = zlib.crc32(doc[:600].encode("utf-8")) ^ (len(doc) << 32)
                if h in seen_hashes:
                    n_dupes += 1
                    continue
                seen_hashes.add(h)
                f.write(json.dumps({"text": doc, "src": name}, ensure_ascii=False) + "\n")
                n_docs += 1
                n_chars += len(doc)
                per_src[name] = per_src.get(name, 0) + 1
                if n_docs % 20_000 == 0:
                    print(f"  … {n_docs:,} documents ({n_chars / 1e6:.0f} M caractères)")
            print(f"  ✓ {name:<16} {n_docs - n_before:,} documents")
        for name, it in chat_sources():
            n_before = n_docs
            try:
                for conv in it:
                    msgs = conv["messages"]
                    for _ in range(conv_repeat if not name.startswith("persona") else 1):  # persona a déjà son x{persona_repeat}
                        f.write(json.dumps({"messages": msgs, "src": name}, ensure_ascii=False) + "\n")
                        n_docs += 1
                        n_chars += sum(len(m["content"]) for m in msgs)
                        per_src[name] = per_src.get(name, 0) + 1
            except Exception as e:                                                         # noqa: BLE001
                print(f"  ✗ {name} interrompu en cours de route ({type(e).__name__}: {str(e)[:120]}) "
                      f"— {n_docs - n_before:,} exemples déjà écrits conservés")
            else:
                print(f"  ✓ {name:<16} {n_docs - n_before:,} exemples écrits "
                      f"(x{conv_repeat if not name.startswith('persona') else persona_repeat})")
    print(f"\nCorpus : {n_docs:,} documents, {n_chars / 1e6:.0f} M caractères, {n_dupes} doublons (texte) ignorés -> {path}")
    print("Détail par source :", ", ".join(f"{k}={v:,}" for k, v in per_src.items()))
    if stats.get("sent_total"):
        print(f"Nettoyage Wikipédia : {stats['sent_dropped'] / stats['sent_total'] * 100:.1f} % des phrases supprimées "
              f"(phrases trouées) ; {stats.get('wiki_raw', 0):,} articles lus")
    if n_docs == 0:
        raise RuntimeError("Corpus vide : vérifie tes sources (--wiki_docs / --web_docs / --local_txt / --sharegpt_conv …).")
    return path


def _record_text(rec: dict) -> str:
    """Texte brut d'un enregistrement, pour l'entraînement du tokenizer (les tokens spéciaux <|user|>/<|assistant|>/
    <|end|> sont de toute façon ajoutés explicitement au vocabulaire, pas besoin de leur forme textuelle ici)."""
    return rec["text"] if "messages" not in rec else "\n".join(m["content"] for m in rec["messages"])


def _iter_jsonl(path: str, stride: int = 1) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                yield _record_text(json.loads(line))


def _iter_records(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_tokenizer(out: str, vocab_size: int = 32_000, max_mb: int = 300) -> MiniTokenizer:
    corpus = os.path.join(out, "corpus", "corpus.jsonl")
    size = os.path.getsize(corpus)
    stride = max(1, int(size / (max_mb * 1e6)))            # sous-échantillonne si le corpus dépasse max_mb
    print(f"Entraînement du tokenizer BPE (vocab={vocab_size}) sur ~{min(size / 1e6, max_mb):.0f} Mo (1 doc sur {stride})…")
    tok = train_tokenizer(_iter_jsonl(corpus, stride), vocab_size=vocab_size, save_path=os.path.join(out, "tokenizer.json"))
    print(f"Tokenizer : {tok.vocab_size} tokens (embedding paddé à {tok.padded_vocab_size})")
    return tok


def tokenize_corpus(out: str, val_permille: int = 10, max_tokens: int = 0, batch_docs: int = 2000,
                    refilter: bool = False) -> dict:
    tok = MiniTokenizer(os.path.join(out, "tokenizer.json"))
    dtype = np.uint16 if tok.padded_vocab_size <= 65_535 else np.uint32
    pre_dir = os.path.join(out, "pretrain")
    os.makedirs(pre_dir, exist_ok=True)
    corpus = os.path.join(out, "corpus", "corpus.jsonl")
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    files = {s: open(os.path.join(pre_dir, f"{s}.bin"), "wb") for s in counts}
    total_chars = 0
    tokens_per_src: Dict[str, int] = {}

    def _write(key_text: str, ids: List[int], src: str):
        nonlocal total_chars
        split = "val" if (zlib.crc32(key_text[:400].encode("utf-8")) % 1000) < val_permille else "train"
        arr = np.asarray(ids + [tok.eot_id], dtype=dtype)          # <|endoftext|> après chaque document
        files[split].write(arr.tobytes())
        counts[split] += len(arr)
        docs[split] += 1
        total_chars += len(key_text)
        tokens_per_src[src] = tokens_per_src.get(src, 0) + len(arr)

    def flush(records: List[dict]):
        text_recs = [r for r in records if "messages" not in r]
        chat_recs = [r for r in records if "messages" in r]
        if text_recs:
            encoded = tok.encode_batch([r["text"] for r in text_recs])
            for r, ids in zip(text_recs, encoded):
                _write(r["text"], ids, r.get("src", "?"))
        for r in chat_recs:                                        # vrais tokens spéciaux <|user|>/<|assistant|>/<|end|>
            ids, _mask = tok.encode_chat(r["messages"])             # mask ignoré : pré-entraînement = loss sur tout
            key = " ".join(m["content"] for m in r["messages"])
            _write(key, ids, r.get("src", "?"))

    n_in = n_kept = chars_in = chars_kept = 0
    try:
        batch: List[dict] = []
        for rec in _iter_records(corpus):
            if refilter and "text" in rec:                 # re-nettoyage avec les règles actuelles, sans retélécharger
                n_in += 1
                chars_in += len(rec["text"])
                text = refilter_doc(rec["text"], rec.get("src", ""))
                if not text:
                    continue
                n_kept += 1
                chars_kept += len(text)
                rec = {**rec, "text": text}
            batch.append(rec)
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
        "tokens_per_src": tokens_per_src,
        "tokenizer": "tokenizer.json", "tokenizer_sha": tok.sha,
    }
    if refilter:
        meta["refiltered"] = True
        print(f"Re-nettoyage : {n_kept:,}/{n_in:,} documents gardés, {(1 - chars_kept / max(1, chars_in)) * 100:.1f} % des caractères retirés")
    save_meta(os.path.join(pre_dir, "meta.json"), meta)
    print(f"\nTokens : train={counts['train']:,} ({docs['train']:,} docs) | val={counts['val']:,} ({docs['val']:,} docs) "
          f"| {meta['chars_per_token']} caractères/token")
    print("Tokens par source :", ", ".join(f"{k}={v / 1e6:.0f}M" for k, v in sorted(tokens_per_src.items(), key=lambda kv: -kv[1])))
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
    # ── mix conversationnel dans le pré-entraînement (mêmes sources que sft_data.py, réutilisées) ────────────
    p.add_argument("--alpaca_conv", type=int, default=0, help="nb d'exemples French-Alpaca réutilisés en pré-entraînement")
    p.add_argument("--oasst_conv", type=int, default=0, help="nb d'exemples OASST-FR réutilisés en pré-entraînement")
    p.add_argument("--magpie_conv", type=int, default=0, help="nb d'exemples Magpie-FR réutilisés en pré-entraînement")
    p.add_argument("--piaf_conv", type=int, default=0, help="nb d'exemples PIAF réutilisés en pré-entraînement")
    p.add_argument("--alpaca_gpt4_conv", type=int, default=0, help="nb d'exemples Alpaca-GPT4-FR réutilisés")
    p.add_argument("--evol_conv", type=int, default=0, help="nb d'exemples Evol-Instruct-FR réutilisés")
    p.add_argument("--sharegpt_conv", type=int, default=0, help="nb de dialogues ShareGPT-FR (~5.6k dispo max)")
    p.add_argument("--french_instruct_conv", type=int, default=0, help="nb de dialogues MaziyarPanahi/french_instruct_sharegpt (multi-tours, ~85M tok dispo)")
    p.add_argument("--vonewman_conv", type=int, default=0, help="nb de dialogues vonewman/french-instruction-dataset (multi-tours, ~109k dispo)")
    p.add_argument("--fquad_conv", type=int, default=0, help="nb d'exemples FQuAD (QA extractive, peut nécessiter --hf_token)")
    p.add_argument("--synthetic_pretrain", type=int, default=0, help="nb de passes de synthetic_qa.py dans le pré-entraînement")
    p.add_argument("--persona", default=None, help="chemin vers personnalite.jsonl (optionnel, côté pré-entraînement)")
    p.add_argument("--persona_repeat_pretrain", type=int, default=50)
    p.add_argument("--conv_repeat", type=int, default=4, help="combien de fois chaque source conversationnelle est dupliquée")
    p.add_argument("--hf_token", default=None, help="token HF si une source devient gated (sinon lu depuis $HF_TOKEN)")
    p.add_argument("--vocab_size", type=int, default=32_000)
    p.add_argument("--tok_mb", type=int, default=300, help="Mo de texte max pour entraîner le tokenizer")
    p.add_argument("--val_permille", type=int, default=10, help="‰ de documents envoyés en validation")
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--refilter", action="store_true",
                   help="tokenize : ré-applique le nettoyage actuel à corpus.jsonl (sans retélécharger) ; le tokenizer existant est conservé")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.stage in ("corpus", "all"):
        build_corpus(a.out, a.wiki_docs, a.web_docs, a.local_txt,
                    alpaca_conv=a.alpaca_conv, oasst_conv=a.oasst_conv, magpie_conv=a.magpie_conv,
                    piaf_conv=a.piaf_conv, alpaca_gpt4_conv=a.alpaca_gpt4_conv, evol_conv=a.evol_conv,
                    sharegpt_conv=a.sharegpt_conv, french_instruct_conv=a.french_instruct_conv,
                    vonewman_conv=a.vonewman_conv, fquad_conv=a.fquad_conv,
                    synthetic_repeat_pretrain=a.synthetic_pretrain,
                    persona=a.persona, persona_repeat=a.persona_repeat_pretrain, conv_repeat=a.conv_repeat,
                    seed=a.seed, hf_token=a.hf_token)
    if a.stage in ("tokenizer", "all"):
        build_tokenizer(a.out, a.vocab_size, a.tok_mb)
    if a.stage in ("tokenize", "all"):
        tokenize_corpus(a.out, a.val_permille, a.max_tokens, refilter=a.refilter)


if __name__ == "__main__":
    main()

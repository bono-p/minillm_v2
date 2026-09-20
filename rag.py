"""
rag.py — mini-RAG : le modèle répond A PARTIR D'UN TEXTE que tu lui donnes (ta base de connaissances).

Pourquoi : un modèle de 49 M de paramètres n'a presque aucune connaissance factuelle fiable, mais le SFT lui apprend
(via PIAF) à répondre « à partir du texte ». On retrouve donc le bon paragraphe dans ta base (BM25, sans dépendance)
et on le lui donne dans le MÊME format que pendant le SFT :

    Réponds à la question à partir du texte.\\n\\nTexte : …\\n\\nQuestion : …

Base de connaissances : un dossier (ou fichier) de .txt / .md (paragraphes séparés par une ligne vide) et/ou .jsonl
({"text": "..."}). Voir knowledge/ pour un exemple.

Usage :
    python rag.py --ckpt checkpoints/sft/best.pt --kb knowledge/
    python rag.py --ckpt checkpoints/sft/best.pt --kb knowledge/ --question "Quelle est la capitale du Cameroun ?"

⚠️ Expérimental : la qualité dépend du SFT (parts d'exemples "à partir du texte") et de la base ; le passage retrouvé est
toujours affiché pour que tu puisses juger si la réponse est fidèle.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import List, Optional, Sequence, Tuple

_STOP = set("""le la les un une des du de d l et ou a à au aux en dans sur par pour avec sans que qui quoi quel quelle quels quelles
est sont etre c ce cet cette ces se sa son ses ne pas plus tu vous il elle on nous ils elles y mon ma mes ton ta tes je me te
comment combien pourquoi quand ou est-ce qu""".split())

RAG_INSTRUCTION = "Réponds à la question à partir du texte."


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    return [w for w in words if w not in _STOP and (len(w) > 1 or w.isdigit())]


def terms(text: str) -> List[str]:
    """Mots + bigrammes consécutifs (sans mots vides) : « capitale cameroun » pèse plus que « capitale » et « cameroun » séparés."""
    w = tokenize(text)
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


class BM25Index:
    """BM25 (Okapi) minimal sur une liste de passages (unigrammes + bigrammes)."""

    def __init__(self, passages: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self.docs = [Counter(terms(p)) for p in self.passages]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 3) -> List[Tuple[float, int]]:
        q = terms(query)
        scored = []
        for i, d in enumerate(self.docs):
            score = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if score > 0:
                scored.append((score, i))
        return sorted(scored, reverse=True)[:k]


# ── base de connaissances ───────────────────────────────────────────────────
def _split_long(paragraph: str, max_chars: int = 700) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def load_knowledge(path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)) if os.path.isdir(path) else [path]
    passages: List[str] = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if not os.path.isfile(f) or ext not in (".txt", ".md", ".jsonl"):
            continue
        with open(f, encoding="utf-8", errors="ignore") as fh:
            if ext == ".jsonl":
                for line in fh:
                    if line.strip():
                        text = json.loads(line).get("text", "").strip()
                        passages += _split_long(text) if text else []
            else:
                for para in fh.read().split("\n\n"):
                    para = " ".join(para.split())
                    if len(para) >= 30 and not para.startswith("#"):
                        passages += _split_long(para)
    return passages


# ── prompt ──────────────────────────────────────────────────────────────────
def focus_passage(passage: str, question: str, max_chars: int) -> str:
    """Si le passage est long, garde la fenêtre de phrases la plus proche de la question."""
    if len(passage) <= max_chars:
        return passage
    sentences = re.split(r"(?<=[.!?…])\s+", passage)
    q = set(tokenize(question))
    overlaps = [len(q & set(tokenize(s))) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: overlaps[i])
    lo = hi = best
    size = len(sentences[best])
    while True:
        grew = False
        if lo > 0 and size + len(sentences[lo - 1]) + 1 <= max_chars:
            lo -= 1
            size += len(sentences[lo]) + 1
            grew = True
        if hi < len(sentences) - 1 and size + len(sentences[hi + 1]) + 1 <= max_chars:
            hi += 1
            size += len(sentences[hi]) + 1
            grew = True
        if not grew:
            break
    return " ".join(sentences[lo:hi + 1])


def build_rag_messages(question: str, passage: str, max_chars: int = 600) -> List[dict]:
    ctx = focus_passage(passage, question, max_chars)
    return [{"role": "user", "content": f"{RAG_INSTRUCTION}\n\nTexte : {ctx}\n\nQuestion : {question.strip()}"}]


def rag_answer(model, tok, index: BM25Index, question: str, *, max_new_tokens: int = 60, max_chars: int = 600,
               **gen_kw) -> Tuple[str, Optional[str]]:
    """Renvoie (réponse, passage utilisé). Sans passage pertinent : réponse "libre" du modèle (passage=None)."""
    from generate import chat_reply
    hits = index.search(question, k=1)
    if not hits:
        return chat_reply(model, tok, [{"role": "user", "content": question}], max_new_tokens=max_new_tokens, **gen_kw), None
    passage = index.passages[hits[0][1]]
    budget = model.cfg.max_seq_len - max_new_tokens
    chars = max_chars
    while True:                                              # le prompt doit tenir dans la fenêtre du modèle
        messages = build_rag_messages(question, passage, chars)
        n_tok = len(tok.encode_chat(messages, add_generation_prompt=True)[0])
        if n_tok <= budget or chars <= 120:
            break
        chars = int(chars * 0.8)
    return chat_reply(model, tok, messages, max_new_tokens=max_new_tokens, **gen_kw), passage


def main():
    from generate import load_model
    p = argparse.ArgumentParser(description="MiniLLM — mini-RAG")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--kb", default="knowledge", help="fichier ou dossier (.txt/.md/.jsonl)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--question", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.2)
    a = p.parse_args()
    passages = load_knowledge(a.kb)
    if not passages:
        raise SystemExit(f"Base de connaissances vide : {a.kb}")
    index = BM25Index(passages)
    model, tok, _ = load_model(a.ckpt, a.tokenizer, a.device)
    print(f"{len(passages)} passages indexés.")
    gen = dict(temperature=a.temperature, top_k=20, top_p=0.9, repetition_penalty=1.05, no_repeat_ngram=0)
    while True:
        q = a.question or input("\nQuestion > ").strip()
        if not q:
            break
        ans, ctx = rag_answer(model, tok, index, q, **gen)
        print(f"\n📄 Passage : {ctx[:300] if ctx else '(aucun passage pertinent : réponse libre)'}\n💬 Réponse : {ans}")
        if a.question:
            break


if __name__ == "__main__":
    main()

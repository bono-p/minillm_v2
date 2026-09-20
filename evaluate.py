"""
evaluate.py — mesures qui parlent vraiment de la qualité (pas seulement la loss).

  perplexity : val_loss / perplexité sur le val FIXE du pré-entraînement.
  qa         : sur le val du SFT, génère (greedy) la réponse à N questions et calcule
               Exact-Match et F1 (normalisés : minuscules, sans accents/ponctuation/articles) + % de réponses
               qui se terminent proprement (<|end|>) au lieu de partir dans la nature.
  demo       : quelques prompts fixes pour "voir" le modèle.

Exemples :
  python evaluate.py perplexity --ckpt checkpoints/pretrain/best.pt --data_dir data/pretrain
  python evaluate.py qa --ckpt checkpoints/sft/best.pt --sft_dir data/sft --n 300
  python evaluate.py demo --ckpt checkpoints/sft/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from typing import Dict, List

import numpy as np
import torch

from data import PretrainData
from generate import chat_reply, load_model, stream_tokens

DEMO_PROMPTS = [
    "Bonjour", "Qui es-tu ?", "Comment tu t'appelles ?", "Qui t'a créé ?", "Quelle est la capitale du Cameroun ?", "Combien font 12 plus 7 ?",
    "Quel jour vient après le mardi ?", "Quel est le contraire de grand ?", "Combien de jours y a-t-il dans une semaine ?",
    "Quel temps fera-t-il demain ?", "Explique ce qu'est la photosynthèse.", "Raconte-moi une blague.",
]

_ARTICLES = {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux"}


def normalize(s: str) -> List[str]:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if t not in _ARTICLES]


def f1_score(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return float(p == r)
    common = {}
    for t in p:
        if t in r:
            common[t] = min(p.count(t), r.count(t))
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def perplexity(ckpt: str, data_dir: str, seq_len: int = 512, n_batches: int = 200, batch_size: int = 16, device: str = "auto") -> Dict:
    from train import evaluate as run_eval
    import contextlib
    model, tok, meta = load_model(ckpt, device=device)
    data = PretrainData(data_dir, seq_len, batch_size)
    dev = next(model.parameters()).device
    loss, n = run_eval(model, data.val_batches(n_batches), contextlib.nullcontext(), dev, False)
    res = {"val_loss": loss, "perplexity": math.exp(loss), "tokens": n, "iter": meta.get("iter")}
    print(json.dumps(res, indent=2))
    return res


@torch.inference_mode()
def qa_eval(ckpt: str, sft_dir: str, n: int = 200, max_new: int = 64, device: str = "auto", show: int = 12) -> Dict:
    model, tok, _ = load_model(ckpt, device=device)
    tokens = np.load(os.path.join(sft_dir, "val.tokens.npy"), mmap_mode="r")
    mask = np.load(os.path.join(sft_dir, "val.mask.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(sft_dir, "val.offsets.npy"))
    total = len(offsets) - 1
    idxs = np.random.default_rng(0).permutation(total)[:n]
    em = f1 = ended = done = 0
    shown = 0
    for i in idxs:
        ids = [int(t) for t in tokens[offsets[i]:offsets[i + 1]]]
        m = np.asarray(mask[offsets[i]:offsets[i + 1]])
        first = int(np.argmax(m))
        try:
            j = ids.index(tok.end_id, first)
        except ValueError:
            continue
        prompt, ref = ids[:first], tok.decode(ids[first:j], skip_special=True)
        if len(prompt) + max_new > model.cfg.max_seq_len:
            continue
        pred_ids = list(stream_tokens(model, prompt, max_new_tokens=max_new, temperature=0.0, repetition_penalty=1.0,
                                      stop_ids=tok.stop_ids(), forbidden=tok.forbidden_in_answer(),
                                      valid_vocab=tok.vocab_size))
        pred = tok.decode(pred_ids, skip_special=True).strip()
        done += 1
        em += normalize(pred) == normalize(ref)
        f1 += f1_score(pred, ref)
        ended += len(pred_ids) < max_new                       # s'est arrêté tout seul
        if shown < show:
            q = tok.decode(prompt, skip_special=True)[-160:].replace("\n", " ")
            print(f"\nQ : …{q}\n  attendu : {ref[:140]}\n  obtenu  : {pred[:140]}")
            shown += 1
    res = {"n": done, "exact_match": round(em / max(1, done), 4), "f1": round(f1 / max(1, done), 4),
           "ends_properly": round(ended / max(1, done), 4)}
    print("\n" + json.dumps(res, indent=2))
    return res


def demo(ckpt: str, device: str = "auto", prompts: List[str] = None) -> None:
    model, tok, _ = load_model(ckpt, device=device)
    for q in prompts or DEMO_PROMPTS:
        a = chat_reply(model, tok, [{"role": "user", "content": q}], max_new_tokens=80, temperature=0.5, top_k=30,
                       top_p=0.9, repetition_penalty=1.1, no_repeat_ngram=3, seed=0)
        print(f"Toi      : {q}\nMiniLLM  : {a}\n")


def main():
    p = argparse.ArgumentParser(description="MiniLLM v2 — évaluation")
    p.add_argument("what", choices=["perplexity", "qa", "demo"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/pretrain")
    p.add_argument("--sft_dir", default="data/sft")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.what == "perplexity":
        perplexity(a.ckpt, a.data_dir, a.seq_len, device=a.device)
    elif a.what == "qa":
        qa_eval(a.ckpt, a.sft_dir, a.n, device=a.device)
    else:
        demo(a.ckpt, a.device)


if __name__ == "__main__":
    main()

"""
skills_eval.py — mesure de compétences CHIFFRÉES, sur des problèmes que le SFT v3.1 n'a jamais vus.

Les problèmes testés sont ceux que `synthetic_plus.is_heldout()` écarte de l'entraînement (~10 %). Donc :
  * un score élevé = le modèle a généralisé la règle (pas récité) ;
  * on peut comparer n'importe quels checkpoints (SFT v3, v3.1…) sur EXACTEMENT les mêmes problèmes.

Correction : le dernier nombre de la réponse doit être le bon résultat (calculs, suites, problèmes).
(Pas de test sur les jours de la semaine : il n'y a que 7 jours, impossible d'en tenir une part à l'écart honnêtement.)

Usage : python skills_eval.py --ckpt checkpoints/sft_v3_1/best.pt [--n 60]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from typing import Dict, List, Tuple

from synthetic_plus import (PROBLEM_KINDS, _arith_item, arithmetic_pairs, arithmetic_question, is_heldout,
                            problem_keys_and_text)


def _last_int(text: str):
    nums = re.findall(r"-?\d+", text.replace(" ", "").replace("\u202f", "")) or re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def build_items(n: int = 60, seed: int = 123) -> Dict[str, List[Tuple[str, object]]]:
    """{compétence: [(question, attendu), ...]} — uniquement des problèmes absents de l'entraînement."""
    rng = random.Random(seed)
    items: Dict[str, List[Tuple[str, object]]] = {}
    for op, label in (("add", "addition"), ("sub", "soustraction"), ("mul", "multiplication"), ("div", "division")):
        pairs = [p for p in arithmetic_pairs(op) if is_heldout(f"{op}|{p[0]}|{p[1]}")]
        rng.shuffle(pairs)
        items[label] = [(arithmetic_question(op, a, b, random.Random(0))["messages"][0]["content"], _arith_item(op, a, b)[0])
                        for a, b in pairs[:n]]
    seqs = []
    for step in range(1, 11):
        for start in range(0, 31):
            for length in (4, 5):
                if is_heldout(f"seq|{start}|{step}|{length}"):
                    seq = [start + step * i for i in range(length)]
                    seqs.append((f"Quel nombre vient après {', '.join(map(str, seq))} ?", start + step * length))
    rng.shuffle(seqs)
    items["suites"] = seqs[:n]
    probs, seen, tries = [], set(), 0
    while len(probs) < n and tries < 20000:
        tries += 1
        key, q, _, r = problem_keys_and_text(rng.choice(PROBLEM_KINDS), rng)
        if is_heldout(key) and key not in seen:
            seen.add(key)
            probs.append((q, r))
    items["problèmes"] = probs
    return items


def main():
    p = argparse.ArgumentParser(description="MiniLLM — compétences chiffrées sur problèmes tenus à l'écart")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    import torch  # noqa: F401
    from evaluate import chat_reply, load_model

    model, tok, _ = load_model(a.ckpt, device=a.device)
    results: Dict[str, float] = {}
    for skill, qa in build_items(a.n).items():
        ok = 0
        shown = 0
        for q, expected in qa:
            reply = chat_reply(model, tok, [{"role": "user", "content": q}], max_new_tokens=60,
                               temperature=0.0, repetition_penalty=1.0)
            good = _last_int(reply) == expected
            ok += bool(good)
            if not good and shown < 2:
                print(f"  ✗ [{skill}] {q}\n      attendu {expected} | obtenu : {reply[:90]}")
                shown += 1
        results[skill] = round(ok / max(1, len(qa)), 3)
        print(f"{skill:<15} {ok:>3}/{len(qa):<3} = {results[skill] * 100:5.1f} %")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

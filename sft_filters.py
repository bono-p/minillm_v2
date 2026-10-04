"""
sft_filters.py — filtres de qualité pour les sources EXTERNES du SFT (French Alpaca, GPT-4 FR, OASST, PIAF…).

Ils ciblent trois défauts observés dans les sorties du SFT v3 / v3.1 :
  * refus génériques  : « Je suis désolé, je ne peux pas générer de texte » (apparus après l'ajout d'une source GPT-4) ;
  * boucles           : « 1. Le Louvre 2. Le Louvre 3. Le Louvre … » (le modèle imite des listes dégénérées) ;
  * écho de question  : la « réponse » reformule la question (« Dans quelle région se trouve Garoua ? »).

Jamais appliqués au synthétique ni à la personnalité (écrits à la main : « Non, je ne peux pas te donner la météo »
y est une bonne réponse). Aucun import lourd : testable hors GPU.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

Conv = Dict

_REFUSALS = (
    "je suis désolé", "je suis désolée", "désolé,", "désolée,", "je ne peux pas", "je ne suis pas en mesure",
    "je n'ai pas la capacité", "je ne suis pas capable", "je n'ai pas accès", "en tant qu'ia", "en tant qu'intelligence artificielle",
    "en tant que modèle", "en tant qu'assistant", "as an ai", "i'm sorry", "i cannot",
)
_WORD = re.compile(r"[a-zà-ÿœ']+")


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


def is_refusal(answer: str) -> bool:
    a = _norm(answer)
    return any(k in a for k in _REFUSALS)


def has_repetition_loop(answer: str) -> bool:
    """Vrai si une paire de mots « pleins » se répète ≥ 4 fois, ou si le vocabulaire est très pauvre (≥ 15 mots)."""
    words = _WORD.findall(_norm(answer))
    if len(words) >= 15 and len(set(words)) / len(words) < 0.40:
        return True
    pairs = Counter((a, b) for a, b in zip(words, words[1:]) if len(a) > 3 or len(b) > 3)
    return bool(pairs) and max(pairs.values()) >= 4


def echoes_question(question: str, answer: str) -> bool:
    """Vrai si la réponse est la question (identique, ou question reformulée qui finit par « ? »)."""
    q, a = _norm(question).strip(), _norm(answer).strip()
    qw, aw = set(_WORD.findall(q)), set(_WORD.findall(a))
    if not aw:
        return True
    if re.sub(r"\W+", " ", q).strip() == re.sub(r"\W+", " ", a).strip():
        return True
    jac = len(qw & aw) / len(qw | aw) if (qw | aw) else 0.0
    return a.rstrip().endswith("?") and jac >= 0.6


def problem(conv: Conv) -> Optional[str]:
    """None si l'exemple est correct, sinon la raison du rejet ('refus', 'boucle', 'écho')."""
    msgs = conv["messages"]
    question = next((m["content"] for m in msgs if m["role"] == "user"), "")
    for m in msgs:
        if m["role"] != "assistant":
            continue
        if is_refusal(m["content"]):
            return "refus"
        if has_repetition_loop(m["content"]):
            return "boucle"
        if echoes_question(question, m["content"]):
            return "écho"
    return None


def filter_quality(convs: List[Conv]) -> Tuple[List[Conv], Dict[str, int]]:
    kept: List[Conv] = []
    counts: Dict[str, int] = {}
    for c in convs:
        why = problem(c)
        if why is None:
            kept.append(c)
        else:
            counts[why] = counts.get(why, 0) + 1
    return kept, counts


def persona_repeat_for_share(share: float, n_other: int, n_persona: int, minimum: int = 1) -> int:
    """Nombre de copies de chaque Q/R de personnalité pour qu'elle pèse `share` (ex. 0.06) du train final."""
    if share <= 0 or n_persona <= 0:
        return minimum
    return max(minimum, round(share * n_other / ((1.0 - share) * n_persona)))

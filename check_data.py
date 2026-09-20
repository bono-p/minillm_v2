"""
check_data.py — vérifie tes fichiers .jsonl AVANT de lancer un entraînement (personnalité, datasets de faits, données extra).

    python check_data.py personnalite.jsonl datasets/faits_cameroun_afrique.jsonl

Erreurs (code de sortie 1) : fichier illisible, ligne qui n'est pas du JSON, question/réponse vide ou manquante, token spécial « <| ».
Avertissements (ne bloquent pas) : question en double, réponse très longue, faits qui vieillissent (président, « depuis 2019 »…),
double espace, pas de ponctuation finale, apostrophes typographiques mélangées.
Utilise `--strict` pour que les avertissements bloquent aussi.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from typing import List, Tuple

_AGING = re.compile(r"\b(président|premier ministre|actuellement|actuel|depuis (?:19|20)\d\d|nommé|au pouvoir|en (?:2024|2025|2026|2027))\b", re.I)


def check_file(path: str, max_answer_chars: int = 300) -> Tuple[List[str], List[str], int]:
    errors: List[str] = []
    warns: List[str] = []
    try:
        raw = open(path, "rb").read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [f"{path} : fichier illisible ({e})"], [], 0
    seen = collections.defaultdict(list)
    n = 0
    for i, line in enumerate(raw.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"ligne {i} : JSON invalide ({e.msg}) -> {line[:60]!r}")
            continue
        if not isinstance(row, dict):
            errors.append(f"ligne {i} : ce n'est pas un objet JSON")
            continue
        if "messages" in row:                                   # format multi-tours : on vérifie juste la structure
            ok = isinstance(row["messages"], list) and row["messages"] and all(
                isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") and str(m.get("content", "")).strip()
                for m in row["messages"])
            if not ok:
                errors.append(f"ligne {i} : 'messages' invalide")
            n += 1
            continue
        q, a = row.get("question"), row.get("answer", row.get("text"))
        if "text" in row and "question" not in row:
            q = row["text"][:40]
        if not isinstance(q, str) or not q.strip() or not isinstance(a, str) or not a.strip():
            errors.append(f"ligne {i} : 'question' ou 'answer' manquant ou vide -> {line[:60]!r}")
            continue
        n += 1
        if "<|" in q + a:
            errors.append(f"ligne {i} : contient « <| » (réservé aux tokens spéciaux)")
        seen[q.strip().lower()].append(i)
        if len(a) > max_answer_chars:
            warns.append(f"ligne {i} : réponse très longue ({len(a)} caractères) : un petit modèle apprend mieux sur du court")
        if _AGING.search(a) or _AGING.search(q):
            warns.append(f"ligne {i} : fait qui peut vieillir ({_AGING.search(a + ' ' + q).group(0)!r}) : {q[:50]!r}")
        if "  " in a or "  " in q:
            warns.append(f"ligne {i} : double espace")
        if a.rstrip()[-1] not in ".!?»)…\"":
            warns.append(f"ligne {i} : la réponse ne se termine pas par une ponctuation")
        if "’" in q + a and "'" in q + a:
            warns.append(f"ligne {i} : apostrophes ' et ’ mélangées")
    for q, lines in seen.items():
        if len(lines) > 1:
            warns.append(f"question en double (lignes {lines}) : « {q[:60]} »")
    return errors, warns, n


def main() -> int:
    p = argparse.ArgumentParser(description="Vérifie des fichiers .jsonl de Q/R")
    p.add_argument("files", nargs="+")
    p.add_argument("--strict", action="store_true", help="les avertissements font aussi échouer")
    a = p.parse_args()
    bad = False
    for f in a.files:
        errors, warns, n = check_file(f)
        status = "✓" if not errors else "✗"
        print(f"{status} {f} : {n} lignes valides, {len(errors)} erreur(s), {len(warns)} avertissement(s)")
        for e in errors:
            print("   ERREUR :", e)
        for w in warns[:25]:
            print("   avertissement :", w)
        if len(warns) > 25:
            print(f"   … et {len(warns) - 25} autres avertissements")
        bad = bad or bool(errors) or (a.strict and bool(warns))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

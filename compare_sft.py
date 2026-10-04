"""
compare_sft.py — compare plusieurs checkpoints SFT sur EXACTEMENT les mêmes mesures, dans un seul tableau.

Pourquoi : la val_loss ne suffit pas à choisir (jeux de validation différents d'un run à l'autre, et elle ignore la
persona, les faits et les refus). Mesures par checkpoint :
  persona   EM / F1 sur les 132 Q/R de personnalité (test de mémorisation de l'identité)
  qa        EM / F1 / ends_properly sur un jeu de validation COMMUN à tous les modèles (--val_dir)
  basics    % de connaissances de base conservées (basics_eval.py)
  openqa    sur les 337 questions : % de réponses distinctes, nombre de refus, nombre de boucles

Chaque mesure est lancée dans un sous-processus (evaluate.py / basics_eval.py) et son JSON final est lu : aucune
dépendance à l'intérieur d'evaluate.py. Les sorties brutes de l'openqa sont écrites dans --out_dir.

Usage : python compare_sft.py --val_dir data/sft_A --out_dir evals --ckpt A=checkpoints/sft_A/final.pt --ckpt A_best=...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

from sft_filters import has_repetition_loop, is_refusal


def last_json(text: str) -> Optional[Dict]:
    """Dernier objet JSON {...} de la sortie (accolades équilibrées en partant de la fin)."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:end + 1])
                    except ValueError:
                        break
        end = text.rfind("}", 0, end)
    return None


def openqa_stats(text: str) -> Dict[str, float]:
    """Statistiques de collapse sur les lignes « A : … » de la sortie de `evaluate.py openqa`."""
    answers = [m.group(1).strip() for m in re.finditer(r"^A : (.*)$", text, flags=re.M)]
    if not answers:
        return {"n": 0, "distinct": 0.0, "refusals": 0, "loops": 0}
    return {"n": len(answers), "distinct": round(len(set(answers)) / len(answers), 3),
            "refusals": sum(is_refusal(a) for a in answers), "loops": sum(has_repetition_loop(a) for a in answers)}


def _run(args: List[str]) -> str:
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ⚠️  échec : {' '.join(args)}\n{r.stderr[-400:]}")
    return r.stdout


def evaluate_one(ckpt: str, val_dir: str, qa_n: int = 300) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["persona"] = last_json(_run(["evaluate.py", "persona", "--ckpt", ckpt])) or {}
    out["qa"] = last_json(_run(["evaluate.py", "qa", "--ckpt", ckpt, "--sft_dir", val_dir, "--n", str(qa_n)])) or {}
    out["basics"] = last_json(_run(["basics_eval.py", "--ckpt", ckpt])) or {}
    raw = _run(["evaluate.py", "openqa", "--ckpt", ckpt, "--n", "1000"])
    out["openqa_raw"] = raw
    out["openqa"] = openqa_stats(raw)
    return out


def _pct(x) -> str:
    return "  -  " if x is None else f"{x * 100:5.1f}"


def _f1(x) -> str:
    return "  -  " if x is None else f"{x * 100:5.1f}"


def table(results: Dict[str, Dict[str, object]]) -> str:
    head = f"{'modèle':<10} {'pers.EM':>7} {'pers.F1':>7} {'qa.EM':>6} {'qa.F1':>6} {'fin ok':>6} {'basics':>6} {'distinct':>8} {'refus':>5} {'boucles':>7}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        p, q, b, o = r["persona"], r["qa"], r["basics"], r["openqa"]
        lines.append(f"{name:<10} {_pct(p.get('exact_match')):>7} {_f1(p.get('f1_moyen')):>7} "
                     f"{_pct(q.get('exact_match')):>6} {_f1(q.get('f1')):>6} {_pct(q.get('ends_properly')):>6} "
                     f"{_pct(b.get('basics')):>6} {_pct(o.get('distinct')):>8} {o.get('refusals', 0):>5} {o.get('loops', 0):>7}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="MiniLLM — tableau comparatif de plusieurs checkpoints SFT")
    ap.add_argument("--ckpt", action="append", required=True, help="NOM=chemin/vers/checkpoint.pt (répétable)")
    ap.add_argument("--val_dir", required=True, help="dossier de validation SFT COMMUN à tous les modèles")
    ap.add_argument("--out_dir", default="evals")
    ap.add_argument("--qa_n", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"(absent : {path})")
            continue
        print(f"→ évaluation de {name} …")
        results[name] = evaluate_one(path, a.val_dir, a.qa_n)
        with open(os.path.join(a.out_dir, f"openqa_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(results[name].pop("openqa_raw"))                       # type: ignore[arg-type]
    print("\n" + table(results))
    print("\nLecture : pers.EM haut = identité mémorisée ; basics haut = faits conservés ; distinct bas = réponses qui se "
          "ressemblent toutes (effondrement) ; refus/boucles = défauts à éviter.")


if __name__ == "__main__":
    main()

"""
compare_knowledge.py — compare plusieurs checkpoints sur le test de connaissances (204 questions), de façon APPARIÉE.

Pour chaque modèle : réussite et intervalle de confiance, puis écart avec le modèle de référence (le premier) et p-valeur du
test exact de McNemar sur les mêmes questions. Chaque évaluation est mise en cache dans --out_dir : une interruption ne
fait pas refaire les modèles déjà évalués.

Usage : python compare_knowledge.py --ckpt PUB=/content/pub/best.pt --ckpt C=checkpoints/sft_C/final.pt --out_dir evals_connaissances
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from compare_sft import ensure_tokenizer, last_json
from knowledge_eval import mcnemar_exact


def _sig(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def evaluate(name: str, ckpt: str, out_dir: str, runner=None) -> Optional[Dict]:
    """Évalue (ou relit depuis le cache si le checkpoint n'a pas changé) un modèle. `runner(args) -> (stdout, ok)` injectable."""
    os.makedirs(out_dir, exist_ok=True)
    js_path, sig_path = os.path.join(out_dir, f"{name}.json"), os.path.join(out_dir, f"{name}.sig")
    sig = _sig(ckpt) if os.path.exists(ckpt) else ""
    if os.path.exists(js_path) and os.path.exists(sig_path) and open(sig_path).read() == sig:
        d = json.load(open(js_path, encoding="utf-8"))
        if "ok_ids" in d:
            print(f"✓ {name} : déjà évalué")
            return d
    if runner is None:
        def runner(args):
            r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"❌ {name} : échec\n{r.stderr[-500:]}")
            return r.stdout, r.returncode == 0
    out, ok = runner(["knowledge_eval.py", "--ckpt", ckpt])
    d = last_json(out) if ok else None
    if not d or "ok_ids" not in d:
        return None
    json.dump(d, open(js_path, "w", encoding="utf-8"), ensure_ascii=False)
    open(sig_path, "w").write(sig)
    open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8").write(out)
    print(f"✓ {name} : {d['correct']}/{d['n']}")
    return d


def table(results: Dict[str, Dict], ref: str) -> str:
    head = f"{'modèle':<8} {'juste':>9} {'%':>6} {'IC95':>13} {'Δ vs ' + ref:>12} {'+/-':>7} {'p':>7}"
    lines = [head, "-" * len(head)]
    for name, d in results.items():
        lo, hi = d["ci95"]
        base = f"{name:<8} {d['correct']:>4}/{d['n']:<4} {d['knowledge'] * 100:6.1f} [{lo * 100:4.1f};{hi * 100:5.1f}]"
        if name == ref or ref not in results:
            lines.append(base + f"{'(réf.)':>12}")
            continue
        a, b, p = mcnemar_exact(d["ok_ids"], results[ref]["ok_ids"])
        delta = (d["knowledge"] - results[ref]["knowledge"]) * 100
        lines.append(base + f" {delta:+11.1f} {a:>3}/{b:<3} {p:7.3f}")
    return "\n".join(lines)


def category_table(results: Dict[str, Dict]) -> str:
    """Réussite (%) par catégorie et par modèle : sépare connaissances du monde et questions de langue."""
    cats = sorted({c for d in results.values() for c in d.get("par_categorie", {})})
    names = list(results)
    head = f"{'catégorie':<14} {'n':>3} " + " ".join(f"{n:>6}" for n in names)
    lines = [head, "-" * len(head)]
    for c in cats:
        n = next((d["par_categorie"][c]["n"] for d in results.values() if c in d.get("par_categorie", {})), 0)
        lines.append(f"{c:<14} {n:>3} " + " ".join(
            f"{results[m]['par_categorie'][c]['acc'] * 100:6.1f}" if c in results[m].get("par_categorie", {}) else f"{'-':>6}" for m in names))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="MiniLLM — connaissances : comparaison appariée (McNemar)")
    ap.add_argument("--ckpt", action="append", required=True, help="NOM=chemin (répétable) ; le 1er est la référence")
    ap.add_argument("--out_dir", default="evals_connaissances")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data_dir", default=None, help="dossier data/ contenant tokenizer.json (si absent à côté des checkpoints)")
    a = ap.parse_args()
    results: Dict[str, Dict] = {}
    ref = None
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"(absent : {path})")
            continue
        if ensure_tokenizer(path, a.tokenizer, os.path.join(a.data_dir, "x") if a.data_dir else "") is None:
            print(f"❌ {name} : tokenizer.json introuvable")
            continue
        d = evaluate(name, path, a.out_dir)
        if d:
            results[name] = d
            ref = ref or name
    if results:
        print("\n" + table(results, ref))
        print("\nRéussite par catégorie (%) :\n" + category_table(results))
        print("\n+/- = questions réussies par ce modèle seulement / par la référence seulement ; p = test exact de McNemar "
              "(p < 0,05 : l'écart n'est probablement pas dû au hasard des questions).")


if __name__ == "__main__":
    main()

"""
release_tools.py — outils de publication d'un modèle miniLLM (nom du dépôt, graphiques, fiche de chiffres RÉELS).

Principe : tout chiffre qui finira dans le README est LU ici (checkpoint, log.jsonl, meta.json, sorties d'évaluation),
jamais recopié à la main. `collect_stats` écrit une fiche JSON ; le README est écrit à partir d'elle.

Les fonctions « log » et « nom » n'ont pas besoin de torch ; seules `read_checkpoint_summary` et `collect_stats`
chargent des checkpoints (import paresseux).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

CORPUS_TRAIN_TOKENS = 3_226_198_113          # corpus d'entraînement du pré-entraînement (data/meta du corpus)
BASE_USER = "bonopassale"


# ── nom du dépôt : miniLLM_v{version}-{params}M-{it}it_{tokens:.2f}Btoks_{epoques:.1f}ep_{AAAAMMJJ} ──────────────────────
def repo_name(version: str, n_params: int, iters: int, tokens_seen: int, date: Optional[_dt.date] = None,
              corpus_tokens: int = CORPUS_TRAIN_TOKENS) -> str:
    """Même nomenclature que miniLLM_v2.1-49M-42500it_2.79Btoks_0.9ep_20261001 (tokens à 2 décimales, époques à 1)."""
    date = date or _dt.date.today()
    return (f"miniLLM_v{version}-{round(n_params / 1e6)}M-{int(iters)}it_{tokens_seen / 1e9:.2f}Btoks_"
            f"{tokens_seen / corpus_tokens:.1f}ep_{date:%Y%m%d}")


# ── lecture d'un log.jsonl ────────────────────────────────────────────────────────────────────────────────────────────
def read_log(path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """(train, val) : {itération: loss}. Doublons (sessions reprises) : la dernière ligne gagne. Lignes illisibles ignorées."""
    train: Dict[int, float] = {}
    val: Dict[int, float] = {}
    if not os.path.exists(path):
        return train, val
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "train" and "loss" in e:
                train[int(e["it"])] = float(e["loss"])
            elif e.get("type") == "eval" and "val_loss" in e:
                val[int(e["it"])] = float(e["val_loss"])
    return train, val


def log_summary(path: str) -> Dict[str, object]:
    """Résumé honnête d'un log : combien de lignes, où est le minimum, jusqu'où va le log (il peut être incomplet)."""
    train, val = read_log(path)
    out: Dict[str, object] = {"log": path, "n_train_points": len(train), "n_evals": len(val)}
    if train:
        out["last_train_it"] = max(train)
    if val:
        best_it = min(val, key=lambda i: val[i])
        last_it = max(val)
        out.update(val_min=round(val[best_it], 4), val_min_it=best_it, val_last=round(val[last_it], 4), val_last_it=last_it,
                   val_first=round(val[min(val)], 4), val_first_it=min(val))
    return out


# ── graphiques : même style que assets/loss_pretrain.png et assets/loss_sft.png (2 panneaux) ─────────────────────────────────
def make_loss_plot(log_path: str, out_png: str, smooth: int = 1) -> Dict[str, object]:
    """Gauche : loss d'entraînement. Droite : val_loss (val FIXE) + points verts « nouveau best ». Renvoie le résumé du log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = read_log(log_path)
    if not train and not val:
        raise ValueError(f"aucune donnée exploitable dans {log_path}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    its = sorted(train)
    ys = [train[i] for i in its]
    if smooth > 1 and len(ys) > smooth:
        ys = [sum(ys[max(0, k - smooth + 1):k + 1]) / len(ys[max(0, k - smooth + 1):k + 1]) for k in range(len(ys))]
    ax.plot(its, ys, linewidth=1.0 if len(its) < 1000 else 0.7)
    ax.set_title("loss d'entraînement")
    ax.set_xlabel("itération")
    ax = axes[1]
    vits = sorted(val)
    ax.plot(vits, [val[i] for i in vits], marker=".", label="val_loss (val FIXE)")
    best, bx, by = math.inf, [], []
    for i in vits:                                   # « nouveau best » = nouveau minimum courant
        if val[i] < best:
            best = val[i]
            bx.append(i)
            by.append(val[i])
    ax.scatter(bx, by, color="green", zorder=3, label="nouveau best")
    ax.set_title("validation")
    ax.set_xlabel("itération")
    ax.legend()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return log_summary(log_path)


# ── les 337 questions ouvertes : combien figurent mot pour mot dans les données SYNTHÉTIQUES du SFT ? ───────────────────────────
def open_questions() -> List[str]:
    try:
        from evaluate import UNSEEN_QA_PROMPTS          # type: ignore
        return list(UNSEEN_QA_PROMPTS)
    except Exception:                                    # noqa: BLE001 (torch absent)
        here = os.path.dirname(os.path.abspath(__file__))
        src = open(os.path.join(here, "evaluate.py"), encoding="utf-8").read()
        i = src.index("UNSEEN_QA_PROMPTS = [")
        ns: dict = {}
        exec(src[i:src.index("]\n", i) + 1], ns)
        return list(ns["UNSEEN_QA_PROMPTS"])


def test_question_overlap(seed: int = 0, repeat: int = 2, plus: bool = True) -> Dict[str, object]:
    """Reproduit la génération du synthétique de `sft_data.build_sft` et compte les questions de test présentes à l'identique."""
    from synthetic_plus import build_synthetic_plus, drop_heldout_arithmetic, norm_q
    from synthetic_qa import build_synthetic_qa
    raw = [c for r in range(repeat) for c in build_synthetic_qa(seed=seed + r)]
    if plus:
        raw = drop_heldout_arithmetic(raw)
    train_q = {norm_q(m["content"]) for c in raw for m in c["messages"] if m["role"] == "user"}
    if plus:
        train_q |= {norm_q(m["content"]) for c in build_synthetic_plus(seed=seed, exclude=open_questions(), light=True)
                    for m in c["messages"] if m["role"] == "user"}
    qs = open_questions()
    inside = [q for q in qs if norm_q(q) in train_q]
    return {"n_questions": len(qs), "n_in_synthetic_train": len(inside), "questions": inside}


# ── fiche de chiffres réels ───────────────────────────────────────────────────────────────────────────────────────────
_CFG_KEYS = ("seq_len", "batch_size", "grad_accum", "max_iters", "epochs", "lr", "min_lr", "weight_decay", "warmup_iters",
             "schedule", "decay_frac", "dropout", "eval_every", "save_every", "patience", "grad_clip", "dtype")


def read_checkpoint_summary(path: str) -> Dict[str, object]:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tc = ck.get("train_config") or {}
    return {"path": path, "iter": ck.get("iter"), "val_loss": ck.get("val_loss", ck.get("best_val_loss")),
            "tokens_seen": ck.get("tokens_seen"), "tokenizer_sha": ck.get("tokenizer_sha"),
            "train_config": {k: tc[k] for k in _CFG_KEYS if k in tc}, "model_config": ck.get("config")}


def _load_json(path: str):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_stats(pretrain_ckpt: str, sft_ckpt: str, sft_data_dir: str, pretrain_log: str, sft_log: str,
                  eval_dir: str, version: str = "2.2", date: Optional[_dt.date] = None) -> Dict[str, object]:
    pt, sft = read_checkpoint_summary(pretrain_ckpt), read_checkpoint_summary(sft_ckpt)
    meta = _load_json(os.path.join(sft_data_dir, "meta.json")) or {}
    n_params = 48_508_672
    tr = meta.get("train", {})
    ans_share = (tr.get("n_answer_tokens", 0) / tr["n_tokens"]) if tr.get("n_tokens") else None
    stats: Dict[str, object] = {
        "repo_name": repo_name(version, n_params, int(pt["iter"]), int(pt["tokens_seen"]), date),
        "pretrain": {**pt, "epochs_on_corpus": round(int(pt["tokens_seen"]) / CORPUS_TRAIN_TOKENS, 3),
                     "perplexity": round(math.exp(float(pt["val_loss"])), 2) if pt.get("val_loss") else None,
                     "log": log_summary(pretrain_log)},
        "sft": {**sft, "data": {"sources": meta.get("sources"), "train": tr, "val": meta.get("val"),
                                "answer_token_share": round(ans_share, 4) if ans_share else None},
                "log": log_summary(sft_log)},
        "overlap": test_question_overlap(),
        "evals": {},
    }
    for name in sorted(os.listdir(eval_dir)) if os.path.isdir(eval_dir) else []:
        if name.endswith(".json"):
            stats["evals"][name[:-5]] = _load_json(os.path.join(eval_dir, name))
    return stats


def last_json(text: str) -> Optional[dict]:
    from compare_sft import last_json as _lj
    return _lj(text)


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s)


# ── exécution et cache des évaluations ───────────────────────────────────────────────────────────────────────────────────
def run_eval(script_args: List[str], name: str, eval_dir: str, force: bool = False, parse_json: bool = True):
    """Lance `python <script_args>`, garde la sortie dans eval_dir/<name>.txt (et le JSON final dans <name>.json).
    Déjà fait -> relu sans relancer (un run interrompu ne refait pas les évaluations terminées). Renvoie (texte, json|None)."""
    import subprocess
    import sys
    os.makedirs(eval_dir, exist_ok=True)
    txt_path, js_path = os.path.join(eval_dir, name + ".txt"), os.path.join(eval_dir, name + ".json")
    if os.path.exists(txt_path) and not force:
        text = open(txt_path, encoding="utf-8").read()
        print(f"✓ {name} : déjà fait (relu depuis {txt_path})")
    else:
        r = subprocess.run([sys.executable] + script_args, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"❌ {name} : échec\n{r.stderr[-600:]}")
            return None, None
        text = r.stdout
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✓ {name} : terminé")
    js = last_json(text) if parse_json else None
    if js is not None:
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)
    return text, js


def extract_openqa(text: str) -> str:
    """Garde la sortie d'`evaluate.py openqa` à partir de son en-tête « ==== » (retire les messages de chargement éventuels)."""
    lines = text.splitlines()
    for k, ln in enumerate(lines):
        if ln.startswith("=" * 20):
            return "\n".join(lines[k:]).rstrip() + "\n"
    return text

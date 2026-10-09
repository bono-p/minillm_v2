"""
prepare_mix.py — construit les données de pré-entraînement v2.3 (deux dossiers : stable/ et decay/) à partir de mixplan.DEFAULT_PLAN.

  python prepare_mix.py plan                                  # affiche le plan, ses proportions, l'espace disque, les avertissements
  python prepare_mix.py inspect                               # (réseau) colonnes et premier exemple de chaque source : À LANCER AVANT build
  python prepare_mix.py build --out data/v23 --tokenizer data/tokenizer.json --existing data/pretrain
  python prepare_mix.py status --out data/v23

`build` est reprenable : relance la même commande après une coupure de session ; les fichiers sont tronqués à la dernière position
enregistrée, donc le résultat est identique à un run sans coupure. Entraînement ensuite (voir docs/V2_3_PLAN.md) :
  python train.py --data_dir data/v23/stable --stop_at <début de la décroissance> ...   puis   --data_dir data/v23/decay
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterator, List, Optional

import mixplan
from mixdata import (Decontaminator, MixWriter, State, chat_ok, clean_ocr, eval_questions, quality_ok, run_existing_bin, run_source,
                     to_messages)


# ── adaptateurs Hugging Face (réseau : NON testés hors ligne) ─────────────────────────────────────────────────────────────────────
def hf_stream(src: mixplan.Source, skip: int, epoch: int) -> Iterator[dict]:
    from datasets import load_dataset
    kw = {"streaming": True, "split": "train"}
    if src.hf_data_files:
        kw["data_files"] = src.hf_data_files
    ds = load_dataset(src.hf_path, src.hf_config or None, **kw)
    if src.filters:
        ds = ds.filter(lambda r, f=dict(src.filters): all(str(r.get(k)) == v for k, v in f.items()))
    ds = ds.shuffle(seed=1234 + epoch, buffer_size=10_000)           # déterministe : la reprise saute exactement les mêmes documents
    if skip:
        ds = ds.skip(skip)
    yield from ds


def make_prepare(src: mixplan.Source, deco: Decontaminator):
    """Transforme une ligne brute de la source en document prêt à tokeniser (str ou messages), ou None si rejetée."""
    if src.kind == "hf_text":
        kind = src.clean or "web"

        def prep(row) -> Optional[str]:
            t = row.get(src.text_field) if isinstance(row, dict) else None
            if not isinstance(t, str):
                return None
            t = clean_ocr(t, kind) if src.clean else t
            return t if quality_ok(t, kind) else None
        return prep
    if src.kind == "hf_chat":
        def prep(row):
            m = to_messages(row.get(src.text_field))
            return m if m and chat_ok(m) and not deco.contaminated(m) else None
        return prep
    if src.kind == "hf_qa":
        from sft_data import _piaf_window

        def prep(row):
            try:
                ans = row["answers"]["text"][0]
                window = _piaf_window(row["context"], ans, 1200)
                m = [{"role": "user", "content": f"Contexte : {window}\n\nQuestion : {row['question']}"}, {"role": "assistant", "content": ans}]
            except Exception:                                   # noqa: BLE001
                return None
            return m if not deco.contaminated(m) else None
        return prep
    raise ValueError(src.kind)


# ── commandes ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan(a) -> None:
    plan = mixplan.scaled(mixplan.DEFAULT_PLAN, int(a.budget)) if a.budget else mixplan.DEFAULT_PLAN
    print(mixplan.summary(plan))
    for p in mixplan.validate(plan, allow_nc=a.allow_nc):
        print("⚠️ ", p)


def cmd_inspect(a) -> None:
    from datasets import load_dataset
    seen = set()
    for src in mixplan.DEFAULT_PLAN:
        if not src.hf_path or (src.hf_path, src.hf_data_files) in seen:
            continue
        seen.add((src.hf_path, src.hf_data_files))
        print("=" * 80, f"\n{src.hf_path}  (utilisée par : {', '.join(s.name for s in mixplan.DEFAULT_PLAN if s.hf_path == src.hf_path)})")
        try:
            kw = {"streaming": True, "split": "train"}
            if src.hf_data_files:
                kw["data_files"] = src.hf_data_files
            ds = load_dataset(src.hf_path, src.hf_config or None, **kw)
            for k, row in zip(range(a.n), ds):
                print({kk: (str(v)[:90] + "…" if len(str(v)) > 90 else v) for kk, v in row.items()})
            print("-> filtres du plan :", {s.name: s.filters for s in mixplan.DEFAULT_PLAN if s.hf_path == src.hf_path and s.filters})
        except Exception as e:                                  # noqa: BLE001
            print(f"❌ {type(e).__name__}: {e}")


def _free_bytes(path: str) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except Exception:                                           # noqa: BLE001
        return None


def cmd_build(a) -> None:
    from mini_tokenizer import MiniTokenizer
    plan = mixplan.scaled(mixplan.DEFAULT_PLAN, int(a.budget)) if a.budget else list(mixplan.DEFAULT_PLAN)
    only = [x for x in a.only.split(",") if x]
    if only:
        plan = [s for s in plan if s.name in only]
    issues = mixplan.validate(plan, allow_nc=a.allow_nc)
    for p in issues:
        print("⚠️ ", p)
    if any("licence" in p for p in issues):
        sys.exit("Licence non compatible : corrige le plan (ou --allow_nc pour une variante de recherche NON publiée).")
    os.makedirs(a.out, exist_ok=True)
    need = mixplan.storage_bytes(plan) + 200_000_000
    free = _free_bytes(a.out)
    st = State(os.path.join(a.out, "state.json"))
    already = sum(os.path.getsize(os.path.join(a.out, ph, "train.bin")) for ph in ("stable", "decay") if os.path.exists(os.path.join(a.out, ph, "train.bin")))
    if free is not None and free + already < need:
        sys.exit(f"Espace disque insuffisant : {need / 1e9:.1f} Go nécessaires, {(free + already) / 1e9:.1f} Go disponibles dans {a.out}. "
                 f"Réduis --budget (ex. {int(5e9 * (free + already) / need * 0.9):,}) ou libère de la place.")
    shutil.copy(a.tokenizer, os.path.join(a.out, "tokenizer.json"))
    tok = MiniTokenizer(os.path.join(a.out, "tokenizer.json"))
    deco = Decontaminator(eval_questions("."))
    print(f"Décontamination : {len(deco)} questions d'évaluation à écarter des conversations.")
    writer = MixWriter(a.out, tok, st, val_permille=a.val_permille)
    try:
        for src in plan:
            quotas = {"stable": src.stable, "decay": src.decay}
            print(f"\n→ {src.name} ({src.kind}) : stable {src.stable / 1e9:.2f} Md + decay {src.decay / 1e9:.2f} Md")
            if src.kind == "existing_bin":
                run_existing_bin(writer, src.name, quotas, os.path.join(a.existing, "train.bin"))
            else:
                run_source(writer, src.name, quotas, lambda skip, ep, s=src: hf_stream(s, skip, ep), make_prepare(src, deco), src.max_epochs)
            s = st.src(src.name)
            print(f"   fait : stable {s['tokens']['stable']:,} · decay {s['tokens']['decay']:,} · répétitions {s['epochs']} · rejetés {s['dropped']}")
    finally:
        writer.close()
    finalize(a.out, a.existing, tok, plan, st)


def finalize(out: str, existing: str, tok, plan: List[mixplan.Source], st: State) -> None:
    import numpy as np
    old_val = os.path.join(existing, "val.bin")
    meta_old = json.load(open(os.path.join(existing, "meta.json"))) if os.path.exists(os.path.join(existing, "meta.json")) else {}
    for ph in ("stable", "decay"):
        d = os.path.join(out, ph)
        if os.path.exists(old_val):
            shutil.copy(old_val, os.path.join(d, "val.bin"))             # même validation que la v2.2 : val_loss comparable à 3,0431
        n_train = os.path.getsize(os.path.join(d, "train.bin")) // 2
        n_val = os.path.getsize(os.path.join(d, "val.bin")) // 2 if os.path.exists(os.path.join(d, "val.bin")) else 0
        meta = {"dtype": "uint16", "vocab_size": tok.vocab_size, "padded_vocab_size": tok.padded_vocab_size, "eot_id": tok.eot_id,
                "n_train_tokens": n_train, "n_val_tokens": n_val, "tokenizer_sha": getattr(tok, "sha", None), "phase": ph,
                "val_note": "val.bin = validation de la v2.2 (comparable à 3,0431) ; val_new.bin (dossier parent) = documents des nouvelles sources",
                "corpus_ref": {k: meta_old.get(k) for k in ("n_train_docs", "n_val_docs")}}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    tot = sum(s["tokens"]["stable"] + s["tokens"]["decay"] for s in st.d["sources"].values()) or 1
    by_cat: Dict[str, int] = {}
    for s in plan:
        if s.name in st.d["sources"]:
            x = st.d["sources"][s.name]
            by_cat[s.category] = by_cat.get(s.category, 0) + x["tokens"]["stable"] + x["tokens"]["decay"]
    report = {"total_tokens": tot, "parts_par_categorie": {c: round(v / tot, 4) for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
              "sources": st.d["sources"]}
    json.dump(report, open(os.path.join(out, "mix_report.json"), "w"), ensure_ascii=False, indent=1)
    print("\nTotal :", f"{tot:,} tokens ;", "parts :", report["parts_par_categorie"])


def cmd_status(a) -> None:
    st = State(os.path.join(a.out, "state.json"))
    for name, s in st.d["sources"].items():
        print(f"{name:<30} stable {s['tokens']['stable']:>14,}  decay {s['tokens']['decay']:>13,}  époques {s['epochs']}  rejetés {s['dropped']}")
    print("fichiers :", st.d["files"])


def main() -> None:
    p = argparse.ArgumentParser(description="MiniLLM v2.3 — préparation du mélange de données")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan"); q.add_argument("--budget", type=float, default=0); q.add_argument("--allow_nc", action="store_true"); q.set_defaults(f=cmd_plan)
    q = sub.add_parser("inspect"); q.add_argument("--n", type=int, default=2); q.set_defaults(f=cmd_inspect)
    q = sub.add_parser("build")
    q.add_argument("--out", default="data/v23"); q.add_argument("--tokenizer", default="data/tokenizer.json")
    q.add_argument("--existing", default="data/pretrain", help="dossier du corpus v2.2 déjà tokenisé (train.bin, val.bin, meta.json)")
    q.add_argument("--budget", type=float, default=0, help="total de tokens (ex. 3e9) : mêmes proportions que le plan par défaut")
    q.add_argument("--only", default="", help="noms de sources séparés par des virgules (ex. essai rapide)")
    q.add_argument("--val_permille", type=int, default=3); q.add_argument("--allow_nc", action="store_true"); q.set_defaults(f=cmd_build)
    q = sub.add_parser("status"); q.add_argument("--out", default="data/v23"); q.set_defaults(f=cmd_status)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

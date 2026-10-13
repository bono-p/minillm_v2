"""
prepare_mix.py — construit les données de pré-entraînement v2.3 (deux dossiers : stable/ et decay/) à partir de mixplan.DEFAULT_PLAN.

  python prepare_mix.py plan                                  # affiche le plan, ses proportions, l'espace disque, les avertissements
  python prepare_mix.py inspect                               # (réseau) colonnes et premier exemple de chaque source : À LANCER AVANT build
  python prepare_mix.py tree --repo PleIAs/common_corpus      # (réseau, rapide) organisation d'un gros dépôt : collections, langues, fichiers
  python prepare_mix.py build --out data/v23 --tokenizer data/tokenizer.json --existing data/pretrain
  python prepare_mix.py status --out data/v23
  python prepare_mix.py verify --out data/v23                 # (nouvelle session) confirme que Drive a bien tout synchronisé

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
            if src.clean == "wiki":                                  # mêmes nettoyages que le pipeline v2.2 (text_cleaning.py)
                from text_cleaning import clean_wikipedia_text
                return clean_wikipedia_text(t)
            if src.clean == "web":
                from text_cleaning import clean_web_text
                return clean_web_text(t)
            t = clean_ocr(t, kind) if src.clean else t
            return t if quality_ok(t, kind) else None
        return prep
    if src.kind == "hf_chat":
        def prep(row):
            m = to_messages(row.get(src.text_field))
            return m if m and chat_ok(m) and not deco.contaminated(m) else None
        return prep
    if src.kind == "hf_qa":
        from sft_data import _piaf_window                        # _piaf_window(context, ans_start, ans_len, width) : INDICES, pas le texte

        def prep(row):
            try:
                texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
                ans = texts[0].strip()
                if not ans:
                    return None
                window = _piaf_window(row["context"], int(starts[0]), len(texts[0]))
                if ans not in window:
                    return None
                m = [{"role": "user", "content": f"Contexte : {window}\n\nQuestion : {row['question'].strip()}"},
                     {"role": "assistant", "content": ans if ans[-1] in ".!?" else ans + "."}]
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


def group_paths(files, depth: int = 2) -> Dict[str, tuple]:
    """[(chemin, taille)] -> {préfixe: (nombre de fichiers, octets)} ; sert à repérer comment un gros dépôt est organisé (collection, langue…)."""
    out: Dict[str, list] = {}
    for path, size in files:
        key = "/".join(path.split("/")[:depth]) if "/" in path else "(racine)"
        e = out.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += size or 0
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def cmd_tree(a) -> None:
    """Liste les fichiers d'un dépôt Hugging Face (API, rapide : ne résout pas tout le jeu de données comme load_dataset)."""
    from huggingface_hub import HfApi
    items = list(HfApi().list_repo_tree(a.repo, path_in_repo=a.path or None, repo_type="dataset", recursive=True))
    files = [(i.path, getattr(i, "size", 0) or 0) for i in items if hasattr(i, "size") and i.__class__.__name__ == "RepoFile"]
    if a.grep:
        files = [(p, sz) for p, sz in files if a.grep.lower() in p.lower()]
    print(f"{a.repo} {a.path or ''} : {len(files):,} fichiers, {sum(sz for _, sz in files) / 1e9:.1f} Go" + (f" (filtre « {a.grep} »)" if a.grep else ""))
    for k, (n, b) in group_paths(files, a.depth).items():
        print(f"  {k:<60} {n:>7,} fichiers {b / 1e9:9.2f} Go")
    print("exemples :", *[p for p, _ in files[:6]], sep="\n  ")


def cmd_inspect(a) -> None:
    from datasets import load_dataset
    seen = set()
    for src in mixplan.DEFAULT_PLAN:
        if not src.hf_path or (src.hf_path, src.hf_data_files) in seen:
            continue
        seen.add((src.hf_path, src.hf_data_files))
        if src.hf_path == "PleIAs/common_corpus" and not src.hf_data_files and not a.all:
            print("=" * 80, f"\n{src.hf_path} : ignoré (dépôt énorme : load_dataset met très longtemps à résoudre ses fichiers).",
                  "\n  Utilise d'abord :  python prepare_mix.py tree --repo PleIAs/common_corpus  pour voir son organisation (collections, langues).")
            continue
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


def ensure_tokenizer_file(path: str, repo: str) -> None:
    """Si tokenizer.json est absent, le télécharge depuis le dépôt Hugging Face public (même tokenizer BPE 32 000 que la v2.2)."""
    if os.path.exists(path):
        return
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    got = hf_hub_download(repo, "tokenizer.json")
    shutil.copy(got, path)
    print(f"tokenizer.json téléchargé depuis {repo}")


def _free_bytes(path: str) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except Exception:                                           # noqa: BLE001
        return None


def probe_filters(src: mixplan.Source, n_scan: int = 400) -> Optional[str]:
    """Avant de consommer le moindre quota : vérifie que les filtres du plan matchent au moins une ligne parmi les `n_scan`
    premières. Si aucune ne matche, renvoie un message d'erreur listant les valeurs RÉELLEMENT vues pour chaque clé filtrée
    (au lieu de tourner pour rien, ou pire, de construire un mélange avec une source vide sans que personne ne le remarque)."""
    if not src.filters or src.kind != "hf_text":
        return None
    from datasets import load_dataset
    kw = {"streaming": True, "split": "train"}
    if src.hf_data_files:
        kw["data_files"] = src.hf_data_files
    ds = load_dataset(src.hf_path, src.hf_config or None, **kw)
    seen: Dict[str, set] = {k: set() for k in src.filters}
    for i, row in enumerate(ds):
        if i >= n_scan:
            break
        for k in src.filters:
            if k in row:
                seen[k].add(row[k])
        if all(str(row.get(k)) == v for k, v in src.filters.items()):
            return None                                             # au moins une ligne matche : tout va bien
    details = " ; ".join(f"{k} attendu « {v} », valeurs vues : {sorted(seen.get(k, []))[:8]}" for k, v in src.filters.items())
    return f"{src.name} : aucune des {n_scan} premières lignes ne correspond au filtre ({details})"


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
    if not a.skip_probe:
        print("\nVérification des filtres avant de commencer (évite de découvrir une source vide après des heures) :")
        probe_errors = [e for e in (probe_filters(s) for s in plan) if e]
        for e in probe_errors:
            print("❌ ", e)
        if probe_errors:
            sys.exit("Corrige mixplan.py (ou relance avec --skip_probe si tu es sûr que c'est bon), puis relance.")
        print("✓ filtres vérifiés\n")
    os.makedirs(a.out, exist_ok=True)
    need = mixplan.storage_bytes(plan) + 200_000_000
    free = _free_bytes(a.out)
    st = State(os.path.join(a.out, "state.json"))
    already = sum(os.path.getsize(os.path.join(a.out, ph, "train.bin")) for ph in ("stable", "decay") if os.path.exists(os.path.join(a.out, ph, "train.bin")))
    if free is not None and free + already < need:
        sys.exit(f"Espace disque insuffisant : {need / 1e9:.1f} Go nécessaires, {(free + already) / 1e9:.1f} Go disponibles dans {a.out}. "
                 f"Réduis --budget (ex. {int(5e9 * (free + already) / need * 0.9):,}) ou libère de la place.")
    ensure_tokenizer_file(a.tokenizer, a.tokenizer_repo)
    if any(s.kind == "existing_bin" for s in plan) and not (a.existing and os.path.exists(os.path.join(a.existing, "train.bin"))):
        sys.exit("Le plan contient une source existing_bin mais --existing ne pointe pas vers un dossier avec train.bin.")
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
    """Écrit meta.json de chaque phase. val.bin : celui d'un corpus existant s'il y en a un (val_loss comparable à celle de ce corpus),
    sinon les documents de validation des nouvelles sources (val_new.bin)."""
    old_val = os.path.join(existing, "val.bin") if existing else ""
    meta_old = json.load(open(os.path.join(existing, "meta.json"))) if existing and os.path.exists(os.path.join(existing, "meta.json")) else {}
    val_src = old_val if old_val and os.path.exists(old_val) else os.path.join(out, "val_new.bin")
    if not os.path.exists(val_src) or os.path.getsize(val_src) == 0:
        print("⚠️  aucun document de validation : augmente --val_permille")
    for ph in ("stable", "decay"):
        d = os.path.join(out, ph)
        if os.path.exists(val_src):
            shutil.copy(val_src, os.path.join(d, "val.bin"))
        n_train = os.path.getsize(os.path.join(d, "train.bin")) // 2
        n_val = os.path.getsize(os.path.join(d, "val.bin")) // 2 if os.path.exists(os.path.join(d, "val.bin")) else 0
        meta = {"dtype": "uint16", "vocab_size": tok.vocab_size, "padded_vocab_size": tok.padded_vocab_size, "eot_id": tok.eot_id,
                "n_train_tokens": n_train, "n_val_tokens": n_val, "tokenizer_sha": getattr(tok, "sha", None), "phase": ph,
                "val_note": ("val.bin = validation du corpus existant" if val_src == old_val else
                             "val.bin = documents de validation des nouvelles sources (3 pour mille des documents de chaque source, écartés de l'entraînement) : "
                             "val_loss NON comparable à celle des versions précédentes"),
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


def cmd_verify(a) -> None:
    """Compare la taille RÉELLE des fichiers sur disque à celle enregistrée dans state.json au moment de l'écriture.
    Utile après avoir changé de session (ou d'ordinateur) : Google Drive peut mettre du temps à synchroniser un gros fichier
    entre deux sessions ; `os.fsync` garantit l'écriture LOCALE au moment du `build`, pas la disponibilité immédiate ailleurs.
    Lancer cette commande dans une session FRAÎCHE (remonte le Drive d'abord) donne une réponse sûre, sans deviner."""
    st_path = os.path.join(a.out, "state.json")
    if not os.path.exists(st_path):
        sys.exit(f"{st_path} introuvable.")
    st = State(st_path)
    ok = True
    print(f"Vérification de {a.out} (état enregistré lors du dernier `build`) :\n")
    for key, expected in st.d["files"].items():
        if key == "stable":
            path = os.path.join(a.out, "stable", "train.bin")
        elif key == "decay":
            path = os.path.join(a.out, "decay", "train.bin")
        else:
            path = os.path.join(a.out, "val_new.bin")
        got = os.path.getsize(path) if os.path.exists(path) else -1
        same = got == expected
        ok = ok and same
        mark = "✓" if same else ("❌ MANQUANT" if got < 0 else "❌ TAILLE DIFFÉRENTE")
        gap = "" if same else f"  (écart {expected - got:+,} o)"
        print(f"  {mark}  {key:<10} attendu {expected:>14,} o | trouvé {got:>14,} o{gap}")
    if ok:
        print("\n✓ Tout correspond : les données sont intactes et complètes telles qu'enregistrées par `build`.")
    else:
        print("\n⚠️  Écart détecté. Deux causes possibles, dans cet ordre de probabilité :")
        print("  1) Synchronisation Drive pas terminée entre la session qui a écrit et celle-ci : remonte le Drive avec "
              "force_remount=True, attends quelques minutes (jusqu'à 15-30 min pour un fichier de plusieurs Go), puis relance "
              "cette commande. Le fichier est très probablement déjà complet côté Drive, juste pas encore visible ici.")
        print("  2) La dernière session a été interrompue avant la fin de l'écriture : dans ce cas, relance `prepare_mix.py build` "
              "(reprenable) — il continuera depuis la taille réellement présente, rien n'est à refaire depuis le début.")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="MiniLLM v2.3 — préparation du mélange de données")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan"); q.add_argument("--budget", type=float, default=0); q.add_argument("--allow_nc", action="store_true"); q.set_defaults(f=cmd_plan)
    q = sub.add_parser("inspect"); q.add_argument("--n", type=int, default=2); q.add_argument("--all", action="store_true"); q.set_defaults(f=cmd_inspect)
    q = sub.add_parser("tree"); q.add_argument("--repo", default="PleIAs/common_corpus"); q.add_argument("--path", default="")
    q.add_argument("--depth", type=int, default=2); q.add_argument("--grep", default=""); q.set_defaults(f=cmd_tree)
    q = sub.add_parser("build")
    q.add_argument("--out", default="data/v23"); q.add_argument("--tokenizer", default="data/tokenizer.json")
    q.add_argument("--existing", default="", help="(facultatif) dossier d'un corpus déjà tokenisé (train.bin, val.bin, meta.json), si le plan contient une source existing_bin")
    q.add_argument("--tokenizer_repo", default="bonopassale/miniLLM_v2.2-49M-50200it_3.29Btoks_1.0ep_20261005",
                   help="dépôt Hugging Face d'où récupérer tokenizer.json s'il est absent (même tokenizer que la v2.2)")
    q.add_argument("--budget", type=float, default=0, help="total de tokens (ex. 3e9) : mêmes proportions que le plan par défaut")
    q.add_argument("--only", default="", help="noms de sources séparés par des virgules (ex. essai rapide)")
    q.add_argument("--val_permille", type=int, default=3); q.add_argument("--allow_nc", action="store_true")
    q.add_argument("--skip_probe", action="store_true", help="ne pas vérifier les filtres avant de commencer (déconseillé)")
    q.set_defaults(f=cmd_build)
    q = sub.add_parser("status"); q.add_argument("--out", default="data/v23"); q.set_defaults(f=cmd_status)
    q = sub.add_parser("verify"); q.add_argument("--out", default="data/v23"); q.set_defaults(f=cmd_verify)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

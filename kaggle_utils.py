"""
kaggle_utils.py — outils pour travailler sur Kaggle (et ailleurs) sans dépendre d'un montage Google Drive.

Kaggle ne sait pas monter Google Drive. Trois façons d'y amener tes données / checkpoints, essayées dans cet ordre :
  1. /kaggle/input : un Dataset Kaggle, OU la sortie d'une version précédente de CE notebook (« Add Input → Notebook Output Files »).
  2. Google Drive via `gdown` : des liens de partage « Toute personne disposant du lien » (transfert serveur à serveur : rien ne passe par ton téléphone).
  3. Reconstruction complète depuis internet (prepare_data.py).

Tout est vérifié après copie (tailles des .bin cohérentes avec meta.json, checkpoint chargeable) pour ne jamais démarrer
un entraînement sur un téléchargement tronqué.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np

from checkpoint import list_checkpoints, load_checkpoint, rolling_cleanup

_LIGHT_CKPT_FILES = ("best.pt", "best.json", "final.pt", "log.jsonl", "tokenizer.json")


def _gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  copie {os.path.relpath(src, '/') if src.startswith('/') else src}  ->  {dst}  ({_gb(src):.2f} Go)")
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def extract_drive_id(url: str) -> Optional[str]:
    m = re.search(r"(?:/d/|id=|folders/)([A-Za-z0-9_-]{20,})", url)
    return m.group(1) if m else None


# ── 1. restauration depuis /kaggle/input ────────────────────────────────────
def _dirs_named(root: str, name: str, must_have: str) -> List[str]:
    hits = []
    for p in glob.glob(os.path.join(root, "**", name), recursive=True):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, must_have)):
            hits.append(p)
    return sorted(hits)


def restore_from_inputs(input_root: str = "/kaggle/input", work: str = "/kaggle/working/MiniLLM") -> Dict[str, str]:
    """Copie (une seule fois) données et checkpoints trouvés dans les entrées attachées vers `work`. Renvoie ce qui a été trouvé."""
    found: Dict[str, str] = {}
    if not os.path.isdir(input_root):
        return found

    # données de pré-entraînement
    dst = os.path.join(work, "data", "pretrain")
    if not os.path.exists(os.path.join(dst, "meta.json")):
        for src in _dirs_named(input_root, "pretrain", "meta.json"):
            if os.path.exists(os.path.join(src, "train.bin")):
                print(f"Données de pré-entraînement trouvées : {src}")
                for f in ("train.bin", "val.bin", "meta.json"):
                    _copy(os.path.join(src, f), os.path.join(dst, f))
                found["pretrain_data"] = src
                break
    # données SFT
    dst = os.path.join(work, "data", "sft")
    if not os.path.exists(os.path.join(dst, "meta.json")):
        for src in _dirs_named(input_root, "sft", "meta.json"):
            if os.path.exists(os.path.join(src, "train.tokens.npy")):
                print(f"Données SFT trouvées : {src}")
                for f in os.listdir(src):
                    if os.path.isfile(os.path.join(src, f)):
                        _copy(os.path.join(src, f), os.path.join(dst, f))
                found["sft_data"] = src
                break
    # tokenizer
    tdst = os.path.join(work, "data", "tokenizer.json")
    if not os.path.exists(tdst):
        cands = sorted(glob.glob(os.path.join(input_root, "**", "tokenizer.json"), recursive=True))
        if cands:
            _copy(cands[0], tdst)
            found["tokenizer"] = cands[0]
    # checkpoints (pretrain / sft) : on garde, pour chaque mode, le plus avancé
    for mode in ("pretrain", "sft"):
        out = os.path.join(work, "checkpoints", mode)
        have = list_checkpoints(out)
        have_it = have[-1][0] if have else -1
        best_src, best_it = None, have_it
        for d in glob.glob(os.path.join(input_root, "**", mode), recursive=True):
            if not os.path.isdir(d) or "checkpoints" not in d:
                continue
            ck = list_checkpoints(d)
            if ck and ck[-1][0] > best_it:
                best_src, best_it = d, ck[-1][0]
        if best_src:
            print(f"Checkpoint {mode} it={best_it} trouvé : {best_src}")
            _copy(list_checkpoints(best_src)[-1][1], os.path.join(out, os.path.basename(list_checkpoints(best_src)[-1][1])))
            for f in _LIGHT_CKPT_FILES:
                if os.path.exists(os.path.join(best_src, f)):
                    _copy(os.path.join(best_src, f), os.path.join(out, f))
            found[f"ckpt_{mode}"] = best_src
    return found


# ── 2. Google Drive (gdown) ─────────────────────────────────────────────────
def download_drive(mapping: Dict[str, str], work: str = "/kaggle/working/MiniLLM") -> List[str]:
    """
    mapping : {"chemin/relatif/dans/work": "lien de partage Drive"} ; les entrées vides sont ignorées, les fichiers déjà là aussi.
    Chaque fichier doit être partagé en « Toute personne disposant du lien : lecteur » (tu peux retirer le partage ensuite).
    """
    import gdown                                              # pip install gdown
    done = []
    for rel, url in mapping.items():
        if not url or not url.strip():
            continue
        dst = os.path.join(work, rel)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  déjà présent : {rel}")
            continue
        if not extract_drive_id(url):
            raise ValueError(f"Lien Drive non reconnu pour {rel} : {url[:60]}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        print(f"Téléchargement Drive -> {rel}")
        out = gdown.download(url, tmp, quiet=False, fuzzy=True)
        if not out or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError(f"Échec du téléchargement de {rel}. Vérifie que le fichier est partagé en « Toute personne disposant "
                               "du lien » et réessaie (quota Drive : attends un peu si « Too many users »).")
        os.replace(tmp, dst)
        done.append(rel)
    return done


# ── vérifications ───────────────────────────────────────────────────────────
def verify_pretrain_data(data_dir: str) -> dict:
    """Les .bin doivent avoir exactement la taille annoncée par meta.json (détecte un téléchargement tronqué)."""
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item = np.dtype(meta["dtype"]).itemsize
    for split, key in (("train", "n_train_tokens"), ("val", "n_val_tokens")):
        p = os.path.join(data_dir, f"{split}.bin")
        expected = int(meta[key]) * item
        actual = os.path.getsize(p)
        if actual != expected:
            raise ValueError(f"{p} : {actual:,} octets au lieu de {expected:,} -> fichier tronqué/corrompu, re-télécharge-le.")
    return meta


def verify_checkpoint(path: str) -> dict:
    ck = load_checkpoint(path)
    info = {"iter": ck.get("iter"), "val_loss": ck.get("val_loss", ck.get("best_val_loss")), "config": ck["config"]["n_layers"]}
    if "model" not in ck:
        raise ValueError(f"{path} : pas de poids de modèle.")
    return info


def prune_checkpoints(out_dir: str, keep: int = 1) -> None:
    """Réduit la taille de la sortie Kaggle : ne garde que le(s) `keep` checkpoint(s) complet(s) le(s) plus récent(s) (+ best/final)."""
    rolling_cleanup(out_dir, keep)


def summarize(work: str = "/kaggle/working/MiniLLM") -> None:
    print(f"\nContenu de {work} ({dir_size_gb(work):.2f} Go) :" if os.path.isdir(work) else f"{work} n'existe pas encore.")
    if not os.path.isdir(work):
        return
    for root, _, files in sorted(os.walk(work)):
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.getsize(p) > 1e6:
                print(f"  {os.path.relpath(p, work):<50} {_gb(p):>6.2f} Go")

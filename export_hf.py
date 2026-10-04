"""
export_hf.py — exporte un checkpoint SFT en dépôt Hugging Face (model.safetensors + config.json + generation_config.json
+ requirements.txt). Reprend, étape pour étape, la cellule d'export déjà exécutée avec succès sur le modèle v2.1.

  * la config écrite est la VRAIE config stockée dans le checkpoint ;
  * refus d'exporter si `tie_embeddings` est annoncé mais que tok_emb ≠ lm_head ;
  * relecture du fichier exporté (écart max vs checkpoint) et test de fumée avec inference.py (CPU, fp32).

Nécessite torch + safetensors (Colab). Rien n'est envoyé sur Hugging Face ici.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from typing import Dict, Optional

NAMES = {"vocab_size", "n_layers", "d_model", "n_heads", "kv_heads", "ffn_hidden", "max_seq_len",
         "dropout", "rope_theta", "tie_embeddings", "qk_norm", "norm_eps"}
GEN_CONFIG = {"max_new_tokens": 128, "temperature": 0.7, "top_k": 40, "top_p": 0.9,
              "repetition_penalty": 1.1, "no_repeat_ngram_size": 3}


def _strip(k: str) -> str:
    for p in ("_orig_mod.", "module."):
        while k.startswith(p):
            k = k[len(p):]
    return k


def export_package(ckpt_path: str, out_dir: str, dtype: str = "fp32") -> Dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = None
    for name in ("model", "model_state_dict", "state_dict"):
        if isinstance(ck.get(name), dict):
            sd = ck[name]
            break
    assert sd is not None, f"état du modèle introuvable, clés : {list(ck)[:20]}"
    sd = {_strip(k): v.detach().to("cpu") for k, v in sd.items() if isinstance(v, torch.Tensor)}
    tdtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype]

    # clone : évite l'erreur safetensors sur les poids partagés (embedding / lm_head)
    tensors = {k: v.clone().contiguous().to(tdtype) for k, v in sd.items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    raw = ck.get("config")
    if raw is not None and not isinstance(raw, dict):
        raw = dataclasses.asdict(raw) if dataclasses.is_dataclass(raw) else dict(vars(raw))
    assert raw, "pas de config dans le checkpoint"
    cfg = {k: v for k, v in raw.items() if k in NAMES}
    missing = sorted(NAMES - set(cfg))
    same = torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"])
    tied = bool(cfg.get("tie_embeddings", True))
    assert not (tied and not same), "config 'tied' mais poids différents : ne pas publier"
    n_params = sum(v.numel() for v in sd.values()) - (sd["lm_head.weight"].numel() if same else 0)
    json.dump({"model_type": "minillm", "n_params": n_params, **cfg}, open(os.path.join(out_dir, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(GEN_CONFIG, open(os.path.join(out_dir, "generation_config.json"), "w"), indent=2)
    open(os.path.join(out_dir, "requirements.txt"), "w").write("torch>=2.3\ntokenizers>=0.15\nsafetensors>=0.4\n")

    chk = load_file(os.path.join(out_dir, "model.safetensors"))
    diff = max((chk[k].float() - sd[k].float()).abs().max().item() for k in sd)
    return {"n_tensors": len(sd), "n_params": n_params, "tied": tied, "tok_emb_equals_lm_head": same,
            "missing_config_fields": missing, "max_diff_vs_checkpoint": diff, "iter": ck.get("iter"),
            "files": {f: os.path.getsize(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))}}


def smoke_test(out_dir: str, tokenizer_path: str, questions=("Qui es-tu ?", "Quelle est la capitale du Cameroun ?",
                                                             "Combien font 2 plus 3 ?")) -> Dict[str, str]:
    """Le dépôt exporté doit répondre TOUT SEUL (inference.py + safetensors + config), sans le code d'entraînement."""
    import importlib
    sys.path.insert(0, out_dir)
    inference = importlib.import_module("inference")
    importlib.reload(inference)
    m = inference.load_model(os.path.join(out_dir, "model.safetensors"), os.path.join(out_dir, "config.json"),
                             device="cpu", dtype="fp32")
    tk = inference.MiniTokenizer(tokenizer_path)
    return {q: inference.chat_reply(m, tk, [{"role": "user", "content": q}], temperature=0.0) for q in questions}


def assemble_static(out_dir: str, release_dir: str, tokenizer_path: str, extra: Optional[Dict[str, str]] = None) -> None:
    """Copie les fichiers fixes (LICENSE, inference.py, assets, tokenizer) puis, si présents, README et evals du dossier release."""
    shutil.copy(tokenizer_path, os.path.join(out_dir, "tokenizer.json"))
    for name in ("LICENSE", "inference.py"):
        shutil.copy(os.path.join(release_dir, name), os.path.join(out_dir, name))
    os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
    for f in os.listdir(os.path.join(release_dir, "assets")):
        shutil.copy(os.path.join(release_dir, "assets", f), os.path.join(out_dir, "assets", f))
    for src, dst in (extra or {}).items():
        os.makedirs(os.path.dirname(os.path.join(out_dir, dst)) or out_dir, exist_ok=True)
        shutil.copy(src, os.path.join(out_dir, dst))

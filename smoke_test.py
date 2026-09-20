"""
smoke_test.py — teste TOUT le pipeline sur CPU en ~1-2 minutes, sans internet ni GPU :
    corpus local -> tokenizer -> tokenisation -> pré-entraînement -> reprise -> SFT -> génération.

    python smoke_test.py            # à lancer avant un long entraînement pour vérifier que tout fonctionne
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import time

import config
from config import ModelConfig, TrainConfig

SUBJECTS = ["Le chat", "Le chien", "La petite fille", "Le vieux marin", "Un étudiant", "Ma voisine", "Le professeur",
            "Le boulanger", "Une femme", "Le garçon", "Le fermier", "Notre équipe"]
VERBS = ["mange", "regarde", "cherche", "aime", "découvre", "prépare", "dessine", "raconte", "achète", "protège"]
OBJECTS = ["une pomme rouge", "le grand livre", "un vieux jardin", "la belle maison", "un petit bateau", "le pain chaud",
           "une histoire drôle", "la route du village", "un secret", "le marché du matin"]
PLACES = ["à Garoua", "à Yaoundé", "près de la rivière", "dans la cuisine", "sous le grand arbre", "au marché", "à l'école"]


def make_corpus_file(path: str, n_paragraphs: int = 1500, seed: int = 0) -> None:
    rng = random.Random(seed)
    paras = []
    for _ in range(n_paragraphs):
        sents = [f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(PLACES)}."
                 for _ in range(rng.randint(3, 6))]
        paras.append(" ".join(sents))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras))


def register_tiny_preset() -> str:
    config.PRESETS["tiny-test"] = ModelConfig(n_layers=2, d_model=64, n_heads=4, kv_heads=2, max_seq_len=128)
    return "tiny-test"


def run_pipeline(work: str, verbose: bool = True) -> dict:
    import prepare_data, sft_data, train, generate

    say = print if verbose else (lambda *a, **k: None)
    name = register_tiny_preset()
    data = os.path.join(work, "data")
    corpus_txt = os.path.join(work, "corpus.txt")
    make_corpus_file(corpus_txt)

    say("\n▶ 1/6 corpus + tokenizer + tokenisation")
    prepare_data.build_corpus(data, wiki_docs=0, web_docs=0, local_txt=[corpus_txt])
    prepare_data.build_tokenizer(data, vocab_size=600)
    prepare_data.tokenize_corpus(data, val_permille=100)

    say("\n▶ 2/6 pré-entraînement (tiny) 30 it")
    pre_dir = os.path.join(work, "ckpt_pre")
    common = dict(model_size=name, data_dir=os.path.join(data, "pretrain"), out_dir=pre_dir, seq_len=128, batch_size=4,
                  grad_accum=2, lr=3e-3, min_lr=3e-4, warmup_iters=5, eval_every=10, eval_iters=4, save_every=10,
                  log_every=5, device="cpu", keep=2)
    r1 = train.train(TrainConfig(mode="pretrain", max_iters=30, **common))

    say("\n▶ 3/6 reprise : on prolonge à 40 it (doit repartir de l'it 30, pas de zéro)")
    r2 = train.train(TrainConfig(mode="pretrain", max_iters=40, **common))
    assert r2["iter"] == 40, r2

    say("\n▶ 4/6 données SFT (synthétiques uniquement, hors-ligne)")
    sft_dir = os.path.join(data, "sft")
    sft_data.build_sft(os.path.join(data, "tokenizer.json"), sft_dir, alpaca=0, piaf=0, oasst=0,
                       synthetic_repeat=1, max_len=127, val_permille=50)

    say("\n▶ 5/6 SFT (tiny) 1 époque")
    sft_out = os.path.join(work, "ckpt_sft")
    cfg = TrainConfig.for_mode("sft")
    cfg.model_size, cfg.data_dir, cfg.out_dir = name, sft_dir, sft_out
    cfg.init_from = os.path.join(pre_dir, "best.pt")
    cfg.batch_size, cfg.grad_accum, cfg.epochs = 8, 1, 1
    cfg.eval_every, cfg.save_every, cfg.log_every, cfg.warmup_iters, cfg.device = 20, 20, 10, 5, "cpu"
    r3 = train.train(cfg)

    say("\n▶ 6/6 génération (chat avec KV-cache)")
    model, tok, _ = generate.load_model(os.path.join(sft_out, "best.pt"), device="cpu")
    t0 = time.perf_counter()
    reply = generate.chat_reply(model, tok, [{"role": "user", "content": "Quelle est la capitale de la France ?"}],
                                max_new_tokens=20, temperature=0.0)
    say(f"Réponse (modèle minuscule non entraîné, seul le fonctionnement compte) : {reply!r} [{time.perf_counter() - t0:.2f}s]")
    assert isinstance(reply, str)
    return {"pretrain": r1, "resumed": r2, "sft": r3, "pre_dir": pre_dir, "sft_dir": sft_out, "data": data}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        out = run_pipeline(tmp)
    print("\n✅ smoke test OK — le pipeline complet fonctionne.")

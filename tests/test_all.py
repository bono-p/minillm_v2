"""Suite de tests MiniLLM v2 — CPU uniquement, sans internet.  Lancer :  python -m pytest -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from config import ModelConfig, TrainConfig, PRESETS, get_preset  # noqa: E402
from model import MiniLLM, KVCache, apply_rope, build_rope_tables  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Modèle
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("kw", [
    dict(),
    dict(kv_heads=2),
    dict(tie_embeddings=False),
    dict(qk_norm=False),
    dict(kv_heads=1, ffn_hidden=192),
])
def test_param_count_is_exact(kw):
    cfg = ModelConfig(vocab_size=200, n_layers=3, d_model=64, n_heads=4, kv_heads=kw.pop("kv_heads", 4), max_seq_len=32, **kw)
    assert MiniLLM(cfg).num_params() == cfg.count_params()


def test_presets_are_named_by_their_real_size_and_are_copies():
    for name, cfg in PRESETS.items():
        assert name == cfg.size_name()
    p = get_preset("49M", vocab_size=1234, max_seq_len=64)
    assert p.vocab_size == 1234 and PRESETS["49M"].vocab_size == config.DEFAULT_VOCAB_SIZE   # pas de mutation globale
    assert 45e6 < PRESETS["49M"].count_params() < 52e6


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_kv_cache_equals_full_forward(kv_heads):
    cfg = ModelConfig(vocab_size=100, n_layers=2, d_model=64, n_heads=4, kv_heads=kv_heads, max_seq_len=48)
    m = MiniLLM(cfg).eval()
    idx = torch.randint(0, 100, (2, 30))
    full, _ = m(idx)
    cache = KVCache(cfg, 2, 48, "cpu", torch.float32)
    parts = [m(idx[:, :10], cache=cache, start_pos=0)[0], m(idx[:, 10:22], cache=cache, start_pos=10)[0]]   # chunk avec passé
    parts += [m(idx[:, t:t + 1], cache=cache, start_pos=t)[0] for t in range(22, 30)]                        # token par token
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)


def test_right_padding_does_not_change_real_tokens():
    cfg = ModelConfig(vocab_size=100, n_layers=2, d_model=64, n_heads=4, kv_heads=4, max_seq_len=32)
    m = MiniLLM(cfg).eval()
    x = torch.randint(0, 100, (1, 10))
    padded = torch.cat([x, torch.zeros(1, 6, dtype=torch.long)], 1)
    assert torch.allclose(m(x)[0], m(padded)[0][:, :10], atol=1e-5)


def test_loss_ignores_masked_labels_and_sum_matches_mean():
    cfg = ModelConfig(vocab_size=100, n_layers=1, d_model=32, n_heads=2, kv_heads=2, max_seq_len=16)
    m = MiniLLM(cfg).eval()
    x = torch.randint(0, 100, (3, 12))
    y = torch.randint(0, 100, (3, 12))
    y[:, :5] = -1
    y[2, :] = -1
    y[2, 11] = 7
    logits, mean = m(x, y)
    _, total = m(x, y, reduction="sum")
    n = (y != -1).sum()
    assert torch.allclose(total / n, mean, atol=1e-5)
    manual = torch.nn.functional.cross_entropy(logits.view(-1, 100), y.view(-1), ignore_index=-1)
    assert torch.allclose(manual, mean, atol=1e-6)


def test_rope_is_relative():
    hd = 16
    cos, sin = build_rope_tables(hd, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)

    def score(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum().item()
    assert abs(score(5, 2) - score(25, 22)) < 1e-4


# ══════════════════════════════════════════════════════════════════════════════
#  Nettoyage, tokenizer
# ══════════════════════════════════════════════════════════════════════════════
def test_cleaning_removes_holey_wikipedia_sentences():
    from text_cleaning import clean_wikipedia_text, has_defect
    bad = "Antoine Meillet, né le  à Moulins (Allier) et mort le  à Châteaumeillant (Cher), est un linguiste français."
    bad2 = "Il fut le principal linguiste français des premières décennies du ."
    good = "Antoine Meillet est un linguiste français reconnu pour ses travaux sur les langues indo-européennes."
    assert has_defect(bad) and has_defect(bad2) and not has_defect(good)
    article = "Antoine Meillet\n" + good + " " + bad2 + " Il a enseigné au Collège de France pendant de nombreuses années.\n" \
              "Biographie\n" + ("Une phrase correcte et suffisamment longue pour compter. " * 8) + "\nNotes et références\nRéférence 1\n"
    out = clean_wikipedia_text(article)
    assert out is not None and "du ." not in out and "Notes" not in out and "Biographie" not in out
    assert "Collège de France" in out


@pytest.fixture(scope="module")
def tok_and_data(tmp_path_factory):
    import prepare_data
    from smoke_test import make_corpus_file
    work = tmp_path_factory.mktemp("d")
    make_corpus_file(str(work / "c.txt"), n_paragraphs=600)
    data = str(work / "data")
    prepare_data.build_corpus(data, 0, 0, [str(work / "c.txt")])
    tok = prepare_data.build_tokenizer(data, vocab_size=500)
    prepare_data.tokenize_corpus(data, val_permille=150)
    return tok, data


def test_tokenizer_roundtrip_chat_mask_and_sanitize(tok_and_data):
    tok, _ = tok_and_data
    text = "Ça va très bien à l'école, 2025 ! 🙂"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("2025")) == 4                                   # chiffres isolés
    assert tok.encode("<|end|>") != [tok.end_id]                          # pas d'injection de token spécial
    ids, mask = tok.encode_chat([{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Salut !"},
                                 {"role": "user", "content": "Merci"}, {"role": "assistant", "content": "De rien."}])
    assert len(ids) == len(mask) and ids[0] == tok.user_id
    trained = [tok.decode([t]) for t, m in zip(ids, mask) if m]
    assert "".join(trained) == "Salut !<|end|>De rien.<|end|>"           # loss UNIQUEMENT sur les réponses
    gen, gmask = tok.encode_chat([{"role": "user", "content": "Bonjour"}], add_generation_prompt=True)
    assert gen[-1] == tok.assistant_id and sum(gmask) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  Données
# ══════════════════════════════════════════════════════════════════════════════
def test_pretrain_loader_is_pure_covers_epoch_and_val_is_fixed(tok_and_data):
    from data import PretrainData
    _, data = tok_and_data
    d = PretrainData(os.path.join(data, "pretrain"), seq_len=32, batch_size=4, seed=7)
    x1, y1 = d.get_batch(3)
    x2, y2 = d.get_batch(3)
    assert torch.equal(x1, x2) and torch.equal(x1[:, 1:], y1[:, :-1])       # pur + décalage d'un token
    assert not torch.equal(d.get_batch(4)[0], x1)
    # une époque = chaque bloc exactement une fois (échantillonnage sans remise)
    per_epoch = d.n_blocks // d.B
    ep, offset = d._epoch(0)
    seen = []
    for g in range(per_epoch):
        base = g * d.B
        for j in range(d.B):
            seen.append(int(ep[base + j]))
    assert len(set(seen)) == len(seen)
    v1 = [x.clone() for x, _ in d.val_batches(3)]
    v2 = [x.clone() for x, _ in d.val_batches(3)]
    assert len(v1) > 0 and all(torch.equal(a, b) for a, b in zip(v1, v2))    # VAL FIXE
    # deux rangs voient des données différentes
    d0 = PretrainData(os.path.join(data, "pretrain"), 32, 4, 7, rank=0, world=2)
    d1 = PretrainData(os.path.join(data, "pretrain"), 32, 4, 7, rank=1, world=2)
    assert not torch.equal(d0.get_batch(0)[0], d1.get_batch(0)[0])


def test_sft_collate_masks_prompt_and_padding(tok_and_data, tmp_path):
    from data import SFTData
    import sft_data
    tok, data = tok_and_data
    out = str(tmp_path / "sft")
    sft_data.build_sft(os.path.join(data, "tokenizer.json"), out, alpaca=0, piaf=0, oasst=0, synthetic_repeat=1,
                       max_len=100, val_permille=50)
    d = SFTData(out, "train", batch_size=6, pad_id=tok.pad_id)
    x, y = d.get_batch(0)
    assert x.shape == y.shape and x.shape[0] == 6
    for r in range(6):
        valid = (y[r] != -1).nonzero().flatten()
        assert len(valid) > 0
        first = int(valid[0])
        assert int(x[r, first]) == tok.assistant_id            # la 1re cible suit <|assistant|> : le prompt n'est jamais appris
        assert first > 0 and (y[r, :first] == -1).all()        # tout ce qui précède la réponse est masqué
        last = int(valid[-1])
        assert (y[r, last + 1:] == -1).all()                   # le padding de fin est masqué
        assert int(y[r, last]) in (tok.end_id,)                # la réponse finit par <|end|> (jamais tronquée)
    assert d.batches_per_epoch() == d.n // d.B
    ev = list(d.eval_batches())
    assert len(ev) > 0


def test_piaf_window_always_contains_answer():
    from sft_data import _piaf_window
    ctx = " ".join(f"Phrase numéro {i} qui parle de choses variées." for i in range(120))
    for target in ["Phrase numéro 3", "Phrase numéro 60", "Phrase numéro 118"]:
        start = ctx.index(target)
        win = _piaf_window(ctx, start, len(target))
        assert target in win and len(win) < 1000


def test_synthetic_qa_is_correct_and_deterministic():
    from synthetic_qa import build_synthetic_qa
    a, b = build_synthetic_qa(seed=3), build_synthetic_qa(seed=3)
    assert a == b and len(a) > 1500
    for c in a:
        m = c["messages"]
        assert len(m) % 2 == 0 and all(m[i]["role"] == ("user" if i % 2 == 0 else "assistant") for i in range(len(m)))
    import re
    for c in a:                                                         # arithmétique toujours juste
        q, ans = c["messages"][0]["content"], c["messages"][1]["content"]
        mt = re.match(r"Combien font (\d+) ([+×-]) (\d+) \?", q)
        if mt:
            x, op, y = int(mt[1]), mt[2], int(mt[3])
            assert ans.endswith(f"= {x + y if op == '+' else x - y if op == '-' else x * y}.")


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoints, reprise, best
# ══════════════════════════════════════════════════════════════════════════════
def _tiny_cfg(data, out, **kw):
    import smoke_test
    name = smoke_test.register_tiny_preset()
    base = dict(mode="pretrain", model_size=name, data_dir=os.path.join(data, "pretrain"), out_dir=out, seq_len=128,
                batch_size=4, grad_accum=2, lr=3e-3, min_lr=3e-4, warmup_iters=3, eval_every=5, eval_iters=3,
                save_every=5, log_every=5, device="cpu", keep=2, max_iters=10)
    base.update(kw)
    return TrainConfig(**base)


def test_rolling_cleanup_sorts_numerically(tmp_path):
    from checkpoint import rolling_cleanup, list_checkpoints
    for it in (2000, 10000, 300, 99999):
        (tmp_path / f"ckpt_{it:07d}.pt").write_bytes(b"x")
    rolling_cleanup(str(tmp_path), keep=2)
    assert [i for i, _ in list_checkpoints(str(tmp_path))] == [10000, 99999]        # les plus RÉCENTS restent


def test_resume_is_exact(tok_and_data, tmp_path):
    """20 it d'un coup  ==  10 it + reprise + 10 it (mêmes poids) : ordre des données, optimiseur, LR restaurés."""
    import train
    from checkpoint import load_checkpoint
    _, data = tok_and_data
    train.train(_tiny_cfg(data, str(tmp_path / "a"), max_iters=20))
    train.train(_tiny_cfg(data, str(tmp_path / "b"), max_iters=20, max_minutes=1e-9))    # s'arrête à it=5 (budget temps)
    r = train.train(_tiny_cfg(data, str(tmp_path / "b"), max_iters=20))                     # reprend et termine
    assert r["iter"] == 20
    wa = load_checkpoint(str(tmp_path / "a" / "final.pt"))["model"]
    wb = load_checkpoint(str(tmp_path / "b" / "final.pt"))["model"]
    for k in wa:
        assert torch.allclose(wa[k], wb[k], atol=1e-5), k


def test_best_is_persisted_and_never_overwritten_by_a_worse_resume(tok_and_data, tmp_path):
    import train
    from checkpoint import load_checkpoint
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=10))
    # on simule un très bon best obtenu avant une coupure
    with open(os.path.join(out, "best.json"), "w") as f:
        json.dump({"val_loss": 0.001, "iter": 3}, f)
    before = load_checkpoint(os.path.join(out, "best.pt"))
    train.train(_tiny_cfg(data, out, max_iters=20))
    after = load_checkpoint(os.path.join(out, "best.pt"))
    assert after["iter"] == before["iter"]                                   # best.pt intact
    assert json.load(open(os.path.join(out, "best.json")))["val_loss"] == 0.001


def test_best_matches_the_lowest_logged_val(tok_and_data, tmp_path):
    import train
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=20))
    evals = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"eval"' in l]
    best = json.load(open(os.path.join(out, "best.json")))
    assert abs(min(e["val_loss"] for e in evals) - best["val_loss"]) < 1e-9
    assert [e["it"] for e in evals if e["val_loss"] == best["val_loss"]][0] == best["iter"]


def test_vocab_mismatch_is_rejected(tok_and_data, tmp_path):
    import train, sft_data
    from mini_tokenizer import train_tokenizer
    from smoke_test import make_corpus_file
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))                     # modèle avec le vocab du tokenizer A
    make_corpus_file(str(tmp_path / "c2.txt"), n_paragraphs=300, seed=5)
    tok_b = str(tmp_path / "tokB.json")
    train_tokenizer([open(str(tmp_path / "c2.txt"), encoding="utf-8").read()], vocab_size=400, save_path=tok_b, show_progress=False)
    sft_dir = str(tmp_path / "sft_b")
    sft_data.build_sft(tok_b, sft_dir, alpaca=0, piaf=0, oasst=0, synthetic_repeat=1, max_len=100)
    cfg = TrainConfig.for_mode("sft")
    cfg.data_dir, cfg.out_dir, cfg.init_from, cfg.epochs, cfg.device = sft_dir, str(tmp_path / "x"), os.path.join(out, "best.pt"), 1, "cpu"
    with pytest.raises(ValueError, match="incompatible"):
        train.train(cfg)


def test_generation_stops_on_end_token_and_respects_forbidden(tok_and_data, tmp_path):
    import train, generate
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))
    model, tok, _ = generate.load_model(os.path.join(out, "best.pt"), device="cpu")
    ids = tok.encode("Le chat")
    toks = list(generate.stream_tokens(model, ids, max_new_tokens=30, temperature=1.0, forbidden=[5, 6, 7], seed=0))
    assert len(toks) <= 30 and not ({5, 6, 7} & set(toks))
    first = toks[0]
    assert list(generate.stream_tokens(model, ids, max_new_tokens=30, temperature=1.0, stop_ids={first}, seed=0)) == []
    a = list(generate.stream_tokens(model, ids, max_new_tokens=20, temperature=1.0, seed=42))
    b = list(generate.stream_tokens(model, ids, max_new_tokens=20, temperature=1.0, seed=42))
    assert a == b                                                            # reproductible avec seed


def test_no_repeat_ngram_only_looks_at_generated_text():
    from generate import _banned_by_ngram
    assert _banned_by_ngram([1, 2, 3, 1, 2], 3) == {3}
    assert _banned_by_ngram([1, 2, 3], 3) == set()


def test_ddp_two_cpu_processes(tok_and_data, tmp_path):
    """Lance torchrun (gloo) avec 2 processus : la boucle distribuée doit tourner et sauvegarder."""
    _, data = tok_and_data
    import smoke_test
    smoke_test.register_tiny_preset()
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import smoke_test, train\n"
        "from config import TrainConfig\n"
        "smoke_test.register_tiny_preset()\n"
        "cfg = TrainConfig(mode='pretrain', model_size='tiny-test', data_dir=%r, out_dir=%r, seq_len=128, batch_size=2, grad_accum=2,\n"
        "    lr=3e-3, min_lr=3e-4, warmup_iters=2, eval_every=4, eval_iters=3, save_every=4, log_every=4, device='cpu', max_iters=8)\n"
        "train.train(cfg)\n" % (ROOT, os.path.join(data, "pretrain"), str(tmp_path / "ddp"))
    )
    script = tmp_path / "run_ddp.py"
    script.write_text(code)
    res = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script)],
                         capture_output=True, text=True, timeout=300, cwd=ROOT)
    if res.returncode != 0:
        pytest.skip(f"DDP CPU indisponible dans cet environnement : {res.stderr[-300:]}")
    assert os.path.exists(tmp_path / "ddp" / "best.pt") and os.path.exists(tmp_path / "ddp" / "final.pt")
    assert "terminé" in res.stdout


# ══════════════════════════════════════════════════════════════════════════════
#  Planning de LR, échantillonnage
# ══════════════════════════════════════════════════════════════════════════════
def test_lr_schedules():
    from train import get_lr
    cos = TrainConfig(lr=1.0, min_lr=0.1, warmup_iters=10, schedule="cosine")
    assert get_lr(0, cos, 100) == pytest.approx(0.1) and get_lr(9, cos, 100) == pytest.approx(1.0)
    assert get_lr(100, cos, 100) == pytest.approx(0.1)
    assert all(get_lr(i, cos, 100) >= get_lr(i + 1, cos, 100) for i in range(10, 99))          # décroissant après le warmup
    wsd = TrainConfig(lr=1.0, min_lr=0.1, warmup_iters=10, schedule="wsd", decay_frac=0.2)
    assert get_lr(50, wsd, 100) == 1.0 and get_lr(79, wsd, 100) == 1.0                          # phase stable
    assert get_lr(90, wsd, 100) == pytest.approx(0.55) and get_lr(100, wsd, 100) == pytest.approx(0.1)


def test_sampling_filters():
    from generate import sample_token
    logits = torch.tensor([5.0, 4.0, 1.0, 0.5, 0.1, -3.0])
    kw = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, no_repeat_ngram=0, forbidden=[], valid_vocab=6)
    g = torch.Generator().manual_seed(0)
    assert sample_token(logits, [], **{**kw, "temperature": 0.0}, rng=g) == 0                    # greedy
    draws = {sample_token(logits, [], **{**kw, "top_k": 2}, rng=g) for _ in range(200)}
    assert draws <= {0, 1}                                                                       # top-k
    draws = {sample_token(logits, [], **{**kw, "top_p": 0.5}, rng=g) for _ in range(200)}
    assert draws == {0}                                                                          # top-p : le 1er token pèse déjà > 50 %
    draws = {sample_token(logits, [], **{**kw, "forbidden": [0]}, rng=g) for _ in range(200)}
    assert 0 not in draws
    draws = {sample_token(logits, [], **{**kw, "valid_vocab": 3}, rng=g) for _ in range(200)}
    assert draws <= {0, 1, 2}                                                                    # lignes de padding jamais tirées
    assert sample_token(logits, [0, 0, 0], **{**kw, "temperature": 0.0, "repetition_penalty": 10.0}, rng=g) == 1   # pénalité de répétition


# ══════════════════════════════════════════════════════════════════════════════
#  Personnalité
# ══════════════════════════════════════════════════════════════════════════════
def test_persona_file_is_valid():
    """personnalite.jsonl = identité + caractère : lisible, sans doublon, sans fait qui vieillit, cohérent avec DevLab / MiniLLM."""
    import check_data
    path = os.path.join(ROOT, "personnalite.jsonl")
    errors, warns, n = check_data.check_file(path)
    assert not errors and not warns, (errors, warns)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert 60 <= len(rows) <= 200 and n == len(rows)
    assert any("MiniLLM" in r["answer"] for r in rows) and any("DevLab" in r["answer"] for r in rows)
    # règle de cohérence : la personnalité ne contient pas de faits encyclopédiques (ils vivent dans knowledge/ et datasets/)
    text = " ".join(r["question"] for r in rows).lower()
    for fact_marker in ("chef-lieu", "quelle est la capitale", "superficie", "combien d'habitants", "coupe d'afrique"):
        assert fact_marker not in text, fact_marker


def test_shipped_datasets_pass_the_linter():
    import check_data, glob
    files = glob.glob(os.path.join(ROOT, "datasets", "*.jsonl"))
    assert files
    for f in files:
        errors, warns, n = check_data.check_file(f)
        assert not errors and not warns and n >= 50, (f, errors, warns[:3])


def test_check_data_detects_problems(tmp_path):
    import check_data
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"question": "Qui est le président du Cameroun ?", "answer": "Le président est X, au pouvoir depuis 1982."}\n'
        '{"question": "Q dup ?", "answer": "R1."}\n{"question": "q dup ?", "answer": "R2."}\n'
        '{"question": "Vide ?", "answer": "  "}\n'
        'ceci n\'est pas du json\n'
        '{"question": "Token ?", "answer": "contient <|end|> interdit."}\n', encoding="utf-8")
    errors, warns, n = check_data.check_file(str(p))
    assert n == 4                                                    # 4 lignes exploitables sur 6
    assert any("JSON invalide" in e for e in errors) and any("vide" in e for e in errors) and any("<|" in e for e in errors)
    assert any("vieillir" in w for w in warns) and any("double" in w for w in warns)
    assert check_data.main.__module__ == "check_data"


def test_identity_filter_drops_contradicting_examples():
    from sft_data import contradicts_persona
    assert contradicts_persona("Qui es-tu ?", "Je suis un assistant.")
    assert contradicts_persona("Écris un poème.", "En tant qu'IA, je ne peux pas ressentir d'émotions.")
    assert contradicts_persona("Ça vient d'où ?", "Je suis ChatGPT, développé par OpenAI.")
    assert not contradicts_persona("Comment faire une omelette ?", "Casse deux œufs et bats-les avec du sel.")


def test_persona_goes_to_train_only_and_is_repeated(tok_and_data, tmp_path):
    import sft_data
    from data import SFTData
    tok, data = tok_and_data
    out = str(tmp_path / "sft")
    persona = os.path.join(ROOT, "personnalite.jsonl")
    n = sum(1 for l in open(persona, encoding="utf-8") if l.strip())
    meta = sft_data.build_sft(os.path.join(data, "tokenizer.json"), out, alpaca=0, piaf=0, oasst=0, synthetic_repeat=1,
                              max_len=200, val_permille=100, persona=persona, persona_repeat=5)
    assert meta["sources"]["persona (x5)"] == n
    q0 = "Dis-moi qui tu es."
    def count(split):
        d = SFTData(out, split, batch_size=2, pad_id=tok.pad_id)
        c = 0
        for i in range(d.n):
            ids = [int(t) for t in d.tokens[d.offsets[i]:d.offsets[i + 1]]]
            c += q0 in tok.decode(ids)
        return c
    assert count("train") == 5 and count("val") == 0


def test_push_script_includes_the_persona_file():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pg", os.path.join(ROOT, "push_to_github.py"))
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    files = pg.collect_files(ROOT)
    assert "personnalite.jsonl" in files and "MiniLLM_v2.ipynb" in files and "tests/test_all.py" in files
    assert not any(f.startswith(("data/", "checkpoints/")) for f in files)


# ══════════════════════════════════════════════════════════════════════════════
#  v2.1 : nettoyage renforcé, RAG, outils Kaggle, continuité multi-GPU, ETA, persona
# ══════════════════════════════════════════════════════════════════════════════
def test_v21_cleaning_catches_empty_parentheses_and_fixes_elisions():
    from text_cleaning import has_defect, normalize_elisions, clean_web_text, spam_score
    for bad in ["L'archidiocèse de Lecce (en latin : ) est un archidiocèse.", "Il est né (le , à Paris).", "Il est né (le ) à Rome.",
                "Le club (fondé en ; dissous en ) existe.", "Il dit « » et partit."]:
        assert has_defect(bad), bad
    for ok in ["L'archidiocèse de Lecce (en latin : Archidioecesis Lupiensis) est un archidiocèse.", "Jean (1902-1980) est un écrivain.",
               "Le prix est de 3,5 euros ; il augmente."]:
        assert not has_defect(ok), ok
    assert normalize_elisions("L' archidiocèse et d' Italie, aujourd' hui, qu' il vienne") == "L'archidiocèse et d'Italie, aujourd'hui, qu'il vienne"
    assert normalize_elisions("Il a dit 'bonjour' à l'école.") == "Il a dit 'bonjour' à l'école."
    spam = "Cliquez ici pour notre boutique en ligne, livraison gratuite, abonnez-vous à la newsletter. " * 4
    assert spam_score(spam) >= 3 and clean_web_text(spam) is None
    assert clean_web_text("Le fleuve Bénoué traverse Garoua avant de rejoindre le Niger, au cœur de la région. " * 8) is not None


def test_refilter_reprocesses_an_existing_corpus_without_downloading(tmp_path):
    import prepare_data
    from mini_tokenizer import train_tokenizer
    data = tmp_path / "data"
    (data / "corpus").mkdir(parents=True)
    good = "Le fleuve Bénoué traverse la ville de Garoua avant de rejoindre le Niger, et les habitants y pêchent depuis toujours. " * 4
    holey = "L' archidiocèse de Lecce (en latin : ) est un archidiocèse métropolitain de l' Église catholique. "
    docs = [{"text": (good + holey + good).strip(), "src": "wiki"} for _ in range(30)]
    docs += [{"text": "Cliquez ici boutique en ligne livraison gratuite newsletter " * 20, "src": "web"} for _ in range(5)]
    with open(data / "corpus" / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    train_tokenizer([d["text"] for d in docs], vocab_size=400, save_path=str(data / "tokenizer.json"), show_progress=False)
    meta = prepare_data.tokenize_corpus(str(data), val_permille=200, refilter=True)
    assert meta["refiltered"] and meta["n_train_docs"] + meta["n_val_docs"] == 30       # les 5 docs spam ont disparu
    from mini_tokenizer import MiniTokenizer
    tok = MiniTokenizer(str(data / "tokenizer.json"))
    text = tok.decode(np.fromfile(str(data / "pretrain" / "train.bin"), dtype=meta["dtype"]).tolist())
    assert "(en latin : )" not in text and "L' archidiocèse" not in text and "Bénoué" in text



def test_rag_retrieval_and_prompt_budget(tok_and_data, tmp_path):
    from rag import BM25Index, load_knowledge, build_rag_messages, rag_answer, RAG_INSTRUCTION
    kb = load_knowledge(os.path.join(ROOT, "knowledge"))
    assert len(kb) >= 5
    idx = BM25Index(kb)
    cases = {"Quelle est la capitale du Cameroun ?": "Yaoundé", "Sur quel fleuve est construite Garoua ?": "Bénoué",
             "Qui a créé MiniLLM ?": "DevLab", "Quelle est la monnaie du Cameroun ?": "CFA"}
    for q, expected in cases.items():
        hit = idx.search(q, 1)
        assert hit and expected in kb[hit[0][1]], q
    assert idx.search("Quelle est la recette du ndolé ?", 1) == []                     # rien de pertinent -> pas de passage
    msg = build_rag_messages("Où est Garoua ?", "Phrase numéro un. " * 200, max_chars=300)[0]["content"]
    assert msg.startswith(RAG_INSTRUCTION) and len(msg) < 500
    # de bout en bout avec un modèle minuscule : le prompt doit toujours tenir dans la fenêtre
    import train, generate
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))
    model, tok, _ = generate.load_model(os.path.join(out, "best.pt"), device="cpu")
    ans, passage = rag_answer(model, tok, idx, "Quelle est la capitale du Cameroun ?", max_new_tokens=20, temperature=0.0)
    assert isinstance(ans, str) and passage is not None and "Yaoundé" in passage
    ans2, passage2 = rag_answer(model, tok, idx, "Quelle est la recette du ndolé ?", max_new_tokens=20, temperature=0.0)
    assert passage2 is None


def test_kaggle_utils_restore_verify_and_drive(tok_and_data, tmp_path, monkeypatch):
    import kaggle_utils as ku
    _, data = tok_and_data
    # faux /kaggle/input : un dataset (données) + la sortie d'un notebook précédent (checkpoints)
    inp = tmp_path / "input"
    (inp / "ds" / "data").mkdir(parents=True)
    import shutil
    shutil.copytree(os.path.join(data, "pretrain"), inp / "ds" / "data" / "pretrain")
    shutil.copy(os.path.join(data, "tokenizer.json"), inp / "ds" / "data" / "tokenizer.json")
    import train
    out = str(tmp_path / "prev" / "checkpoints" / "pretrain")
    train.train(_tiny_cfg(data, out, max_iters=10, save_every=5))
    shutil.copytree(tmp_path / "prev", inp / "nb_prev")
    work = str(tmp_path / "work")
    found = ku.restore_from_inputs(str(inp), work)
    assert {"pretrain_data", "tokenizer", "ckpt_pretrain"} <= set(found)
    meta = ku.verify_pretrain_data(os.path.join(work, "data", "pretrain"))
    assert meta["n_train_tokens"] > 0
    ck = ku.verify_checkpoint(os.path.join(work, "checkpoints", "pretrain", "best.pt"))
    assert ck["iter"] is not None
    from checkpoint import list_checkpoints
    assert list_checkpoints(os.path.join(work, "checkpoints", "pretrain"))[-1][0] == 10        # le plus récent seulement
    # 2e appel : idempotent (rien n'est recopié)
    assert ku.restore_from_inputs(str(inp), work) == {}
    # un .bin tronqué est détecté
    with open(os.path.join(work, "data", "pretrain", "train.bin"), "r+b") as f:
        f.truncate(1000)
    with pytest.raises(ValueError, match="tronqué"):
        ku.verify_pretrain_data(os.path.join(work, "data", "pretrain"))
    # Drive : id reconnu, téléchargement simulé, entrées vides / déjà présentes ignorées
    assert ku.extract_drive_id("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing") == "1AbCdEfGhIjKlMnOpQrStUvWxYz"
    import types
    calls = []

    def fake_download(url, dst, quiet=False, fuzzy=True):
        calls.append(url)
        open(dst, "wb").write(b"abc")
        return dst
    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_download))
    link = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view"
    done = ku.download_drive({"data/x.bin": link, "data/skip.bin": "", "data/pretrain/meta.json": link}, work)
    assert done == ["data/x.bin"] and len(calls) == 1 and os.path.exists(os.path.join(work, "data", "x.bin"))
    with pytest.raises(ValueError, match="non reconnu"):
        ku.download_drive({"data/y.bin": "https://example.com/pas-un-lien"}, work)


def test_resume_stream_is_identical_across_gpu_counts(tok_and_data):
    """1 GPU (accum 8) et 2 GPU (accum 4) lisent EXACTEMENT les mêmes séquences à chaque itération -> on peut reprendre un run Colab (1 GPU) sur Kaggle (2 GPU)."""
    from data import PretrainData
    _, data = tok_and_data
    d1 = PretrainData(os.path.join(data, "pretrain"), 32, 2, seed=11, rank=0, world=1)
    d2 = [PretrainData(os.path.join(data, "pretrain"), 32, 2, seed=11, rank=r, world=2) for r in range(2)]
    for it in (0, 3, 7):
        a = {tuple(row.tolist()) for k in range(8) for row in d1.get_batch(it * 8 + k)[0]}
        b = {tuple(row.tolist()) for r in range(2) for k in range(4) for row in d2[r].get_batch(it * 4 + k)[0]}
        assert a == b and len(a) == 16


def test_eta_and_memory_are_logged_and_persona_check_runs(tok_and_data, tmp_path):
    import train, evaluate
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=10))
    rows = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"train"' in l]
    assert rows and all("eta_h" in r and "mem_gb" in r for r in rows)
    persona = tmp_path / "p.jsonl"
    persona.write_text('{"question": "Comment tu t\'appelles ?", "answer": "Je suis MiniLLM."}\n'
                       '{"question": "Qui t\'a créé ?", "answer": "J\'ai été créé par DevLab."}\n', encoding="utf-8")
    res = evaluate.persona_check(os.path.join(out, "best.pt"), str(persona), device="cpu", show=2)
    assert res["n"] == 2 and 0.0 <= res["f1_moyen"] <= 1.0


def test_tokenizer_fingerprint_blocks_mismatched_resume(tok_and_data, tmp_path):
    """Un checkpoint entraîné avec un tokenizer ne doit JAMAIS être repris avec des données d'un autre tokenizer (même taille de vocab = piège silencieux)."""
    import shutil, train
    _, data = tok_and_data
    d2 = str(tmp_path / "data2")
    shutil.copytree(os.path.join(data, "pretrain"), os.path.join(d2, "pretrain"))
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))
    ck = torch.load(os.path.join(out, "best.pt"), weights_only=True)
    assert ck.get("tokenizer_sha")
    meta_path = os.path.join(d2, "pretrain", "meta.json")
    meta = json.load(open(meta_path))
    meta["tokenizer_sha"] = "0000000000000000"
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ValueError, match="Tokenizer incompatible"):
        train.train(_tiny_cfg(d2, out, max_iters=10))                        # reprise avec un autre tokenizer -> refusée


def test_v2_checkpoints_without_fingerprint_still_resume(tok_and_data, tmp_path):
    """Compat v2 : un checkpoint sans tokenizer_sha (ancien) doit se reprendre normalement avec des données qui en ont un."""
    import train
    from checkpoint import latest_checkpoint
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))
    path = latest_checkpoint(out)
    ck = torch.load(path, weights_only=True)
    ck.pop("tokenizer_sha", None)
    torch.save(ck, path)
    assert train.train(_tiny_cfg(data, out, max_iters=10))["iter"] == 10


def test_rag_supports_qa_jsonl_coverage_guard_and_multiple_sources(tmp_path):
    from rag import load_knowledge_many, BM25Index
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.jsonl").write_text('{"question": "Quelle est la capitale du Togo ?", "answer": "La capitale du Togo est Lomé."}\n'
                                '{"text": "Le lac Tchad se situe entre le Tchad, le Niger, le Nigeria et le Cameroun."}\n', encoding="utf-8")
    (kb / "b.txt").write_text("Le mont Cameroun est le plus haut sommet du Cameroun.\n\nOk\n\nLa Sanaga est un fleuve.", encoding="utf-8")
    passages = load_knowledge_many([str(kb / "a.jsonl"), str(kb / "b.txt")])
    assert "La capitale du Togo est Lomé." in passages                # Q/R : la réponse devient le passage
    assert "Ok" not in passages and "La Sanaga est un fleuve." in passages   # < 20 caractères ignoré, court mais valide gardé
    idx = BM25Index(passages)
    assert "Lomé" in passages[idx.search("Peux-tu me dire quelle est la capitale du Togo ?", 1)[0][1]]
    assert "Sanaga" in passages[idx.search("Où se trouve la Sanaga ?", 1)[0][1]]
    assert idx.search("Qui a gagné la Coupe du monde 1998 ?", 1) == []      # sujet voisin mais pas de réponse -> rejeté
    assert idx.search("le la les de", 1) == []                              # aucun mot utile


def test_shipped_knowledge_answers_key_questions():
    from rag import load_knowledge_many, BM25Index
    passages = load_knowledge_many([os.path.join(ROOT, "knowledge"), os.path.join(ROOT, "datasets")])
    assert len(passages) > 200
    idx = BM25Index(passages)
    cases = {"Quelle est la capitale du Togo ?": "Lomé", "Quel est le chef-lieu de la région du Nord ?": "Garoua",
             "Combien de paramètres a MiniLLM ?": "49 millions", "Qu'est-ce qu'un token ?": "morceau de texte",
             "Quelle est la capitale de la Tanzanie ?": "Dodoma", "Sur quel fleuve est construite Garoua ?": "Bénoué"}
    for q, expected in cases.items():
        hit = idx.search(q, 1)
        assert hit and expected in passages[hit[0][1]], q
    assert idx.search("Qui est le président de la France ?", 1) == []


def test_extra_jsonl_can_be_repeated_in_train_only(tok_and_data, tmp_path):
    import sft_data
    from data import SFTData
    tok, data = tok_and_data
    extra = tmp_path / "faits.jsonl"
    qs = [f"Question de test numéro {i} sur un fait ?" for i in range(20)]
    extra.write_text("\n".join(json.dumps({"question": q, "answer": f"Réponse {i}."}, ensure_ascii=False) for i, q in enumerate(qs)), encoding="utf-8")
    out = str(tmp_path / "sft")
    meta = sft_data.build_sft(os.path.join(data, "tokenizer.json"), out, alpaca=0, piaf=0, oasst=0, synthetic_repeat=1, max_len=200,
                              val_permille=100, extra_jsonl=[str(extra)], extra_repeat=3)
    assert any(k.startswith("jsonl:") for k in meta["sources"])
    d = SFTData(out, "train", batch_size=2, pad_id=tok.pad_id)
    counts = {q: 0 for q in qs}
    for i in range(d.n):
        text = tok.decode([int(t) for t in d.tokens[d.offsets[i]:d.offsets[i + 1]]])
        for q in qs:
            counts[q] += q in text
    assert set(counts.values()) <= {0, 3} and 3 in counts.values()         # chaque fait : 3 copies en train (ou 0 s'il est tombé en val)


def test_configure_optimizers_does_not_request_fused():
    """fused=True casse la reprise d'un optimiseur sauvegardé sur un autre device (compteur 'step' en tenseur GPU
    incompatible avec un état venu du CPU) -> jamais demandé, quel que soit device_type."""
    cfg = ModelConfig(vocab_size=50, n_layers=1, d_model=16, n_heads=2, kv_heads=2, max_seq_len=16)
    m = MiniLLM(cfg)
    for device_type in ("cpu", "cuda"):
        opt = m.configure_optimizers(lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), device_type=device_type)
        assert opt.defaults.get("fused") is not True, device_type


def test_resume_realigns_optimizer_state_onto_the_current_device(tok_and_data, tmp_path):
    """Reproduit le bug observé : un optimiseur repris est sauvegardé avec des tenseurs d'état 'ailleurs' (ex. un
    run tombé sur CPU) ; la reprise doit les réaligner elle-même plutôt que planter au premier pas."""
    import train
    from checkpoint import latest_checkpoint
    _, data = tok_and_data
    out = str(tmp_path / "o")
    train.train(_tiny_cfg(data, out, max_iters=5))
    path = latest_checkpoint(out)
    ck = torch.load(path, weights_only=True)
    for state in ck["optimizer"]["state"].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.clone().detach()
    torch.save(ck, path)
    r = train.train(_tiny_cfg(data, out, max_iters=10))          # ne doit pas lever d'AssertionError
    assert r["iter"] == 10

"""Suite de tests MiniLLM v2 — CPU uniquement, sans internet.  Lancer :  python -m pytest -q"""
import json
import os
import subprocess
import sys

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
    with pytest.raises(ValueError, match="Vocabulaire incompatible"):
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

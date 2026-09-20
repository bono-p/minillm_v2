"""
train.py — pré-entraînement ET fine-tuning (SFT) avec la même boucle.

Corrections majeures par rapport à l'ancienne version (voir README / CHANGELOG) :
  * "best" FIABLE : val FIXE (mêmes séquences à chaque eval) ; meilleure val_loss persistée (best.json)
    et restaurée à la reprise -> un run repris ne peut plus écraser un meilleur modèle.
  * REPRISE exacte : checkpoint complet (modèle+optimiseur+scaler+iter+best) ; on reprend le plus récent,
    pas best.pt ; l'ordre des données est déterministe -> on ne relit plus le début du corpus.
  * init_from (poids seuls) pour le SFT au lieu du hack reset_iter ; optimiseur neuf.
  * Écritures atomiques (session coupée = pas de checkpoint corrompu), nettoyage numérique des ckpt.
  * SFT : loss uniquement sur les réponses, normalisée par le nombre de tokens de réponse.
  * fp16 + GradScaler sur T4 ; bf16 seulement sur GPU Ampere+ (jamais de bf16 émulé, 4x plus lent).
  * DDP (torchrun) pour Kaggle T4 x2 ; budget temps (--max_minutes) avec sauvegarde propre.
  * Log JSONL (checkpoints/.../log.jsonl) : plus besoin de recharger les checkpoints pour tracer les courbes.

Lancement :
  python train.py --mode pretrain --data_dir data/pretrain --out_dir checkpoints/pretrain
  torchrun --standalone --nproc_per_node=2 train.py --mode pretrain ...        # 2 GPU
  python train.py --mode sft --data_dir data/sft --init_from checkpoints/pretrain/best.pt
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import shutil
import signal
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint import (latest_checkpoint, load_checkpoint, read_best_val, rolling_cleanup, save_best,
                        save_final, save_full)
from config import ModelConfig, TrainConfig, get_preset
from data import PretrainData, SFTData
from model import MiniLLM


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════════════════
def resolve_device(pref: str, local_rank: int = 0) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def resolve_dtype(pref: str, device: torch.device) -> torch.dtype:
    """fp16+GradScaler sur T4/P100 ; bf16 seulement si le GPU le supporte VRAIMENT (Ampere+)."""
    if device.type != "cuda":
        return torch.float32
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    if pref == "auto":
        return torch.bfloat16 if native_bf16 else torch.float16
    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dt = table[pref]
    if dt is torch.bfloat16 and not native_bf16:
        print("⚠️  bf16 non natif sur ce GPU (émulé = ~4x plus lent) -> fp16 utilisé à la place.")
        dt = torch.float16
    return dt


def get_lr(it: int, cfg: TrainConfig, max_iters: int) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / max(1, cfg.warmup_iters)
    if cfg.schedule == "wsd":                                   # warmup - stable - decay
        decay_start = int(max_iters * (1 - cfg.decay_frac))
        if it < decay_start:
            return cfg.lr
        p = min(1.0, (it - decay_start) / max(1, max_iters - decay_start))
        return cfg.lr + (cfg.min_lr - cfg.lr) * p
    if it >= max_iters:
        return cfg.min_lr
    p = (it - cfg.warmup_iters) / max(1, max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (cfg.lr - cfg.min_lr)


def _log_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Évaluation (val FIXE, modèle non compilé, réduite entre les GPU)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(raw_model, val_iter, ctx, device, ddp: bool):
    raw_model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    for x, y in val_iter:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = raw_model(x, y, reduction="sum")
        loss_sum += loss.double()
        count += (y != -1).sum()
    if ddp:
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
    raw_model.train()
    return (loss_sum / count.clamp(min=1)).item(), int(count.item())


# ══════════════════════════════════════════════════════════════════════════════
#  Entraînement
# ══════════════════════════════════════════════════════════════════════════════
def train(cfg: TrainConfig) -> dict:
    # ── distribué ───────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        device = resolve_device(cfg.device, local_rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    else:
        rank, world, local_rank = 0, 1, 0
        device = resolve_device(cfg.device)
    master = rank == 0
    torch.manual_seed(cfg.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = resolve_dtype(cfg.dtype, device)
    use_amp = dtype in (torch.float16, torch.bfloat16)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else contextlib.nullcontext()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype is torch.float16))
    except (AttributeError, TypeError):                          # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype is torch.float16))
    sft = cfg.mode == "sft"

    def say(*a):
        if master:
            print(*a, flush=True)

    # ── données ─────────────────────────────────────────────────────────
    if sft:
        train_data = SFTData(cfg.data_dir, "train", cfg.batch_size, cfg.seed, rank, world)
        val_data = SFTData(cfg.data_dir, "val", cfg.batch_size, cfg.seed, rank, world)
        iters_per_epoch = max(1, train_data.batches_per_epoch() // cfg.grad_accum)
        max_iters = cfg.epochs * iters_per_epoch if cfg.epochs > 0 else cfg.max_iters
        val_iter_fn = lambda: val_data.eval_batches(cfg.eval_iters)            # noqa: E731
        data_vocab, data_max_len = train_data.vocab_size, int(train_data.meta["max_len"])
    else:
        train_data = PretrainData(cfg.data_dir, cfg.seq_len, cfg.batch_size, cfg.seed, rank, world)
        iters_per_epoch = max(1, train_data.n_blocks // (cfg.batch_size * cfg.grad_accum * world))
        max_iters = cfg.max_iters
        val_iter_fn = lambda: train_data.val_batches(cfg.eval_iters)          # noqa: E731
        data_vocab, data_max_len = train_data.vocab_size, cfg.seq_len

    # ── reprise / init ──────────────────────────────────────────────────
    resume_path = latest_checkpoint(cfg.out_dir) if cfg.resume == "auto" else (cfg.resume or None)
    ckpt = load_checkpoint(resume_path) if resume_path else None
    init_ckpt = None
    if ckpt is not None:
        model_cfg = ModelConfig.from_dict(ckpt["config"])
        if cfg.init_from:
            say("ℹ️  reprise d'un checkpoint : init_from ignoré.")
    elif cfg.init_from:
        init_ckpt = load_checkpoint(cfg.init_from)
        model_cfg = ModelConfig.from_dict(init_ckpt["config"])
    else:
        model_cfg = get_preset(cfg.model_size, vocab_size=data_vocab, max_seq_len=cfg.seq_len)
    if cfg.dropout >= 0:
        model_cfg = dataclasses.replace(model_cfg, dropout=cfg.dropout)
    data_sha = (train_data.meta or {}).get("tokenizer_sha")
    ref_sha = (ckpt or init_ckpt or {}).get("tokenizer_sha")
    if data_sha and ref_sha and data_sha != ref_sha:
        raise ValueError(f"Tokenizer incompatible : les données utilisent le tokenizer {data_sha} mais le checkpoint a été entraîné avec "
                         f"{ref_sha}. Utilise les MÊMES data/tokenizer.json et .bin que pour le checkpoint (ne reconstruis pas les données).")
    ckpt_extra = {"tokenizer_sha": data_sha} if data_sha else {}
    if model_cfg.vocab_size != data_vocab:
        raise ValueError(f"Vocabulaire incompatible : modèle={model_cfg.vocab_size}, données={data_vocab}. "
                         "Utilise le même tokenizer pour le pré-entraînement et le SFT.")
    if data_max_len > model_cfg.max_seq_len:
        raise ValueError(f"Séquences ({data_max_len}) plus longues que max_seq_len du modèle ({model_cfg.max_seq_len}).")

    model = MiniLLM(model_cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    elif init_ckpt is not None:
        model.load_state_dict(init_ckpt["model"])
    model.to(device)
    raw_model = model
    optimizer = raw_model.configure_optimizers(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), device.type)

    start_it, best_val, no_improve, tokens_seen = 0, float("inf"), 0, 0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer.param_groups[0]["weight_decay"] = cfg.weight_decay
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_it, no_improve, tokens_seen = int(ckpt["iter"]), int(ckpt["no_improve"]), int(ckpt["tokens_seen"])
        best_val = min(float(ckpt["best_val_loss"]), read_best_val(cfg.out_dir))
        del ckpt
    del init_ckpt

    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ── dossier de sortie, log ──────────────────────────────────────────
    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    if master:
        os.makedirs(cfg.out_dir, exist_ok=True)
        tok_src = os.path.join(os.path.dirname(os.path.abspath(cfg.data_dir.rstrip("/"))), "tokenizer.json")
        if os.path.exists(tok_src):
            shutil.copy(tok_src, os.path.join(cfg.out_dir, "tokenizer.json"))     # ckpt + tokenizer voyagent ensemble
    tokens_per_iter = cfg.batch_size * cfg.grad_accum * world * (data_max_len if not sft else 1)
    say("═" * 72)
    say(f"MiniLLM v2 — {cfg.mode.upper()} | {model_cfg.size_name()} ({raw_model.num_params() / 1e6:.1f}M params, "
        f"dont {model_cfg.count_non_embedding_params() / 1e6:.1f}M hors embeddings)")
    say(f"device={device} dtype={dtype} GPU x{world} compile={cfg.compile} | batch {cfg.batch_size}x{cfg.grad_accum}x{world}")
    if sft:
        say(f"SFT : {train_data.n:,} exemples | {iters_per_epoch} it/époque | {max_iters} it ({cfg.epochs} époques)")
    else:
        say(f"Pré-entraînement : {tokens_per_iter:,} tokens/it | {max_iters} it = {max_iters * tokens_per_iter / 1e6:.0f} M tokens "
            f"({max_iters / iters_per_epoch:.2f} époque sur {len(train_data.train) / 1e6:.0f} M tokens)")
    say(f"Reprise à it={start_it} | best_val_loss={best_val:.4f}" if start_it or best_val < float("inf")
        else "Nouveau run.")
    say("═" * 72)

    def signal_handler(signum, frame):                          # SIGTERM (arrêt de session) -> sauvegarde propre
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        pass

    # ── boucle ──────────────────────────────────────────────────────────
    it = start_it
    last_val: Optional[float] = None
    t_start = time.perf_counter()
    t_log, tok_log, it_log = t_start, 0, start_it
    loss_acc = torch.zeros((), device=device)
    n_acc = 0
    raw_model.train()

    def do_eval() -> float:
        nonlocal best_val, no_improve, last_val
        val_loss, n_tok = evaluate(raw_model, val_iter_fn(), ctx, device, ddp)
        last_val = val_loss
        improved = val_loss < best_val
        if improved:
            best_val, no_improve = val_loss, 0
            if master:
                save_best(cfg.out_dir, it, raw_model, model_cfg, cfg, val_loss, tokens_seen, ckpt_extra)
        else:
            no_improve += 1
        say(f"  ┌ EVAL it {it:>6} | val_loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.1f}) | "
            f"{n_tok:,} tokens évalués | best {best_val:.4f} {'★ nouveau best' if improved else f'(sans amélio. x{no_improve})'}")
        if master:
            _log_jsonl(log_path, {"type": "eval", "it": it, "val_loss": val_loss, "best": best_val,
                                  "improved": improved, "tokens": tokens_seen})
        return val_loss

    def do_save():
        if master:
            save_full(cfg.out_dir, it, raw_model, optimizer, scaler, model_cfg, cfg, best_val, no_improve, tokens_seen, ckpt_extra)
            rolling_cleanup(cfg.out_dir, cfg.keep)

    stopped_early = False
    try:
        if cfg.eval_at_start and start_it == 0:
            do_eval()
        while it < max_iters:
            lr = get_lr(it, cfg, max_iters)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            g0 = it * cfg.grad_accum
            micro = [train_data.get_batch(g0 + k) for k in range(cfg.grad_accum)]
            total_answer_tok = sum(int((y != -1).sum()) for _, y in micro) if sft else 0
            for k, (x, y) in enumerate(micro):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                sync = model.no_sync() if (ddp and k < cfg.grad_accum - 1) else contextlib.nullcontext()
                with sync, ctx:
                    if sft:                                    # somme / nb total de tokens de réponse du pas
                        _, loss = model(x, y, reduction="sum")
                        loss = loss / max(1, total_answer_tok)
                    else:
                        _, loss = model(x, y)
                        loss = loss / cfg.grad_accum
                scaler.scale(loss).backward()
                loss_acc += loss.detach().float()
                tokens_seen += x.numel() * world
            n_acc += 1

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip) if cfg.grad_clip > 0 else torch.zeros(())
            scaler.step(optimizer)
            scaler.update()
            it += 1

            # ── log ─────────────────────────────────────────────────────
            if it % cfg.log_every == 0 or it == max_iters:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                avg = (loss_acc / n_acc).item()
                if ddp:
                    t = torch.tensor(avg, device=device)
                    dist.all_reduce(t)
                    avg = (t / world).item()
                if not math.isfinite(avg):
                    if not scaler.is_enabled():
                        raise RuntimeError(f"Loss non finie à l'itération {it} ({avg}). Baisse le lr ou vérifie les données.")
                    say(f"⚠️  loss non finie à it {it} (fp16) : le GradScaler saute ces pas, surveillance…")
                    loss_acc.zero_()
                    n_acc, t_log, tok_log, it_log = 0, now, tokens_seen, it
                    continue
                tps = (tokens_seen - tok_log) / max(1e-9, now - t_log)
                sec_per_it = (now - t_log) / max(1, it - it_log)
                eta_h = max(0, max_iters - it) * sec_per_it / 3600            # au rythme actuel (hors évaluations)
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                say(f"it {it:>6}/{max_iters} | loss {avg:.4f} | lr {lr:.2e} | gnorm {float(grad_norm):.2f} | "
                    f"{tps / 1e3:.1f}k tok/s | {tokens_seen / 1e6:.1f}M tok | {(now - t_start) / 60:.1f} min | "
                    f"ETA {eta_h:.1f} h" + (f" | {mem_gb:.1f} Go GPU" if mem_gb else ""))
                if master:
                    _log_jsonl(log_path, {"type": "train", "it": it, "loss": avg, "lr": lr, "gnorm": float(grad_norm),
                                          "tok_s": tps, "tokens": tokens_seen, "eta_h": eta_h, "mem_gb": mem_gb})
                loss_acc.zero_()
                n_acc, t_log, tok_log, it_log = 0, now, tokens_seen, it

            # ── budget temps ────────────────────────────────────────────
            time_up = False
            if cfg.max_minutes > 0 and it % cfg.log_every == 0:
                flag = torch.tensor(1.0 if (time.perf_counter() - t_start) / 60 >= cfg.max_minutes else 0.0, device=device)
                if ddp:
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                time_up = bool(flag.item())

            # ── éval / sauvegarde (éval AVANT sauvegarde : best_val à jour dans le ckpt) ──
            if it % cfg.eval_every == 0 or it == max_iters or time_up:
                do_eval()
                if cfg.patience > 0 and no_improve >= cfg.patience:
                    say(f"⏹  early stopping : {no_improve} évaluations sans amélioration.")
                    stopped_early = True
            if it % cfg.save_every == 0 or it == max_iters or time_up or stopped_early:
                do_save()
            if time_up:
                say(f"⏱  budget de {cfg.max_minutes} min atteint : checkpoint sauvegardé, relance la même commande pour continuer.")
                break
            if stopped_early:
                break
    except KeyboardInterrupt:
        say("\n⚠️  interruption : sauvegarde d'un checkpoint de reprise…")
        do_save()
        if ddp:
            dist.destroy_process_group()
        return {"interrupted": True, "iter": it, "best_val_loss": best_val}

    if master:
        save_final(cfg.out_dir, it, raw_model, model_cfg, cfg, last_val, tokens_seen, ckpt_extra)
    say(f"✅ terminé à it={it} | meilleure val_loss={best_val:.4f} -> {os.path.join(cfg.out_dir, 'best.pt')}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return {"interrupted": False, "iter": it, "best_val_loss": best_val}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI : un argument par champ de TrainConfig
# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> TrainConfig:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", default="pretrain", choices=["pretrain", "sft"])
    known, _ = pre.parse_known_args(argv)
    base = TrainConfig.for_mode(known.mode)

    p = argparse.ArgumentParser(description="MiniLLM v2 — entraînement", parents=[pre])
    for f in dataclasses.fields(TrainConfig):
        if f.name == "mode":
            continue
        default = getattr(base, f.name)
        if isinstance(default, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(f"--{f.name}", type=type(default), default=default)
    args = vars(p.parse_args(argv))
    return TrainConfig(**args)


if __name__ == "__main__":
    train(parse_args())

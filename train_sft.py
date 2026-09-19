"""Supervised fine-tuning entry point for PIAF/FQuAD-style QA data.

This deliberately remains separate from the pre-training loop: pre-training uses
binary token streams, while SFT needs per-example masked labels.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
import tiktoken

from config import PRESETS
from model import MiniLLM
from sft_data import QASFTDataset, sft_collate


def load_model(checkpoint: str, size: str, seq_len: int, device: str):
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = state["config"]
        model = MiniLLM(cfg).to(device)
        weights = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
        model.load_state_dict(weights)
        return model, cfg, state
    cfg = PRESETS[size]
    cfg.max_seq_len = seq_len
    return MiniLLM(cfg).to(device), cfg, None


def run(args):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding(args.encoding)
    model, model_cfg, previous = load_model(args.checkpoint, args.size, args.seq, device)
    model_cfg.max_seq_len = args.seq
    model.train()

    train_set = QASFTDataset(args.data, enc, args.seq, split="train", val_ratio=args.val_ratio)
    val_set = QASFTDataset(args.data, enc, args.seq, split="validation", val_ratio=args.val_ratio)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=sft_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, collate_fn=sft_collate)

    optimizer = model.configure_optimizers(args.lr, args.lr * 0.1, args.weight_decay, 0.9, 0.95, device)
    if previous and "optimizer" in previous:
        optimizer.load_state_dict(previous["optimizer"])
    os.makedirs(args.out, exist_ok=True)
    best = float(previous.get("best_val_loss", "inf")) if previous else float("inf")

    for step in range(args.iters):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if step % args.log_every == 0:
                print(f"sft step={step} loss={loss.item():.4f}")
            break

        if step % args.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    _, loss = model(inputs.to(device), labels.to(device))
                    losses.append(loss.item())
                    if len(losses) >= args.eval_batches:
                        break
            val_loss = sum(losses) / max(1, len(losses))
            print(f"sft step={step} val_loss={val_loss:.4f}")
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {"iter": step, "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                       "val_loss": val_loss, "best_val_loss": min(best, val_loss),
                       "best_iter": step if val_loss < best else previous.get("best_iter", 0) if previous else 0,
                       "config": model_cfg, "mode": "sft"}
            torch.save(payload, os.path.join(args.out, "last.pt"))
            if val_loss < best:
                best = val_loss
                torch.save(payload, os.path.join(args.out, "best.pt"))
            model.train()


def main():
    p = argparse.ArgumentParser(description="MiniLLM v3 — SFT français")
    p.add_argument("--data", nargs="+", required=True, help="JSON PIAF/FQuAD/SQuAD")
    p.add_argument("--checkpoint", default="", help="Checkpoint pretrain optionnel")
    p.add_argument("--size", default="15M", choices=list(PRESETS))
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--encoding", default="cl100k_base")
    p.add_argument("--device", default="auto")
    run(p.parse_args())


if __name__ == "__main__":
    main()

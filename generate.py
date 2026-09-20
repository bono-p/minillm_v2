"""
generate.py — génération de texte et chat.

Corrections / améliorations :
  * KV-cache : chaque token coûte O(1) au lieu de recalculer tout le contexte (5 à 20x plus rapide).
  * Le chat utilise EXACTEMENT le template du SFT (<|user|> … <|end|> <|assistant|>) — avant, generate.py
    envoyait du texte brut au modèle fine-tuné.
  * Arrêt sur <|end|> / <|endoftext|> ; les tokens de rôle sont interdits dans la réponse.
  * Contrôles anti-boucle (le défaut n°1 des petits modèles) : pénalité de répétition et interdiction des
    n-grammes répétés, appliqués UNIQUEMENT au texte généré (donc pas de blocage quand on recopie le contexte).
  * Affichage en streaming, historique tronqué proprement pour rester dans la fenêtre du modèle.

Usage :
  python generate.py --ckpt checkpoints/sft/best.pt --mode chat
  python generate.py --ckpt checkpoints/pretrain/best.pt --mode complete --prompt "La ville de Garoua"
  python generate.py --ckpt checkpoints/sft/best.pt --prompt "Quelle est la capitale du Cameroun ?"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import torch

from checkpoint import load_checkpoint
from config import ModelConfig
from mini_tokenizer import MiniTokenizer
from model import KVCache, MiniLLM


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement
# ══════════════════════════════════════════════════════════════════════════════
def pick_device_dtype(device: str = "auto", dtype: str = "auto") -> Tuple[torch.device, torch.dtype]:
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    if dev.type != "cuda":
        return dev, torch.float32
    if dtype == "auto":
        return dev, torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    return dev, {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def load_model(ckpt_path: str, tokenizer_path: Optional[str] = None, device: str = "auto", dtype: str = "auto"):
    ckpt = load_checkpoint(ckpt_path)
    cfg = ModelConfig.from_dict(ckpt["config"])
    dev, dt = pick_device_dtype(device, dtype)
    model = MiniLLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to_inference(dev, dt)
    candidates = [tokenizer_path, os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer.json"),
                  "data/tokenizer.json"]
    tok_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError("tokenizer.json introuvable : passe --tokenizer chemin/vers/tokenizer.json")
    return model, MiniTokenizer(tok_path), ckpt


# ══════════════════════════════════════════════════════════════════════════════
#  Échantillonnage
# ══════════════════════════════════════════════════════════════════════════════
def _banned_by_ngram(generated: List[int], n: int) -> Set[int]:
    """Tokens qui compléteraient un n-gramme DÉJÀ présent dans le texte généré."""
    if n < 2 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


def sample_token(logits: torch.Tensor, generated: List[int], *, temperature: float, top_k: int, top_p: float,
                 min_p: float, repetition_penalty: float, no_repeat_ngram: int, forbidden: Sequence[int],
                 valid_vocab: int, rng: Optional[torch.Generator]) -> int:
    logits = logits.float().clone()
    if valid_vocab < logits.numel():
        logits[valid_vocab:] = -float("inf")                    # lignes de padding de l'embedding
    if forbidden:
        logits[list(forbidden)] = -float("inf")
    if repetition_penalty != 1.0 and generated:
        ids = torch.tensor(sorted(set(generated[-256:])), device=logits.device)
        score = logits[ids]
        logits[ids] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    if no_repeat_ngram >= 2:
        banned = _banned_by_ngram(generated, no_repeat_ngram)
        if banned:
            logits[list(banned)] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if min_p > 0:
        probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[(cum - sp) > top_p * cum[-1]] = 0.0                  # garde le plus petit ensemble dont la masse >= top_p
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=rng))


@torch.inference_mode()
def stream_tokens(model: MiniLLM, prompt_ids: List[int], *, max_new_tokens: int = 128, temperature: float = 0.7,
                  top_k: int = 40, top_p: float = 0.9, min_p: float = 0.0, repetition_penalty: float = 1.1,
                  no_repeat_ngram: int = 0, stop_ids: Set[int] = frozenset(), forbidden: Sequence[int] = (),
                  valid_vocab: Optional[int] = None, seed: Optional[int] = None) -> Iterator[int]:
    """Génère token par token (KV-cache). S'arrête sur un stop_id (qui n'est PAS renvoyé)."""
    cfg = model.cfg
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(prompt_ids) >= cfg.max_seq_len:
        prompt_ids = prompt_ids[-(cfg.max_seq_len - 1):]
    budget = min(max_new_tokens, cfg.max_seq_len - len(prompt_ids))
    rng = None
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    valid_vocab = valid_vocab or cfg.vocab_size
    cache = KVCache(cfg, 1, cfg.max_seq_len, device, dtype)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(idx, cache=cache, start_pos=0, last_only=True)
    pos = len(prompt_ids)
    generated: List[int] = []
    for _ in range(budget):
        tok = sample_token(logits[0, -1], generated,
                           temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                           repetition_penalty=repetition_penalty, no_repeat_ngram=no_repeat_ngram,
                           forbidden=forbidden, valid_vocab=valid_vocab, rng=rng)
        if tok in stop_ids:
            return
        generated.append(tok)
        yield tok
        if pos >= cfg.max_seq_len:
            return
        logits, _ = model(torch.tensor([[tok]], dtype=torch.long, device=device), cache=cache, start_pos=pos, last_only=True)
        pos += 1


class _Printer:
    """Affichage incrémental : attend qu'un caractère UTF-8 soit complet avant de l'écrire."""

    def __init__(self, tok: MiniTokenizer, write=sys.stdout.write):
        self.tok, self.ids, self.printed, self.write = tok, [], "", write

    def push(self, tid: int):
        self.ids.append(tid)
        text = self.tok.decode(self.ids, skip_special=True)
        if text.endswith("\ufffd"):
            return
        self.write(text[len(self.printed):])
        sys.stdout.flush()
        self.printed = text

    def text(self) -> str:
        return self.tok.decode(self.ids, skip_special=True)


def generate_text(model, tok: MiniTokenizer, prompt: str, **kw) -> str:
    """Complétion de texte brut (modèle de base)."""
    ids = tok.encode(prompt)
    out = list(stream_tokens(model, ids, stop_ids={tok.eot_id}, valid_vocab=tok.vocab_size, **kw))
    return tok.decode(ids + out, skip_special=True)


def chat_reply(model, tok: MiniTokenizer, messages: List[dict], *, max_new_tokens: int = 128, stream: bool = False,
               **kw) -> str:
    """Répond au dernier message utilisateur. Tronque l'historique (tours les plus anciens d'abord) si besoin."""
    msgs = list(messages)
    max_ctx = model.cfg.max_seq_len
    max_new_tokens = min(max_new_tokens, max_ctx // 2)
    while True:
        ids, _ = tok.encode_chat(msgs, add_generation_prompt=True)
        if len(ids) + max_new_tokens <= max_ctx or len(msgs) <= 1:
            break
        system = [msgs[0]] if msgs[0]["role"] == "system" else []
        rest = msgs[len(system):][1:]                            # oublie le tour le plus ancien
        while rest and rest[0]["role"] == "assistant":           # l'historique doit commencer par l'utilisateur
            rest = rest[1:]
        msgs = system + rest
    if len(ids) + 16 > max_ctx:                                  # un seul message trop long : on garde la fin
        ids = ids[-(max_ctx - max_new_tokens):]
    printer = _Printer(tok) if stream else None
    out: List[int] = []
    for t in stream_tokens(model, ids, max_new_tokens=max_new_tokens, stop_ids=tok.stop_ids(),
                           forbidden=tok.forbidden_in_answer(), valid_vocab=tok.vocab_size, **kw):
        out.append(t)
        if printer:
            printer.push(t)
    return tok.decode(out, skip_special=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="MiniLLM v2 — génération / chat")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--mode", choices=["chat", "complete"], default="chat")
    p.add_argument("--prompt", default=None, help="un seul prompt (sinon boucle interactive)")
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--rep_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    a = p.parse_args()

    model, tok, ckpt = load_model(a.ckpt, a.tokenizer, a.device, a.dtype)
    print(f"Modèle {model.cfg.size_name()} | iter {ckpt.get('iter')} | val_loss {ckpt.get('val_loss')} | "
          f"device {next(model.parameters()).device}")
    sampling = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, min_p=a.min_p,
                    repetition_penalty=a.rep_penalty, no_repeat_ngram=a.no_repeat_ngram, seed=a.seed)

    if a.mode == "complete":
        prompts = [a.prompt] if a.prompt else None
        while True:
            text = prompts[0] if prompts else input("\nTexte de départ > ").strip()
            if not text:
                break
            t0 = time.perf_counter()
            out = generate_text(model, tok, text, max_new_tokens=a.max_new, **sampling)
            print(out, f"\n[{time.perf_counter() - t0:.1f}s]")
            if prompts:
                break
        return

    history: List[dict] = []
    if a.prompt:
        print(chat_reply(model, tok, [{"role": "user", "content": a.prompt}], max_new_tokens=a.max_new, **sampling))
        return
    print("Chat — /reset pour oublier l'historique, /quit pour sortir.\n")
    while True:
        try:
            user = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = []
            print("(historique effacé)")
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        print("MiniLLM > ", end="", flush=True)
        reply = chat_reply(model, tok, history, max_new_tokens=a.max_new, stream=True, **sampling)
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

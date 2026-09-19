# generate.py — MiniLLM v2
#
# Génération de texte depuis un modèle entraîné.
#
# Usage :
#   python generate.py                                     # prompt interactif
#   python generate.py --prompt "Il était une fois"
#   python generate.py --checkpoint checkpoints/best.pt --prompt "Bonjour"
#   python generate.py --prompt "..." --temp 0.7 --top_k 40 --max_new 300

import argparse
import os

import torch
import tiktoken

from config import ModelConfig
from model import MiniLLM

DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile, précis et concis."
DEFAULT_END_TAG = "<|end|>"


def trim_stop_tag(text: str, stop_tag: str = DEFAULT_END_TAG) -> str:
    idx = text.find(stop_tag)
    if idx >= 0:
        return text[:idx].strip()
    return text.strip()


def build_prompt(question: str, context: str | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    parts = ["<|system|>", system_prompt.strip()]
    if context and context.strip():
        parts.extend(["<|context|>", context.strip()])
    parts.extend(["<|user|>", question.strip(), "<|assistant|>"])
    return "\n".join(parts) + "\n"


def load_model(checkpoint_path: str, device: str = "auto") -> tuple[MiniLLM, ModelConfig, str]:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    assert os.path.exists(checkpoint_path), (
        f"Checkpoint introuvable : {checkpoint_path}\n"
        f"Lance d'abord l'entraînement : python train.py"
    )

    print(f"Chargement de {checkpoint_path} sur {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt["config"]
    model = MiniLLM(cfg).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    val_info = f"val_loss={ckpt['val_loss']:.4f}" if ckpt.get("val_loss") is not None else ""
    print(f"  Modèle chargé : {cfg} | iter={ckpt.get('iter', '?')} {val_info}")
    return model, cfg, device


@torch.no_grad()
def generate(
    model: MiniLLM,
    prompt_tokens: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
    stop_token: int | None = None,
) -> list[int]:
    model.eval()
    generated = list(prompt_tokens)

    for _ in range(max_new_tokens):
        ctx = generated[-model.cfg.max_seq_len:]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(idx)
        logits = logits[0, -1, :]

        logits = logits / max(temperature, 1e-8)

        if top_k > 0 and top_k < logits.size(-1):
            kth_val = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_t)

        if stop_token is not None and next_t == stop_token:
            break

    return generated


def chat(
    checkpoint_path: str = "checkpoints/best.pt",
    device: str = "auto",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    model, cfg, device = load_model(checkpoint_path, device)
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"  Tokenizer : cl100k_base | vocab={enc.n_vocab:,}")
    print(f"\n{'═'*60}")
    print("  MiniLLM v2 — Mode génération")
    print("  Commandes : 'quit' pour quitter, 'reset' pour effacer le contexte")
    print(f"{'═'*60}\n")

    history: list[str] = []
    system_prompt = DEFAULT_SYSTEM_PROMPT

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "quitter"):
            print("Au revoir !")
            break
        if user_input.lower() == "reset":
            history = []
            print("(contexte effacé)\n")
            continue

        history.append(user_input)
        prompt_text = build_prompt(user_input, context=None, system_prompt=system_prompt)
        prompt_tokens = enc.encode(prompt_text)

        all_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            stop_token=enc.encode(DEFAULT_END_TAG)[0] if enc.encode(DEFAULT_END_TAG) else None,
        )

        text = enc.decode(all_tokens[len(prompt_tokens):])
        text = trim_stop_tag(text)
        print(f"\n{text}\n")


def generate_from_prompt(
    prompt: str,
    checkpoint_path: str = "checkpoints/best.pt",
    device: str = "auto",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    n_samples: int = 1,
) -> list[str]:
    model, cfg, device = load_model(checkpoint_path, device)
    enc = tiktoken.get_encoding("cl100k_base")
    prompt_text = build_prompt(prompt, context=None)
    tokens = enc.encode(prompt_text)
    results = []

    for i in range(n_samples):
        out_tokens = generate(
            model=model,
            prompt_tokens=tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            stop_token=enc.encode(DEFAULT_END_TAG)[0] if enc.encode(DEFAULT_END_TAG) else None,
        )
        text = enc.decode(out_tokens[len(tokens):])
        text = trim_stop_tag(text)
        results.append(text)
        if n_samples > 1:
            print(f"── Sample {i+1} ──────────────────────────")
            print(text)
            print()

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MiniLLM v2 — Génération de texte")
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="Chemin vers le checkpoint")
    p.add_argument("--prompt", default=None, help="Prompt de départ (si absent : mode interactif)")
    p.add_argument("--max_new", type=int, default=200, help="Nombre max de tokens à générer")
    p.add_argument("--temp", type=float, default=0.8, help="Temperature (0.1=conservateur, 1.5=créatif)")
    p.add_argument("--top_k", type=int, default=50, help="Top-K sampling (0 = désactivé)")
    p.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling (1.0 = désactivé)")
    p.add_argument("--samples", type=int, default=1, help="Nombre de générations (si --prompt fourni)")
    p.add_argument("--device", default="auto", help="Appareil : auto | cuda | mps | cpu")
    args = p.parse_args()

    if args.prompt:
        generate_from_prompt(
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            device=args.device,
            max_new_tokens=args.max_new,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            n_samples=args.samples,
        )
    else:
        chat(
            checkpoint_path=args.checkpoint,
            device=args.device,
            max_new_tokens=args.max_new,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
        )

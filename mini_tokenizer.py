"""
mini_tokenizer.py — tokenizer BPE (byte-level) entraîné sur TON corpus français + format de chat.

Pourquoi ne plus utiliser cl100k_base (100 277 tokens) :
  * avec d_model=512 les embeddings représentaient 62 % des paramètres et ~62 % du calcul du forward ;
  * beaucoup de tokens (code, autres langues) n'étaient presque jamais vus -> lignes d'embedding sous-entraînées.
Un vocab de 32 000 tokens entraîné sur du français est plus compact ET ~1,7x plus rapide à entraîner.
Les ids tiennent dans un uint16 (< 65 536) -> fichiers .bin 2x plus petits.

Format de chat (loss uniquement sur la réponse de l'assistant) :
    <|user|>Question<|end|><|assistant|>Réponse<|end|>
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, List, Sequence, Tuple

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from config import round_up

EOT = "<|endoftext|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
SPECIAL_TOKENS = [EOT, SYSTEM, USER, ASSISTANT, END]

# Découpage façon GPT-4 mais : chiffres isolés (arithmétique/dates) et apostrophes françaises (l'homme, aujourd'hui) collées au mot.
_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def train_tokenizer(text_iterator: Iterable[str], vocab_size: int = 32_000, save_path: str = "data/tokenizer.json",
                    min_frequency: int = 2, show_progress: bool = True) -> "MiniTokenizer":
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(_SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    tok.train_from_iterator(text_iterator, trainer=trainer)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tok.save(save_path)
    return MiniTokenizer(save_path)


class MiniTokenizer:
    def __init__(self, path: str):
        self.path = path
        self.tok = Tokenizer.from_file(path)
        self.eot_id = self.tok.token_to_id(EOT)
        self.system_id = self.tok.token_to_id(SYSTEM)
        self.user_id = self.tok.token_to_id(USER)
        self.assistant_id = self.tok.token_to_id(ASSISTANT)
        self.end_id = self.tok.token_to_id(END)
        if None in (self.eot_id, self.system_id, self.user_id, self.assistant_id, self.end_id):
            raise ValueError(f"{path} ne contient pas les tokens spéciaux attendus {SPECIAL_TOKENS}")
        self.pad_id = self.eot_id
        with open(path, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()[:16]     # empreinte : détecte un tokenizer reconstruit ≠ celui du checkpoint

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def padded_vocab_size(self) -> int:
        """Taille de la matrice d'embedding (multiple de 64 : meilleurs kernels GPU)."""
        return round_up(self.vocab_size, 64)

    @staticmethod
    def sanitize(text: str) -> str:
        """Empêche un texte de contenir les tokens spéciaux (injection de <|end|> dans un document)."""
        return text.replace("<|", "< |")

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(self.sanitize(text), add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch([self.sanitize(t) for t in texts], add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    # ── chat ────────────────────────────────────────────────────────────
    def encode_chat(self, messages: Sequence[dict], add_generation_prompt: bool = False) -> Tuple[List[int], List[int]]:
        """
        messages : [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Renvoie (ids, mask) ; mask=1 sur les tokens sur lesquels on calcule la loss
        (contenu de l'assistant + son <|end|>), 0 ailleurs.
        """
        role_ids = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        ids: List[int] = []
        mask: List[int] = []
        for m in messages:
            train = 1 if m["role"] == "assistant" else 0
            ids.append(role_ids[m["role"]])
            mask.append(0)
            content_ids = self.encode(m["content"].strip())
            ids += content_ids
            mask += [train] * len(content_ids)
            ids.append(self.end_id)
            mask.append(train)
        if add_generation_prompt:
            ids.append(self.assistant_id)
            mask.append(0)
        return ids, mask

    def stop_ids(self) -> set:
        return {self.end_id, self.eot_id}

    def forbidden_in_answer(self) -> List[int]:
        return [self.system_id, self.user_id, self.assistant_id]

    def info(self) -> dict:
        return {"vocab_size": self.vocab_size, "padded_vocab_size": self.padded_vocab_size,
                "eot_id": self.eot_id, "end_id": self.end_id}


def save_meta(path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

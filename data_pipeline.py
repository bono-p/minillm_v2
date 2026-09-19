"""Reusable MiniLLM v3 data utilities.

This module keeps general pre-training text and supervised QA separate:
- ``pretrain`` emits ordinary next-token documents.
- ``sft`` emits ``input_ids`` and labels with ``-1`` outside the assistant answer.

PIAF/FQuAD/SQuAD-like JSON can be consumed without downloading anything. The
caller remains responsible for checking each dataset's licence and provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

END_TAG = "<|end|>"
DEFAULT_SYSTEM = "Tu es un assistant utile, précis et concis."
IGNORE_INDEX = -1


@dataclass(frozen=True)
class QAExample:
    context: str
    question: str
    answer: str
    source: str = "unknown"
    group: str = "unknown"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_answer(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("text"))
    if isinstance(value, list) and value:
        return _first_answer(value[0])
    return _text(value)


def iter_squad_examples(payload: dict[str, Any], source: str = "qa") -> Iterable[QAExample]:
    """Yield PIAF/FQuAD/SQuAD-shaped examples, grouped by article/title."""
    articles = payload.get("data", []) if isinstance(payload, dict) else payload
    for article_index, article in enumerate(articles or []):
        article_group = _text(article.get("title")) or f"{source}:article:{article_index}"
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = _text(paragraph.get("context"))
            group = f"{article_group}:paragraph:{paragraph_index}"
            for qa in paragraph.get("qas", []):
                question = _text(qa.get("question"))
                answer = _first_answer(qa.get("answers"))
                if context and question and answer:
                    yield QAExample(context, question, answer, source, group)


def normalise_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(example: QAExample, system: str = DEFAULT_SYSTEM) -> str:
    return "\n".join(
        [
            "<|system|>", system,
            "<|context|>", example.context,
            "<|user|>", example.question,
            "<|assistant|>",
        ]
    ) + "\n"


def encode_sft_example(encoder, example: QAExample, system: str = DEFAULT_SYSTEM):
    """Return input ids and labels; only assistant answer + END_TAG are trained."""
    prompt_ids = encoder.encode(build_prompt(example, system), allowed_special=())
    answer_ids = encoder.encode(" " + example.answer + END_TAG, allowed_special=())
    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return input_ids, labels


def stable_split(examples: list[QAExample], val_ratio: float = 0.1):
    """Split by source group to prevent context leakage between train/validation."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio doit être compris entre 0 et 1")
    groups = sorted({example.group for example in examples})
    val_groups = {
        group for group in groups
        if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_ratio
    }
    if not val_groups and groups:
        val_groups = {groups[-1]}
    train = [e for e in examples if e.group not in val_groups]
    valid = [e for e in examples if e.group in val_groups]
    return train, valid


def audit_examples(examples: Iterable[QAExample]) -> dict[str, int]:
    report = {"total": 0, "empty": 0, "too_short": 0, "suspicious": 0, "duplicates": 0}
    seen = set()
    suspicious = re.compile(r"\b(né le|du)\s+(à|\.)|\s{2,}|\s[,.;:]")
    for example in examples:
        report["total"] += 1
        key = (example.context, example.question, example.answer)
        if key in seen:
            report["duplicates"] += 1
        seen.add(key)
        joined = " ".join((example.context, example.question, example.answer))
        if not all((example.context, example.question, example.answer)):
            report["empty"] += 1
        if len(example.answer) < 2:
            report["too_short"] += 1
        if suspicious.search(joined):
            report["suspicious"] += 1
    return report


def load_json_examples(path: str, source: str = "qa") -> list[QAExample]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    examples = list(iter_squad_examples(payload, source))
    if not examples:
        raise ValueError(f"Aucun exemple QA compatible trouvé dans {path}")
    return examples

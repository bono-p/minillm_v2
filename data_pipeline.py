"""QA parsing and masked-label formatting for MiniLLM v3."""
from __future__ import annotations
import hashlib, json, re
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

def _answer(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("text"))
    if isinstance(value, list):
        return _answer(value[0]) if value else ""
    return _text(value)

def iter_squad_examples(payload: Any, source: str = "qa") -> Iterable[QAExample]:
    """Accept SQuAD/PIAF JSON and the flat {context, question, answer} export."""
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            if isinstance(item, dict) and _text(item.get("context")) and _text(item.get("question")) and _answer(item.get("answer")):
                yield QAExample(_text(item["context"]), _text(item["question"]), _answer(item["answer"]), source, f"{source}:item:{i}")
        return
    articles = payload.get("data", []) if isinstance(payload, dict) else []
    for ai, article in enumerate(articles):
        title = _text(article.get("title")) or f"{source}:article:{ai}"
        for pi, paragraph in enumerate(article.get("paragraphs", [])):
            context = _text(paragraph.get("context")); group = f"{title}:paragraph:{pi}"
            for qa in paragraph.get("qas", []):
                question, answer = _text(qa.get("question")), _answer(qa.get("answers"))
                if context and question and answer:
                    yield QAExample(context, question, answer, source, group)

def build_prompt(example: QAExample, system: str = DEFAULT_SYSTEM) -> str:
    return "\n".join(["<|system|>", system, "<|context|>", example.context, "<|user|>", example.question, "<|assistant|>"]) + "\n"

def encode_sft_example(encoder, example: QAExample, system: str = DEFAULT_SYSTEM):
    prompt = encoder.encode(build_prompt(example, system), allowed_special=())
    answer = encoder.encode(" " + example.answer + END_TAG, allowed_special=())
    return prompt + answer, [IGNORE_INDEX] * len(prompt) + answer

def stable_split(examples: list[QAExample], val_ratio: float = .1):
    groups = sorted({x.group for x in examples})
    if len(groups) < 2: return examples, []
    n_val = max(1, round(len(groups) * val_ratio))
    ranked = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    val_groups = set(ranked[:n_val])
    return [x for x in examples if x.group not in val_groups], [x for x in examples if x.group in val_groups]

def load_json_examples(path: str, source: str = "qa") -> list[QAExample]:
    with open(path, encoding="utf-8") as f: payload = json.load(f)
    result = list(iter_squad_examples(payload, source))
    if not result: raise ValueError(f"Aucun exemple QA valide dans {path}")
    return result

def audit_examples(examples):
    seen=set(); report={"total":0,"duplicates":0,"empty":0,"too_short":0}
    for x in examples:
        report["total"] += 1; key=(x.context,x.question,x.answer)
        report["duplicates"] += key in seen; seen.add(key)
        report["empty"] += not all((x.context,x.question,x.answer)); report["too_short"] += len(x.answer)<2
    return report

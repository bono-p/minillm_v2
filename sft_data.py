"""SFT dataset and collator for PIAF/FQuAD/SQuAD JSON files."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from data_pipeline import QAExample, encode_sft_example, iter_squad_examples, stable_split


@dataclass
class SFTRecord:
    input_ids: list[int]
    labels: list[int]
    group: str
    source: str


class QASFTDataset(Dataset):
    def __init__(self, paths: list[str], encoder, max_length: int, split: str = "train", val_ratio: float = 0.1):
        examples: list[QAExample] = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                payload: Any = json.load(handle)
            examples.extend(iter_squad_examples(payload, source=path))
        train, valid = stable_split(examples, val_ratio) if len(examples) > 1 else (examples, [])
        selected = train if split == "train" else valid
        self.records: list[SFTRecord] = []
        for example in selected:
            ids, labels = encode_sft_example(encoder, example)
            # Keep the answer whenever possible; truncate only the left context.
            if len(ids) > max_length:
                ids = ids[-max_length:]
                labels = labels[-max_length:]
            if any(label != -1 for label in labels):
                self.records.append(SFTRecord(ids, labels, example.group, example.source))
        if not self.records:
            raise ValueError(f"Aucun exemple SFT exploitable pour split={split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return torch.tensor(record.input_ids, dtype=torch.long), torch.tensor(record.labels, dtype=torch.long)


def sft_collate(batch, pad_id: int = 0):
    max_len = max(item[0].numel() for item in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -1, dtype=torch.long)
    for row, (ids, target) in enumerate(batch):
        inputs[row, :ids.numel()] = ids
        labels[row, :target.numel()] = target
    return inputs, labels

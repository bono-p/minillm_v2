"""SFT Dataset with compatible PIAF/FQuAD parsing and masked labels."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from data_pipeline import QAExample, encode_sft_example, load_json_examples, stable_split

class QASFTDataset(Dataset):
    def __init__(self, paths, encoder, max_length, split="train", val_ratio=.1):
        examples=[]
        for path in paths: examples.extend(load_json_examples(path, path))
        train, valid = stable_split(examples, val_ratio)
        selected = train if split == "train" else valid
        self.records=[]
        for ex in selected:
            ids, labels = encode_sft_example(encoder, ex)
            # Preserve the assistant answer: drop only prompt tokens when needed.
            if len(ids)>max_length:
                first_answer=next((i for i,v in enumerate(labels) if v != -1), len(ids)-1)
                start=max(0, min(first_answer, len(ids)-max_length))
                ids, labels=ids[start:start+max_length], labels[start:start+max_length]
            if any(v != -1 for v in labels): self.records.append((ids,labels,ex))
        if not self.records: raise ValueError(f"Aucun exemple SFT pour split={split}")
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        ids, labels, _ = self.records[i]
        return torch.tensor(ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long)

def sft_collate(batch, pad_id=0):
    n=max(x.numel() for x,_ in batch); inputs=torch.full((len(batch),n),pad_id,dtype=torch.long); labels=torch.full((len(batch),n),-1,dtype=torch.long)
    for i,(x,y) in enumerate(batch): inputs[i,:len(x)]=x; labels[i,:len(y)]=y
    return inputs, labels

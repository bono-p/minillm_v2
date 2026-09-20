"""
data.py — chargement des données (pré-entraînement et SFT).

Corrige à l'audit :
  * VAL FIXE : toujours les mêmes séquences à chaque évaluation -> val_loss comparable d'un eval à l'autre
    (avant : le loader avançait dans le fichier, chaque eval mesurait une autre tranche de 1/16 du val).
  * L'ordre des données est une FONCTION PURE de (seed, compteur de micro-batchs) : une reprise continue
    exactement là où le run s'est arrêté (avant : le loader repartait à 0 et relisait toujours le début).
  * Échantillonnage SANS remise par époque (permutation de blocs + décalage aléatoire par époque).
  * SFT : padding à droite + labels = -1 sur le prompt et le padding (la loss ne porte que sur les réponses),
    batchs regroupés par longueur pour limiter le padding.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
#  Pré-entraînement : fichiers binaires plats (train.bin / val.bin / meta.json)
# ══════════════════════════════════════════════════════════════════════════════
class PretrainData:
    def __init__(self, data_dir: str, seq_len: int, batch_size: int, seed: int = 1337,
                 rank: int = 0, world: int = 1):
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            self.meta: Dict = json.load(f)
        self.dtype = np.dtype(self.meta["dtype"])
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=self.dtype, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=self.dtype, mode="r")
        self.T, self.B, self.seed, self.rank, self.world = seq_len, batch_size, seed, rank, world
        self.n_blocks = len(self.train) // seq_len - 1
        if self.n_blocks < 1:
            raise ValueError(f"train.bin trop petit ({len(self.train)} tokens) pour seq_len={seq_len}")
        if len(self.val) < seq_len + 1:
            raise ValueError("val.bin trop petit : relance prepare_data avec un val_permille plus grand")
        self._perm_cache: Dict[int, Tuple[np.ndarray, int]] = {}

    @property
    def vocab_size(self) -> int:
        return int(self.meta["padded_vocab_size"])

    def _epoch(self, ep: int) -> Tuple[np.ndarray, int]:
        if ep not in self._perm_cache:
            rng = np.random.default_rng([self.seed, ep])
            perm = rng.permutation(self.n_blocks)
            offset = int(rng.integers(0, self.T))            # décale l'alignement des blocs à chaque époque
            if len(self._perm_cache) >= 3:
                self._perm_cache.pop(next(iter(self._perm_cache)))
            self._perm_cache[ep] = (perm, offset)
        return self._perm_cache[ep]

    def epoch_of(self, g: int) -> float:
        """Époque (fractionnaire) atteinte après g micro-batchs par rang."""
        return g * self.B * self.world / self.n_blocks

    def get_batch(self, g: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Micro-batch numéro g pour CE rang. Pur : mêmes (seed, g, rank) -> mêmes données."""
        base = (g * self.world + self.rank) * self.B
        rows = []
        for j in range(self.B):
            ep, pos = divmod(base + j, self.n_blocks)
            perm, offset = self._epoch(ep)
            s = offset + int(perm[pos]) * self.T
            rows.append(self.train[s:s + self.T + 1])
        arr = np.stack(rows).astype(np.int64)
        return torch.from_numpy(arr[:, :-1]), torch.from_numpy(arr[:, 1:])

    def val_batches(self, n_batches: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Séquences de val FIXES (espacées régulièrement dans val.bin), réparties entre les rangs."""
        n_val_blocks = len(self.val) // self.T - 1
        need = max(1, n_batches) * self.B * self.world
        idx = np.unique(np.linspace(0, max(0, n_val_blocks - 1), num=min(need, n_val_blocks)).astype(np.int64))
        mine = idx[self.rank::self.world]
        for i in range(0, len(mine), self.B):
            rows = [self.val[int(b) * self.T: int(b) * self.T + self.T + 1] for b in mine[i:i + self.B]]
            arr = np.stack(rows).astype(np.int64)
            yield torch.from_numpy(arr[:, :-1]), torch.from_numpy(arr[:, 1:])


# ══════════════════════════════════════════════════════════════════════════════
#  SFT : exemples (tokens + masque) stockés à plat + offsets
# ══════════════════════════════════════════════════════════════════════════════
class SFTData:
    def __init__(self, data_dir: str, split: str, batch_size: int, seed: int = 1337,
                 rank: int = 0, world: int = 1, bucket_mult: int = 32, pad_id: int = 0):
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.tokens = np.load(os.path.join(data_dir, f"{split}.tokens.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(data_dir, f"{split}.mask.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(data_dir, f"{split}.offsets.npy"))
        self.lengths = np.diff(self.offsets)
        self.n = len(self.lengths)
        self.B, self.seed, self.rank, self.world = batch_size, seed, rank, world
        self.bucket_mult, self.pad_id = bucket_mult, pad_id
        self._cache: Dict[int, List[np.ndarray]] = {}
        if split == "train":
            per_rank = self.n // world
            if per_rank < batch_size:
                raise ValueError(f"Trop peu d'exemples SFT ({self.n}) pour batch_size={batch_size} x {world} GPU")

    @property
    def vocab_size(self) -> int:
        return int(self.meta["padded_vocab_size"])

    # ── batchs d'entraînement ───────────────────────────────────────────
    def _epoch_batches(self, ep: int) -> List[np.ndarray]:
        if ep not in self._cache:
            rng = np.random.default_rng([self.seed, ep])
            perm = rng.permutation(self.n)
            per_rank = (self.n // self.world // self.B) * self.B          # tronque : même nb de batchs sur chaque rang
            perm = perm[: per_rank * self.world][self.rank::self.world]
            mega = self.B * self.bucket_mult
            batches: List[np.ndarray] = []
            for i in range(0, len(perm), mega):
                chunk = perm[i:i + mega]
                chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]     # tri par longueur dans le chunk
                for j in range(0, len(chunk), self.B):
                    batches.append(chunk[j:j + self.B])
            order = rng.permutation(len(batches))
            self._cache = {ep: [batches[k] for k in order]}                     # 1 seule époque en cache
        return self._cache[ep]

    def batches_per_epoch(self) -> int:
        return self.n // self.world // self.B            # batchs complets uniquement (identique sur chaque rang)

    def get_batch(self, g: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = self.batches_per_epoch()
        ep, i = divmod(g, nb)
        return self._collate(self._epoch_batches(ep)[i])

    def epoch_of(self, g: int) -> float:
        return g / self.batches_per_epoch()

    # ── évaluation (ordre fixe) ─────────────────────────────────────────
    def eval_batches(self, max_batches: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        order = np.argsort(self.lengths, kind="stable")
        batches = [order[i:i + self.B] for i in range(0, len(order), self.B)]
        batches = batches[self.rank::self.world]
        if max_batches > 0:
            batches = batches[:max_batches]
        for b in batches:
            yield self._collate(b)

    # ── assemblage d'un batch ───────────────────────────────────────────
    def _collate(self, idxs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        L = int(max(self.lengths[i] for i in idxs)) - 1
        x = np.full((len(idxs), L), self.pad_id, dtype=np.int64)
        y = np.full((len(idxs), L), -1, dtype=np.int64)
        for r, i in enumerate(idxs):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            ids = np.asarray(self.tokens[s:e], dtype=np.int64)
            m = np.asarray(self.mask[s:e])
            n = len(ids) - 1
            x[r, :n] = ids[:-1]
            y[r, :n] = np.where(m[1:] > 0, ids[1:], -1)
        return torch.from_numpy(x), torch.from_numpy(y)


def write_sft_split(out_dir: str, split: str, examples: List[Tuple[List[int], List[int]]], token_dtype) -> dict:
    """examples : liste de (ids, mask). Écrit {split}.tokens.npy / .mask.npy / .offsets.npy."""
    lengths = np.array([len(ids) for ids, _ in examples], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens = np.empty(int(offsets[-1]), dtype=token_dtype)
    mask = np.empty(int(offsets[-1]), dtype=np.uint8)
    for k, (ids, m) in enumerate(examples):
        tokens[offsets[k]:offsets[k + 1]] = ids
        mask[offsets[k]:offsets[k + 1]] = m
    np.save(os.path.join(out_dir, f"{split}.tokens.npy"), tokens)
    np.save(os.path.join(out_dir, f"{split}.mask.npy"), mask)
    np.save(os.path.join(out_dir, f"{split}.offsets.npy"), offsets)
    return {"n_examples": len(examples), "n_tokens": int(offsets[-1]), "n_answer_tokens": int(mask.sum())}

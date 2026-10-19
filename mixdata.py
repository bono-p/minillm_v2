"""
mixdata.py — moteur du mélange v2.3 : filtres de qualité, décontamination, écriture reprenable des fichiers de tokens.

Sans torch : testable hors GPU. Les accès réseau (Hugging Face) sont dans prepare_mix.py ; ici tout est injecté
(`iter_factory`), ce qui permet de tester les quotas, la reprise après interruption et les répétitions avec des sources factices.

Garanties :
  * un fichier de tokens est TOUJOURS cohérent avec state.json : à la reprise, il est tronqué à la taille enregistrée ;
  * reprise = résultat identique à un run sans interruption (testé) ;
  * un document de validation (hash du texte) n'est jamais écrit en entraînement ;
  * les conversations qui contiennent une question de nos jeux d'évaluation sont écartées (décontamination).
"""
from __future__ import annotations

import json
import os
import re
import zlib
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

Doc = object            # str (texte) ou liste de messages [{"role", "content"}] (conversation)

FRENCH_WORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "en", "que", "qui", "dans", "pour", "pas", "sur",
                "au", "ce", "il", "elle", "se", "par", "plus", "avec", "son", "sa", "ses", "mais", "ou", "nous", "vous", "sont"}


# ── nettoyage et qualité ──────────────────────────────────────────────────────────────────────────────────────────────────────
def clean_ocr(text: str, kind: str = "livre") -> str:
    """Recolle les mots coupés en fin de ligne, retire les numéros de page isolés, normalise les espaces."""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                       # mots coupés en fin de ligne
    t = re.sub(r"(?m)^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", "", t)   # numéros de page seuls sur une ligne
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)                       # retours à la ligne simples -> espace (le texte reste en paragraphes)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def quality_ok(text: str, kind: str = "web", min_chars: int = 200) -> bool:
    """Filtre grossier mais efficace : assez de lettres, assez de mots français courants, pas de répétition folle."""
    if len(text) < min_chars:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < (0.70 if kind in ("livre", "presse") else 0.60):
        return False
    words = re.findall(r"[a-zà-ÿœ']+", text.lower())
    if len(words) < 30:
        return False
    fr = sum(w in FRENCH_WORDS for w in words) / len(words)
    if fr < (0.12 if kind in ("livre", "presse") else 0.08):       # trop peu de mots français courants : autre langue ou bruit OCR
        return False
    if len(set(words)) / len(words) < 0.15:                      # texte en boucle
        return False
    avg = sum(map(len, words)) / len(words)
    return 3.0 <= avg <= 9.0


_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "system": "system"}


def to_messages(raw) -> Optional[List[dict]]:
    """Conversation ShareGPT ([{from, value}]) ou OpenAI ([{role, content}]) -> messages ; None si inexploitable."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out: List[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = _ROLE.get(str(m.get("from", m.get("role", ""))).lower())
        content = m.get("value", m.get("content", m.get("text")))
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content.strip()})
    if not any(m["role"] == "user" for m in out) or not any(m["role"] == "assistant" for m in out):
        return None
    return out


def chat_ok(messages: Sequence[dict]) -> bool:
    """Écarte refus génériques, boucles et échos (mêmes filtres que le SFT), et les réponses minuscules."""
    from sft_filters import problem
    question = ""
    for m in messages:
        if m["role"] == "user":
            question = m["content"]
        elif m["role"] == "assistant":
            if len(m["content"]) < 3:
                return False
            if problem({"messages": [{"role": "user", "content": question}, m]}):
                return False
    return True


# ── décontamination ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().replace("’", "'")).strip(" ?!.")


class Decontaminator:
    """Écarte les conversations dont un message utilisateur est exactement une question de nos jeux d'évaluation.
    Limite assumée : correspondance exacte (après normalisation) sur les tours utilisateur ; elle ne détecte pas les reformulations,
    et ne s'applique pas aux livres / pages web (où une question isolée a peu de chances d'apparaître telle quelle)."""

    def __init__(self, questions: Iterable[str]):
        self.q = {_norm(q) for q in questions if q and len(q.strip()) > 8}

    def __len__(self) -> int:
        return len(self.q)

    def contaminated(self, messages: Sequence[dict]) -> bool:
        return any(m["role"] == "user" and _norm(m["content"]) in self.q for m in messages)


def eval_questions(root: str = ".") -> List[str]:
    """Toutes les questions de nos évaluations : 337 ouvertes, 204 de connaissances, 36 de base, problèmes chiffrés, persona."""
    qs: List[str] = []
    try:
        import release_tools as rt
        qs += rt.open_questions()
    except Exception:                                   # noqa: BLE001
        pass
    from basics_eval import BASICS
    from knowledge_eval import KNOWLEDGE
    qs += [q for _, q in [(0, b[0]) for b in BASICS]] + [q for _, q, _ in KNOWLEDGE]
    try:
        from skills_eval import build_items
        qs += [q for items in build_items(200).values() for q, _ in items]
    except Exception:                                   # noqa: BLE001
        pass
    pj = os.path.join(root, "personnalite.jsonl")
    if os.path.exists(pj):
        for line in open(pj, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if isinstance(d.get("question"), str):
                    qs.append(d["question"])
    return qs


# ── écriture reprenable ──────────────────────────────────────────────────────────────────────────────────────────────────────
def is_val_doc(key_text: str, permille: int) -> bool:
    return (zlib.crc32(key_text[:400].encode("utf-8")) % 1000) < permille


def doc_key(doc: Doc) -> str:
    return doc if isinstance(doc, str) else " ".join(m["content"] for m in doc[:2])


class State:
    """state.json : progression par source et par phase, écrit de façon atomique."""

    def __init__(self, path: str):
        self.path = path
        self.d: Dict = {"sources": {}, "files": {}}
        if os.path.exists(path):
            self.d = json.load(open(path, encoding="utf-8"))

    def src(self, name: str) -> Dict:
        return self.d["sources"].setdefault(name, {"docs_consumed": 0, "epochs": 0, "tokens": {"stable": 0, "decay": 0},
                                                   "val_tokens": 0, "dropped": {}, "repeated_tokens": 0})

    def save(self) -> None:
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


def shard_name(idx: int) -> str:
    return f"shard_{idx:05d}.bin"


class MixWriter:
    """Écrit `<out>/<phase>/shard_NNNNN.bin` (tokens du mélange, en PETITS fichiers plafonnés à `shard_tokens`) et
    `<out>/val_new.bin` (documents de validation des nouvelles sources).

    Pourquoi des shards et pas un seul `train.bin` géant : un fichier de plusieurs Go, réécrit en continu, est le cas le
    plus défavorable pour la synchronisation Google Drive entre deux sessions (vécu le 11-12/10/2026 : `os.fsync` confirme
    l'écriture locale dans la session qui écrit, pas sa disponibilité dans une AUTRE session, et ce délai peut dépasser
    plusieurs heures pour un gros fichier sans cesse modifié). Un shard fermé est un petit fichier IMMUABLE (par défaut
    100 M tokens ≈ 200 Mo) : il ne change plus jamais après sa fermeture, ce qui est le cas le plus favorable pour Drive.
    `train.py` (via `data.ShardedMemmap`) lit directement ces fragments, sans jamais les recoller en un seul fichier.
    """

    def __init__(self, out_dir: str, tok, state: State, dtype=np.uint16, val_permille: int = 3, shard_tokens: int = 100_000_000):
        self.out, self.tok, self.state, self.dtype, self.val_permille = out_dir, tok, state, dtype, val_permille
        self.itemsize = np.dtype(dtype).itemsize
        self.shard_tokens = shard_tokens
        for ph in ("stable", "decay"):
            os.makedirs(os.path.join(out_dir, ph), exist_ok=True)
            self.state.d.setdefault("shards", {}).setdefault(ph, [])     # liste de {"file","tokens"} : shards FERMÉS (immuables)
        self.val_path = os.path.join(out_dir, "val_new.bin")
        self.cur_fh: Dict[str, "object"] = {}
        self.cur_tokens: Dict[str, int] = {}
        for ph in ("stable", "decay"):
            idx = len(self.state.d["shards"][ph])
            path = os.path.join(out_dir, ph, shard_name(idx))
            want = self.state.d["files"].get(f"{ph}_current", 0)
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have < want:
                raise RuntimeError(f"{path} est plus petit ({have} o) que l'état enregistré ({want} o) : fichier corrompu, supprime le dossier de sortie.")
            if have > want:
                with open(path, "r+b") as f:
                    f.truncate(want)
            elif not os.path.exists(path):
                open(path, "wb").close()
            self.cur_fh[ph] = open(path, "ab")
            self.cur_tokens[ph] = want // self.itemsize
        want_val = self.state.d["files"].get("val", 0)
        have_val = os.path.getsize(self.val_path) if os.path.exists(self.val_path) else 0
        if have_val < want_val:
            raise RuntimeError(f"{self.val_path} est plus petit ({have_val} o) que l'état enregistré ({want_val} o) : supprime le dossier de sortie.")
        if have_val > want_val:
            with open(self.val_path, "r+b") as f:
                f.truncate(want_val)
        elif not os.path.exists(self.val_path):
            open(self.val_path, "wb").close()
        self.val_fh = open(self.val_path, "ab")

    def encode(self, doc: Doc) -> List[int]:
        if isinstance(doc, str):
            ids = self.tok.encode(doc)
        else:
            ids, _ = self.tok.encode_chat(doc)
        return ids + [self.tok.eot_id]

    def write(self, key: str, ids: List[int]) -> None:
        if key == "val":
            self.val_fh.write(np.asarray(ids, dtype=self.dtype).tobytes())
            return
        self.write_tokens(key, np.asarray(ids, dtype=self.dtype))

    def write_tokens(self, phase: str, arr) -> None:
        """Comme `write`, mais prend directement un tableau déjà tokenisé (utilisé par `run_existing_bin`)."""
        arr = np.asarray(arr, dtype=self.dtype)
        pos = 0
        while pos < len(arr):
            room = self.shard_tokens - self.cur_tokens[phase]
            take = min(room, len(arr) - pos)
            if take > 0:
                self.cur_fh[phase].write(arr[pos:pos + take].tobytes())
                self.cur_tokens[phase] += take
                pos += take
            if self.cur_tokens[phase] >= self.shard_tokens:
                self._close_shard(phase)

    def _close_shard(self, phase: str) -> None:
        f = self.cur_fh[phase]
        f.flush()
        os.fsync(f.fileno())
        f.close()
        idx = len(self.state.d["shards"][phase])
        self.state.d["shards"][phase].append({"file": shard_name(idx), "tokens": self.cur_tokens[phase]})
        self.state.d["files"][f"{phase}_current"] = 0
        self.state.save()
        new_path = os.path.join(self.out, phase, shard_name(idx + 1))
        self.cur_fh[phase] = open(new_path, "wb")
        self.cur_tokens[phase] = 0

    def checkpoint(self) -> None:
        for ph in ("stable", "decay"):
            f = self.cur_fh[ph]
            f.flush()
            os.fsync(f.fileno())
            self.state.d["files"][f"{ph}_current"] = self.cur_tokens[ph] * self.itemsize
        self.val_fh.flush()
        os.fsync(self.val_fh.fileno())
        self.state.d["files"]["val"] = self.val_fh.tell()
        self.state.save()

    def close(self) -> None:
        self.checkpoint()
        for f in self.cur_fh.values():
            f.close()
        self.val_fh.close()

    def total_tokens(self, phase: str) -> int:
        """Shards fermés + shard courant (partiel) — toujours exact, utilisé par `finalize()`."""
        return sum(s["tokens"] for s in self.state.d["shards"][phase]) + self.cur_tokens[phase]


def run_source(writer: MixWriter, name: str, quotas: Dict[str, int], iter_factory: Callable[[int, int], Iterator[Doc]],
               prepare: Callable[[Doc], Optional[Doc]], max_epochs: float, checkpoint_every: int = 2_000_000,
               log: Callable[[str], None] = print) -> Dict:
    """Remplit les quotas {phase: tokens} d'une source. `iter_factory(skip_docs, epoch)` renvoie le flux de documents à partir du
    n-ième ; `prepare(doc)` filtre/nettoie (None = rejeté). Le flux est relancé (époque suivante) s'il est épuisé avant le quota."""
    st = writer.state.src(name)
    since = 0
    for phase in ("stable", "decay"):
        need = quotas.get(phase, 0) - st["tokens"][phase]
        while need > 0:
            progressed = False
            for doc in iter_factory(st["docs_consumed"], st["epochs"]):
                progressed = True
                st["docs_consumed"] += 1
                doc = prepare(doc)
                if doc is None:
                    st["dropped"]["filtre"] = st["dropped"].get("filtre", 0) + 1
                    continue
                ids = writer.encode(doc)
                if is_val_doc(doc_key(doc), writer.val_permille):
                    if st["epochs"] == 0:                          # la validation n'est écrite qu'une fois, jamais répétée
                        writer.write("val", ids)
                        st["val_tokens"] += len(ids)
                else:
                    writer.write(phase, ids)
                    st["tokens"][phase] += len(ids)
                    if st["epochs"] > 0:
                        st["repeated_tokens"] += len(ids)
                    need -= len(ids)
                    since += len(ids)
                if since >= checkpoint_every:
                    writer.checkpoint()
                    since = 0
                if need <= 0:
                    break
            if need <= 0:
                break
            if not progressed:
                log(f"⚠️  {name} : flux vide, quota {phase} non atteint ({need:,} tokens manquants)")
                break
            st["epochs"] += 1                                    # flux épuisé avant le quota -> répétition
            st["docs_consumed"] = 0
            if st["epochs"] > max_epochs:
                log(f"⚠️  {name} : {st['epochs']} répétitions (> {max_epochs:g}) : arrêt, quota {phase} non atteint ({need:,} tokens manquants)")
                break
        writer.checkpoint()
    return st


def run_existing_bin(writer: MixWriter, name: str, quotas: Dict[str, int], bin_path: str, block: int = 1 << 20, seed: int = 0,
                     log: Callable[[str], None] = print) -> Dict:
    """Copie des blocs (de `block` tokens) d'un corpus DÉJÀ tokenisé, dans un ordre pseudo-aléatoire fixe ; la phase decay reprend
    les blocs suivants de la même permutation (donc jamais ceux de la phase stable). Aucun retéléchargement ni retokenisation."""
    data = np.memmap(bin_path, dtype=writer.dtype, mode="r")
    n_blocks = len(data) // block
    order = np.random.default_rng(seed).permutation(n_blocks)
    st = writer.state.src(name)
    since = 0
    for phase in ("stable", "decay"):
        need = quotas.get(phase, 0) - st["tokens"][phase]
        while need > 0 and st["docs_consumed"] < n_blocks:           # docs_consumed = nombre de blocs utilisés
            b = int(order[st["docs_consumed"]])
            chunk = np.asarray(data[b * block:(b + 1) * block])
            if len(chunk) > need:
                chunk = chunk[:need]
            writer.write_tokens(phase, chunk)
            st["docs_consumed"] += 1
            st["tokens"][phase] += len(chunk)
            need -= len(chunk)
            since += len(chunk)
            if since >= 50_000_000:
                writer.checkpoint()
                since = 0
        if need > 0:
            log(f"⚠️  {name} : seulement {n_blocks} blocs disponibles, quota {phase} non atteint ({need:,} tokens manquants)")
        writer.checkpoint()
    return st

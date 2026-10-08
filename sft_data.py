"""
sft_data.py — construit le jeu de données de fine-tuning (questions -> réponses).

Ce qui change par rapport à l'ancien SFT (PIAF seul) :
  * plusieurs sources : Q/R synthétiques propres (hors-ligne, incluant des refus variés : météo, heure,
    Internet, infos personnelles...), French-Alpaca, Magpie-FR (filtré, réponses courtes), OpenAssistant FR,
    PIAF (lecture de texte, fenêtre de contexte CENTRÉE sur la réponse) ;
  * template chat avec tokens spéciaux  <|user|> … <|end|> <|assistant|> … <|end|> ;
  * la loss ne porte QUE sur les réponses (mask) — avant, la réponse ne pesait que 2-4 % de la loss ;
  * exemples trop longs écartés (jamais tronqués au milieu d'une réponse) ;
  * split train/val par hachage stable (une même question reste toujours du même côté).

Chaque source externe est protégée par try/except : si un dataset HF change de format, on le signale
et on continue avec les autres.

Équilibre par défaut (sur ~35-40k exemples) : French-Alpaca reste la plus grosse source à elle seule ; pour
éviter qu'elle écrase le reste (identité, refus, PIAF), ses valeurs par défaut ont été réduites et Magpie-FR
(source française, filtrée courte) ajoutée pour diversifier sans revenir à un seul gros bloc. Voir --alpaca,
--magpie, --oasst, --piaf pour ajuster les proportions.

Exemple :
  python sft_data.py --tokenizer data/tokenizer.json --out data/sft --alpaca 15000 --magpie 6000 --piaf 4000 --oasst 3000

Personnalité : `personnalite.jsonl` (≤ 30 Q/R sur le modèle lui-même) est ajouté automatiquement, toujours en train,
répété --persona_repeat fois. Modifie ce fichier puis reconstruis le SFT (supprime data/sft) et ré-entraîne.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from data import write_sft_split
from mini_tokenizer import MiniTokenizer, save_meta
from synthetic_qa import build_synthetic_qa

Conv = Dict  # {"messages": [{"role":..., "content":...}, ...]}


def _conv(q: str, a: str) -> Conv:
    return {"messages": [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()}]}


def _first(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  Cohérence de personnalité : on écarte des sources externes tout ce qui contredit "je suis MiniLLM, créé par DevLab"
# ══════════════════════════════════════════════════════════════════════════════
_IDENTITY_Q = ("qui es-tu", "qui êtes-vous", "tu es qui", "comment t'appelles", "comment vous appelez", "ton nom",
               "ton prénom", "qui t'a créé", "qui t'a conçu", "es-tu un", "es-tu une", "présente-toi", "présentez-vous")
_IDENTITY_A = ("openai", "chatgpt", "gpt-3", "gpt-4", "en tant qu'ia", "en tant qu'intelligence artificielle",
               "en tant que modèle de langage", "en tant qu'assistant", "en tant qu'ai", "je suis une ia",
               "je suis une intelligence artificielle", "je suis un assistant", "assistant virtuel",
               "open assistant", "open-assistant", "laion")


def contradicts_persona(question: str, answer: str) -> bool:
    """Vrai si l'exemple parle de l'identité de l'assistant (ChatGPT, OpenAI, 'en tant qu'IA'...) : il brouillerait la personnalité."""
    q = question.lower().replace("’", "'")
    a = answer.lower().replace("’", "'")
    return any(k in q for k in _IDENTITY_Q) or any(k in a for k in _IDENTITY_A)


# ══════════════════════════════════════════════════════════════════════════════
#  Sources externes (HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════
_MARKDOWN_HEAVY = ("```", "###", "|---", "- **", "1. **")


def ok_magpie(instr: str, resp: str, max_answer_chars: int = 350, max_question_chars: int = 300) -> bool:
    """
    Filtre pur (sans réseau, testable) pour Magpie-FR : questions/réponses générées par un gros modèle, donc
    souvent longues, très formatées (listes, tableaux, titres markdown) et vouvoyées — on ne garde que ce qui
    ressemble à une réponse courte, en phrases, du genre de ce que le reste du SFT enseigne déjà.
    """
    instr, resp = instr.strip(), resp.strip()
    if not (3 <= len(resp) <= max_answer_chars) or not instr or len(instr) > max_question_chars:
        return False
    if any(m in resp for m in _MARKDOWN_HEAVY) or "http" in resp:
        return False
    if contradicts_persona(instr, resp):
        return False
    return True


def _first_qa_any(row: dict) -> Optional[Tuple[str, str]]:
    """Essaie d'extraire une paire (question, réponse) d'une ligne, qu'elle soit au format plat
    (instruction/input/output ou équivalent) OU au format ShareGPT (conversations=[{from/role, value/content}, ...]).
    Filet de sécurité pour les sources dont le nom de champ réel diffère de ce qu'on attendait."""
    instr = _first(row, "instruction", "question", "prompt", "query")
    inp = _first(row, "input", "context")
    ans = _first(row, "output", "response", "answer", "completion")
    if instr and ans:
        return (f"{instr}\n\n{inp}" if inp else instr), ans
    turns = row.get("conversations") or row.get("conversation") or row.get("messages") or []
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
    q = a = None
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = role_map.get(str(t.get("from") or t.get("role") or "").lower())
        content = (t.get("value") or t.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and q is None:
            q = content
        elif role == "assistant" and q is not None and a is None:
            a = content
            break
    return (q, a) if q and a else None


def load_magpie_fr(max_examples: int, seed: int = 0) -> List[Conv]:
    """
    bofenghuang/magpie-fr : questions + réponses en français générées par de gros modèles (Magpie), notées et
    filtrées par leurs auteurs. On cible le sous-ensemble le plus petit et le plus tractable (50k lignes) plutôt
    que les fichiers bruts (plusieurs Go par modèle). Champs non garantis d'une révision à l'autre du dataset :
    plusieurs noms candidats sont essayés (dont le format ShareGPT en repli), en cas d'échec la source est
    simplement ignorée (voir _safe()).
    """
    from datasets import load_dataset
    ds = load_dataset("bofenghuang/magpie-fr", data_files="magpie_inst_resp_mininstscore28_subset50k_responded_processed.jsonl",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out: List[Conv] = []
    first_row_keys = None
    for row in ds:
        if first_row_keys is None:
            first_row_keys = list(row.keys())
        qa = _first_qa_any(row)
        if qa and ok_magpie(*qa):
            out.append(_conv(*qa))
            if len(out) >= max_examples:
                break
    if not out and first_row_keys:
        print(f"  ⚠️  magpie_fr : 0 exemple retenu — colonnes réelles du dataset : {first_row_keys}")
    return out


def load_alpaca_style(dataset_name: str, max_examples: int, max_answer_chars: int = 350,
                      max_question_chars: int = 300, seed: int = 0) -> List[Conv]:
    """Charge n'importe quel dataset au format Alpaca (instruction/input/output ou équivalent), avec repli
    ShareGPT (conversations=[...]) si le dataset s'avère en fait dans ce format (voir _first_qa_any)."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    idx = np.random.default_rng(seed).permutation(len(ds))
    out: List[Conv] = []
    first_row_keys = None
    for i in idx:
        row = ds[int(i)]
        if first_row_keys is None:
            first_row_keys = list(row.keys())
        qa = _first_qa_any(row)
        if not qa:
            continue
        q, ans = qa
        if len(q) > max_question_chars or not (3 <= len(ans) <= max_answer_chars):
            continue                                    # un tout petit modèle apprend mieux sur des réponses COURTES
        if "###" in ans or "http" in ans or contradicts_persona(q, ans):
            continue
        out.append(_conv(q, ans))
        if len(out) >= max_examples:
            break
    if not out and first_row_keys:
        print(f"  ⚠️  {dataset_name} : 0 exemple retenu — colonnes réelles du dataset : {first_row_keys}")
    return out


def load_french_alpaca(max_examples: int, max_answer_chars: int = 350, max_question_chars: int = 300, seed: int = 0) -> List[Conv]:
    return load_alpaca_style("jpacifico/French-Alpaca-dataset-Instruct-110K", max_examples, max_answer_chars, max_question_chars, seed)


def _piaf_window(context: str, ans_start: int, ans_len: int, width: int = 600) -> str:
    """Fenêtre d'environ `width` caractères qui CONTIENT toujours la réponse (avant : context[:800] la coupait parfois)."""
    start = max(0, min(ans_start - width // 3, len(context) - width))
    end = min(len(context), start + width)
    end = max(end, ans_start + ans_len)
    if start > 0:                                       # aligne le début sur un début de phrase/mot
        cut = context.rfind(". ", 0, ans_start)
        start = cut + 2 if cut >= start - 200 and cut != -1 else context.find(" ", start) + 1
    if end < len(context):
        cut = context.find(". ", ans_start + ans_len)
        end = cut + 1 if cut != -1 and cut - start <= width + 250 else end
    return context[start:end].strip()


def load_piaf(max_examples: int, seed: int = 0) -> List[Conv]:
    from datasets import load_dataset
    ds = load_dataset("etalab-ia/piaf", split="train")
    idx = np.random.default_rng(seed).permutation(len(ds))
    out: List[Conv] = []
    for i in idx:
        row = ds[int(i)]
        texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
        if not texts:
            continue
        ans = texts[0].strip()
        if not ans:
            continue
        ctx = _piaf_window(row["context"], int(starts[0]), len(texts[0]))
        if ans not in ctx:
            continue
        q = f"Réponds à la question à partir du texte.\n\nTexte : {ctx}\n\nQuestion : {row['question'].strip()}"
        a = ans if ans[-1] in ".!?" else ans + "."
        out.append(_conv(q, a))
        if len(out) >= max_examples:
            break
    return out


def load_oasst_fr(max_examples: int, max_answer_chars: int = 500, seed: int = 0) -> List[Conv]:
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("OpenAssistant/oasst1")
    rows = concatenate_datasets([ds["train"], ds["validation"]])
    by_id = {r["message_id"]: r for r in rows if r["lang"] == "fr"}
    out: List[Conv] = []
    for r in by_id.values():
        if r["role"] != "assistant" or r.get("rank") not in (0, None):
            continue
        parent = by_id.get(r["parent_id"])
        if not parent or parent["role"] != "prompter":
            continue
        q, a = parent["text"].strip(), r["text"].strip()
        if len(q) > 400 or not (3 <= len(a) <= max_answer_chars) or contradicts_persona(q, a):
            continue
        out.append(_conv(q, a))
    np.random.default_rng(seed).shuffle(out)
    return out[:max_examples]


def load_sharegpt_style(dataset_name: str, max_examples: int, text_field: str = "conversations",
                        max_turns: int = 4, max_msg_chars: int = 400, seed: int = 0,
                        data_files: Optional[str] = None) -> List[Conv]:
    """Charge n'importe quel dataset au format ShareGPT ({text_field}=[{from/role, value/content}, ...]) ->
    diversité de FORME (relances, contexte qui s'accumule) que les sources question->réponse à un tour n'apportent pas."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, data_files=data_files, split="train") if data_files else \
        load_dataset(dataset_name, split="train")
    role_map = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
    idx = np.random.default_rng(seed).permutation(len(ds))
    out: List[Conv] = []
    for i in idx:
        row = ds[int(i)]
        turns = row.get(text_field) or row.get("conversation") or row.get("messages") or []
        messages = []
        for t in turns[:max_turns]:
            role = role_map.get(str(t.get("from") or t.get("role") or "").lower())
            content = (t.get("value") or t.get("content") or "").strip()
            if not role or not content or len(content) > max_msg_chars:
                messages = []
                break
            messages.append({"role": role, "content": content})
        if len(messages) < 2 or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
            continue
        text = " ".join(m["content"] for m in messages)
        if "###" in text or "http" in text or any(contradicts_persona(m["content"], m2["content"])
                                                    for m, m2 in zip(messages, messages[1:])):
            continue
        out.append({"messages": messages})
        if len(out) >= max_examples:
            break
    return out


def load_sharegpt_fr(max_examples: int, max_turns: int = 4, max_msg_chars: int = 400, seed: int = 0) -> List[Conv]:
    """Conversations multi-tours (format ShareGPT : conversations=[{from, value}, ...]) -> diversité de FORME
    que les sources question->réponse à un tour n'apportent pas (relances, contexte qui s'accumule)."""
    return load_sharegpt_style("FreedomIntelligence/sharegpt-french", max_examples, "conversations",
                               max_turns, max_msg_chars, seed)


def load_jsonl(path: str) -> List[Conv]:
    """Tes propres données : lignes {"messages":[...]} ou {"question":..., "answer":...}."""
    out: List[Conv] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row:
                out.append({"messages": row["messages"]})
            else:
                q, a = _first(row, "question", "instruction", "prompt"), _first(row, "answer", "response", "output")
                if q and a:
                    out.append(_conv(q, a))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Construction
# ══════════════════════════════════════════════════════════════════════════════
def _safe(name: str, fn, *args, **kwargs) -> List[Conv]:
    try:
        res = fn(*args, **kwargs)
        print(f"  ✓ {name:<16} {len(res):>7,} exemples")
        return res
    except Exception as e:                                       # noqa: BLE001
        print(f"  ✗ {name:<16} ignoré ({type(e).__name__}: {str(e)[:120]})")
        return []


def build_sft(tokenizer_path: str, out_dir: str, alpaca: int = 15_000, piaf: int = 4_000, oasst: int = 3_000,
              magpie: int = 6_000, alpaca_gpt4: int = 0, evol: int = 0, sharegpt: int = 0,
              synthetic_repeat: int = 2, extra_jsonl: Optional[List[str]] = None, max_len: int = 512,
              val_permille: int = 20, seed: int = 0, persona: Optional[str] = None, persona_repeat: int = 8, extra_repeat: int = 1) -> dict:
    tok = MiniTokenizer(tokenizer_path)
    os.makedirs(out_dir, exist_ok=True)
    print("Sources :")
    sources: Dict[str, List[Conv]] = {}
    sources["synthetic"] = [c for r in range(synthetic_repeat) for c in build_synthetic_qa(seed=seed + r)]
    print(f"  ✓ {'synthetic':<16} {len(sources['synthetic']):>7,} exemples")
    if alpaca > 0:
        sources["french_alpaca"] = _safe("french_alpaca", load_french_alpaca, alpaca, seed=seed)
    if oasst > 0:
        sources["oasst_fr"] = _safe("oasst_fr", load_oasst_fr, oasst, seed=seed)
    if magpie > 0:
        sources["magpie_fr"] = _safe("magpie_fr", load_magpie_fr, magpie, seed=seed)
    if piaf > 0:
        sources["piaf"] = _safe("piaf", load_piaf, piaf, seed=seed)
    # Nouvelles sources (diversité) : réponses GPT-4 (plus soignées), instructions plus complexes, multi-tours.
    if alpaca_gpt4 > 0:
        sources["alpaca_gpt4_fr"] = _safe("alpaca_gpt4_fr", load_alpaca_style,
                                          "FreedomIntelligence/alpaca-gpt4-french", alpaca_gpt4, seed=seed)
    if evol > 0:
        sources["evol_instruct_fr"] = _safe("evol_instruct_fr", load_alpaca_style,
                                            "FreedomIntelligence/evol-instruct-french", evol,
                                            max_answer_chars=450, seed=seed)
    if sharegpt > 0:
        sources["sharegpt_fr"] = _safe("sharegpt_fr", load_sharegpt_fr, sharegpt, seed=seed)
    for p in extra_jsonl or []:
        sources[f"jsonl:{os.path.basename(p)}"] = _safe(p, load_jsonl, p)

    persona_convs: List[Conv] = _safe("persona", load_jsonl, persona) if persona else []

    dtype = np.uint16 if tok.padded_vocab_size <= 65_535 else np.uint32
    splits: Dict[str, List[Tuple[List[int], List[int]]]] = {"train": [], "val": []}
    n_too_long = 0
    per_source: Dict[str, int] = {}
    for name, convs in sources.items():
        for conv in convs:
            ids, mask = tok.encode_chat(conv["messages"])
            if len(ids) > max_len + 1 or sum(mask) == 0:
                n_too_long += 1
                continue
            key = conv["messages"][0]["content"].encode("utf-8")
            split = "val" if zlib.crc32(key) % 1000 < val_permille else "train"
            copies = extra_repeat if (name.startswith("jsonl:") and split == "train") else 1
            splits[split] += [(ids, mask)] * copies
            per_source[name] = per_source.get(name, 0) + 1
    if len(splits["val"]) < 50:                                  # petit corpus : garantit un val exploitable
        step = max(2, len(splits["train"]) // 50)
        moved = splits["train"][::step]
        splits["val"] += moved
        splits["train"] = [e for i, e in enumerate(splits["train"]) if i % step != 0]
    # Personnalité : TOUJOURS dans train (jamais en val), répétée persona_repeat fois pour peser face aux ~30 000 autres exemples
    n_persona = 0
    for conv in persona_convs:
        ids, mask = tok.encode_chat(conv["messages"])
        if len(ids) > max_len + 1 or sum(mask) == 0:
            print(f"  ⚠️  persona ignoré (trop long) : {conv['messages'][0]['content'][:60]}")
            continue
        splits["train"] += [(ids, mask)] * persona_repeat
        n_persona += 1
    if n_persona:
        per_source[f"persona (x{persona_repeat})"] = n_persona
    rng = np.random.default_rng(seed)
    for s in splits:
        order = rng.permutation(len(splits[s]))
        splits[s] = [splits[s][i] for i in order]

    stats = {s: write_sft_split(out_dir, s, splits[s], dtype) for s in splits}
    tr = stats["train"]
    meta = {
        "vocab_size": tok.vocab_size, "padded_vocab_size": tok.padded_vocab_size, "pad_id": tok.pad_id,
        "max_len": max_len, "dtype": np.dtype(dtype).name, "train": stats["train"], "val": stats["val"],
        "sources": per_source, "skipped_too_long": n_too_long, "tokenizer_sha": tok.sha,
    }
    save_meta(os.path.join(out_dir, "meta.json"), meta)
    print(f"\nSFT : train={tr['n_examples']:,} ex. / {tr['n_tokens']:,} tokens | val={stats['val']['n_examples']:,} ex. "
          f"| {n_too_long} exemples trop longs écartés")
    print(f"Part des tokens qui portent de la loss (réponses) : {tr['n_answer_tokens'] / tr['n_tokens'] * 100:.1f} % "
          f"(ancien SFT : ~2-4 %)")
    ids, mask = splits["train"][0]
    print("Exemple :", repr(tok.decode(ids)[:250]))
    return meta


def main():
    p = argparse.ArgumentParser(description="MiniLLM v2 — données de fine-tuning")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="data/sft")
    p.add_argument("--alpaca", type=int, default=15_000, help="French-Alpaca ; réduit par défaut pour ne pas écraser le reste (~76%% du SFT sinon)")
    p.add_argument("--piaf", type=int, default=4_000, help="exemples PIAF (le jeu en contient ~3 800 : tous par défaut)")
    p.add_argument("--oasst", type=int, default=3_000)
    p.add_argument("--magpie", type=int, default=6_000, help="Magpie-FR, filtré (réponses courtes, sans markdown lourd)")
    p.add_argument("--alpaca_gpt4", type=int, default=0, help="FreedomIntelligence/alpaca-gpt4-french (réponses GPT-4, ~50k dispo)")
    p.add_argument("--evol", type=int, default=0, help="FreedomIntelligence/evol-instruct-french (instructions plus complexes, ~59k dispo)")
    p.add_argument("--sharegpt", type=int, default=0, help="FreedomIntelligence/sharegpt-french (conversations multi-tours, ~5.6k dispo)")
    p.add_argument("--synthetic_repeat", type=int, default=2)
    p.add_argument("--extra_jsonl", nargs="*", default=None)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--val_permille", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    default_persona = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personnalite.jsonl")
    p.add_argument("--persona", default=default_persona if os.path.exists(default_persona) else "",
                   help="fichier .jsonl de Q/R sur le modèle lui-même (personnalité) ; \"\" pour désactiver")
    p.add_argument("--persona_repeat", type=int, default=8, help="nb de copies de chaque Q/R de personnalité dans train")
    p.add_argument("--extra_repeat", type=int, default=1, help="nb de copies de chaque exemple de --extra_jsonl dans train (ex. 3 pour datasets/faits_cameroun_afrique.jsonl)")
    a = p.parse_args()
    build_sft(a.tokenizer, a.out, alpaca=a.alpaca, piaf=a.piaf, oasst=a.oasst, magpie=a.magpie,
              alpaca_gpt4=a.alpaca_gpt4, evol=a.evol, sharegpt=a.sharegpt,
              synthetic_repeat=a.synthetic_repeat, extra_jsonl=a.extra_jsonl, max_len=a.max_len,
              val_permille=a.val_permille, seed=a.seed, persona=a.persona or None,
              persona_repeat=a.persona_repeat, extra_repeat=a.extra_repeat)


if __name__ == "__main__":
    main()

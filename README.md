# 🧠 MiniLLM v2

Un petit LLM **français** entraîné *from scratch* (Transformer décodeur : RMSNorm · RoPE · SwiGLU · GQA · QK-norm),
pensé pour être entraîné sur **Kaggle / Colab (T4)** et même pilotable depuis un téléphone.

> **Attentes réalistes.** Avec ~1 milliard de tokens sur un GPU T4, ce modèle apprend à écrire du français court et
> plausible, et — après le fine-tuning — à **répondre en phrases complètes à des questions simples** (identité,
> salutations, capitales, calendrier, contraires, petites opérations…) et à dire « je ne sais pas » quand il ne peut pas
> savoir. Ce n'est pas un assistant généraliste : hors de ce qu'il a vu, il inventera. C'est normal pour un modèle de cette taille.

---

## Démarrage rapide

**Notebook (le plus simple)** — ouvre `MiniLLM_v2.ipynb` sur Kaggle ou Colab (GPU activé) et exécute les cellules *dans l'ordre, une par une*.
Chaque cellule d'entraînement **reprend seule** après une coupure. Sur Colab, les données sont copiées automatiquement sur le disque local
avant l'entraînement (Drive est trop lent pour les lectures aléatoires) ; seuls les checkpoints restent sur Drive.

**Ligne de commande**
```bash
pip install -r requirements.txt
python smoke_test.py                                   # 2 min, CPU, sans internet : vérifie tout le pipeline

# 1. données (Wikipédia FR nettoyé + web FR) -> tokenizer 32k -> binaires
python prepare_data.py all --out data --wiki_docs 250000 --web_docs 300000

# 2. pré-entraînement  (2 GPU : remplacer "python" par "torchrun --standalone --nproc_per_node=2")
python train.py --mode pretrain --data_dir data/pretrain --out_dir checkpoints/pretrain --model_size 49M \
                --max_iters 10000 --batch_size 16 --grad_accum 8

# 3. questions -> réponses
python sft_data.py --tokenizer data/tokenizer.json --out data/sft     # ajoute automatiquement personnalite.jsonl
python train.py --mode sft --data_dir data/sft --out_dir checkpoints/sft --init_from checkpoints/pretrain/best.pt

# 4. discuter / évaluer
python generate.py --ckpt checkpoints/sft/best.pt --mode chat
python evaluate.py qa --ckpt checkpoints/sft/best.pt --sft_dir data/sft
```
100 % hors-ligne possible : `prepare_data.py all --wiki_docs 0 --local_txt mes_textes/*.txt`.

---

## Presets (les noms sont le **vrai** nombre de paramètres, calculé automatiquement)

| preset | couches | d_model | têtes (Q/KV) | contexte | paramètres totaux | dont hors embeddings |
|---|---|---|---|---|---|---|
| **13M** | 6 | 256 | 4/4 | 512 | 13,012,992 | 4,820,992 |
| **23M** | 6 | 384 | 6/6 | 512 | 22,910,592 | 10,622,592 |
| **49M** | 10 | 512 | 8/8 | 512 | 48,508,672 | 32,124,672 |
| **110M** | 12 | 768 | 12/12 | 1024 | 109,531,392 | 84,955,392 |
| **311M** | 24 | 1024 | 16/8 | 2048 | 311,218,176 | 278,450,176 |
| **952M** | 20 | 2048 | 16/4 | 2048 | 951,671,808 | 886,135,808 |

Chiffres pour un vocabulaire de 32 000 tokens. (Avec l'ancien tokenizer de 100 277 tokens les mêmes architectures
faisaient 49M / 83M / 162M / 381M / 1 091M : les embeddings pesaient 62 % du plus petit modèle.)
`python inspect_model.py` affiche le tableau, la mémoire estimée et vérifie qu'un modèle est sain.

---

## Fichiers

| fichier | rôle |
|---|---|
| `config.py` | `ModelConfig`, presets nommés d'après leur taille réelle, `TrainConfig` (pré-entraînement **et** SFT) |
| `model.py` | Transformer : RoPE réel, QK-norm, GQA, **KV-cache**, loss masquée (`-1` ignoré) |
| `mini_tokenizer.py` | BPE 32k byte-level (chiffres isolés, apostrophes françaises) + **template de chat** |
| `text_cleaning.py` | nettoyage Wikipédia (phrases trouées, sections de fin, titres) |
| `prepare_data.py` | corpus → tokenizer → `train.bin` / `val.bin` (split **par document**, `<|endoftext|>` entre documents) |
| `synthetic_qa.py` | ~2 000 Q/R propres générées par code (faits sûrs, arithmétique calculée) |
| `personnalite.jsonl` | **≤ 30 Q/R sur le modèle lui-même** (nom, créateur, caractère) — à éditer librement |
| `sft_data.py` | jeu SFT : synthétique + French-Alpaca + OpenAssistant FR + PIAF + personnalité ; masque de loss ; filtre d'identité |
| `data.py` | loaders **déterministes** : val fixe, époques sans remise, SFT groupé par longueur |
| `checkpoint.py` | sauvegardes atomiques, `best`/`final`/reprise, nettoyage numérique |
| `train.py` | boucle unique (pretrain + SFT), DDP, budget temps, log JSONL |
| `generate.py` | génération avec KV-cache, chat au bon template, streaming, anti-répétition |
| `evaluate.py` | perplexité (val fixe), Exact-Match / F1, prompts de démonstration |
| `inspect_model.py` · `smoke_test.py` · `tests/` | vérifications |
| `push_to_github.py` | pousse le projet via l'API GitHub (token par variable d'environnement) |

---

## Ce qui rend l'entraînement fiable

**Meilleur modèle (`best.pt`)** — la `val_loss` est mesurée sur des séquences **toujours identiques** (espacées dans `val.bin`),
donc comparable d'une évaluation à l'autre. La meilleure valeur est écrite dans `best.json` et **relue à la reprise** :
un run repris ne peut plus écraser un meilleur modèle. En SFT la val_loss ne porte que sur les réponses.

**Reprise** — `ckpt_XXXXXXX.pt` contient tout (modèle, optimiseur, GradScaler, itération, best, tokens vus) ; l'ordre des
données est une fonction de `(seed, numéro de batch)`, donc on repart exactement là où on s'est arrêté (test automatisé :
20 itérations d'un coup = 10 + reprise + 10, mêmes poids). Les écritures sont atomiques (fichier temporaire + `os.replace`).
`--max_minutes` sauvegarde et s'arrête proprement avant la fin d'une session.

**Fichiers produits dans `out_dir`** : `best.pt` (poids, léger) · `final.pt` (dernière it.) · `ckpt_*.pt` (état complet, 2 gardés)
· `best.json` · `log.jsonl` (courbes) · `tokenizer.json` (copié : checkpoint et tokenizer voyagent ensemble).

**Précision** — `auto` : bf16 seulement sur GPU Ampere+ ; sinon fp16 + GradScaler (T4/P100). Jamais de bf16 émulé.

**Personnalité** — `personnalite.jsonl` (≤ 30 Q/R : « Comment tu t'appelles ? » → « Je suis MiniLLM… ») est ajouté au SFT, toujours en train
(jamais en val) et répété 20 fois ; les exemples externes qui parlent de l'identité de l'assistant (« en tant qu'IA », ChatGPT…) sont écartés.
Pas de system prompt : la personnalité vit dans les poids, le contexte reste libre pour la conversation.

**Fine-tuning** — template `<|user|>…<|end|><|assistant|>…<|end|>`, loss **uniquement sur la réponse**, exemples
jamais tronqués, batchs regroupés par longueur (padding à droite : sans effet grâce au masque causal), 3 époques max avec
early stopping, optimiseur neuf (`--init_from` charge les poids seulement).

**Génération** — KV-cache (≈ 6× plus rapide que l'ancien recalcul complet dès 200 tokens, mesuré sur CPU ; l'écart grandit avec la
longueur), arrêt sur `<|end|>`, jamais de token de rôle dans la réponse, pénalité de répétition + interdiction des trigrammes
répétés **appliquées au texte généré uniquement** (pas de blocage quand on recopie le contexte).

---

## Réglages utiles

| envie | option |
|---|---|
| plus de tokens/s | 2 GPU (`torchrun`), `--compile` (à tester), `--batch_size` maximal qui tient en mémoire |
| out of memory | `--batch_size 8` (monte `--grad_accum` pour garder le batch effectif) |
| entraînement interruptible à durée libre | `--schedule wsd` (warmup–stable–decay) : le LR reste haut puis décroît sur les `--decay_frac` derniers % |
| modèle plus petit / plus rapide | `--model_size 23M` (LR 1e-3) |
| réponses plus « sages » | `--temperature 0.3` dans `generate.py` |
| tes propres Q/R | `sft_data.py --extra_jsonl mes_qr.jsonl` (lignes `{"question":…,"answer":…}` ou `{"messages":[…]}`) |
| changer la personnalité | édite `personnalite.jsonl`, supprime `data/sft`, relance `sft_data.py` puis le SFT (pas de system prompt : la personnalité est dans les poids) |

---

## Tests

```bash
python -m pytest -q          # 33 tests, CPU, ~20 s : KV-cache = forward complet, nombre de paramètres exact, masque de loss,
                             # nettoyage, tokenizer, loaders, reprise exacte, best jamais écrasé, DDP 2 processus…
python smoke_test.py         # pipeline complet sur un mini modèle
```

## Limites connues

* Le code d'entraînement a été validé sur **CPU** (tests, DDP en gloo, mini-entraînement Q/R de bout en bout). Les performances GPU
  (tok/s, mémoire) doivent être lues dans les logs de ta session : la 1re fois, regarde le `k tok/s` affiché.
* Les chargeurs de datasets HuggingFace (Wikipédia, FineWeb-2, French-Alpaca, OpenAssistant, PIAF) ne peuvent pas être testés hors-ligne :
  chacun est protégé (`try/except`) et le SFT continue avec les sources restantes s'il change de format.
* Les Q/R synthétiques donnent la *forme* d'une conversation, pas de la connaissance ; les réponses arithmétiques d'un très petit modèle restent fragiles.

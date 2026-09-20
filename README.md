# 🧠 MiniLLM v2.1

Un petit LLM **français** entraîné *from scratch* (Transformer décodeur : RMSNorm · RoPE · SwiGLU · GQA · QK-norm),
pensé pour être entraîné sur **Kaggle / Colab (GPU T4)** et pilotable depuis un téléphone.

> **Attentes réalistes.** Le corpus par défaut fait ~340 M de tokens (le run par défaut en traite ~655 M, soit ~2 époques).
> Le modèle de base écrit alors du français **correct et plausible mais inventé** ; après le fine-tuning il répond en
> **phrases complètes à des questions simples** (identité, salutations, capitales, calendrier, contraires, petites opérations…)
> et dit « je ne sais pas » quand il ne peut pas savoir. Ce n'est pas un assistant généraliste : hors de ce qu'il a vu, il inventera.

**Sommaire** — [Versions](#versions-et-branches) · [Démarrer](#démarrer) · [Kaggle pas à pas](#kaggle-pas-à-pas-avec-google-drive) ·
[Données](#données) · [Presets](#presets) · [Fichiers](#fichiers) · [Fiabilité et reprise](#fiabilité-et-reprise) ·
[Personnalité, SFT, génération](#personnalité-sft-génération) · [Mini-RAG](#mini-rag-expérimental) · [Réglages](#réglages-utiles) ·
[Repères observés](#repères-observés-run-réel) · [Dépannage](#dépannage) · [Tests](#tests) · [Limites](#limites-connues-et-idées-non-faites)

---

## Versions et branches

| branche | contenu |
|---|---|
| `main` | **v2** : architecture, pipeline de données/entraînement corrigés après audit (voir `CHANGELOG.md`) |
| `v2.1` | v2 **+** notebook Kaggle, garde-fou tokenizer, nettoyage renforcé, mini-RAG, `evaluate.py persona`, logs ETA/mémoire |

**Aucun changement d'architecture** entre v2 et v2.1 : le code v2.1 relit les **checkpoints et les données v2** (le champ
`tokenizer_sha` est optionnel). Deux usages :

| | continuer ton run v2 | run neuf v2.1 |
|---|---|---|
| `BRANCH` (notebook Kaggle) | `main` | `v2.1` |
| données | les **mêmes** `train.bin` / `val.bin` / `tokenizer.json` (Drive ou `/kaggle/input`) | reconstruites (corpus mieux nettoyé) |
| `PT_ITERS` / `SCHEDULE` | ex. 8 000 / `cosine` (reprend à l'it. du checkpoint) | 10 000 / `wsd` conseillé |
| résultat | prolonge le même modèle | repart de **zéro** |

⚠️ Si tu fusionnes `v2.1` dans `main` : `MiniLLM_v2.ipynb` a été modifié des deux côtés (sauvegardes Colab avec sorties d'un côté, version v2.1 de l'autre) → garde celui de `v2.1`.

---

## Démarrer

| plateforme | comment |
|---|---|
| **Colab** | ouvre `MiniLLM_v2.ipynb` : <https://colab.research.google.com/github/bono-p/minillm_v2/blob/v2.1/MiniLLM_v2.ipynb> (GPU T4). Données copiées sur le disque local avant l'entraînement (Drive est trop lent en accès aléatoire) ; seuls les checkpoints restent sur Drive. Limite de GPU gratuite : voir Kaggle ↓ |
| **Kaggle** | importe `MiniLLM_v2_Kaggle.ipynb` (File → Import Notebook) → [pas à pas](#kaggle-pas-à-pas-avec-google-drive) |
| **Ligne de commande** | ci-dessous |

Exécute les cellules **une par une**. Chaque cellule d'entraînement **reprend seule** après une coupure.

```bash
pip install -r requirements.txt
python smoke_test.py                                   # 2 min, CPU, sans internet : vérifie tout le pipeline

# 1. données (Wikipédia FR nettoyé + web FR) -> tokenizer 32k -> binaires
python prepare_data.py all --out data --wiki_docs 250000 --web_docs 300000

# 2. pré-entraînement  (2 GPU : remplacer "python" par "torchrun --standalone --nproc_per_node=2")
python train.py --mode pretrain --data_dir data/pretrain --out_dir checkpoints/pretrain --model_size 49M \
                --max_iters 10000 --batch_size 16 --grad_accum 8

# 3. questions -> réponses (+ personnalité)
python sft_data.py --tokenizer data/tokenizer.json --out data/sft     # ajoute automatiquement personnalite.jsonl
python train.py --mode sft --data_dir data/sft --out_dir checkpoints/sft --init_from checkpoints/pretrain/best.pt

# 4. discuter / évaluer
python generate.py --ckpt checkpoints/sft/best.pt --mode chat
python evaluate.py qa      --ckpt checkpoints/sft/best.pt --sft_dir data/sft
python evaluate.py persona --ckpt checkpoints/sft/best.pt
python rag.py --ckpt checkpoints/sft/best.pt --kb knowledge datasets --question "Quelle est la capitale du Cameroun ?"
```
100 % hors-ligne possible : `prepare_data.py all --wiki_docs 0 --local_txt mes_textes/*.txt`.

---

## Kaggle pas à pas (avec Google Drive)

Kaggle **ne peut pas monter Google Drive**. `MiniLLM_v2_Kaggle.ipynb` récupère donc données et checkpoints dans cet ordre :
**1)** `/kaggle/input` (Dataset ou sortie d'une version précédente du notebook) → **2)** Google Drive via `gdown` (liens de partage, transfert
serveur à serveur : rien ne passe par ton téléphone) → **3)** reconstruction depuis internet, **seulement si aucun checkpoint n'existe**.

**Préparer (une fois)**
1. Compte Kaggle vérifié par téléphone (nécessaire pour *Internet : On*). Quota gratuit indicatif : ~30 h de GPU par semaine, sessions ≤ 12 h.
2. Nouveau notebook → *File → Import Notebook* → `MiniLLM_v2_Kaggle.ipynb`. Panneau *Settings* : **Accelerator = GPU T4 ×2**, **Internet = On**, **Persistence = Files only**.
3. *(Uniquement pour reprendre un run existant)* Dans l'app Drive, pour chaque fichier : ⋮ → *Partager* → *Accès général : Toute personne disposant du lien* (Lecteur) → *Copier le lien*, et colle-le dans le dictionnaire `DRIVE` de la cellule 1. Tu pourras retirer le partage ensuite.

| fichier (dans `MiniLLM_v2/` sur Drive) | clé `DRIVE` | taille indicative |
|---|---|---|
| `data/tokenizer.json` | `data/tokenizer.json` | ~2 Mo |
| `data/pretrain/meta.json` · `val.bin` · `train.bin` | `data/pretrain/…` | 670 Mo pour `train.bin` (335 M tokens × 2 octets) |
| `checkpoints/pretrain/ckpt_XXXXXXX.pt` (le plus récent) | `checkpoints/pretrain/ckpt_XXXXXXX.pt` (**garde ce nom**) | ~585 Mo (état complet : reprise exacte) |
| `checkpoints/pretrain/best.pt` · `best.json` · `log.jsonl` | idem | ~195 Mo · petits |

**Configurer la cellule 1** (voir le tableau « Versions ») : `PHASE`, `BRANCH`, `PT_ITERS`, `SCHEDULE`, `MAX_MINUTES` (défaut 640 min : arrêt propre avant la limite de 12 h).

**Lancer sans risque**
1. Exécute d'abord **à la main** les cellules 1 à 3 avec `PHASE = "pretrain"` : tu dois voir `✓ données de pré-entraînement OK` et `✓ checkpoint de reprise : it …` (sinon corrige les liens *avant* de lancer 10 h).
2. Puis *Save Version → Save & Run All (Commit)*. Les cellules sont protégées par `PHASE` : un « Run All » est toujours sûr.
3. Session suivante : *Add Input → Notebook Output Files → ce notebook* : le code recopie tout seul le dernier checkpoint et les données, puis **reprend**.

| `PHASE` | fait |
|---|---|
| `data` | récupère / vérifie / (re)construit les données |
| `pretrain` | données si besoin, puis pré-entraînement (`--resume auto`) |
| `sft` | données SFT (+ personnalité) puis fine-tuning depuis `best.pt` |
| `chat` | évaluation (`qa`, `demo`, `persona`), mini-RAG, export zip |
| `all` | tout, si le temps le permet |

**Sécurité** : si un checkpoint existe mais pas ses données, le notebook **refuse de reconstruire** (voir *garde-fou tokenizer*). Reprendre un run 1 GPU (Colab)
sur 2 GPU (Kaggle) est sûr : le flux de données est identique (testé). Durée indicative : Colab T4 mesuré à ~21 k tok/s ≈ 3,05 s/it ; sur T4×2 compte
~2× plus vite **(estimation, non mesurée)**.

---

## Données

### Pré-entraînement (`prepare_data.py`)
| source | défaut | remarque |
|---|---|---|
| Wikipédia FR (`wikimedia/wikipedia`, streaming) | 250 000 articles | **nettoyée** (voir ci-dessous) |
| FineWeb-2 FR (`fra_Latn`, streaming) | 300 000 documents | web, plus « naturel », filtre anti-spam en v2.1 |
| tes `.txt` (`--local_txt`) | aucun | découpés en documents d'~4 000 caractères |

Résultat observé sur un vrai run (Colab) : **549 774 documents · 1 431 M caractères · 335,7 M tokens d'entraînement · 4,22 caractères/token**,
Wikipédia : 464 452 articles lus pour 250 000 gardés, **24,5 % des phrases supprimées** (trous des templates du dump).

**Homogénéisation** : chaque source a son nettoyage ; format commun (paragraphes séparés par une ligne vide, Unicode NFC, doublons écartés,
`<|endoftext|>` après chaque document) ; **un seul tokenizer** BPE 32 000 entraîné sur un échantillon des deux sources.
**Mélange** : l'entraînement tire les blocs de 512 tokens au hasard **sans remise** sur tout le fichier → les sources sont mélangées au niveau du bloc,
dans leurs proportions naturelles. **Validation** : 1 % des documents choisis par hachage du texte (jamais recoupés avec l'entraînement).

**Nettoyage Wikipédia** (`text_cleaning.py`) : phrases « trouées » supprimées (« né le  à Moulins », « du . », **v2.1** : « (en latin : ) », « (le ) »,
guillemets vides), paragraphes trop abîmés supprimés, sections de fin (« Notes et références », « Liens externes »…) et titres retirés,
**v2.1** : élisions espacées recollées (« L' archidiocèse » → « L'archidiocèse »). `prepare_data.py tokenize --refilter` ré-applique ces règles à un
`corpus.jsonl` existant **sans re-télécharger** (le tokenizer existant est conservé). ⚠️ Ne l'utilise pas pour *continuer* un run déjà commencé : les `.bin` changeraient.
Les apostrophes ’ et ' ne sont pas unifiées (choix : sans gravité).

### Fine-tuning (`sft_data.py`)
| source | volume max | forme |
|---|---|---|
| Q/R synthétiques (`synthetic_qa.py`, générées par code) | ~4 200 (2 × ~2 100) | phrase complète, faits sûrs, arithmétique calculée |
| French-Alpaca (`jpacifico/French-Alpaca-dataset-Instruct-110K`) | 30 000 | réponses courtes (3–350 caractères) |
| OpenAssistant FR (`OpenAssistant/oasst1`) | 3 000 (souvent moins) | meilleure réponse en français |
| PIAF (`etalab-ia/piaf`) | 4 000 (~3 800 dispo.) | « réponds à partir du texte », fenêtre de 600 caractères **centrée sur la réponse** |
| `personnalite.jsonl` | 132 lignes × 8 copies (≈ 3 % du SFT) | **identité et caractère** uniquement (voir `PERSONNALITE.md`), toujours en `train` |
| `datasets/faits_cameroun_afrique.jsonl` (optionnel) | 109 × `--extra_repeat` | faits Cameroun/Afrique en Q/R ; **non utilisé par défaut** (les faits passent par le RAG) ; `--extra_jsonl datasets/faits_cameroun_afrique.jsonl --extra_repeat 3` pour les faire apprendre |
| tes données (`--extra_jsonl`) | libre | `{"question":…,"answer":…}` ou `{"messages":[…]}` |

Tout est converti en `<|user|>…<|end|><|assistant|>…<|end|>`, la loss ne porte que sur les réponses, les exemples trop longs sont écartés (jamais tronqués), les
exemples externes qui parlent de l'identité de l'assistant (« en tant qu'IA », ChatGPT, OpenAI…) sont filtrés, puis tout est mélangé et groupé par longueur.
Il n'y a **pas de pondération par source** : les proportions viennent des quantités demandées (`--alpaca`, `--oasst`, `--piaf`, `--synthetic_repeat`, `--persona_repeat`, `--extra_repeat`).

---

## Presets

Les noms sont le **vrai** nombre de paramètres, calculé automatiquement (vocabulaire de 32 000 tokens).

| preset | couches | d_model | têtes (Q/KV) | contexte | paramètres totaux | dont hors embeddings |
|---|---|---|---|---|---|---|
| **13M** | 6 | 256 | 4/4 | 512 | 13,012,992 | 4,820,992 |
| **23M** | 6 | 384 | 6/6 | 512 | 22,910,592 | 10,622,592 |
| **49M** | 10 | 512 | 8/8 | 512 | 48,508,672 | 32,124,672 |
| **110M** | 12 | 768 | 12/12 | 1024 | 109,531,392 | 84,955,392 |
| **311M** | 24 | 1024 | 16/8 | 2048 | 311,218,176 | 278,450,176 |
| **952M** | 20 | 2048 | 16/4 | 2048 | 951,671,808 | 886,135,808 |

(Avec l'ancien tokenizer de 100 277 tokens ces architectures faisaient 49M / 83M / 162M / 381M / 1 091M : les embeddings pesaient 62 % du plus petit.)
Le nom affiché dans les logs est calculé avec le vocabulaire **réel** des données. `python inspect_model.py` affiche le tableau et vérifie qu'un modèle est sain.

---

## Fichiers

| fichier | rôle |
|---|---|
| `config.py` | `ModelConfig`, presets nommés d'après leur taille réelle, `TrainConfig` (pré-entraînement **et** SFT) |
| `model.py` | Transformer : RoPE réel, QK-norm, GQA, **KV-cache**, loss masquée (`-1` ignoré) |
| `mini_tokenizer.py` | BPE 32k byte-level (chiffres isolés, élisions françaises) + template de chat + empreinte (`sha`) |
| `text_cleaning.py` | nettoyage Wikipédia (trous, élisions, sections de fin), filtre web anti-spam, `refilter_doc` |
| `prepare_data.py` | `corpus` → `tokenizer` → `tokenize` (`--refilter`) ; split par document, `<|endoftext|>` entre documents |
| `synthetic_qa.py` | ~2 100 Q/R propres générées par code |
| `personnalite.jsonl` · `PERSONNALITE.md` | identité + caractère du modèle (132 Q/R) · guide pour en ajouter sans créer de contradictions |
| `datasets/` | jeux de Q/R de faits (Cameroun, Afrique), utilisables au SFT (`--extra_jsonl`) **et** par le RAG |
| `check_data.py` | vérifie tes `.jsonl` (JSON valide, doublons, faits qui vieillissent…) avant d'entraîner |
| `sft_data.py` | jeu SFT multi-sources, masque de loss, filtre d'identité, `--persona` |
| `data.py` | loaders **déterministes** : val fixe, époques sans remise, SFT groupé par longueur |
| `checkpoint.py` | sauvegardes atomiques, `best`/`final`/reprise, nettoyage numérique |
| `train.py` | boucle unique (pretrain + SFT), DDP, budget temps, log JSONL (ETA, mémoire GPU), garde-fou tokenizer |
| `generate.py` | génération KV-cache, chat au bon template, streaming, anti-répétition |
| `evaluate.py` | `perplexity`, `qa` (EM/F1), `demo`, **`persona`** |
| `rag.py` · `knowledge/` | mini-RAG (BM25) · base de connaissances : `minillm.txt`, `ia_bases.txt`, `cameroun.txt`, `capitales_monde.txt` |
| `kaggle_utils.py` | restauration `/kaggle/input`, Drive (`gdown`), vérifications (tailles des `.bin`, checkpoint), élagage |
| `MiniLLM_v2.ipynb` · `MiniLLM_v2_Kaggle.ipynb` | notebooks Colab · Kaggle |
| `inspect_model.py` · `smoke_test.py` · `tests/` | vérifications |
| `push_to_github.py` | pousse le projet via l'API GitHub (token par variable d'environnement `GITHUB_TOKEN`) |
| `CHANGELOG.md` | chaque problème de l'audit → correction → fichier |

---

## Fiabilité et reprise

**Meilleur modèle (`best.pt`)** — la `val_loss` est mesurée sur des séquences **toujours identiques** (espacées dans `val.bin`), donc comparable d'une
évaluation à l'autre. La meilleure valeur est écrite dans `best.json` et **relue à la reprise** : un run repris ne peut plus écraser un meilleur modèle.
En SFT la val_loss ne porte que sur les réponses.

**Reprise** — `ckpt_XXXXXXX.pt` contient tout (modèle, optimiseur, GradScaler, itération, best, tokens vus) ; l'ordre des données est une fonction de
`(seed, numéro de batch)` : on repart exactement où on s'est arrêté (test : 20 it. d'un coup = 10 + reprise + 10, mêmes poids). Écritures atomiques.
`--resume auto` (défaut) reprend le plus récent ; `--init_from` charge des poids seuls (SFT) avec un optimiseur neuf. `--max_minutes` sauvegarde et s'arrête proprement.

**Garde-fou tokenizer (v2.1)** — l'empreinte du tokenizer est écrite dans `meta.json` et les checkpoints. Reprendre ou affiner avec un tokenizer différent
(même taille de vocabulaire, autre contenu : erreur silencieuse) est **refusé** avec le message « Tokenizer incompatible ». Les checkpoints v2 sans empreinte restent acceptés.

**Fichiers produits dans `out_dir`** : `best.pt` (poids seuls) · `final.pt` (dernière it.) · `ckpt_*.pt` (état complet, `--keep 2`) · `best.json` · `log.jsonl` · `tokenizer.json`.

**Pour reprendre ailleurs** (Colab → Kaggle, ou autre compte Drive) il faut **le même jeu** : `tokenizer.json`, `pretrain/{train.bin,val.bin,meta.json}`, le dernier
`ckpt_*.pt`, `best.pt`, `best.json`, `log.jsonl`. Ne reconstruis jamais les données pour un checkpoint existant.

**Précision** — `auto` : bf16 seulement sur GPU Ampere+ ; sinon fp16 + GradScaler (T4/P100). Jamais de bf16 émulé.

---

## Personnalité, SFT, génération

**Personnalité** — `personnalite.jsonl` (132 Q/R d'**identité et de caractère** : « Comment tu t'appelles ? » → « Je suis MiniLLM… ») est ajouté au SFT, toujours en `train`
(jamais en `val`) et répété 8 fois (≈ 3 % du SFT). Les **faits** n'y sont plus : ils sont dans `knowledge/` (RAG, modifiables sans ré-entraîner) et `datasets/` (optionnel au SFT).
Règles d'écriture et fiche de personnage : `PERSONNALITE.md` ; vérification avant entraînement : `python check_data.py personnalite.jsonl datasets/*.jsonl`. **Pas de system prompt** : la personnalité vit dans les poids (un modèle de cette taille suit mal les consignes, et le contexte de 512 tokens reste libre).
Vérification : `python evaluate.py persona --ckpt …` (F1 ≈ 1 = appris par cœur ; les reformulations se jugent avec `evaluate.py demo`).
Pour la changer : édite le fichier, supprime `data/sft`, relance `sft_data.py` puis le SFT (pas besoin de refaire le pré-entraînement).

**Fine-tuning** — loss **uniquement sur la réponse**, exemples jamais tronqués, padding à droite (sans effet grâce au masque causal), 3 époques max avec early stopping.

**Génération** — KV-cache (≈ 6× plus rapide que le recalcul complet dès 200 tokens, mesuré sur CPU ; l'écart grandit avec la longueur), arrêt sur `<|end|>`, jamais de token de
rôle dans la réponse, pénalité de répétition + interdiction des trigrammes répétés **appliquées au texte généré uniquement** (pas de blocage quand on recopie le contexte).

---

## Mini-RAG (expérimental)

Un modèle de 49 M de paramètres n'a presque aucune connaissance fiable, mais le SFT lui apprend (via PIAF) à répondre « à partir du texte ». `rag.py` retrouve donc
le bon paragraphe dans **ta** base (BM25 unigrammes + bigrammes, sans dépendance) et le lui donne au format du SFT :
`Réponds à la question à partir du texte.\n\nTexte : …\n\nQuestion : …`.

```bash
python rag.py --ckpt checkpoints/sft/best.pt --kb knowledge datasets    # interactif ; le passage utilisé est toujours affiché
```
**Architecture** : `rag.py` = (1) *retriever* BM25 sans IA, (2) le modèle SFT qui lit le passage et répond. **Découpage (chunking)** : un morceau = un paragraphe (bloc séparé par une ligne vide) ;
< 20 caractères ou titre `#` : ignoré ; > 700 caractères : coupé par phrases ; pas de chevauchement ; une ligne `.jsonl` = un texte, ou la *réponse* d'une Q/R. **Recherche** : mots sans accents/majuscules/mots vides,
unigrammes + bigrammes, 1 seul meilleur passage ; **garde-fou** : il faut que **plus de la moitié** des mots utiles de la question soient dans le passage, sinon aucun passage n'est renvoyé.
À l'affichage, un passage > 600 caractères est recentré sur la phrase la plus proche de la question, puis réduit pour tenir dans les 512 tokens.
* Base fournie (`knowledge/` + `datasets/`, ~270 passages) : `minillm.txt` (le projet), `ia_bases.txt` (définitions IA), `cameroun.txt`, `capitales_monde.txt` (95 pays), `faits_cameroun_afrique.jsonl`.
* **Ajouter des infos** : mets un `.txt`/`.md` dans `knowledge/` (ou un `.jsonl`), **un fait par paragraphe, sujet dans la phrase** (« La capitale du Cameroun est Yaoundé. », pas « Elle est Yaoundé. »). Aucun ré-entraînement.
* Sans passage pertinent, le modèle répond « librement » (et peut inventer). Pas de recherche par sens : « président » ne retrouve pas « chef de l'État » ; un passage voisin mais sans la réponse peut encore être renvoyé si la question partage plus de la moitié de ses mots.
* ⚠️ Testé : retrouver le bon passage — 34/34 sur mes 34 questions (dont 3 sans réponse à rejeter) : **évaluation optimiste**, j'ai écrit la base et les questions. **Non mesuré** : la qualité des réponses du modèle (dépend du SFT) → juge sur le passage affiché.

---

## Réglages utiles

| envie | option |
|---|---|
| plus de tokens/s | 2 GPU (`torchrun`), `--compile` (à tester), `--batch_size` maximal qui tient en mémoire |
| out of memory | `--batch_size 8` (le notebook ajuste `--grad_accum` pour garder le batch effectif) |
| entraînement interruptible à durée libre | `--schedule wsd` : LR stable puis décroissance sur les `--decay_frac` (20 %) derniers % ; `cosine` exige de connaître `--max_iters` d'avance |
| changer `--max_iters` en cours de route (ex. 10 000 → 8 000) | possible à la reprise : le LR est recalculé pour la nouvelle durée (petit saut, ex. ~4,3e-4 → ~3,5e-4, sans risque) |
| modèle plus petit / plus rapide | `--model_size 23M` avec `--lr 1e-3` |
| réponses plus « sages » | `--temperature 0.3` dans `generate.py` |
| tes propres Q/R | `sft_data.py --extra_jsonl mes_qr.jsonl` |
| changer la personnalité | voir plus haut |
| re-nettoyer un corpus existant | `prepare_data.py tokenize --out data --refilter` (nouveau run seulement) |

---

## Repères observés (run réel)

Colab T4, 49M, fp16, batch effectif 65 536 tokens/it., corpus 335,7 M tokens : **~21,3 k tok/s constants (≈ 3,05 s/it.)**, gnorm stable ~0,32, aucun pic.

| itération | tokens vus | val_loss | perplexité |
|---|---|---|---|
| 500 | 33 M | 4,61 | 100,7 |
| 1 000 | 66 M | 3,99 | 54,3 |
| 2 000 | 131 M | 3,69 | 39,9 |
| 3 000 | 197 M | 3,56 | 35,0 |
| 4 000 | 262 M | 3,47 | ~32 |

Chaque évaluation est un nouveau « best » (la val fixe fonctionne). Textes du modèle de base à l'it. 4000 : français correct, registre cohérent, faits inventés,
répétitions au bout de quelques phrases, ne répond pas aux questions (normal avant SFT).

---

## Dépannage

| symptôme | cause / solution |
|---|---|
| `Tokenizer incompatible` | les données ne sont pas celles du checkpoint : reprends **les mêmes** `tokenizer.json` / `.bin` |
| `Vocabulaire incompatible` | même remède (tokenizer de taille différente) |
| `CUDA out of memory` | `--batch_size 8` |
| `Loss non finie` (bf16/fp32) | baisse `--lr` de ~30 % et relance (reprise automatique) ; en fp16 le GradScaler saute les pas |
| `gdown` : « Cannot retrieve the public link » | fichier non partagé en « Toute personne disposant du lien », ou quota Drive : réessaie plus tard |
| fichier tronqué (`fichier tronqué/corrompu`) | re-télécharge ce fichier |
| `torchrun` bloque sur Kaggle | `N_GPU = 1` dans la cellule 2, ou `NCCL_P2P_DISABLE=1` (déjà positionné) |
| Colab : limite de GPU atteinte | reprends sur Kaggle (voir plus haut) : même checkpoint, mêmes données |
| réponses en boucle | `--temperature 0.3` ; `no_repeat_ngram=3` et `repetition_penalty=1.1` sont déjà actifs |

---

## Tests

```bash
python -m pytest -q          # 44 tests, CPU, ~1 min : KV-cache = forward complet, nombre de paramètres exact, masque de loss, nettoyage,
                             # tokenizer, loaders, reprise exacte, best jamais écrasé, DDP 2 processus, continuité 1↔2 GPU, RAG, outils Kaggle…
python smoke_test.py         # pipeline complet sur un mini modèle (corpus local -> tokenizer -> pretrain -> reprise -> SFT -> génération)
```

## Limites connues et idées non faites

* Validé sur **CPU** (tests, DDP en gloo, mini-entraînement Q/R de bout en bout) et sur le run Colab T4 ci-dessus. **Jamais testés sur Kaggle** : le notebook Kaggle, `gdown`, `torchrun` sur T4×2 (le code est prudent et vérifie ses fichiers, mais lis les premières lignes).
* Les chargeurs HuggingFace (Wikipédia, FineWeb-2, French-Alpaca, OpenAssistant, PIAF) ne peuvent pas être testés hors-ligne : chacun est protégé (`try/except`) et le SFT continue avec les sources restantes.
* Le nettoyage v2.1 est testé sur des phrases types, **pas mesuré sur le vrai corpus** (utilise `--refilter` pour voir le pourcentage retiré).
* Le mini-RAG est expérimental (voir plus haut). Les Q/R synthétiques donnent la *forme* d'une conversation, pas de la connaissance ; l'arithmétique d'un très petit modèle reste fragile.
* Idées **non implémentées** (gains non validés) : modèle plus profond/étroit à taille égale, optimiseur Muon, davantage de tokens (le facteur limitant le plus probable), pondération par source.

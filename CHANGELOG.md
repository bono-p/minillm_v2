# Changelog — MiniLLM v2 (refonte après audit)

## SFT v3.2 — ablation A/B/C/D (branche `v2.1.1`)

Constat : le SFT v3.1 (+8 000 exemples synthétiques, source GPT-4 FR) n'a pas amélioré le modèle — calcul toujours non appris (0-7 %), persona 95 % → 59 %, régressions sur des faits simples, définitions qui s'effondrent sur une même phrase, réponses de refus (« Je suis désolé, je ne peux pas générer de texte »).

* **`sft_filters.py`** (sans torch) : filtre des sources externes — refus, boucles (« Le Louvre » ×5), échos de la question. `sft_data.py --quality_filter 1`. Jamais appliqué au synthétique ni à la persona.
* **`--persona_share 0.06`** : la personnalité garde ~6 % du train quelle que soit la taille du corpus (au lieu d'un nombre de copies fixe qui se diluait).
* **`--plus_light 1`** : `synthetic_plus` sans définitions, suites ni problèmes (0-5 % de réussite) ; ~1 000 calculs.
* **`basics_eval.py`** : test de régression sur 36 connaissances de base (mots entiers : « 7 » ne valide pas « 17 »).
* **`compare_sft.py`** : tableau unique persona / qa / basics / % de réponses distinctes / refus / boucles, sur un jeu de validation commun ; compare `final.pt` ET `best.pt` (le `best.pt` choisi par val_loss tombe en milieu d'entraînement et mémorise moins la persona).
* **Notebook** `colab/SFT_ablation.ipynb` : 4 variantes (A témoin v3 / B +plus allégé / C +filtre +persona 6 % / D +GPT-4 FR filtré), chacune dans `checkpoints/sft_X`.
* **Tests** : 4 nouveaux (filtres, mode allégé sans fuite vers l'éval, scoring en mots entiers, lecture des sorties d'evaluate).
* **Mesure `top5g`** (compare_sft) : nombre de réponses de l'openqa qui partagent le même 5-gramme de mots ; remplace le taux de réponses distinctes, qui restait à 100 % même quand une définition était resservie pour une douzaine de notions. `basics_<nom>.txt` liste les connaissances de base ratées.
* **`colab/Compare_publie.ipynb`** : compare le modèle publié sur Hugging Face (téléchargé, rien n'est modifié sur le dépôt) à C/B/A `final.pt` dans un seul tableau ; `compare_sft.py` affiche aussi la phrase la plus reprise de chaque modèle.
* **Correctif** : `compare_sft.py` recopie `tokenizer.json` à côté du checkpoint s'il manque (le notebook d'ablation ne le fournissait pas : les 8 évaluations échouaient), s'arrête avec un message clair au lieu d'afficher un tableau de zéros, et affiche `-` quand une mesure n'a pas abouti.

## SFT v3.1 (branche `v2.1.1`)

* **`synthetic_plus.py`** : ~8 000 exemples calculés par code (additions/soustractions jusqu'à 100, tables de multiplication/division, pourcentages, petits problèmes avec le calcul écrit avant le résultat, suites de nombres, intrus, logique, calendrier, langue, 65 définitions courtes). Activé par `sft_data.py --plus 1`.
* **Pas de fuite vers l'évaluation** : les questions de `UNSEEN_QA_PROMPTS` sont exclues de `synthetic_plus`, et ~10 % des problèmes chiffrés (`is_heldout`) ne sont JAMAIS entraînés — y compris dans l'ancien synthétique (`drop_heldout_arithmetic`).
* **`skills_eval.py`** : exactitude sur ces problèmes tenus à l'écart (addition, soustraction, multiplication, division, suites, problèmes) — compare deux checkpoints sur exactement les mêmes problèmes.
* **Correctif `sft_data.py`** : `load_alpaca_style` lit aussi le format ShareGPT (`conversations`: human/gpt). Avant, `alpaca-gpt4-french` ne chargeait aucun exemple sans le dire ; le chargeur affiche maintenant les colonnes du dataset.
* **Tests** : 3 nouveaux (exactitude des réponses calculées, absence des questions de test, absence des problèmes tenus à l'écart).

## v2.1 (branche `v2.1`)

* **Kaggle** : `MiniLLM_v2_Kaggle.ipynb` + `kaggle_utils.py` (restauration depuis `/kaggle/input`, Google Drive via `gdown`, vérification des tailles de `.bin` et du checkpoint, élagage de la sortie).
* **Garde-fou tokenizer** : `tokenizer_sha` dans `meta.json` (pré-entraînement et SFT) et dans les checkpoints ; reprise/`init_from` refusés si les empreintes diffèrent.
* **Nettoyage** : motifs `(en latin : )`, `(le )`, guillemets vides ; recollage des élisions (`L' archidiocèse` → `L'archidiocèse`) ; filtre anti-spam web (≥ 3 termes de boilerplate) ; `prepare_data.py tokenize --refilter`.
* **Mini-RAG** : `rag.py` (BM25 unigrammes + bigrammes, passage centré sur la question, prompt au format PIAF, tient dans la fenêtre du modèle) + `knowledge/exemple.txt`.
* **Évaluation** : `evaluate.py persona`.
* **Logs** : ETA et mémoire GPU (`log.jsonl` inclus).
* **SFT** : PIAF complet par défaut (4 000 max).
* **Tests** : 44 (continuité du flux de données entre 1 et 2 GPU, RAG, outils Kaggle, garde-fou tokenizer, refilter de bout en bout…).
* **Personnalité redessinée** : `personnalite.jsonl` = identité + caractère (132 Q/R, une seule règle de ton, plus de faits ni de contradictions « pas de sentiments » / « je suis fier ») + `PERSONNALITE.md` (fiche de personnage et règles d'écriture). `--persona_repeat` 8 par défaut.
* **Faits séparés** : `datasets/faits_cameroun_afrique.jsonl` (109 Q/R nettoyées : doublons, faits qui vieillissent retirés, réponses corrigées) — utilisable au SFT (`--extra_jsonl`, nouveau `--extra_repeat`) et par le RAG.
* **RAG** : base de connaissances élargie (`minillm.txt`, `ia_bases.txt`, `cameroun.txt`, `capitales_monde.txt`), Q/R `.jsonl` acceptées (la réponse sert de passage), `--kb` multi-sources, garde-fou de couverture des mots de la question, mots vides conversationnels.
* **`check_data.py`** : vérifie les `.jsonl` (JSON, champs, doublons, faits qui vieillissent…).
* Compatibilité : aucun changement d'architecture ni de format de checkpoint (champ optionnel `tokenizer_sha`) : les checkpoints et données v2 restent utilisables.

Chaque ligne relie un problème constaté à l'audit à sa correction et au fichier concerné.

## Sauvegarde du « meilleur » modèle

| Problème | Correction | Où |
|---|---|---|
| Chaque évaluation lisait une **autre tranche** du val (1/16 à chaque fois) → la val_loss variait de ±0,1 sans que le modèle change (2,97 → 3,14 → 3,02 avec un LR quasi constant) | **Val fixe** : mêmes séquences à chaque eval, espacées dans `val.bin` | `data.py` (`val_batches`, `eval_batches`) |
| `best_val_loss = inf` à chaque reprise → le 1er eval écrasait `best.pt` même s'il était moins bon | Meilleure val persistée (`best.json` + dans le checkpoint) et relue à la reprise | `checkpoint.py`, `train.py` |
| Reprise depuis `best.pt` (progression perdue) | Reprise depuis le **dernier checkpoint complet** ; `best.pt` = poids seuls | `checkpoint.py`, `train.py` |
| Le loader repartait à 0 à chaque reprise (début du corpus relu, fin jamais vue) | Ordre des données = fonction pure de `(seed, n° de batch)` ; époques sans remise | `data.py` |
| Nettoyage rolling trié alphabétiquement (pouvait supprimer les checkpoints récents) | Tri numérique | `checkpoint.py` |
| Écriture non atomique (session coupée = fichier corrompu) | Fichier temporaire + `os.replace` | `checkpoint.py` |
| Hack `reset_iter` pour le SFT, optimiseur du pré-entraînement rechargé | `init_from` (poids seuls, optimiseur neuf) | `train.py` |
| SFT : val_loss dominée par le contexte | Loss/val **uniquement sur les réponses** ; early stopping ; EM/F1 en plus | `train.py`, `evaluate.py` |
| Courbes = rechargement de tous les checkpoints | `log.jsonl` | `train.py` |

## Qualité des phrases et logique question → réponse

| Problème | Correction | Où |
|---|---|---|
| Wikipédia **trouée** (« né le  à Moulins », « du . ») apprise par le modèle | Suppression des phrases défectueuses, sections de fin coupées, titres retirés | `text_cleaning.py` |
| 128 k articles devenus 4,19 M « articles » ; val = paragraphes des mêmes articles que train | Split **par document** (hash), un document = un article | `prepare_data.py` |
| Pas de token de fin de document (le modèle n'apprenait jamais à s'arrêter) | `<|endoftext|>` après chaque document | `prepare_data.py` |
| SFT sur PIAF seul : extraction de réponses de 1–5 mots avec contexte obligatoire | Mélange : Q/R synthétiques propres + French-Alpaca (réponses courtes) + OpenAssistant FR + PIAF | `sft_data.py`, `synthetic_qa.py` |
| Loss sur toute la séquence (réponse ≈ 2–4 % des tokens) | Masque : loss sur la réponse seule (≈ 47 % des tokens d'un jeu synthétique) | `mini_tokenizer.py`, `data.py`, `model.py` |
| Contexte PIAF coupé à 800 caractères (la réponse pouvait disparaître) | Fenêtre **centrée sur la réponse**, vérifiée | `sft_data.py` |
| 29 époques sur les mêmes 850 k tokens (mémorisation) | 3 époques, early stopping, ~10× plus d'exemples (jusqu'à ~35 000 au lieu de 3 835) | `config.py`, `sft_data.py` |
| `generate.py` (chat) envoyait du texte brut, sans le template du SFT | Chat = template exact du SFT | `generate.py` |
| Pas de stop token → la génération continuait après la réponse | Arrêt sur `<|end|>` / `<|endoftext|>` | `generate.py` |

## Architecture et vitesse

| Problème | Correction | Où |
|---|---|---|
| Vocab 100 277 : embeddings = 62 % des paramètres et ~62 % du calcul du forward | Tokenizer BPE **32 000** entraîné sur le corpus : ~1,7× plus rapide par token, ids en `uint16` | `mini_tokenizer.py` |
| Presets mal nommés (« 15M » = 49M réels, « 50M » = 83M…) | Nom **calculé** depuis le nombre réel de paramètres, vérifié par un test | `config.py` |
| bf16 sur T4 (émulé : 3,3 k tok/s contre 13,8 k en fp16) | fp16+GradScaler sauf GPU Ampere+ | `train.py` |
| RoPE en complexes : `torch.compile` retombe en eager | RoPE réel (cos/sin) | `model.py` |
| Génération O(n²) (recalcul complet à chaque token) | **KV-cache** pré-alloué | `model.py`, `generate.py` |
| Kaggle T4×2 : un seul GPU utilisé | DDP (`torchrun`), val réduite entre GPU | `train.py` |
| Stabilité fp16 | QK-norm, init résiduelle conservée | `model.py` |
| Presets globaux mutés (`mcfg.max_seq_len = …`), `rolling_keep` greffé hors dataclass, `config.py` réécrit à chaud par le notebook | `get_preset()` renvoie une copie ; tout est dans `TrainConfig` ; plus de patch de fichier | `config.py` |
| tok/s faux au 1er eval, test `model(x, x)` trompeur (loss 9,87 au lieu de 11,52) | tok/s mesuré entre deux logs ; sanity check avec cibles indépendantes | `train.py`, `inspect_model.py` |
| `torch.load(weights_only=False)` + config picklée | Config en dict simple, `weights_only=True` (repli explicite avec avertissement pour les anciens fichiers) | `checkpoint.py` |

## Personnalité

* `personnalite.jsonl` (30 Q/R sur le modèle : nom, créateur, caractère) ajouté au SFT : toujours en train, répété 20 fois (`--persona`, `--persona_repeat`).
* Les exemples French-Alpaca / OpenAssistant qui parlent de l'identité de l'assistant (« en tant qu'IA », ChatGPT, OpenAI…) sont écartés pour ne pas contredire la personnalité.
* Pas de system prompt : sur un modèle de cette taille la personnalité s'apprend dans les poids et le contexte (512 tokens) reste pour la conversation.

## Hygiène

* Deux notebooks divergents (DevLab / Kaggle, sorties et chemins mélangés) → **un seul** `MiniLLM_v2.ipynb`.
* Ajout : `.gitignore`, `requirements.txt`, tests (`tests/`, `smoke_test.py`), `evaluate.py`, `push_to_github.py` (token par variable d'environnement).

## Compatibilité

Les anciens checkpoints (vocab 100 277, RoPE complexe) **ne sont pas compatibles** avec cette version : il faut ré-entraîner
avec le nouveau tokenizer. C'est voulu : les anciennes données contenaient les trous décrits plus haut.

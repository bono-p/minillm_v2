# Changelog — MiniLLM v2 (refonte après audit)

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

# miniLLM v2.3 — plan : 5 milliards de tokens, sources variées, contexte 1024

Branche `v2.3`. Objectif : mesurer ce que changent **des données plus variées et plus nombreuses** (et un contexte plus long) pour le 49M, avant de refaire la même expérience en 110M.

## 1. Ce qui change, et ce qui ne change pas (volontairement)

| | v2.2 | v2.3 (essai 49M) |
|---|---|---|
| tokens d'entraînement | 3,29 Md (1,02 époque) | **5,0 Md** |
| sources | Wikipédia FR + web (FineWeb-2) | web/wiki **33 %**, livres et presse **30 %**, web ouvert (transcriptions, StackExchange) **12 %**, conversations **9 %**, science **8 %**, administratif **7 %** |
| contexte | 512 | **1 024** |
| architecture | 10 couches, d_model 512, 8 têtes | **identique** (têtes : essai séparé, voir §5) |
| tokenizer | BPE 32 000 | **identique** (corpus v2.2 réutilisé, sans retokeniser ; les jetons `<|user|>`, `<|assistant|>`, `<|end|>` existent déjà) |

Une seule famille de variables à la fois : données + contexte. Changer aussi le nombre de têtes rendrait impossible de dire ce qui a aidé.

## 2. Planning WSD et deux jeux de données

Le planning *warmup-stable-decay* est déjà implémenté. Le mélange est séparé en deux dossiers :

- **`stable/` (4,25 Md, 85 %)** : mélange large ; le learning rate reste à 6e-4.
- **`decay/` (0,75 Md, 15 %)** : texte moderne de qualité, **conversations multi-tours**, questions-réponses (PIAF), pendant que le learning rate descend à 6e-5.

Pourquoi : les données de style « instruction » ne sont pas mélangées à tout l'entraînement (où elles seraient diluées et marqueraient le modèle trop tôt) mais placées là où le modèle se stabilise. Les livres du domaine public, souvent en français ancien, sont compensés par la phase finale en français moderne. L'arrêt entre les deux phases se fait avec `train.py --stop_at <itération>` (nouveau) : checkpoint complet, puis reprise avec `--data_dir decay`.

## 3. Sources et licences

| source | licence | rôle | statut de la vérification |
|---|---|---|---|
| corpus v2.2 déjà tokenisé (Wikipédia FR, FineWeb-2 FR) | déjà utilisé | base, 1,67 Md | vérifié (c'est le tien) |
| Common Corpus (`PleIAs/common_corpus`) : OpenCulture (livres, presse), OpenWeb, OpenScience, OpenGovernment, filtre langue française | permissive (fiche Hugging Face) | livres, web ouvert, science, administratif | licence vérifiée ; **colonnes et organisation des fichiers non vérifiées** (cellule `inspect`) |
| `french_instruct` (miroir `MaziyarPanahi/french_instruct_sharegpt`) | MIT (original `angeluriot/french_instruct`) ; ≈ 275 000 conversations, ≈ 85 M tokens | conversations multi-tours | fiche vérifiée ; une partie traduite via l'API ChatGPT : voir les conditions d'utilisation |
| `vonewman/french-instruction-dataset` | Apache-2.0 ; 108 991 conversations | conversations | fiche vérifiée |
| PIAF (`etalab-ia/piaf`) | déjà utilisé au SFT | questions-réponses | identifiant repris du SFT |
| **exclus** : Lucie-Training-Dataset, Claire (dialogues) | **CC BY-NC-SA 4.0** (usage non commercial) | — | vérifié ; incompatible avec une publication MIT. `--allow_nc` existe pour une variante de recherche **non publiée** |

**Limite honnête sur les conversations.** Les données de conversation en français sous licence permissive représentent ≈ 130 M tokens (estimation). À 9 % de 5 Md (≈ 0,45 Md), elles sont donc vues **≈ 3 fois chacune**. C'est acceptable (quelques répétitions ne coûtent presque rien), mais on ne peut pas aller plus haut sans nouvelles données : traduction de conversations anglaises, génération synthétique, ou OpenAssistant (messages en français). Le plan avertit dès qu'une source dépasse son nombre de répétitions autorisé.

## 4. Protections

- **Décontamination** : les conversations dont un message utilisateur est exactement l'une de nos questions d'évaluation (337 questions ouvertes, 204 de connaissances, 36 de base, problèmes chiffrés, persona) sont écartées. Limite : correspondance exacte, pas de reformulation ; non appliquée aux livres et pages web.
- **Qualité** : filtre OCR pour les livres (lettres, mots français courants, répétitions), recollage des mots coupés ; refus génériques, boucles et échos écartés des conversations (mêmes filtres que le SFT).
- **Validation** : `val.bin` = la validation de la v2.2 (la `val_loss` reste comparable à 3,0431) ; les documents des nouvelles sources sont dans `val_new.bin` (non évalués pendant l'entraînement).
- **Reprise** : les fichiers sont tronqués à la dernière position enregistrée ; une interruption donne le même résultat qu'un run sans coupure (testé).
- **Espace disque** : refus de démarrer si le Drive est trop plein.

## 5. Contexte 1024 et nombre de têtes

- **Contexte 1024** : un simple réglage (`--seq_len 1024`, la RoPE s'adapte). Utile pour les livres et les conversations longues. Coût : un peu plus de calcul par token.
- **Plus de têtes** : à `d_model = 512`, passer de 8 à 16 têtes réduit chaque tête à 32 dimensions ; le nombre de paramètres ne change pas. Pas de gain attendu : à tester sur un **essai court (≈ 300 M tokens)** avec `--n_heads 16`, à budget égal, avant de s'y engager. Pour le 110M (d_model 768, 12 couches), 12 têtes de 64 dimensions viennent naturellement.

## 6. Estimations (à vérifier avec tes propres mesures)

- **Disque** : 5 Md × 2 octets = **10 Go** (+ le corpus v2.2 de 6,5 Go déjà présent). Tient dans 30 Go.
- **Temps** : `jours = tokens / (tokens_par_seconde × 86 400)`. À 8 000 / 12 000 / 20 000 tokens/s : 7,2 / 4,8 / 2,9 jours de GPU continu pour 5 Md (14 / 9,6 / 5,8 pour 10 Md). Le débit réel du pré-entraînement n'est pas dans ce dépôt : lis `tok/s` dans tes logs.
- **Gain attendu sur la perte** (ordre de grandeur, ajustement de Chinchilla à taille fixe, pas transférable exactement à notre tokenizer ni à ce mélange) : environ −0,10 de `val_loss` à 5 Md, −0,24 à 10 Md. Dans notre expérience, −0,16 de `val_loss` (v2.1 → v2.2) n'a pas suffi à améliorer significativement les connaissances : le gain mesuré est venu du SFT. Un modèle de 49M a aussi une capacité de connaissance très limitée (de l'ordre de 2 bits par paramètre selon la littérature) : plus de tokens aident surtout la langue ; les faits dépendront davantage de la taille du modèle.
- **Budget extensible** : avec WSD, on peut faire la phase de décroissance à n'importe quel moment depuis le plateau. Stable jusqu'à 5 Md, décroissance courte ; ou continuer le plateau jusqu'à 10 Md si le disque (100 Go demain) et le temps le permettent.

## 7. Déroulé

1. `colab/Prepare_v23_data.ipynb` : plan → **inspection** (vérifier colonnes et filtres) → construction reprenable.
2. Essai court optionnel (têtes 8 vs 16, contexte 512 vs 1024) sur ≈ 300 M tokens.
3. `colab/Pretrain_v23.ipynb` : phase stable (`--stop_at`), puis phase decay.
4. SFT avec la recette de la v2.2 (inchangée), puis `compare_sft.py` + `knowledge_eval.py` : comparer à la v2.2 avec les mêmes mesures.
5. Même chose en 110M.

## 8. Ce qui reste à valider

Colonnes et filtres de Common Corpus (étape `inspect`), vitesse de lecture en streaming, tailles réelles des sous-ensembles français, qualité réelle des livres après filtrage (échantillon à lire), et le pourcentage de documents rejetés par source (`mix_report.json`).

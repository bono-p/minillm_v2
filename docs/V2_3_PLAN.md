# miniLLM v2.3 — plan : 5 milliards de tokens, sources variées, contexte 1024

Branche `v2.3`. Objectif : mesurer ce que changent **des données plus variées et plus nombreuses** (et un contexte plus long) pour le 49M, avant de refaire la même expérience en 110M.

## 1. Ce qui change, et ce qui ne change pas (volontairement)

| | v2.2 | v2.3 (essai 49M) |
|---|---|---|
| tokens d'entraînement | 3,29 Md (1,02 époque) | **5,0 Md** |
| sources | Wikipédia FR + web (FineWeb-2) | web/wiki **33 %**, livres et presse **30 %**, web ouvert (transcriptions, StackExchange) **12 %**, conversations **9 %**, science **8 %**, administratif **7 %** |
| contexte | 512 | **1 024** |
| architecture | 10 couches, d_model 512, 8 têtes | **identique** (têtes : essai séparé, voir §5) |
| tokenizer | BPE 32 000 | **identique** (récupéré sur le dépôt Hugging Face v2.2 ; les jetons `<|user|>`, `<|assistant|>`, `<|end|>` existent déjà) |

Une seule famille de variables à la fois : données + contexte. Changer aussi le nombre de têtes rendrait impossible de dire ce qui a aidé.

## 2. Planning WSD et deux jeux de données

Le planning *warmup-stable-decay* est déjà implémenté. Le mélange est séparé en deux dossiers :

- **`stable/` (4,25 Md, 85 %)** : mélange large ; le learning rate reste à 6e-4.
- **`decay/` (0,75 Md, 15 %)** : texte moderne de qualité, **conversations multi-tours**, questions-réponses (PIAF), pendant que le learning rate descend à 6e-5.

Pourquoi : les données de style « instruction » ne sont pas mélangées à tout l'entraînement (où elles seraient diluées et marqueraient le modèle trop tôt) mais placées là où le modèle se stabilise. Les livres du domaine public, souvent en français ancien, sont compensés par la phase finale en français moderne. L'arrêt entre les deux phases se fait avec `train.py --stop_at <itération>` (nouveau) : checkpoint complet, puis reprise avec `--data_dir decay`.

## 3. Sources et licences

| source | licence | rôle | statut de la vérification |
|---|---|---|---|
| Wikipédia FR (`wikimedia/wikipedia`, 20231101.fr) et FineWeb-2 (`fra_Latn`), en streaming | CC BY-SA ; ODC-By | base, 1,67 Md | identifiants et nettoyages repris du pipeline v2.2 (le Drive ayant été vidé, on repart de zéro) |
| Common Corpus (`PleIAs/common_corpus`), filtré sur `open_type` (« Open Culture », « Open Web », « Open Science », « Open Government ») et `language` = « French » | permissive (fiche Hugging Face) | livres, web ouvert, science, administratif | licence vérifiée ; `open_type` et `language` confirmés sur un exemple réel (10 octobre 2026, `open_type="Open Government"`, `language="French"`) ; « Open Culture », « Open Science » et « Open Web » restent des noms déduits de la documentation, pas vus directement — `prepare_mix.py build` sonde chaque filtre avant de consommer le moindre quota et s'arrête en listant les valeurs réelles s'il ne trouve rien |
| `french_instruct` (miroir `MaziyarPanahi/french_instruct_sharegpt`) | MIT (original `angeluriot/french_instruct`) ; ≈ 275 000 conversations, ≈ 85 M tokens | conversations multi-tours | fiche vérifiée ; une partie traduite via l'API ChatGPT : voir les conditions d'utilisation |
| `vonewman/french-instruction-dataset` | Apache-2.0 ; 108 991 conversations | conversations | fiche vérifiée |
| PIAF (`etalab-ia/piaf`) | déjà utilisé au SFT | questions-réponses | identifiant repris du SFT |
| **exclus** : Lucie-Training-Dataset, Claire (dialogues) | **CC BY-NC-SA 4.0** (usage non commercial) | — | vérifié ; incompatible avec une publication MIT. `--allow_nc` existe pour une variante de recherche **non publiée** |

**Limite honnête sur les conversations.** Les données de conversation en français sous licence permissive représentent ≈ 130 M tokens (estimation). À 9 % de 5 Md (≈ 0,45 Md), elles sont donc vues **≈ 3 fois chacune**. C'est acceptable (quelques répétitions ne coûtent presque rien), mais on ne peut pas aller plus haut sans nouvelles données : traduction de conversations anglaises, génération synthétique, ou OpenAssistant (messages en français). Le plan avertit dès qu'une source dépasse son nombre de répétitions autorisé.

## 4. Protections

- **Décontamination** : les conversations dont un message utilisateur est exactement l'une de nos questions d'évaluation (337 questions ouvertes, 204 de connaissances, 36 de base, problèmes chiffrés, persona) sont écartées. Limite : correspondance exacte, pas de reformulation ; non appliquée aux livres et pages web.
- **Qualité** : filtre OCR pour les livres (lettres, mots français courants, répétitions), recollage des mots coupés ; refus génériques, boucles et échos écartés des conversations (mêmes filtres que le SFT).
- **Validation** : 3 pour mille des documents de chaque source sont écartés de l'entraînement et servent de `val.bin`. Cette `val_loss` n'est **pas comparable** à celle des versions précédentes (3,0431) : l'ancienne validation n'existe plus et la distribution est différente.
- **Reprise** : les fichiers sont tronqués à la dernière position enregistrée ; une interruption donne le même résultat qu'un run sans coupure (testé).
- **Espace disque** : refus de démarrer si le Drive est trop plein.

## 5. Contexte 1024 et nombre de têtes

- **Contexte 1024** : un simple réglage (`--seq_len 1024`, la RoPE s'adapte). Utile pour les livres et les conversations longues. Coût : un peu plus de calcul par token.
- **Plus de têtes** : à `d_model = 512`, passer de 8 à 16 têtes réduit chaque tête à 32 dimensions ; le nombre de paramètres ne change pas. Pas de gain attendu : à tester sur un **essai court (≈ 300 M tokens)** avec `--n_heads 16`, à budget égal, avant de s'y engager. Pour le 110M (d_model 768, 12 couches), 12 têtes de 64 dimensions viennent naturellement.

## 6. Estimations (à vérifier avec tes propres mesures)

- **Disque** : 5 Md × 2 octets = **10 Go**, rien d'autre sur le Drive. Tient dans 30 Go.
- **Durée de construction** : non mesurée (streaming + nettoyage Python + tokenisation sur CPU Colab) ; compte plusieurs heures, probablement plusieurs sessions. La commande est reprenable.
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

« Open Culture », « Open Science » et « Open Web » comme valeurs exactes du champ `open_type` (sondées automatiquement au lancement de `build`, avant toute consommation de quota) ; vitesse de lecture en streaming ; tailles réelles des sous-ensembles français ; qualité réelle des livres après filtrage (échantillon à lire) ; pourcentage de documents rejetés par source (`mix_report.json`).

## 9. Commandes utiles en cas de filtre qui ne matche rien

```
python prepare_mix.py build --out data/v23 --tokenizer data/tokenizer.json --budget 5e9   # sonde les filtres puis s'arrête si un seul est vide
python prepare_mix.py tree --repo PleIAs/common_corpus --depth 2                            # organisation des fichiers (collections)
```

Si la sonde échoue, corrige `filters={"open_type": ...}` dans `mixplan.py` avec la valeur réelle qu'elle affiche, puis relance `build` : la commande est reprenable, rien de ce qui a déjà été écrit n'est perdu.


## 10. Décalage de synchronisation Google Drive (gros fichiers, entre deux sessions)

Vécu le 11/10/2026 : une session de `build` a bien écrit et confirmé localement (`os.fsync`, toutes les ~2 M tokens) un
`stable/train.bin` de 8,36 Go. Une **nouvelle** session Colab, ouverte ensuite pour l'entraînement, a remonté le Drive et vu
une version plus ancienne et plus petite du même fichier. `os.fsync` garantit l'écriture dans le cache local du montage
FUSE Drive de la session qui écrit ; il ne garantit pas que la copie cloud soit immédiatement visible depuis une AUTRE
session. Pour un fichier de plusieurs Go, ce délai peut atteindre 15 à 30 minutes.

**Ce n'est pas un problème de fréquence d'écriture** (le moteur checkpoint déjà toutes les ~2 M tokens, soit des centaines de
fois par fichier) : écrire plus souvent ne change rien au délai de synchronisation Drive lui-même, et ferait même plus
d'appels fsync pour un gain nul.

**Procédure à suivre :**
1. Après un `build` qui a beaucoup écrit, laisse quelques minutes avant d'ouvrir une nouvelle session dessus.
2. Dans la nouvelle session, avant d'entraîner : `drive.mount('/content/drive', force_remount=True)`.
3. `python prepare_mix.py verify --out data/v23` — compare la taille réelle des fichiers à celle enregistrée par `build`.
   - Tout correspond : les données sont bonnes, entraîne sans crainte.
   - Écart : attends encore quelques minutes et relance `verify` ; si l'écart persiste longtemps, relance `prepare_mix.py
     build` (reprenable, ne refait pas ce qui est déjà là).
4. `Pretrain_v23.ipynb` appelle maintenant `verify` automatiquement avant de calculer le planning, avec un message clair
   au lieu d'un `AssertionError` brut.


## 11. Écriture locale + synchronisation vérifiée (demande du 11/10/2026)

`prepare_mix.py build --local /content/v23_local --out <Drive>` écrit d'abord sur le disque rapide de la session (aucun
aléa de synchronisation), puis pousse vers `--out` (Drive) **après chaque source**, en ne recopiant que les octets
manquants et en **vérifiant immédiatement** (taille locale == taille Drive) avant de continuer — au lieu d'écrire
directement sur Drive et de découvrir un éventuel décalage seulement dans une autre session, plus tard.

`prepare_mix.py sync --local <local> --out <Drive>` est la « petite cellule » qui peut être relancée seule, à tout moment,
pour repousser et reconfirmer sans rien retélécharger. `colab/Prepare_v23_data.ipynb` l'inclut en cellule 4b.

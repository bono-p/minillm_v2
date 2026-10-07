---
language: fr
license: mit
library_name: pytorch
tags:
  - text-generation
  - french
  - small-language-model
  - from-scratch
  - minillm
  - transformer
pipeline_tag: text-generation
---

![miniLLM](./assets/banner.png)

# miniLLM v2.2 — 49M

![badges](./assets/badges.png)

**`bonopassale/miniLLM_v2.2-49M-50200it_3.29Btoks_1.0ep_20261005`**

Modèle de langage français entraîné from scratch par DevLab (Cameroun) : architecture, tokenizer et poids ont été développés et entraînés dans le cadre de ce projet, sans initialisation à partir des poids d'un modèle de langage préexistant.

L'objectif du projet n'est pas de rivaliser avec les grands modèles généralistes, mais de documenter précisément ce qu'un Transformer d'environ 49M de paramètres peut et ne peut pas faire lorsqu'il est entraîné sérieusement, avec des ressources de calcul limitées.

## Ce qui change par rapport à la v2.1

La v2.2 a la même architecture, le même tokenizer et le même corpus que la v2.1 (`bonopassale/miniLLM_v2.1-49M-42500it_2.79Btoks_0.9ep_20261001`, toujours disponible). Deux choses changent :

- **Pré-entraînement prolongé et terminé** : 50 200 itérations au lieu de 42 500, avec la phase finale de décroissance du learning rate qui manquait à la v2.1. `val_loss` : **3,1998 → 3,0431** (perplexité ≈ 24,5 → ≈ 21,0), sur le même jeu de validation fixe, donc directement comparable.
- **SFT refait** : sources externes filtrées, données synthétiques complémentaires, persona à poids fixe, 2 époques à learning rate décroissant, et publication du **dernier** checkpoint plutôt que de celui de plus faible `val_loss`.

Mesures directes v2.1 → v2.2, avec le même code d'évaluation :

| mesure | v2.1 | v2.2 |
|---|---|---|
| persona, Exact Match (132 exemples, appris par cœur) | 86,4 % | **99,2 %** |
| test de connaissances (204 questions hors SFT) | 10,8 % (22/204) | **20,6 %** (42/204) |
| connaissances de base (36 questions) | 61,1 % | 63,9 % |
| réponses de type refus (337 questions ouvertes, détecteur grossier) | 11 | 3 |
| calcul pur (24 questions ouvertes) | 1/24 | 2/24 |

Ce que ces chiffres permettent d'affirmer, et ce qu'ils ne permettent pas :

- Le gain au test de connaissances est **statistiquement net** (test exact de McNemar sur les mêmes questions : p = 0,001) et se retrouve sur trois entraînements identiques à la graine près (18,1 % à 20,6 %). Il n'apparaît **pas avec le seul pré-entraînement prolongé** : une recette de SFT antérieure (15 000 exemples French Alpaca, synthétique de base, sans filtre) appliquée au nouveau pré-entraînement donne 13,7 %, non significativement différent de la v2.1 (p = 0,31). Il est associé au SFT révisé (17,6 % à 21,6 % selon les variantes testées).
- Ce test mélange des connaissances du monde et 45 questions de langue (féminins, pluriels, contraires, synonymes, participes), dont 39 reprennent un **type de question enseigné par le SFT**, avec d'autres mots. Une part du gain reflète donc l'apprentissage de ces formats, et pas seulement du savoir. Pour la v2.2, la réussite par catégorie est : langue 35,6 % (16/45), culture 36,4 % (8/22), géographie 27,8 % (5/18), informatique 25,0 % (2/8), animaux 13,3 % (2/15), capitales absentes de l'entraînement 12,5 % (6/48), sciences 11,1 % (2/18), Cameroun 6,2 % (1/16), nombres 0 % (0/14). Les questions de langue fournissent donc 16 des 42 bonnes réponses : le gain n'est pas uniquement du savoir. La comparaison catégorie par catégorie avec la v2.1 n'a pas été faite.
- **Aucune amélioration** en calcul : 2 bonnes réponses sur 24 questions de calcul pur, comme avant.
- La persona à 99 % mesure la mémorisation de 132 exemples d'entraînement, pas une capacité nouvelle.

Détails, incertitudes et limites : voir « Méthodologie d'évaluation » et [`evals/resultats_automatiques.md`](./evals/resultats_automatiques.md).

## Caractéristiques du modèle

- **48 508 672 paramètres** (environ 32,1M hors embeddings)
- **10 couches Transformer**
- `d_model = 512`
- **8 têtes d'attention**
- **8 têtes K/V**
- **RMSNorm**
- **RoPE**
- **SwiGLU**
- **QK-Norm**
- Embeddings d'entrée et de sortie **partagés**
- Contexte : **512 tokens**
- Vocabulaire : **32 000 tokens**
- Tokenizer BPE byte-level, entraîné sur le corpus du projet
- Poids et tokenizer publiés intégralement dans ce dépôt

> Avec 8 têtes de requêtes et 8 têtes K/V, l'implémentation correspond à une attention multi-têtes classique (MHA), et non à du GQA au sens strict.

---

## Entraînement

### Pré-entraînement

- Corpus français : **≈ 3,23 milliards de tokens** (3 226 M), **4 567 182 documents** d'entraînement
- Sources : Wikipédia française (quasi complète) nettoyée + données web françaises issues de FineWeb-2
- **50 200 itérations**
- **3 289 956 352 tokens vus** (≈ 3,29 milliards)
- Environ **1,02 époque** sur le corpus : environ 2 % des blocs ont donc été vus deux fois
- Meilleure `val_loss` : **3,0431**, atteinte à la dernière itération
- Perplexité correspondante : **≈ 21,0**

![Courbe de loss — pré-entraînement](./assets/loss_pretrain.png)

Après la reprise, la loss de validation décroît régulièrement (hormis une hausse de 0,001 à l'itération 43 000, au tout début), y compris pendant la phase finale de décroissance du learning rate, et le meilleur point est la dernière évaluation.

Le pré-entraînement s'est déroulé en deux temps, ce qui se voit sur la courbe :

1. **Jusqu'à environ l'itération 42 700** : planning cosine prévu sur 280 000 itérations. L'entraînement a été interrompu à ce stade avec un learning rate encore d'environ 5,7e-4 (c'est le point qui avait été publié en v2.1, au checkpoint de l'itération 42 500).
2. **De l'itération ≈ 42 700 à 50 200** : l'entraînement a été repris depuis ce point avec un plan raccourci à 50 200 itérations (≈ 1 époque), selon un planning *warmup-stable-decay* : learning rate maintenu à 6e-4 jusqu'à l'itération 42 670, puis décroissance linéaire jusqu'à 6e-5 à l'itération 50 200.

La reprise a donc fait passer le learning rate d'environ 5,7e-4 à environ 6e-4, avant la décroissance. Ce changement de planning en cours de route est une particularité de cet entraînement.

### SFT (fine-tuning supervisé)

- **22 706 exemples d'entraînement** (≈ 1,35 million de tokens) et **410 exemples de validation** (≈ 24 700 tokens)
- Format : `<|user|>…<|end|><|assistant|>…<|end|>`
- Loss calculée uniquement sur les tokens de réponse, qui représentent **51,3 %** des tokens du SFT
- 2 époques (709 itérations par époque, **1 418 itérations**), environ 2,8 millions de tokens vus
- **Checkpoint publié : la dernière itération (1 418)**, sans early stopping
- `val_loss` finale : **1,8782** (minimum observé 1,8644 à l'itération 700)

Sources, avant découpage entraînement / validation :

| source | exemples | remarque |
|---|---|---|
| French Alpaca | 14 415 | 15 000 retenus au départ, 585 retirés par le filtre qualité |
| données synthétiques | 2 886 | identité, salutations, faits, capitales, contraires, calculs simples, jours et mois… générés par code |
| données synthétiques complémentaires | 2 281 | logique, intrus, langue (pluriels, féminins, contraires, synonymes), calendrier, environ 1 000 calculs |
| PIAF | 1 881 | questions-réponses extractives |
| OASST-FR | 333 | |
| persona (identité de MiniLLM) | 132 × 10 | 132 exemples répétés 10 fois, soit ≈ 5,8 % de l'entraînement |

Le **filtre qualité** appliqué aux sources externes retire les réponses de refus générique (« je suis désolé, je ne peux pas… »), les boucles de répétition et les réponses qui ne font que reprendre la question. Il n'est appliqué ni aux données synthétiques ni à la persona.

![Courbe de loss — SFT](./assets/loss_sft.png)

La loss de validation atteint son minimum à l'itération 700 (fin de la première époque), puis remonte très légèrement (de 0,014) pendant la seconde époque, avant de se stabiliser. L'écart est faible. Le checkpoint final a été préféré au minimum de `val_loss` parce que, dans nos comparaisons, la `val_loss` ne prédit pas la qualité : le checkpoint de plus faible `val_loss` mémorisait moins bien la persona.

French Alpaca représente encore ≈ 62 % des exemples, ce qui oriente le style des réponses vers des consignes génériques.

**Données et questions de test.** 28 des 337 questions du fichier d'évaluation ouverte figurent **mot pour mot** dans les données synthétiques du SFT (identité, salutations, contraires, repères de temps, 10 des 11 questions de capitales, quelques faits). Les bonnes réponses à ces questions sont donc en partie apprises par cœur ; la liste est donnée dans [`evals/resultats_automatiques.md`](./evals/resultats_automatiques.md).

---

## Prétraitement

### Wikipédia FR

Le corpus Wikipédia a notamment fait l'objet des traitements suivants :

- nettoyage des phrases « trouées » issues de certains templates du dump, par exemple `« né le  à Moulins »` ou `« (en latin : ) »` ;
- suppression de certaines sections de fin, notamment les références et liens externes ;
- suppression des titres selon les règles de préparation du corpus ;
- recollage des élisions françaises espacées, par exemple `« L' archidiocèse »` → `« L'archidiocèse »`.

### Web FR

Les documents français issus de FineWeb-2 ont fait l'objet d'un filtre anti-spam avant leur intégration au corpus.

### Format commun

Les deux sources ont ensuite été normalisées selon un format commun :

- paragraphes séparés par une ligne vide ;
- normalisation Unicode NFC ;
- suppression des doublons ;
- insertion de `<|endoftext|>` après chaque document.

## Tokenizer

Un tokenizer BPE byte-level de 32 000 tokens a été entraîné sur un échantillon représentatif des deux sources du corpus. Taux de compression : ≈ 4,23 caractères par token. Le tokenizer est identique à celui de la v2.1.

La préparation du tokenizer a notamment pris en compte :

- les particularités de la langue française ;
- les chiffres ;
- les élisions françaises ;
- les besoins du corpus utilisé pour le pré-entraînement.

## Split train/validation

- environ 1 % des documents ont été réservés à la validation : 46 422 documents (≈ 32,9 M tokens), contre 4 567 182 documents d'entraînement ;
- la sélection est effectuée par hachage du texte ;
- les documents du jeu de validation ne sont donc pas volontairement réutilisés dans le jeu d'entraînement.

## Séquences d'entraînement

Les données sont utilisées sous forme de blocs de 512 tokens.

Les blocs sont tirés du corpus selon la stratégie de sampling utilisée par le pipeline d'entraînement, avec mélange des différentes sources au niveau des blocs.

## Paramètres de constitution du corpus

- Wikipédia FR (quasi complète) et FineWeb-2 FR ;
- total : ≈ 4,61 millions de documents, ≈ 3,26 milliards de tokens (entraînement + validation) ;
- vocabulaire cible : 32 000 tokens.

---

## Hyperparamètres

### Pré-entraînement

| paramètre | valeur |
|---|---|
| batch size | 16 |
| gradient accumulation | 8 |
| longueur de séquence | 512 |
| itérations | 50 200 |
| tokens / itération | 65 536 |
| learning rate | 6e-4 → 6e-5 |
| scheduler | cosine jusqu'à ≈ l'itération 42 700 (planifié sur 280 000), puis warmup-stable-decay sur 50 200 itérations : plateau à 6e-4 jusqu'à l'itération 42 670, décroissance linéaire sur les 15 % finaux |
| warmup | 300 itérations |
| weight decay | 0,1 |
| optimiseur | AdamW |
| β1 | 0,9 |
| β2 | 0,95 |
| gradient clipping | 1,0 |
| précision | fp16 avec GradScaler (T4) |

Le budget total correspond à 3,29 milliards de tokens vus, soit environ 1,02 passage du corpus d'entraînement (3 226 M tokens).

### SFT

| paramètre | valeur |
|---|---|
| exemples (train / validation) | 22 706 / 410 |
| époques | 2 (1 418 itérations) |
| early stopping | aucun ; checkpoint publié = dernière itération |
| batch size | 16 |
| gradient accumulation | 2 |
| learning rate | 1e-4 → 1e-5 |
| scheduler | cosine decay |
| warmup | 100 itérations |
| weight decay | 0,1 |
| dropout | 0,1 |
| gradient clipping | 1,0 |
| évaluation | toutes les 250 itérations (toutes les 100 jusqu'à l'itération 1 500) |
| graine | 1337 |
| loss | tokens de réponse uniquement |
| initialisation | poids du pré-entraînement (itération 50 200) |
| optimiseur | réinitialisé avant le SFT |

---

## Hardware

L'entraînement a été réalisé sur des GPU NVIDIA T4, principalement via Google Colab et Kaggle.

Selon la session disponible :

- entraînement sur un seul GPU ;
- ou entraînement sur deux GPU avec DDP ;
- précision fp16 avec GradScaler sur T4.

Les conditions matérielles peuvent donc varier légèrement d'une session à l'autre. Un SFT complet de cette version prend environ 5 minutes sur un T4.

---

## Comportement observé

Le modèle produit du français grammaticalement correct et reproduit bien son identité (nom, créateur, origine). Il répond correctement à une partie des questions factuelles fréquentes : 10 des 11 questions de capitales du fichier d'évaluation reçoivent une réponse correcte (« capitale de l'Égypte » reçoit « Nubie »). Mais 10 de ces 11 questions figurent mot pour mot dans les données du SFT : il s'agit en grande partie de récitation. Sur les 48 capitales de pays **absents** de l'entraînement, 6 seulement sont justes (12,5 %).

En revanche, sa fiabilité chute dès que la tâche demande un calcul, un raisonnement ou une définition précise. Sur les 24 questions de calcul pur du fichier d'évaluation (n° 40 à 63), deux réponses sont correctes (100 divisé par 4 et la racine carrée de 144). Sur les 10 petits problèmes (n° 64 à 73), aucune bonne réponse ; sur les 6 suites de nombres (n° 74 à 79), aucune réponse propre (deux commencent par le bon nombre avant de dériver). Sur les définitions techniques (informatique, machine learning), le modèle produit des phrases du type « X est un système de … qui permet … » dont le fond est le plus souvent faux.

Le fichier [`MiniLlmQA.md`](./MiniLlmQA.md) rassemble 337 questions ouvertes, posées sans fournir la réponse attendue au modèle, et conserve les réponses produites, y compris les erreurs. Il constitue principalement un jeu d'observations qualitatives. Les exemples ci-dessous ne représentent qu'une sélection illustrative ; les numéros renvoient au fichier.

### Exemples de réponses cohérentes

**Identité** (n° 3)

**Q :** Qui es-tu ?
**R :** Je suis MiniLLM, un petit modèle de langage qui essaie de répondre à tes questions en français.

**Grammaire**

**Q :** Corrige cette phrase : Les enfant joue dehors.
**R :** Les enfants jouent dehors.

**Q :** Quel est le contraire de grand ?
**R :** Le contraire de grand est petit.

**Repères chiffrés** (appris par le SFT)

**Q :** Combien de minutes y a-t-il dans une heure ?
**R :** Il y a soixante minutes dans une heure.

**Sciences**

**Q :** Pourquoi le ciel est-il bleu ?
**R :** Le ciel est bleu en raison de la diffusion de la lumière du soleil par les molécules d'air.

**Q :** Qu'est-ce que la photosynthèse ?
**R :** La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie chimique.

**Géographie et astronomie**

**Q :** Quelle est la planète la plus proche du Soleil ?
**R :** La planète la plus proche du Soleil est Mercure.

**Q :** Quel est le plus long fleuve d'Afrique ?
**R :** Le Nil est le plus long fleuve d'Afrique.

**Personnages historiques**

**Q :** Qui était Marie Curie ?
**R :** Marie Curie était une physicienne et chimiste polonaise-française, célèbre pour ses travaux sur la radioactivité.

**Logique simple** (formats appris par le SFT)

**Q :** Paul est plus âgé que Jean. Jean est plus âgé que Marc. Qui est le plus jeune ?
**R :** Marc est le plus jeune.

**Q :** Si tous les chats sont des animaux et que Mimi est un chat, que peut-on dire de Mimi ?
**R :** Mimi est aussi un animal.

### Exemples d'échecs représentatifs

**Arithmétique**

**Q :** Combien font 2 plus 3 ?
**R :** 2 plus 3 font 10.

**Q :** Combien font 7 fois 8 ?
**R :** 7 fois 8 font 42.

**Q :** Combien font 100 moins 37 ?
**R :** 100 moins 37 font 24.

**Petit problème**

**Q :** Une classe contient 30 élèves. 12 sont des filles. Combien y a-t-il de garçons ?
**R :** Il y a 30 filles.

**Faits de base**

**Q :** Quelle est la planète sur laquelle nous vivons ?
**R :** La planète sur laquelle nousons vit est Mars.

**Q :** Quelle est la plus grande planète du système solaire ?
**R :** La plus grande planète du système solaire est la Terre.

**Q :** Quelle est la capitale de l'Égypte ?
**R :** La capitale de l'Égypte est Nubie.

**Q :** Quand le Cameroun est-il devenu indépendant ?
**R :** Le Cameroun est devenu indépendant en 1968, lorsque le Cameroun a été scindé en deux.

**Personnes**

**Q :** Qui était Nelson Mandela ?
**R :** Moussa Traoré.

**Q :** Qui était Léonard de Vinci ?
**R :** Le Louvre était le plus grand musée d'Europe, situé à Paris.

**Définitions techniques et persona qui déborde**

**Q :** Qu'est-ce que Git ?
**R :** Git est un langage de programmation développé par Git.com.

**Q :** Qu'est-ce que SQL ?
**R :** SQL est un langage de programmation développé par DevLab.

**Q :** Qu'est-ce que Windows ?
**R :** Windows est un système de sécurité sociale qui permet aux utilisateurs de payer leurs frais de santé en ligne, de se connecter à un service de santé mentale et de suivre les données personnelles.

**Q :** Qu'est-ce qu'un atome ?
**R :** Un atome est un corps céleste qui est composé de deux atomes d'hydrogène et d'un atome d'oxygène, chacun avec une longueur d'onde différente.

**Réponses qui reprennent la question ou n'en sont pas**

**Q :** Dans quelle région se trouve Garoua ?
**R :** Dans quelle région se trouve Garoua?

**Q :** Invente une devinette simple.
**R :** Je suis un petit modèle de langage, mais je ne peux pas apprendre.

Ces exemples montrent que la fluidité linguistique du modèle ne garantit pas la fiabilité de son contenu. Un même type de question peut être correctement traité dans un cas et produire une erreur dans un autre : les résultats doivent être interprétés comme des observations du comportement du modèle, et non comme une garantie de compétence générale.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` :

```
torch>=2.3
tokenizers>=0.15
safetensors>=0.4
```

## Utilisation

Ce dépôt contient notamment :

- les poids du modèle (`model.safetensors`) ;
- la configuration (`config.json`) ;
- le tokenizer (`tokenizer.json`) ;
- les réglages de génération (`generation_config.json`) ;
- un script d'inférence autonome (`inference.py`) ;
- le fichier d'évaluation `MiniLlmQA.md`.

Le script `inference.py` utilise les fichiers nécessaires présents dans le dépôt et permet une utilisation locale du modèle.

### Inférence en ligne de commande

```bash
python inference.py --weights model.safetensors --config config.json --tokenizer tokenizer.json \
    --prompt "Quelle est la capitale du Cameroun ?"
```

### Mode conversationnel

```bash
python inference.py --weights model.safetensors --config config.json --tokenizer tokenizer.json --mode chat
```

### Utilisation en Python

```python
from inference import load_model, MiniTokenizer, chat_reply

model = load_model("model.safetensors", "config.json")
tok = MiniTokenizer("tokenizer.json")

response = chat_reply(
    model,
    tok,
    [
        {"role": "user", "content": "Qui es-tu ?"}
    ]
)

print(response)
```

---

## Méthodologie d'évaluation

### Perplexité

La perplexité est mesurée sur un ensemble de validation fixe afin de permettre la comparaison entre différentes étapes d'entraînement.

Pour le pré-entraînement (même jeu de validation que la v2.1) :

- `val_loss = 3,0431` (v2.1 : 3,1998)
- perplexité ≈ 21,0 (v2.1 : ≈ 24,5)

Pour le SFT :

- `val_loss = 1,8782` sur le jeu de validation du SFT de la v2.2 (410 exemples)
- la loss est calculée uniquement sur les tokens correspondant aux réponses.

La loss du SFT ne doit pas être comparée à celle de la v2.1 (2,0459) : les jeux de validation sont différents (composition des sources, filtrage), et elle ne doit pas être comparée directement à la loss de pré-entraînement.

### Métriques automatiques

Évaluations avec réponse de référence sur le checkpoint SFT publié (décodage glouton) :

| évaluation | n | Exact Match | F1 (moyen) | générations terminées proprement |
|---|---|---|---|---|
| validation du SFT v2.2 (mélange des sources) | 410 | 9,5 % | 0,397 | 88,3 % |
| persona (identité de MiniLLM) | 132 | 99,2 % | 0,994 | n/d |

Ces scores doivent être lus avec précaution :

- le jeu de 410 exemples n'est pas celui de la v2.1 (455 exemples, 6,8 % d'Exact Match, F1 0,311) : **les deux lignes ne sont pas comparables**. Il contient notamment des exemples synthétiques du même type que ceux de l'entraînement, ce qui favorise la v2.2 ;
- l'Exact Match est très pénalisant pour des réponses ouvertes : une réponse correcte mais reformulée compte comme fausse ;
- les 132 exemples de persona font partie des données SFT. Le score de 99,2 % mesure la fidélité avec laquelle le modèle a appris sa persona, et non une capacité de généralisation.

Pour comparer les deux versions sur un même jeu, nous avons évalué la v2.1 et la v2.2 sur 300 exemples d'un jeu de validation commun : F1 de 31,4 (v2.1) contre 33,3 (v2.2) sur le jeu d'une variante intermédiaire (qui peut avantager la v2.1, dont les données d'entraînement ne sont pas contrôlées par rapport à ce jeu), et 34,2 contre 39,4 sur le jeu de la v2.2 (qui avantage la v2.2). L'écart réel se situe entre les deux et reste modeste.

### Test de connaissances (204 questions hors SFT)

Pour mesurer autre chose que de la récitation, nous utilisons un test de 204 questions à réponse courte, dont aucune n'est produite par les générateurs de données du SFT (vérifié automatiquement) : 48 capitales de pays absents de l'entraînement, 10 chefs-lieux de régions du Cameroun et 6 autres questions sur le Cameroun, géographie, sciences, animaux, nombres, culture, informatique, et 45 questions de langue. La liste complète, avec les réponses acceptées, est dans [`evals/test_connaissances.md`](./evals/test_connaissances.md). Une réponse est comptée juste si le mot attendu y figure comme mot entier (accents et casse ignorés).

| modèle | juste | % [IC95] | écart avec la v2.1 | p (McNemar) |
|---|---|---|---|---|
| v2.1 | 22/204 | 10,8 [7,2 ; 15,8] | — | — |
| v2.2 (ce dépôt) | 42/204 | 20,6 [15,6 ; 26,7] | +9,8 | 0,001 |
| v2.2, graine 1 | 38/204 | 18,6 [13,9 ; 24,5] | +7,8 | 0,005 |
| v2.2, graine 2 | 37/204 | 18,1 [13,5 ; 24,0] | +7,3 | 0,011 |
| SFT antérieur (sans filtre ni synthétique complémentaire) sur le pré-entraînement v2.2 | 28/204 | 13,7 [9,7 ; 19,1] | +2,9 | 0,307 |

Les trois lignes v2.2 sont trois entraînements identiques à la graine près : leur dispersion (18,1 % à 20,6 %) donne le bruit à attendre. Le test mêle connaissances du monde et 45 questions de langue dont 39 reprennent un type de question enseigné par le SFT (voir plus haut) : il ne doit pas être lu comme une mesure de « savoir » pur.

### Autres mesures

- **Connaissances de base (36 questions)** : 23/36 (63,9 %) ; trois entraînements à graine différente : 61,1 % à 66,7 %. 22 de ces 36 questions figurent mot pour mot dans les données synthétiques du SFT : c'est un test de régression, pas de généralisation.
- **Compétences chiffrées sur des problèmes tenus à l'écart de l'entraînement** (décodage glouton) : addition 1/60, soustraction 2/60, multiplication 6/19, division 4/17, suites de nombres 0/60, petits problèmes 0/60. Le modèle imite le format (« 44 - 17 = 29 ») mais ne calcule pas.

### Évaluation qualitative

Une évaluation séparée contient 337 questions ouvertes.

Les réponses sont conservées telles qu'elles ont été générées afin de documenter à la fois :

- les réponses correctes ;
- les réponses partiellement correctes ;
- les hallucinations ;
- les erreurs de raisonnement ;
- les erreurs arithmétiques ;
- les réponses hors sujet.

Aucune réponse attendue n'est fournie au modèle lors de cette évaluation. Décodage utilisé pour `MiniLlmQA.md` : glouton, avec pénalité de répétition 1,1 et interdiction de répéter un 3-gramme, 80 tokens maximum.

### Ce qui n'est pas mesuré

Cette évaluation ne constitue pas un benchmark standardisé. Aucun score agrégé n'est calculé sur les 337 questions ouvertes, et les métriques automatiques ci-dessus portent sur des jeux restreints. Un seul run a été publié ; la variabilité entre graines n'a été mesurée que sur les entraînements du SFT (pas sur le pré-entraînement).

Les résultats sont donc principalement destinés à l'analyse qualitative du comportement du modèle.

### Pistes essayées sans succès

- **Learning rate plus bas au SFT** (5e-5 et 3e-5) : meilleure `val_loss` (1,832 et 1,835 contre 1,876 à 1,878), mais persona moins bien apprise (88,6 % puis 28,0 % d'Exact Match), connaissances de base à 50 % et connaissances sans gain. La `val_loss` seule ne suffit pas à choisir un SFT.

---

## Pourquoi publier un modèle de cette taille ?

Un modèle d'environ 49M de paramètres entraîné from scratch constitue un objet d'étude intéressant pour :

- expérimenter des architectures Transformer à petite échelle ;
- observer concrètement l'effet du pré-entraînement et du SFT ;
- tester différentes stratégies de tokenisation ;
- travailler avec des ressources de calcul limitées ;
- comprendre par la pratique le fonctionnement d'un modèle de langage, du corpus à l'inférence ;
- disposer d'un point de comparaison pour de futures versions du projet ;
- étudier les limites de la génération de texte sur des modèles compacts.

miniLLM est avant tout un projet de recherche et d'apprentissage personnel, et non un modèle destiné à remplacer des systèmes généralistes ou à être utilisé comme source d'information fiable.

---

## Usages recommandés

Le modèle peut notamment être utilisé pour :

- expérimenter avec un petit modèle de langage français ;
- apprendre et étudier le fonctionnement d'un LLM de bout en bout ;
- tester localement une architecture Transformer compacte ;
- réaliser de petites conversations en français ;
- expérimenter avec la génération de texte ;
- comparer différentes versions de miniLLM ;
- étudier les effets du pré-entraînement et du SFT sur un modèle de petite taille.

---

## Usages hors objectif

Le modèle n'est pas conçu pour :

- les décisions importantes nécessitant des informations fiables ;
- les calculs financiers ;
- les usages médicaux ou juridiques ;
- la génération de code fiable ;
- le raisonnement multi-étapes exigeant une exactitude élevée ;
- l'arithmétique nécessitant des résultats garantis ;
- les applications de production nécessitant une fiabilité élevée ;
- les langues autres que le français, pour lesquelles il n'a pas été spécifiquement entraîné ;
- les contextes dépassant sa fenêtre de 512 tokens.

---

## Limites

Les évaluations réalisées sur ce modèle montrent notamment :

- des hallucinations factuelles : environ 4 réponses sur 5 du test de connaissances sont fausses ;
- des connaissances du monde très limitées hors de ce que le SFT enseigne : 6 capitales justes sur 48 absentes de l'entraînement, 1 bonne réponse sur 16 questions sur le Cameroun (chefs-lieux de régions compris), 0 sur 14 questions de nombres usuels (côtés des polygones, unités) ;
- des erreurs arithmétiques, y compris sur des opérations simples (2 bonnes réponses sur 24 questions de calcul pur ; aucune sur les petits problèmes ni les suites de nombres) ;
- des définitions techniques le plus souvent fausses, souvent formulées avec la même tournure générique (« est un ensemble de … qui permet … », « il peut être utilisé pour … ») ;
- une persona qui déborde parfois sur d'autres sujets (par exemple, SQL, Python ou JavaScript présentés comme « développés par DevLab ») ;
- des réponses qui reprennent la question au lieu d'y répondre (5 cas sur 337) et quelques réponses de type refus ou non-réponse ;
- une persona apprise par cœur : elle est reproduite à 99 % sur ses exemples d'entraînement, sans que cela dise comment le modèle se comporte sur des formulations nouvelles ;
- une forte variabilité des résultats d'un entraînement à l'autre sur certaines mesures (jusqu'à 8 points sur la proportion de générations terminées proprement), qui rend non significatifs les petits écarts ;
- un SFT dominé par French Alpaca (≈ 62 % des exemples) ;
- un seul pré-entraînement, dont la reprise s'est faite avec un changement de planning de learning rate ;
- des réponses grammaticalement correctes mais incorrectes sur le fond ;
- une capacité de raisonnement limitée ;
- une fenêtre de contexte de 512 tokens ;
- une spécialisation principalement francophone.

La taille du modèle et les données utilisées imposent également des limites importantes par rapport aux modèles de langage beaucoup plus grands.

Le modèle ne doit donc pas être considéré comme une source d'information fiable sans vérification externe.

---

## Évaluation ouverte

Le fichier [`MiniLlmQA.md`](./MiniLlmQA.md) contient les 337 questions et réponses brutes utilisées pour observer le comportement du modèle.

Les réponses sont conservées sans sélection visant à ne montrer que les réussites.

L'objectif est de permettre à d'autres utilisateurs de :

1. reprendre les mêmes questions avec le modèle ;
2. comparer les sorties obtenues ;
3. ajouter de nouvelles questions ;
4. documenter de nouveaux comportements ;
5. identifier aussi bien les réussites que les échecs.

Toute sortie inattendue peut ainsi constituer une observation utile pour les versions futures du projet.

---

## À propos

miniLLM est développé par DevLab, au Cameroun, dans le cadre d'un projet personnel visant à documenter ce qu'un petit modèle de langage entraîné localement, avec des ressources limitées, peut accomplir et quelles sont ses limites.

Cette version constitue un point de mesure dans le développement du projet, et non un aboutissement.

---

## Licence

miniLLM v2.2 est distribué sous licence MIT.

Voir le fichier [`LICENSE`](./LICENSE) pour le texte complet de la licence.

---

## Fichiers

- `config.json` — configuration de l'architecture du modèle
- `generation_config.json` — paramètres par défaut de génération
- `model.safetensors` — poids du modèle (float32, ≈ 260 Mo : la matrice d'embeddings, partagée avec la tête de sortie, y figure deux fois)
- `tokenizer.json` — tokenizer BPE entraîné sur le corpus
- `requirements.txt` — dépendances nécessaires à l'inférence
- `inference.py` — script d'inférence autonome
- [`MiniLlmQA.md`](./MiniLlmQA.md) — 337 questions/réponses brutes d'évaluation
- [`evals/resultats_automatiques.md`](./evals/resultats_automatiques.md) — métriques automatiques, comparaisons et tests de conversation
- [`evals/test_connaissances.md`](./evals/test_connaissances.md) — les 204 questions du test de connaissances
- `assets/banner.png` — bannière du projet
- `assets/icon.png` — icône du projet
- `assets/badges.png` — badges du projet
- `assets/loss_pretrain.png` — courbe de loss du pré-entraînement
- `assets/loss_sft.png` — courbe de loss du SFT
- `LICENSE` — licence MIT

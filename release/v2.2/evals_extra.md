## 7. Comparaison directe avec la v2.1 (mêmes mesures, même code)

Évaluations faites le 5 octobre 2026, avec le même code, sur le modèle v2.1 publié (`best.pt`, SFT itération 750) et sur la v2.2.

| mesure | v2.1 | v2.2 |
|---|---|---|
| persona, Exact Match / F1 (132 exemples d'entraînement) | 86,4 % / 93,5 | 99,2 % / 99,4 |
| test de connaissances (204 questions), IC95 | 10,8 % [7,2 ; 15,8] | 20,6 % [15,6 ; 26,7] |
| connaissances de base (36 questions, 22 dans les données synthétiques du SFT) | 61,1 % | 63,9 % |
| qa, jeu de validation d'une variante intermédiaire (300 ex.) : EM / F1 / fins correctes | 6,7 % / 31,4 / 84,0 % | 6,7 % / 33,3 / 87,7 % |
| qa, jeu de validation de la v2.2 (300 ex.) : EM / F1 / fins correctes | 6,7 % / 34,2 / 87,0 % | 10,0 % / 39,4 / 87,0 % |
| 337 questions ouvertes : réponses de type refus / boucles (détecteur grossier) | 11 / 0 | 3 / 1 |
| 337 questions ouvertes : réponses partageant le même 5-gramme de mots | 9 | 8 |

Les deux lignes `qa` ne sont pas équivalentes : la première vient d'un jeu qui n'est pas celui de la v2.2 et peut avantager la v2.1 (ses données d'entraînement ne sont pas contrôlées par rapport à ce jeu) ; la seconde vient du jeu de la v2.2, qui contient des exemples synthétiques du même type que son entraînement, et l'avantage. L'écart réel se situe entre les deux.

Le détecteur de refus repère des formules comme « je suis désolé », « je ne peux pas », « en tant qu'IA » : il compte aussi des réponses légitimes de la persona (par exemple « Je ne peux pas apprendre »). Le détecteur de boucle repère un même couple de mots répété au moins 4 fois.

## 8. Trois entraînements identiques à la graine près, et deux learning rates plus bas

Tous sur le jeu de validation de la v2.2, `final.pt`, SFT de 2 époques. `Clr5` et `Clr3` n'ont que le learning rate maximal qui change (5e-5 et 3e-5 au lieu de 1e-4).

| modèle | val_loss finale | persona EM / F1 | qa EM / F1 | fins correctes | connaissances de base | connaissances (204) | refus / boucles |
|---|---|---|---|---|---|---|---|
| v2.1 (publié) | n/d | 86,4 % / 93,5 | 6,7 % / 34,2 | 87,0 % | 61,1 % | 10,8 % | 11 / 0 |
| **v2.2 (ce dépôt, graine 1337)** | 1,8782 | 99,2 % / 99,4 | 10,0 % / 39,4 | 87,0 % | 63,9 % | 20,6 % | 3 / 1 |
| v2.2, graine 1 | 1,8774 | 98,5 % / 98,8 | 10,0 % / 39,2 | 87,0 % | 61,1 % | 18,6 % | 3 / 2 |
| v2.2, graine 2 | 1,8761 | 100,0 % / 100,0 | 9,0 % / 38,6 | 79,3 % | 66,7 % | 18,1 % | 0 / 5 |
| lr 5e-5 | 1,8320 | 88,6 % / 93,7 | 8,0 % / 38,5 | 82,0 % | 50,0 % | 16,7 % | 3 / 2 |
| lr 3e-5 | 1,8348 | 28,0 % / 60,6 | 7,0 % / 37,0 | 80,0 % | 50,0 % | 17,2 % | 1 / 2 |

Bruit entre les trois entraînements à 1e-4 : persona 98,5 % à 100 %, qa F1 38,6 à 39,4, connaissances de base 61,1 % à 66,7 % (deux questions sur 36), connaissances 18,1 % à 20,6 %, fins correctes 79,3 % à 87,0 %. Une différence plus petite que ces étendues ne doit pas être interprétée.

Un learning rate plus bas améliore nettement la `val_loss` (environ 0,045 de moins, soit une vingtaine de fois l'écart entre graines de 0,002) mais détériore la persona et les connaissances de base : la `val_loss` ne permet pas de choisir un SFT.

## 9. Variantes de recette de SFT, sur le pré-entraînement v2.2

`final.pt` de chaque variante, évaluées sur le jeu de validation d'une variante intermédiaire (300 exemples). Le test de connaissances est apparié avec la v2.1 par test exact de McNemar.

- **A** : French Alpaca (15 000 exemples), synthétique de base, persona répétée 8 fois, sans filtre qualité.
- **B** : A + synthétique complémentaire allégé (logique, intrus, langue, calendrier, environ 1 000 calculs).
- **C** : B + filtre qualité + persona à 6 % de l'entraînement. **C est la v2.2 publiée.**
- **D** : C + une source supplémentaire de réponses, non retenue.

| variante | persona EM / F1 | qa EM / F1 | fins correctes | connaissances de base | refus / boucles | connaissances (204) | écart avec la v2.1 | p |
|---|---|---|---|---|---|---|---|---|
| A | 94,7 % / 96,4 | 7,0 % / 33,2 | 86,7 % | 47,2 % | 5 / 2 | 28 (13,7 %) | +2,9 | 0,307 |
| B | 98,5 % / 98,8 | 6,3 % / 33,0 | 85,7 % | 66,7 % | 3 / 1 | 36 (17,6 %) | +6,8 | 0,016 |
| **C** | 99,2 % / 99,4 | 6,7 % / 33,3 | 87,7 % | 63,9 % | 3 / 1 | 42 (20,6 %) | +9,8 | 0,001 |
| D | 100,0 % / 100,0 | 4,3 % / 31,4 | 86,0 % | 61,1 % | 1 / 2 | 44 (21,6 %) | +10,8 | < 0,001 |

Checkpoint `best.pt` (choisi par `val_loss`) contre `final.pt` : identiques pour A ; légèrement moins bons pour B ; nettement moins bons pour C (persona 83,3 % contre 99,2 %, qa F1 30,8 contre 33,3) et pour D (persona 90,9 % contre 100 %). C'est la raison pour laquelle `final.pt` est publié.

## 10. Questions du fichier ouvert présentes mot pour mot dans les données synthétiques du SFT

28 des 337 questions (la persona n'en contient aucune) :

Bonjour. · Qui es-tu ? · Comment t'appelles-tu ? · Que peux-tu faire ? · Quel est le contraire de grand ? · Quel est le contraire de rapide ? · Quel est le contraire de difficile ? · Combien font 7 fois 8 ? · Combien de jours y a-t-il dans une semaine ? · Combien de mois y a-t-il dans une année ? · Combien d'heures y a-t-il dans une journée ? · Combien de minutes y a-t-il dans une heure ? · Combien de secondes y a-t-il dans une minute ? · Quelle est la planète la plus proche du Soleil ? · Quelle est la plus grande planète du système solaire ? · Quelle est la capitale du Cameroun ? · Quelle est la plus grande ville du Cameroun ? · Quel est le plus grand océan du monde ? · Quelle est la capitale de la France ? · du Nigeria ? · de la République centrafricaine ? · du Gabon ? · de la Côte d'Ivoire ? · du Sénégal ? · du Ghana ? · du Kenya ? · de l'Égypte ? · Quel est le plus long fleuve d'Afrique ?

(Dix de ces questions sont des capitales : les réponses justes qu'elles reçoivent relèvent en grande partie de la récitation. Pour le calcul, « 7 fois 8 » figure dans les données et reçoit pourtant une mauvaise réponse.)

## 11. Notes de méthode

- **Décodage** : glouton pour `qa`, `persona`, connaissances de base, compétences chiffrées et test de connaissances ; glouton avec pénalité de répétition 1,1 et interdiction de répéter un 3-gramme pour les 337 questions ouvertes ; température 0,5, top-k 30, top-p 0,9, mêmes pénalités et graine 0 pour le test de conversation.
- **Test exact de McNemar** : compare deux modèles sur les mêmes questions en ne retenant que celles où ils diffèrent ; p est la probabilité d'un écart au moins aussi grand si les deux modèles étaient équivalents.
- **Un seul pré-entraînement** a été fait : la variabilité entre graines ne concerne que le SFT.

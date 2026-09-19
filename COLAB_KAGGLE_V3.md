# MiniLLM v3 — guide Colab/Kaggle

Les notebooks historiques restent compatibles avec le dépôt, mais les commandes
ci-dessous sont la procédure canonique V3. Exécuter les cellules dans l'ordre.

## Installer et cloner la branche V3

```bash
pip install -r requirements.txt
# Le clone public peut ensuite être positionné sur la branche V3 :
git clone https://github.com/bono-p/minillm_v2.git
cd minillm_v2
git checkout v3/training-pipeline-overhaul
```

## Vérifier les vrais paramètres

Les noms `15M`, `50M`, etc. sont conservés comme identifiants historiques de CLI.
Ils ne représentent pas le nombre réel de paramètres avec `cl100k_base`.

```bash
python inspect_model.py
python inspect_model.py --size 50M
```

Noms réels affichés par PyTorch :

| Identifiant CLI | Nom réel | Paramètres réels |
|---|---:|---:|
| `15M` | MiniLLM-49M | 49,104,? — calculé dynamiquement par `model.n_params` |
| `50M` | MiniLLM-83M | 83,? — calculé dynamiquement par `model.n_params` |
| `125M` | MiniLLM-162M | 162,? — calculé dynamiquement par `model.n_params` |
| `350M` | MiniLLM-381M | 381,? — calculé dynamiquement par `model.n_params` |
| `1B` | MiniLLM-1.09B | 1,091,? — calculé dynamiquement par `model.n_params` |

Le tableau exact, avec les entiers, est produit par `inspect_model.py`; aucune
valeur arrondie écrite à la main ne doit servir de référence.

## Pré-entraînement

```bash
python prepare_data.py data/wikipedia_fr.txt data/autres_textes/ --out data/pretrain --val_ratio 0.01
python train.py --size 50M --data data/pretrain/train.bin --val data/pretrain/val.bin --out checkpoints/pretrain --no-compile
```

## PIAF/FQuAD en SFT masqué

PIAF et FQuAD restent séparés du corpus de pré-entraînement. Fournir les JSON
obtenus légalement et vérifier leur licence avant usage ou redistribution.

```bash
python train_sft.py \
  --data data/piaf-train.json data/fquad-train.json \
  --checkpoint checkpoints/pretrain/best.pt \
  --size 15M --seq 512 --batch 2 --iters 1000 \
  --out checkpoints/sft --device cuda
```

Le SFT entraîne uniquement les tokens de la réponse assistant et `<|end|>`;
le système, le contexte et la question portent le label `-1`.

## Contrôle minimal avant un long entraînement

```bash
python -m py_compile *.py
python inspect_model.py
python train_sft.py --help
```

Les notebooks Colab/Kaggle doivent importer les scripts depuis cette branche et
afficher les paramètres avec `model.n_params`. Les sorties historiques affichées
dans un notebook ne constituent pas une mesure actuelle : relancer la cellule
d'inspection après avoir cloné V3.

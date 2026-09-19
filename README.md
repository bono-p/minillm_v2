# MiniLLM v3 — état actuel

La branche V3 distingue clairement :
- pré-entraînement causal sur fichiers texte `int32` ;
- SFT QA français sur PIAF/FQuAD/SQuAD JSON avec labels masqués.

Les presets CLI `15M`, `50M`, `125M`, `350M`, `1B` sont des alias historiques. Les paramètres réels sont toujours calculés par `model.n_params` et affichés par `inspect_model.py`.

## Smoke test obligatoire

```bash
pip install -r requirements.txt
python -m py_compile *.py
python -m unittest discover -s tests -v
python inspect_model.py
```

## Pré-entraînement

```bash
python prepare_data.py wikipedia_fr.txt autres_textes/ --out data/pretrain --val_ratio 0.01
python train.py --size 50M --data data/pretrain/train.bin --val data/pretrain/val.bin --out checkpoints/pretrain --no-compile
```

Le tokenizer utilise `cl100k_base` et les fichiers binaires sont en `int32`, jamais `uint16`.
Les checkpoints comprennent `best.pt`, `last.pt` et les checkpoints rolling.

## SFT PIAF/FQuAD

Les formats SQuAD et les exports plats `{context, question, answer}` sont acceptés :

```bash
python train_sft.py --data data/piaf.json data/fquad.json --checkpoint checkpoints/pretrain/best.pt --size 15M --seq 512 --batch 2 --iters 1000 --out checkpoints/sft --device cuda
```

Le SFT entraîne uniquement les tokens de la réponse et `<|end|>`. Les contextes, questions et instructions sont masqués avec `-1`. La séparation validation est déterministe par groupe documentaire.

## Limites vérifiées

- Les notebooks historiques contiennent des sorties anciennes : utiliser `COLAB_KAGGLE_V3.md` et relancer les cellules après checkout de la branche V3.
- Le DDP multi-GPU et l'intégration de `train.py --mode sft` restent à traiter séparément ; utilisez `train_sft.py` pour le SFT.
- Aucun entraînement long n'est déclaré validé sans exécution dans votre runtime.

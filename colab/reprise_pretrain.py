# ============================================================
# Reprise du pré-entraînement — plan raccourci à ≈ 1,02 époque
# À lancer À LA PLACE de la cellule 9 du notebook (« Pré-entraînement »),
# APRÈS avoir exécuté les cellules 0 (configuration) et 1 (installation).
#
# Ce qui change par rapport à l'ancien plan (280 000 it., cosine) :
#   - max_iters 50 200 : 50 200 x 65 536 = 3 290 M tokens = 1,02 x 3 226 M
#   - schedule WSD : le LR reste à 6e-4 jusqu'à l'it. 42 670, puis descend
#     linéairement jusqu'à 6e-5 à l'it. 50 200 (il était à 5,7e-4 : pas de saut).
#   - reprise AUTOMATIQUE depuis le dernier checkpoint (it. 42 672).
# Rien n'est écrasé : poids, optimiseur, ordre des données et train.bin sont conservés.
# ============================================================
PT_ITERS   = 50_200
DECAY_FRAC = 0.15          # 1 - 42 670/50 200 -> la décroissance démarre à l'it. 42 670

PT_DATA = local_copy("pretrain")
args = (f"--mode pretrain --data_dir {PT_DATA} --out_dir {PT_OUT} --model_size {MODEL_SIZE} "
        f"--seq_len 512 --batch_size {PT_BATCH} --grad_accum {PT_ACCUM} --lr {PT_LR} --min_lr {PT_LR/10} "
        f"--max_iters {PT_ITERS} --schedule wsd --decay_frac {DECAY_FRAC} "
        f"--warmup_iters 300 --eval_every 500 --save_every 500 --eval_iters 50 "
        f"--max_minutes {MAX_MINUTES}")
if N_GPU > 1:
    !torchrun --standalone --nproc_per_node={N_GPU} train.py {args}
else:
    !python train.py {args}

# Vérifie dans les premières lignes affichées :
#   "Pré-entraînement : 65,536 tokens/it | 50200 it = 3290 M tokens (1.02 époque sur 3226 M tokens)"
#   "Reprise à it=42672 | best_val_loss=3.1998"
#   puis lr ≈ 6.00e-04 au début, et ≈ 6.00e-05 à la fin.
# Si la session est coupée : relance simplement cette cellule (reprise automatique).

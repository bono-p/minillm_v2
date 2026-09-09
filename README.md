# MiniLLM v2 — DevLab

Transformer decoder-only optimisé, scalable de **15M à 1B paramètres**,  
conçu pour être entraîné sur un seul GPU grand public.

## Architecture

| Composant | Choix | Avantage |
|---|---|---|
| Positional embeddings | **RoPE** | Extrapolation longueur, 0 params |
| Normalisation | **RMSNorm** | ~10% plus rapide que LayerNorm |
| Activation FFN | **SwiGLU** | Surpasse GELU empiriquement |
| Attention | **GQA** (configurable) | Réduit le KV cache à grande échelle |
| Flash Attention | **Oui** (PyTorch 2.0+) | Intégré, sans lib externe |
| Biais linéaires | **Non** | Standard moderne (LLaMA, Gemma) |
| Precision | **bfloat16** | Plus stable que float16 |

## Tailles disponibles

| Preset | Params | Layers | d_model | Heads | KV heads |
|--------|--------|--------|---------|-------|----------|
| 15M    | ~15M   | 6      | 384     | 6     | 6 (MHA)  |
| 50M    | ~50M   | 10     | 512     | 8     | 8 (MHA)  |
| 125M   | ~125M  | 12     | 768     | 12    | 12 (MHA) |
| 350M   | ~350M  | 24     | 1024    | 16    | 8 (GQA)  |
| 1B     | ~1B    | 20     | 2048    | 16    | 4 (GQA)  |

---

## Installation

```bash
# 1. Cloner / copier le projet
cd minillm_v2

# 2. Installer les dépendances
pip install -r requirements.txt

# Pour GPU CUDA (remplace la ligne torch dans requirements.txt) :
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Utilisation

### Étape 1 — Préparer les données

Le modèle s'entraîne sur un fichier texte plain (`.txt`).  
Plus il y a de texte, mieux c'est. Minimum recommandé : **50M tokens** (~40MB de texte).

```bash
# Depuis un seul fichier
python prepare_data.py mon_corpus.txt

# Depuis un dossier de fichiers .txt
python prepare_data.py dossier_textes/

# Avec options avancées
python prepare_data.py corpus.txt --out data/ --val_ratio 0.02
```

Cela crée :
```
data/
├── train.bin   ← données d'entraînement (uint16)
├── val.bin     ← données de validation
└── meta.json   ← métadonnées
```

**Sources de corpus recommandées :**
- Texte français : [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [mC4](https://huggingface.co/datasets/mc4), Wikipedia FR
- Vos propres données, articles, livres, etc.
- Pour le Fulfulde : exporter les données du projet Tardigrade

### Étape 2 — Vérifier le modèle

Avant de lancer un long entraînement, vérifier que tout est correct :

```bash
# Tableau comparatif de tous les presets
python inspect_model.py

# Détails du modèle 50M (test forward pass inclus)
python inspect_model.py --size 50M
```

### Étape 3 — Entraîner

```bash
# Entraînement avec le preset 50M (défaut)
python train.py

# Choisir une taille différente
python train.py --size 15M          # plus rapide, pour tester
python train.py --size 125M         # plus puissant, nécessite plus de VRAM

# Options avancées
python train.py \
    --size 50M \
    --seq 1024 \
    --batch 8 \
    --accum 8 \
    --iters 50000 \
    --lr 3e-4 \
    --out checkpoints/

# Reprendre un entraînement interrompu
python train.py --resume checkpoints/best.pt
```

**Paramètres importants :**

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--size`  | `50M`  | Taille du modèle |
| `--batch` | `8`    | Micro-batch par GPU (réduire si OOM) |
| `--accum` | `8`    | Accumulation gradient (batch effectif = batch × accum) |
| `--seq`   | `1024` | Longueur de séquence |
| `--iters` | `50000`| Itérations totales |
| `--lr`    | `3e-4` | Learning rate max (cosine decay) |

**Si mémoire insuffisante (OOM) :**
```bash
# Réduire le batch et augmenter l'accumulation (batch effectif identique)
python train.py --batch 4 --accum 16

# Désactiver la compilation
python train.py --no-compile

# Réduire la longueur de séquence
python train.py --seq 512
```

### Étape 4 — Générer du texte

```bash
# Mode interactif (prompt dans le terminal)
python generate.py

# Avec un prompt fourni en argument
python generate.py --prompt "Il était une fois"

# Régler le style de génération
python generate.py \
    --prompt "Le modèle de langue" \
    --temp 0.7 \
    --top_k 40 \
    --max_new 300

# Depuis un checkpoint spécifique
python generate.py \
    --checkpoint checkpoints/ckpt_010000.pt \
    --prompt "Bonjour"
```

**Paramètres de génération :**

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `--temp`  | `0.8`  | < 1 = conservateur, > 1 = créatif |
| `--top_k` | `50`   | 0 = désactivé (greedy) |
| `--top_p` | `0.95` | Nucleus sampling (1.0 = désactivé) |
| `--max_new` | `200` | Nombre max de tokens générés |
| `--samples` | `1`  | Nombre de générations (avec --prompt) |

---

## Structure du projet

```
minillm_v2/
├── config.py         ← Configuration et presets (modifier ici)
├── model.py          ← Architecture complète (RoPE, RMSNorm, SwiGLU, GQA)
├── train.py          ← Boucle d'entraînement
├── generate.py       ← Génération de texte
├── prepare_data.py   ← Préparation du corpus
├── inspect_model.py  ← Inspection et diagnostics
├── requirements.txt  ← Dépendances Python
└── README.md         ← Ce fichier
```

---

## Scaling vers 1B paramètres

L'architecture est conçue pour scaler sans modifier le code.  
Seule la configuration change :

```python
# Dans config.py, les presets "350M" et "1B" sont déjà définis.
# Pour entraîner un 1B :
python train.py --size 1B --batch 2 --accum 32 --seq 2048
```

**Prérequis matériels approximatifs :**

| Taille | VRAM GPU | Temps (RTX 3090, 1B tokens) |
|--------|----------|------------------------------|
| 15M    | 2 GB     | ~2 heures                    |
| 50M    | 4 GB     | ~4 heures                    |
| 125M   | 8 GB     | ~10 heures                   |
| 350M   | 24 GB    | ~30 heures                   |
| 1B     | 80 GB    | ~plusieurs jours (multi-GPU) |

---

## Personnalisation

### Modifier l'architecture

Tout se passe dans `config.py` :

```python
# Exemple : modèle custom avec contexte plus long
from config import ModelConfig

mon_modele = ModelConfig(
    n_layers    = 12,
    d_model     = 768,
    n_heads     = 12,
    kv_heads    = 4,      # GQA : 3× moins de KV cache
    max_seq_len = 4096,   # contexte plus long
    rope_theta  = 50_000, # RoPE adapté aux longues séquences
)
```

### Entraîner un tokenizer custom (Fulfulde + Français)

```bash
pip install sentencepiece datasets

# Après avoir collecté votre corpus
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='corpus_fuv_fra.txt',
    model_prefix='tokenizer/fuv_fra',
    vocab_size=32000,
    character_coverage=0.9999,
    model_type='bpe',
)
"
```

Puis adapter `vocab_size` dans `ModelConfig` et remplacer `tiktoken` par
votre tokenizer SentencePiece dans `prepare_data.py` et `generate.py`.

---

## Références

- [Gemma (Google, 2024)](https://arxiv.org/abs/2403.08295) — RMSNorm, SwiGLU, RoPE, GQA
- [RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Rotary Positional Embeddings  
- [SwiGLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — Activation GLU
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — Attention efficace
- [nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat) — Inspiration codebase
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) — Pipeline open source LLM

# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

> A modular, reproducible research framework for few-shot **static image**
> sign language recognition using metric learning with cross-lingual transfer.

---

## Overview

This project implements geometry-aware metric learning for sign language
recognition using hand landmarks extracted via MediaPipe from **static images**.
It supports:

- **Metric learning** with Triplet, Supervised Contrastive (SupCon), and ArcFace losses  
- **Few-shot methods**: Prototypical Networks, Siamese Networks, Matching Networks  
- **Three encoders**: MLP, Spatial Transformer (landmark attention), Graph Convolutional Network (GCN)  
- **Cross-lingual transfer**: Pre-train on ASL → zero-shot/few-shot adapt to Thai / LIBRAS / Arabic  
- **Comprehensive ablation study** over representations, losses, models, normalisation, and shots  

### Datasets

| # | Dataset | Kaggle Slug | Role |
|---|---------|-------------|------|
| 🥇 | ASL Alphabet | `grassknoted/asl-alphabet` | Source pre-train (~1 GB, 29 classes) |
| 🥈 | Sign Language MNIST | `datamunge/sign-language-mnist` | Low-resolution domain shift |
| 🥉 | Thai Fingerspelling | `phattarapong58/thai-sign-language` | Target adaptation |
| 🟡 | Brazilian LIBRAS | `williansoliveira/libras` | Cross-language reinforcement |
| 🟡 | Arabic Sign Alphabet | `muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset` | Diversity |

---

## Theoretical Foundation

### Rigid Transformation Invariance

Hand landmarks extracted from images are subject to rigid transformations due to
camera viewpoint, hand position, and distance to the camera. If a rigid transform
is applied:

$$
\mathbf{x}' = R\mathbf{x} + \mathbf{t}
$$

then pairwise distances are preserved:

$$
\|\mathbf{x}'_i - \mathbf{x}'_j\| = \|R(\mathbf{x}_i - \mathbf{x}_j)\| = \|\mathbf{x}_i - \mathbf{x}_j\|
$$

because rotation matrices satisfy $R^T R = I$.

### Normalisation Strategy

Our preprocessing pipeline reduces nuisance variation through two steps:

1. **Translation invariance** — Subtract the wrist landmark (landmark 0) from all
   landmarks, centring the hand at the origin:
   $$\hat{\mathbf{x}}_i = \mathbf{x}_i - \mathbf{x}_0$$

2. **Scale invariance** — Divide by the maximum pairwise distance (hand span):
   $$\tilde{\mathbf{x}}_i = \frac{\hat{\mathbf{x}}_i}{\max_{j,k}\|\hat{\mathbf{x}}_j - \hat{\mathbf{x}}_k\|}$$

This ensures that the input representation is invariant to translation and scale,
focusing the model on the *geometry* (shape) of the hand pose.

---

## Project Structure

```
sign_metric_learning/
├── configs/
│   ├── base.yaml              # Base hyperparameters
│   └── asl_to_bdsl.yaml       # Cross-lingual transfer config
├── data/
│   ├── __init__.py
│   ├── preprocess.py           # MediaPipe landmark extraction from images
│   ├── datasets.py             # PyTorch Dataset classes
│   └── episodes.py             # Episodic N-way K-shot sampler
├── models/
│   ├── __init__.py             # Encoder/model factory functions
│   ├── mlp_encoder.py          # MLP baseline encoder
│   ├── temporal_transformer.py # Spatial Transformer (landmark attention)
│   ├── gcn_encoder.py          # Graph Convolution on hand skeleton
│   ├── prototypical.py         # Prototypical Networks
│   └── siamese.py              # Siamese & Matching Networks
├── losses/
│   ├── __init__.py
│   ├── triplet.py              # Triplet loss with online mining
│   └── supcon.py               # SupCon, ArcFace, and build_loss factory
├── utils/
│   ├── __init__.py
│   ├── seed.py                 # Reproducibility (seed, deterministic mode)
│   ├── logger.py               # Console + file logging
│   └── metrics.py              # Accuracy, CI, confusion matrix, t-SNE
├── tools/
│   └── auto_find_download_and_filter_onehand.py  # Dataset download & filter
├── train.py                    # Episodic training script
├── evaluate.py                 # Evaluation & zero-shot testing
├── adapt.py                    # Few-shot adaptation (1-shot, 5-shot)
├── ablation.py                 # Systematic ablation study
├── run.sh                      # End-to-end pipeline script
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. Dataset Download

Use the included dataset tool:

```bash
# Download ASL Alphabet (source)
python tools/auto_find_download_and_filter_onehand.py \
    --dataset asl-alphabet --download --extract --seed 42

# Download Thai Fingerspelling (target)
python tools/auto_find_download_and_filter_onehand.py \
    --dataset thai-fingerspelling --download --extract --seed 42
```

### 3. Preprocessing

Extract landmarks from images:

```bash
python data/preprocess.py --image_dir data/raw/asl_alphabet --output_dir data/processed/asl
python data/preprocess.py --image_dir data/raw/thai_fingerspelling --output_dir data/processed/thai
```

> **Note:** If real datasets are unavailable, the code automatically falls back to
> synthetic data for demonstration and testing purposes.

---

## Running Experiments

### Quick Start (full pipeline with synthetic data)

```bash
bash run.sh
```

### Individual Steps

#### Pre-train on ASL

```bash
python train.py --config configs/base.yaml --dataset ASL
```

#### Zero-shot evaluation on Thai

```bash
python evaluate.py --dataset Thai --zero-shot
```

#### Few-shot adaptation

```bash
# 1-shot
python adapt.py --dataset Thai --shot 1

# 5-shot
python adapt.py --dataset Thai --shot 5

# With different adaptation strategies
python adapt.py --dataset Thai --shot 5 --mode freeze
python adapt.py --dataset Thai --shot 5 --mode finetune_last
python adapt.py --dataset Thai --shot 5 --mode full_finetune
```

#### Ablation study

```bash
# Full ablation
python ablation.py

# Fast ablation (fewer episodes)
python ablation.py --fast

# Specific axes only
python ablation.py --axes model loss representation
```

---

## Results

All results are saved under `results/`:

| File | Contents |
|------|----------|
| `results/zero_shot.csv` | Zero-shot cross-domain accuracy |
| `results/few_shot.csv` | Few-shot adaptation results |
| `results/ablation.csv` | Full ablation table |
| `results/plots/tsne.png` | t-SNE embedding visualisation |
| `results/plots/tsne_adapt_*.png` | Post-adaptation t-SNE plots |
| `results/checkpoints/` | Model checkpoints |

### Metrics Reported

- **Few-shot accuracy**: mean ± 95% confidence interval over 1000 episodes  
- **Cross-domain accuracy drop**: source vs. target accuracy  
- **Confusion matrix**: per-class classification breakdown  
- **t-SNE visualisation**: embedding space structure  

---

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Embedding dim | 128 |
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Temperature (SupCon) | 0.07 |
| Episodes (eval) | 1000 |
| N-way | 5 |
| K-shot | 1, 5 |
| Q-query | 15 |
| Landmarks | 21 (MediaPipe hand) |
| Input shape | (21, 3) per image |

All hyperparameters are configurable via YAML files in `configs/`.

---

## Ablation Axes

| Axis | Variants |
|------|----------|
| Representation | Raw landmarks · Pairwise distances (210D) · Graph-based |
| Loss | Triplet · SupCon |
| Model | MLP · Transformer · GCN |
| Normalisation | Full (translate+scale) · No scale · None |
| Adaptation | Zero-shot · 1-shot · 5-shot |

---

## Reproducibility

- Fixed random seed (default: 42) across Python, NumPy, and PyTorch  
- Deterministic CuDNN and CUBLAS settings  
- Config-driven hyperparameters (no magic numbers in code)  
- Checkpoint saving with metadata  
- Automatic result directory creation  
- Comprehensive logging to file and console  

---

## Citation

```bibtex
@article{geometry_aware_metric_sign_2026,
  title   = {Geometry-Aware Metric Learning for Cross-Lingual Few-Shot
             Sign Language Recognition},
  author  = {[Author Names]},
  journal = {[Conference/Journal]},
  year    = {2026},
  note    = {Code: https://github.com/[username]/sign_metric_learning}
}
```

---

## License

This project is released for academic research purposes. Please cite if you use
this code in your work.

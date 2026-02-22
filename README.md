# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

> A modular, reproducible research framework for few-shot sign language recognition
> using metric learning with cross-lingual transfer from ASL to BdSL.

---

## Overview

This project implements geometry-aware metric learning for sign language recognition
using hand landmarks extracted via MediaPipe. It supports:

- **Metric learning** with Triplet, Supervised Contrastive (SupCon), and ArcFace losses  
- **Few-shot methods**: Prototypical Networks, Siamese Networks, Matching Networks  
- **Three encoders**: MLP, Temporal Transformer, Graph Convolutional Network (GCN)  
- **Cross-lingual transfer**: Pre-train on ASL → zero-shot/few-shot adapt to BdSL  
- **Comprehensive ablation study** over representations, losses, models, normalisation, and shots  

---

## Theoretical Foundation

### Rigid Transformation Invariance

Hand landmarks extracted from video are subject to rigid transformations due to
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
   landmarks per frame, centring the hand at the origin:
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
│   ├── preprocess.py           # MediaPipe landmark extraction & normalisation
│   ├── datasets.py             # PyTorch Dataset classes
│   └── episodes.py             # Episodic N-way K-shot sampler
├── models/
│   ├── __init__.py             # Encoder/model factory functions
│   ├── mlp_encoder.py          # MLP baseline encoder
│   ├── temporal_transformer.py # Transformer with positional encoding
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
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### MS-ASL (Source)

1. Download MS-ASL from [Microsoft Research](https://www.microsoft.com/en-us/research/project/ms-asl/).
2. Organise videos into class sub-folders:
   ```
   data/raw/ms_asl/
       train/
           class_0/
               video1.mp4
           class_1/
               ...
       test/
           class_0/
               ...
   ```
3. Extract landmarks:
   ```bash
   python data/preprocess.py --video_dir data/raw/ms_asl/train --output_dir data/raw/ms_asl/train
   python data/preprocess.py --video_dir data/raw/ms_asl/test --output_dir data/raw/ms_asl/test
   ```

#### BdSLW60 (Target)

1. Download BdSLW60 from [Mendeley Data](https://data.mendeley.com/) or the original authors.
2. Organise similarly under `data/raw/bdslw60/{train,test}/`.
3. Run the same preprocessing.

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

#### Zero-shot evaluation on BdSL

```bash
python evaluate.py --dataset BdSL --zero-shot
```

#### Few-shot adaptation

```bash
# 1-shot
python adapt.py --dataset BdSL --shot 1

# 5-shot
python adapt.py --dataset BdSL --shot 5

# With different adaptation strategies
python adapt.py --dataset BdSL --shot 5 --mode freeze
python adapt.py --dataset BdSL --shot 5 --mode finetune_last
python adapt.py --dataset BdSL --shot 5 --mode full_finetune
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
| Sequence length | 32 frames |
| Landmarks | 21 (MediaPipe hand) |

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

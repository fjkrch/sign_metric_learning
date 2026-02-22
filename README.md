# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

> Modular, reproducible research framework for **static-image** sign-language
> recognition using metric learning with cross-lingual transfer across four
> sign-language alphabets.
>
> All experiments are deterministic (seed = 42) and fully reproducible from
> a single shell script.

---

## A. Overview

This project implements geometry-aware metric learning for sign-language
recognition using hand landmarks extracted via **MediaPipe** from static images.

**Methods:**
- **Metric learning losses:** Triplet (online mining), Supervised Contrastive (SupCon), ArcFace
- **Few-shot methods:** Prototypical Networks, Siamese Networks, Matching Networks
- **Encoders:** MLP, Spatial Transformer (landmark-attention), GCN (hand skeleton graph)
- **Representations:** `raw` (63-D xyz), `angle` (20-D joint angles), `raw_angle` (83-D concat)

**Evaluation protocol:**
- JSON-based stratified train/test splits (70 / 30, seed = 42)
- N-way K-shot episodic evaluation with strict K+Q feasibility enforcement
- Deterministic per-episode seeding: `rng = np.random.RandomState(seed + e)`
- Cross-domain evaluation: pretrain on source, evaluate on target test split

### Datasets

| Dataset | Kaggle Slug | Classes | Total Samples |
|---------|-------------|---------|---------------|
| ASL Alphabet | `grassknoted/asl-alphabet` | 28 | ~63 500 |
| LIBRAS Alphabet | `williansoliveira/libras` | 21 | ~34 300 |
| Arabic Sign Alphabet | `muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset` | 31 | ~7 100 |
| Thai Fingerspelling | `nickihartmann/thai-letter-sign-language` | 42 | ~2 900 |

### Theoretical Foundation

Hand landmarks from images are subject to rigid transforms.  If

$$\mathbf{x}' = R\mathbf{x} + \mathbf{t}$$

then pairwise distances are preserved because $R^T R = I$:

$$\|\mathbf{x}'_i - \mathbf{x}'_j\| = \|\mathbf{x}_i - \mathbf{x}_j\|$$

**Normalisation:** (1) subtract wrist landmark → translation invariance;
(2) divide by max pairwise distance → scale invariance.

| Repr | Dim | Description |
|------|-----|-------------|
| `raw` | 63 | Flattened (21 × 3) normalised xyz |
| `angle` | 20 | Inter-joint angles from 20 anatomical triplets |
| `raw_angle` | 83 | z-normalised raw ‖ angle |

---

## B. Setup

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- mediapipe, numpy, scikit-learn, matplotlib, pyyaml, tqdm

```bash
pip install -r requirements.txt
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT` | Override the base data directory (default: working directory) |
| `KAGGLE_API_TOKEN` | Kaggle API credentials for dataset download |

---

## C. Dataset Download & Preprocessing

```bash
# 1. Download from Kaggle (requires KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json)
for ds in asl-alphabet thai-fingerspelling libras-alphabet arabic-sign-alphabet; do
    python tools/auto_find_download_and_filter_onehand.py \
        --dataset "$ds" --download --extract --seed 42
done

# 2. Extract MediaPipe hand landmarks (21 × 3 .npy per image)
python data/preprocess.py --image_dir data/raw/asl_alphabet       --output_dir data/processed/asl_alphabet
python data/preprocess.py --image_dir data/raw/libras_alphabet    --output_dir data/processed/libras_alphabet
python data/preprocess.py --image_dir data/raw/arabic_sign_alphabet --output_dir data/processed/arabic_sign_alphabet
python data/preprocess.py --image_dir data/raw/thai_fingerspelling --output_dir data/processed/thai_fingerspelling
```

After preprocessing, each dataset lives in `data/processed/<name>/<class>/*.npy`.

---

## D. Split Generation

Splits are **JSON-based, stratified, and deterministic**.  Each class is
independently shuffled with `numpy.RandomState(seed)` and split at
`floor(ratio × n_c)` train samples, with at least 1 sample in each split.

```bash
python tools/make_splits.py --dataset asl_alphabet         --seed 42 --ratio 0.7
python tools/make_splits.py --dataset libras_alphabet      --seed 42 --ratio 0.7
python tools/make_splits.py --dataset arabic_sign_alphabet --seed 42 --ratio 0.7
python tools/make_splits.py --dataset thai_fingerspelling  --seed 42 --ratio 0.7
```

**Output:** `splits/<dataset>_train.json` and `splits/<dataset>_test.json`.

JSON format:
```json
{
  "class_name": ["class_name/file001.npy", "class_name/file002.npy", ...],
  ...
}
```

Paths inside the JSON are relative to the flat preprocessed directory
(`data/processed/<dataset>`).

### Integrity Guarantees

- No duplicate paths within a split
- Zero overlap between train and test
- At least 1 sample per class in each split
- Validated automatically at load time (`validate_no_leak()`)

---

## E. K+Q Feasibility Rule

For N-way K-shot episodic evaluation with Q query samples, a class is
**eligible** only when:

$$n_c \geq K + Q$$

where $n_c$ is the number of samples for that class in the evaluation split.
At least $N$ classes must be eligible, otherwise the sampler raises a
`ValueError` with a detailed diagnostic.

### Auto-adjust mode

Pass `--auto_adjust_q` to automatically lower $Q$ to the largest feasible
value (minimum 1).  This is useful for datasets with small per-class counts
(e.g. Thai Fingerspelling test split).

### Example diagnostic

```
[thai_fingerspelling/test] K+Q feasibility FAILED.
  Need n_c >= K+Q = 5+15 = 20 for at least N=5 classes.
  Eligible classes: 3/42.
  Min samples/class: 4.
  Classes with too few samples (39): cls 0(4), cls 1(7), ...
  Hint: lower K, Q, or N; or use --auto_adjust_q.
```

### Deterministic Sampling

Each episode `e` uses its own RNG:
```python
rng = np.random.RandomState(seed + e)
```

This makes results **perfectly reproducible** regardless of parallelism or
iteration order.

---

## F. Reproducing Results

### F.1 Within-domain evaluation (no pretraining)

This evaluates each dataset independently using the old directory-based
splits (legacy mode, for backward compatibility with `matrix_final.csv`):

```bash
# ASL + LIBRAS + Arabic (eval on test split)
python tools/run_full_matrix.py \
    --datasets asl_alphabet libras_alphabet arabic_sign_alphabet \
    --encoders mlp transformer \
    --representations raw angle raw_angle \
    --shots 1 3 5 --episodes 600 --seed 42 \
    --eval_split test --output results/matrix_3ds.csv

# Thai (eval on train split — legacy: test split too small at 15% ratio)
python tools/run_full_matrix.py \
    --datasets thai_fingerspelling \
    --encoders mlp transformer \
    --representations raw angle raw_angle \
    --shots 1 3 5 --episodes 600 --seed 42 \
    --eval_split train --output results/matrix_thai.csv

# Merge
head -1 results/matrix_3ds.csv > results/matrix_final.csv
tail -n +2 results/matrix_3ds.csv >> results/matrix_final.csv
tail -n +2 results/matrix_thai.csv >> results/matrix_final.csv
```

### F.2 Cross-domain evaluation (new protocol)

Pretrain on a source dataset, evaluate on every target's test split using
the new JSON-based splits:

```bash
# One-command: pretrain on ASL → evaluate on all 4 datasets
bash tools/run_pretrain_and_eval.sh --source asl_alphabet --seed 42

# Or step-by-step:
# 1. Generate splits
for ds in asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling; do
    python tools/make_splits.py --dataset "$ds" --seed 42 --ratio 0.7
done

# 2. Pretrain on ASL (train split, JSON-based)
python train.py --config configs/base.yaml --dataset asl_alphabet \
    --json_splits --save results/checkpoints/best_asl_alphabet.pt

# 3. Evaluate cross-domain on each target (test split)
for target in asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling; do
    python evaluate.py --config configs/base.yaml \
        --cross_domain_eval \
        --source_dataset asl_alphabet \
        --target_dataset "$target" \
        --source_ckpt results/checkpoints/best_asl_alphabet.pt \
        --episodes 600 --seed 42 \
        --json_splits --split test --auto_adjust_q
done
```

### F.3 Training from scratch

```bash
python train.py --config configs/base.yaml --dataset ASL --epochs 100
python train.py --config configs/base.yaml --dataset ASL --json_splits --epochs 100
```

### Key results (within-domain, no pretraining)

| Dataset | Best Config | 1-shot | 3-shot | 5-shot |
|---------|-------------|--------|--------|--------|
| ASL | Transformer / raw_angle | 90.3 ± 0.7 | 94.9 ± 0.4 | 95.4 ± 0.4 |
| LIBRAS | MLP / angle | 88.8 ± 0.7 | 92.9 ± 0.6 | 94.7 ± 0.5 |
| Arabic | MLP / angle | 82.0 ± 0.9 | 89.0 ± 0.6 | 90.6 ± 0.6 |
| Thai | MLP / angle | 60.8 ± 1.1 | 65.5 ± 1.0 | 67.4 ± 1.0 |

Full results: [results/matrix_final.csv](results/matrix_final.csv).

---

## G. Outputs

| File | Content |
|------|---------|
| `splits/<dataset>_train.json` | JSON split: class → list of relative .npy paths |
| `splits/<dataset>_test.json` | Same, for test partition |
| `results/checkpoints/best_<dataset>.pt` | Best model checkpoint (epoch, state_dict, config) |
| `results/cross_domain.csv` | Cross-domain evaluation results (appended per run) |
| `results/matrix_final.csv` | Within-domain evaluation matrix (72 rows) |
| `results/few_shot.csv` | Single-dataset episodic evaluation |
| `results/plots/tsne.png` | t-SNE embedding visualisation |

### CSV schema (cross-domain)

```
source_dataset, target_dataset, encoder, representation,
k_shot, n_way, q_query, episodes, seed, accuracy_mean, ci95, notes
```

---

## Project Structure

```
sign_metric_learning/
├── configs/
│   ├── base.yaml                # Base hyperparameters
│   ├── asl_to_bdsl.yaml         # Cross-lingual transfer config
│   └── reproduce.yaml           # Reproduction manifest
├── data/
│   ├── __init__.py
│   ├── preprocess.py            # MediaPipe landmark extraction
│   ├── datasets.py              # LandmarkDataset, SplitLandmarkDataset
│   └── episodes.py              # Deterministic episodic N-way K-shot sampler
├── models/
│   ├── __init__.py              # Encoder/model factory
│   ├── mlp_encoder.py           # MLP encoder
│   ├── temporal_transformer.py  # Spatial Transformer
│   ├── gcn_encoder.py           # GCN on hand skeleton
│   ├── prototypical.py          # Prototypical Networks
│   └── siamese.py               # Siamese & Matching Networks
├── losses/
│   └── supcon.py                # SupCon, Triplet, ArcFace, loss factory
├── utils/
│   ├── seed.py                  # Deterministic seed utilities
│   ├── logger.py                # Console + file logging
│   └── metrics.py               # Accuracy, CI, confusion matrix, t-SNE
├── tools/
│   ├── make_splits.py           # JSON stratified splits (new protocol)
│   ├── run_pretrain_and_eval.sh # Pretrain → cross-domain eval pipeline
│   ├── run_full_matrix.py       # Encoder × Repr × Shot evaluation matrix
│   ├── smoke_test.py            # Repo health check
│   └── auto_find_download_and_filter_onehand.py
├── splits/                      # Generated JSON split files
├── results/
│   └── matrix_final.csv         # Published evaluation results
├── train.py                     # Episodic training (supports --json_splits)
├── evaluate.py                  # Evaluation & cross-domain eval
├── adapt.py                     # Few-shot adaptation
├── ablation.py                  # Ablation study
├── requirements.txt
├── CITATION.bib
├── LICENSE                      # MIT
└── README.md
```

---

## Smoke Test

```bash
python tools/smoke_test.py          # full check
python tools/smoke_test.py --quick  # imports only
```

---

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Embedding dim | 128 |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Temperature (SupCon) | 0.07 |
| N-way | 5 |
| K-shot | 1, 3, 5 |
| Q-query | 15 |
| Episodes (eval) | 600 |
| Seed | 42 |
| Train/test ratio | 70 / 30 |

All configurable via YAML in `configs/`.

---

## Citation

```bibtex
@inproceedings{geometry_sign_metric_2025,
  title     = {Geometry-Aware Metric Learning for Cross-Lingual Few-Shot
               Sign Language Recognition},
  author    = {Chyanin},
  booktitle = {Workshop on Sign Language Recognition, CVPR},
  year      = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

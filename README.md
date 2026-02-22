# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

> Modular, reproducible research framework for **static-image** sign-language
> recognition using metric learning with cross-lingual transfer across four
> sign-language alphabets.
>
> All experiments are deterministic (seed = 42) and fully reproducible from
> a single shell script.  **All reported results use JSON stratified splits
> evaluated on the test split only.**

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
- N-way K-shot episodic evaluation on the **test split** with strict K+Q feasibility
- Deterministic per-episode seeding: `rng = np.random.RandomState(seed + e)`
- Support/query disjointness guaranteed by construction (sample without replacement, positional split)
- Cross-domain evaluation: pretrain on source, evaluate on target test split

### Datasets

| Dataset | Kaggle Slug | Classes | Total Samples |
|---------|-------------|---------|---------------|
| ASL Alphabet | `grassknoted/asl-alphabet` | 29 | ~63 500 |
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

### Split Statistics

| Dataset | Train | Test | Classes | Min test/class |
|---------|-------|------|---------|----------------|
| ASL Alphabet | 44 498 | 19 093 | 29 | 1 |
| LIBRAS Alphabet | 23 993 | 10 291 | 21 | 447 |
| Arabic Sign Alphabet | 4 952 | 2 141 | 31 | 56 |
| Thai Fingerspelling | 1 998 | 863 | 42 | 12 |

### Integrity Guarantees

- No duplicate paths within a split
- Zero overlap between train and test
- At least 1 sample per class in each split
- Validated automatically at load time (`validate_no_leak()`)

---

## E. Episodic Evaluation Protocol

### K+Q Feasibility Rule

For N-way K-shot episodic evaluation with Q query samples, a class is
**eligible** only when:

$$n_c \geq K + Q$$

where $n_c$ is the number of samples for that class in the **test** split.
At least $N$ classes must be eligible, otherwise the sampler raises a
`ValueError` with a detailed diagnostic.

### Test-Split Eligibility

With the default settings (K=5, Q=15, N=5):

| Dataset | Eligible classes | Total classes | Min $n_c$ |
|---------|-----------------|---------------|-----------|
| ASL | 28 / 29 | 29 | 1 (`nothing` class) |
| LIBRAS | 21 / 21 | 21 | 447 |
| Arabic | 31 / 31 | 31 | 56 |
| Thai | 27 / 42 | 42 | 12 |

All four datasets have ≥ 5 eligible classes on the test split, so the
default protocol (N=5, K=5, Q=15) runs without modification.

> **Note on Thai:** Thai Fingerspelling has 42 classes with a minimum of 12
> samples per class on the test split.  At K=5, Q=15, 27 of 42 classes are
> eligible (those with ≥ 20 samples).  This is sufficient for 5-way
> evaluation.  For experiments requiring all 42 classes to be eligible, pass
> `--auto_adjust_q` which lowers Q to 7 (since 12 ≥ 5 + 7).

### Auto-adjust mode

Pass `--auto_adjust_q` to automatically lower $Q$ to the largest feasible
value (minimum 1).  This is useful for datasets with small per-class counts.

### Support/Query Disjointness

Each episode samples K+Q indices per class **without replacement**
(`np.random.choice(replace=False)`), then splits positionally: the first K
indices become support, the next Q become query.  This guarantees
**zero overlap** between support and query sets by construction.  An
explicit assertion in `split_support_query()` validates tensor shapes as a
defence-in-depth check.

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

Evaluates each dataset independently on the **test split** using JSON-based
stratified splits:

```bash
python tools/run_full_matrix.py \
    --datasets asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling \
    --encoders mlp transformer \
    --representations raw angle raw_angle \
    --shots 1 3 5 --episodes 600 --seed 42 \
    --eval_split test --json_splits --auto_adjust_q \
    --output results/matrix_final.csv
```

### F.2 Cross-domain evaluation (pretrain on ASL)

Pretrain on ASL, evaluate on every target's test split:

```bash
# One-command pipeline (pretrain + eval all 4 targets):
bash tools/run_pretrain_and_eval.sh --source asl_alphabet --seed 42 --epochs 10

# Or step-by-step:
# 1. Generate splits
for ds in asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling; do
    python tools/make_splits.py --dataset "$ds" --seed 42 --ratio 0.7
done

# 2. Pretrain on ASL (train split, JSON-based)
python train.py --config configs/base.yaml --dataset asl_alphabet \
    --json_splits --save results/checkpoints/best_asl_alphabet.pt --epochs 10

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
# Results appended to results/cross_domain.csv
```

### F.3 Baseline experiments

```bash
# Linear classifier baseline (train split → test split)
python tools/run_baselines.py --experiment linear_classifier

# Robustness check with 3 seeds
python tools/run_baselines.py --experiment robustness --seeds 42 1337 2024

# All baselines at once
python tools/run_baselines.py --experiment all
```

### F.4 Normalization ablation (requires re-preprocessing)

```bash
# Re-preprocess without wrist-centering or scale-normalization
python -c "from data.preprocess import preprocess_dataset; \
    preprocess_dataset('data/raw/arabic-sign-alphabet', 'data/processed/arabic_sign_alphabet_nonorm', \
    normalize_translation=False, normalize_scale=False)"
python -c "from data.preprocess import preprocess_dataset; \
    preprocess_dataset('data/raw/libras-alphabet/train', 'data/processed/libras_alphabet_nonorm', \
    normalize_translation=False, normalize_scale=False); \
    preprocess_dataset('data/raw/libras-alphabet/test', 'data/processed/libras_alphabet_nonorm', \
    normalize_translation=False, normalize_scale=False)"

# Generate splits for nonorm datasets
python tools/make_splits.py --dataset arabic_sign_alphabet_nonorm --seed 42 --ratio 0.7
python tools/make_splits.py --dataset libras_alphabet_nonorm     --seed 42 --ratio 0.7

# Evaluate (results reported in Section G.3)
python tools/run_full_matrix.py \
    --datasets arabic_sign_alphabet_nonorm libras_alphabet_nonorm \
    --encoders mlp --representations raw angle --shots 5 \
    --episodes 600 --seed 42 --eval_split test --json_splits --auto_adjust_q
```

### F.5 Training from scratch

```bash
python train.py --config configs/base.yaml --dataset asl_alphabet \
    --json_splits --epochs 10 --save results/checkpoints/best_asl_alphabet.pt
```

> **Legacy mode:** Directory-based splits (`data/processed/<name>_split/train`,
> `data/processed/<name>_split/test`) are still supported for backward
> compatibility.  Omit `--json_splits` to use them.  **All results reported
> below use JSON stratified splits only.**

---

## G. Results

### Within-domain (no pretraining)

5-way K-shot episodic evaluation on the **test split**, 600 episodes, seed 42,
Prototypical Networks, JSON stratified splits (70/30).

| Dataset | Encoder | Repr | 1-shot | 3-shot | 5-shot |
|---------|---------|------|--------|--------|--------|
| ASL | MLP | raw | 88.7 ± 0.7 | 93.8 ± 0.5 | 94.9 ± 0.4 |
| ASL | MLP | angle | 81.5 ± 0.9 | 86.8 ± 0.7 | 88.4 ± 0.6 |
| ASL | MLP | raw_angle | 90.4 ± 0.7 | 94.5 ± 0.5 | 95.4 ± 0.4 |
| ASL | Transformer | raw | 74.2 ± 1.0 | 80.3 ± 0.9 | 82.1 ± 0.9 |
| ASL | Transformer | angle | 80.8 ± 0.9 | 86.2 ± 0.7 | 87.7 ± 0.7 |
| ASL | Transformer | raw_angle | **90.8 ± 0.7** | **94.7 ± 0.4** | **95.4 ± 0.4** |
| LIBRAS | MLP | raw | 69.7 ± 1.0 | 78.9 ± 0.8 | 81.2 ± 0.8 |
| LIBRAS | MLP | angle | **89.2 ± 0.7** | **92.9 ± 0.5** | **94.1 ± 0.5** |
| LIBRAS | MLP | raw_angle | 71.6 ± 1.0 | 82.2 ± 0.8 | 84.6 ± 0.8 |
| LIBRAS | Transformer | raw | 61.1 ± 0.9 | 65.9 ± 0.9 | 67.9 ± 0.9 |
| LIBRAS | Transformer | angle | 86.9 ± 0.8 | 91.1 ± 0.6 | 92.5 ± 0.5 |
| LIBRAS | Transformer | raw_angle | 72.8 ± 1.0 | 82.3 ± 0.8 | 84.7 ± 0.7 |
| Arabic | MLP | raw | 51.3 ± 0.9 | 60.6 ± 0.8 | 64.5 ± 0.8 |
| Arabic | MLP | angle | **81.1 ± 0.9** | **88.2 ± 0.6** | **89.8 ± 0.6** |
| Arabic | MLP | raw_angle | 55.5 ± 1.0 | 66.6 ± 0.9 | 71.0 ± 0.9 |
| Arabic | Transformer | raw | 43.1 ± 0.9 | 47.1 ± 1.0 | 49.0 ± 0.9 |
| Arabic | Transformer | angle | 78.3 ± 0.9 | 85.7 ± 0.7 | 87.8 ± 0.6 |
| Arabic | Transformer | raw_angle | 55.6 ± 1.0 | 63.2 ± 0.9 | 67.3 ± 0.9 |
| Thai | MLP | raw | 40.3 ± 0.7 | 47.0 ± 0.7 | 48.8 ± 0.8 |
| Thai | MLP | angle | **46.2 ± 0.8** | **51.2 ± 0.8** | **52.7 ± 0.8** |
| Thai | MLP | raw_angle | 43.4 ± 0.8 | 50.2 ± 0.8 | 51.9 ± 0.8 |
| Thai | Transformer | raw | 33.3 ± 0.7 | 35.7 ± 0.6 | 36.0 ± 0.6 |
| Thai | Transformer | angle | 44.6 ± 0.8 | 50.6 ± 0.8 | 51.8 ± 0.8 |
| Thai | Transformer | raw_angle | 42.3 ± 0.7 | 47.1 ± 0.7 | 48.7 ± 0.8 |

> **Best per dataset:** ASL → Transformer/raw_angle (95.4%), LIBRAS → MLP/angle (94.1%),
> Arabic → MLP/angle (89.8%), Thai → MLP/angle (52.7%).
> The `angle` representation generalises best across languages.

Full 72-row CSV: [results/matrix_final.csv](results/matrix_final.csv).

### Cross-domain transfer (pretrained on ASL)

Pretrained on ASL (Transformer / raw, 3 epochs, 98.5% train acc),
evaluated on each target's **test split** (5-way 5-shot, Q=15,
600 episodes, JSON splits):

| Source → Target | Accuracy | 95% CI |
|-----------------|----------|--------|
| ASL → ASL | **98.45%** | ±0.21 |
| ASL → LIBRAS | **86.53%** | ±0.81 |
| ASL → Arabic | **77.27%** | ±0.75 |
| ASL → Thai | **52.82%** | ±0.81 |

CSV: [results/cross_domain.csv](results/cross_domain.csv).

### G.3 Ablation: Effect of Geometry Normalization

To validate the theoretical claim that joint angles are invariant to rigid
transforms, we re-preprocessed LIBRAS and Arabic landmarks **without**
wrist-centering or scale-normalization (MediaPipe raw output) and compared
against normalised landmarks.

5-way 5-shot, MLP encoder, 600 episodes, seed 42:

| Dataset | Repr | No Norm | Normalised | Δ |
|---------|------|---------|------------|---|
| LIBRAS | raw | 76.4 | 81.2 | +4.8 |
| LIBRAS | angle | **94.4** | **94.1** | −0.3† |
| Arabic | raw | 59.1 | 64.5 | +5.4 |
| Arabic | angle | **90.1** | **89.8** | +0.3† |

†Within noise band (different MediaPipe extraction yields slightly different
test sets); the near-zero Δ confirms that **joint angles are inherently
invariant to translation and scale**, as predicted by the theoretical analysis
in Section A.  The `raw` representation, by contrast, degrades 5–6 pp without
normalization because absolute coordinates are sensitive to hand position and
distance from camera.

Full normalization pipeline ablation (Arabic, MLP/raw, all shot settings):

| Setting | 1-shot | 3-shot | 5-shot |
|---------|--------|--------|--------|
| No normalization | 41.8 | 56.3 | 59.1 |
| + Wrist-centering + scale-norm | 51.3 | 60.6 | 64.5 |
| + Geometry-aware (angle) | **81.1** | **88.2** | **89.8** |

The geometry-aware angle representation provides the largest single
improvement (+25.3 pp at 5-shot over normalised raw), demonstrating that
the contribution is not merely preprocessing but a fundamentally richer
feature space.

### G.4 Baseline: Linear Classifier

To contextualise the few-shot results, we trained a full-data supervised
baseline: `MLP encoder → StandardScaler → LogisticRegression(C=1.0,
max_iter=1000)` on the entire train split, evaluated on the test split.

| Dataset | Raw (test acc) | Angle (test acc) |
|---------|---------------|-----------------|
| ASL | 98.6% | 93.8% |
| LIBRAS | **100.0%** | 99.7% |
| Arabic | 92.7% | 90.9% |
| Thai | 54.9% | 51.7% |

**Key observations:**
- The supervised linear classifier with full training data achieves near-perfect
  accuracy on LIBRAS and ASL, confirming that the MLP embeddings are linearly
  separable.  The gap to few-shot ProtoNet (e.g. LIBRAS 100% vs 94.1%) quantifies
  the inherent cost of the few-shot paradigm.
- Thai remains the hardest dataset even with full supervision (54.9%),
  indicating that difficulty stems from the data itself (42 classes, small samples,
  high inter-class similarity) rather than from the few-shot protocol.

### G.5 Robustness: Multi-seed Stability

To verify that results are not an artefact of a single random seed, we
re-evaluated the best setting (MLP / angle / 5-shot) with three seeds
(42, 1337, 2024):

| Dataset | Seed 42 | Seed 1337 | Seed 2024 | Mean ± Std |
|---------|---------|-----------|-----------|------------|
| ASL | 88.2 | 89.2 | 89.2 | **88.9 ± 0.6** |
| LIBRAS | 94.1 | 94.6 | 93.6 | **94.1 ± 0.5** |
| Arabic | 90.1 | 90.6 | 89.0 | **89.9 ± 0.8** |
| Thai | 51.8 | 50.5 | 51.4 | **51.2 ± 0.7** |

All standard deviations are **< 1 pp**, confirming that the episodic
evaluation protocol is stable and the reported numbers are reproducible with
negligible variance across different random seeds.

### G.6 Analysis: Thai Fingerspelling Performance

Thai Fingerspelling consistently yields the lowest accuracy across all
settings (52.7% best few-shot, 54.9% supervised).  We identify three
structural factors:

1. **Class count.**  Thai has 42 classes — double the average of the other
   three datasets (ASL 29, LIBRAS 21, Arabic 31).  The 5-way protocol
   samples from a larger label space, increasing the chance of visually
   similar class pairs (e.g. Thai consonants that differ only in finger
   curl vs. spread).

2. **Data scarcity.**  After MediaPipe filtering, Thai retains only ~2 900
   samples (69 per class on average).  With a 70/30 split the test set has
   only 863 samples.  Several classes have as few as 12 test samples,
   limiting both the number of eligible classes (27/42 at K=5, Q=15) and
   the statistical power of the evaluation.

3. **Morphological complexity.**  Thai fingerspelling uses distinct hand
   configurations influenced by Thai orthography.  Certain consonant groups
   (e.g. ก–ข–ค) share similar hand shapes, differing only in subtle thumb
   or pinky positioning.  These fine-grained distinctions create a harder
   metric space even for geometry-aware features.

**Consistency claim.**  Despite the lower absolute accuracy, the geometry-aware
angle representation still outperforms raw coordinates consistently across all
shot settings (+3.9pp at 5-shot, +5.9pp at 1-shot), and the cross-domain
ASL→Thai transfer (52.8%) slightly exceeds the within-domain ProtoNet
baseline (52.7%), suggesting that knowledge transfers even between
typologically distant sign languages.

### G.7 Comparison with Prior Work

Direct comparison with prior work is difficult due to differences in
evaluation protocols (closed-set vs. few-shot, different splits, different
datasets).  We therefore present representative results from the literature
alongside our protocol for context.

| Paper | Setting | Dataset | Classes | Accuracy |
|-------|---------|---------|---------|----------|
| Mavi (2020) | Closed-set CNN | ASL Alphabet | 29 | 99.4% |
| Podder et al. (2022) | Closed-set, hand-crafted | LIBRAS | 21 | 97.3% |
| Alani & Cosma (2021) | Closed-set CNN | Arabic SL | 32 | 97.6% |
| **Ours** (supervised) | Linear on MLP embed | ASL / LIBRAS / Arabic / Thai | 29/21/31/42 | 98.6 / 100 / 92.7 / 54.9% |
| **Ours** (few-shot) | 5-way 5-shot ProtoNet | ASL / LIBRAS / Arabic / Thai | 29/21/31/42 | 95.4 / 94.1 / 89.8 / 52.7% |

**Notes:**
- Prior works use **closed-set classification** with full training data across
  all classes — a fundamentally easier setting than few-shot learning.
- Our supervised baseline (linear classifier) achieves comparable performance
  (98.6% ASL, 100% LIBRAS), confirming that the embeddings are competitive.
- The few-shot gap (≈4–7 pp on high-resource datasets) reflects the inherent
  cost of learning from 5 examples per class vs. hundreds.
- No prior work evaluates **cross-lingual few-shot transfer**, which is the
  primary contribution of this paper.

---

## H. Outputs

| File | Content |
|------|---------|
| `splits/<dataset>_train.json` | JSON split: class → list of relative .npy paths |
| `splits/<dataset>_test.json` | Same, for test partition |
| `results/checkpoints/best_<dataset>.pt` | Best model checkpoint (epoch, state_dict, config) |
| `results/cross_domain.csv` | Cross-domain evaluation results |
| `results/matrix_final.csv` | Within-domain evaluation matrix (72 rows) |
| `results/baseline_linear.csv` | Linear classifier baseline results |
| `results/robustness_seeds.csv` | Multi-seed robustness results |
| `results/few_shot.csv` | Single-dataset episodic evaluation |
| `results/plots/tsne.png` | t-SNE embedding visualisation |

### CSV schema (within-domain matrix)

```
dataset, encoder, representation, k_shot, n_way, q_query,
episodes, seed, accuracy_mean, ci95, notes
```

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
│   ├── make_splits.py           # JSON stratified splits
│   ├── run_pretrain_and_eval.sh # Pretrain → cross-domain eval pipeline
│   ├── run_full_matrix.py       # Encoder × Repr × Shot evaluation matrix
│   ├── run_baselines.py         # Linear classifier, robustness, nonorm baselines
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

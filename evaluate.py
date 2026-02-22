"""
Evaluation script for sign language few-shot recognition.

Supports zero-shot evaluation and standard episodic evaluation.

Usage:
    python evaluate.py --dataset BdSL --zero-shot
    python evaluate.py --dataset ASL --config configs/base.yaml
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LandmarkDataset, SyntheticLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from models import build_encoder, build_few_shot_model
from utils.logger import get_logger
from utils.metrics import (
    accuracy,
    compute_confusion_matrix,
    cross_domain_accuracy_drop,
    few_shot_accuracy_with_ci,
    plot_confusion_matrix,
    plot_tsne,
)
from utils.seed import set_seed
from train import get_dataset, get_device, load_config


@torch.no_grad()
def collect_embeddings(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """Extract embeddings for the full dataset.

    Args:
        model: Model with ``get_embeddings`` method.
        dataset: Dataset yielding ``(tensor, label)``.
        device: Torch device.
        batch_size: Batch size for extraction.

    Returns:
        Tuple of ``(embeddings_np, labels_np)`` arrays.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_emb, all_lbl = [], []
    for data, labels in tqdm(loader, desc="Extracting embeddings", leave=False):
        data = data.to(device)
        emb = model.get_embeddings(data)
        all_emb.append(emb.cpu().numpy())
        all_lbl.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
    return np.concatenate(all_emb), np.concatenate(all_lbl)


@torch.no_grad()
def zero_shot_evaluate(
    model,
    source_dataset,
    target_dataset,
    device: torch.device,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    episodes: int = 1000,
) -> Dict[str, float]:
    """Zero-shot cross-domain evaluation.

    Builds prototypes from the *source* support set and classifies *target*
    queries. This evaluates how well the embedding space transfers across
    languages without any adaptation.

    Args:
        model: Few-shot model.
        source_dataset: Source language dataset (e.g., ASL).
        target_dataset: Target language dataset (e.g., BdSL).
        device: Device.
        n_way: Ways.
        k_shot: Shots from source.
        q_query: Queries from target.
        episodes: Number of evaluation episodes.

    Returns:
        Dict with accuracy metrics.
    """
    model.eval()

    source_labels = [source_dataset[i][1] for i in range(len(source_dataset))]
    target_labels = [target_dataset[i][1] for i in range(len(target_dataset))]

    # Build class index maps
    src_cls_idx = {}
    for i, lbl in enumerate(source_labels):
        src_cls_idx.setdefault(lbl, []).append(i)
    tgt_cls_idx = {}
    for i, lbl in enumerate(target_labels):
        tgt_cls_idx.setdefault(lbl, []).append(i)

    # Use overlapping class count
    n_common = min(n_way, len(src_cls_idx), len(tgt_cls_idx))
    src_classes = sorted(src_cls_idx.keys())[:n_common]
    tgt_classes = sorted(tgt_cls_idx.keys())[:n_common]

    accs = []
    import random
    for _ in tqdm(range(episodes), desc="Zero-shot eval"):
        # Sample support from source, query from target
        support_data, support_labels_ep = [], []
        query_data, query_labels_ep = [], []

        for new_lbl, (sc, tc) in enumerate(zip(src_classes, tgt_classes)):
            s_idxs = random.sample(src_cls_idx[sc], min(k_shot, len(src_cls_idx[sc])))
            q_idxs = random.sample(tgt_cls_idx[tc], min(q_query, len(tgt_cls_idx[tc])))
            for si in s_idxs:
                support_data.append(source_dataset[si][0])
                support_labels_ep.append(new_lbl)
            for qi in q_idxs:
                query_data.append(target_dataset[qi][0])
                query_labels_ep.append(new_lbl)

        sx = torch.stack(support_data).to(device)
        sy = torch.tensor(support_labels_ep, device=device)
        qx = torch.stack(query_data).to(device)
        qy = torch.tensor(query_labels_ep, device=device)

        log_probs = model(sx, sy, qx, n_common)
        acc = accuracy(log_probs, qy)
        accs.append(acc)

    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return {"accuracy": mean_acc, "ci": ci, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description="Evaluate sign language model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset", type=str, default="ASL")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)

    os.makedirs(cfg.get("output_dir", "results"), exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    logger = get_logger("evaluate", cfg.get("log_file", "results/eval.log"))

    # Build model
    representation = cfg.get("representation", "raw")
    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(
            cfg.get("checkpoint_dir", "results/checkpoints"),
            f"best_{cfg['dataset']['name'].lower()}.pt",
        )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, using untrained model")

    fs_cfg = cfg["few_shot"]
    n_way = fs_cfg["n_way"]
    k_shot = fs_cfg["k_shot"]
    q_query = fs_cfg["q_query"]

    if args.zero_shot:
        # ── Zero-shot cross-domain ──
        logger.info("Running zero-shot cross-domain evaluation")
        source_ds = get_dataset(cfg, split="test")

        # Build target dataset
        target_cfg = cfg.copy()
        target_cfg["dataset"] = {
            **cfg["dataset"],
            "name": args.dataset,
            "root": f"data/raw/{args.dataset.lower()}",
            "num_classes": 60,
        }
        target_ds = get_dataset(target_cfg, split="test")

        results = zero_shot_evaluate(
            model, source_ds, target_ds, device,
            n_way, k_shot, q_query, args.episodes,
        )
        logger.info(f"Zero-shot accuracy: {results['accuracy']:.4f} ± {results['ci']:.4f}")

        # Save results
        csv_path = os.path.join(cfg.get("output_dir", "results"), "zero_shot.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            for k, v in results.items():
                writer.writerow({"metric": k, "value": v})
        logger.info(f"Results saved to {csv_path}")

    else:
        # ── Standard episodic evaluation ──
        logger.info(f"Running episodic evaluation on {args.dataset}")
        eval_ds = get_dataset(cfg, split="test")
        eval_labels = [eval_ds[i][1] for i in range(len(eval_ds))]

        eval_sampler = EpisodicSampler(
            eval_labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes=args.episodes,
        )
        eval_loader = DataLoader(
            eval_ds, batch_sampler=eval_sampler, collate_fn=collate_episode,
        )

        from train import evaluate_episodes
        results = evaluate_episodes(model, eval_loader, device, n_way, k_shot, q_query)
        logger.info(f"Accuracy: {results['accuracy']:.4f} ± {results['ci']:.4f}")

        # Embeddings & plots
        embeddings, labels = collect_embeddings(model, eval_ds, device)
        plot_tsne(embeddings, labels, save_path="results/plots/tsne.png")
        logger.info("t-SNE plot saved to results/plots/tsne.png")

        # Save results
        csv_path = os.path.join(cfg.get("output_dir", "results"), "few_shot.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            for k, v in results.items():
                writer.writerow({"metric": k, "value": v})
        logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

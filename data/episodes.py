"""
Episodic data sampler for N-way K-shot few-shot learning.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class EpisodicSampler(Sampler):
    """Sampler that yields indices for N-way K-shot episodes.

    Each episode selects *n_way* classes and for each class samples
    *k_shot + q_query* examples.

    Args:
        labels: List of integer labels for every sample in the dataset.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        episodes: Number of episodes to generate.
    """

    def __init__(
        self,
        labels: List[int],
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        episodes: int = 1000,
    ) -> None:
        super().__init__()
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        # Build class → indices mapping
        self.class_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(self.labels):
            self.class_indices.setdefault(int(lbl), []).append(idx)

        # Filter classes that have enough samples
        self.valid_classes = [
            c for c, idxs in self.class_indices.items()
            if len(idxs) >= k_shot + q_query
        ]
        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes with >= {k_shot + q_query} samples. "
                f"Found {len(self.valid_classes)}, need {n_way}."
            )

    def __iter__(self):
        for _ in range(self.episodes):
            episode_classes = random.sample(self.valid_classes, self.n_way)
            indices: List[int] = []
            for cls in episode_classes:
                cls_idxs = random.sample(
                    self.class_indices[cls], self.k_shot + self.q_query,
                )
                indices.extend(cls_idxs)
            yield indices

    def __len__(self) -> int:
        return self.episodes


def split_support_query(
    batch: Tuple[torch.Tensor, torch.Tensor],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat episode batch into support and query sets.

    The batch is assumed to be ordered: for each of the *n_way* classes,
    the first *k_shot* samples are support, the next *q_query* are query.

    Args:
        batch: Tuple of (data_tensor, label_tensor).
        n_way: Number of classes.
        k_shot: Support shots.
        q_query: Query shots.

    Returns:
        (support_x, support_y, query_x, query_y)
    """
    data, labels = batch
    per_class = k_shot + q_query

    support_x, support_y, query_x, query_y = [], [], [], []

    for i in range(n_way):
        start = i * per_class
        s_end = start + k_shot
        q_end = s_end + q_query

        support_x.append(data[start:s_end])
        support_y.append(labels[start:s_end])
        query_x.append(data[s_end:q_end])
        query_y.append(labels[s_end:q_end])

    support_x = torch.cat(support_x, dim=0)
    support_y = torch.cat(support_y, dim=0)
    query_x = torch.cat(query_x, dim=0)
    query_y = torch.cat(query_y, dim=0)

    # Re-label to 0..n_way-1
    unique_labels = support_y.unique()
    label_map = {int(old): new for new, old in enumerate(unique_labels.tolist())}
    support_y = torch.tensor([label_map[int(l)] for l in support_y])
    query_y = torch.tensor([label_map[int(l)] for l in query_y])

    return support_x, support_y, query_x, query_y


def collate_episode(batch):
    """Custom collate that stacks episode samples into a single tensor.

    Args:
        batch: List of lists of (tensor, label) from EpisodicSampler.

    Returns:
        (data, labels) tensors.
    """
    # batch is a list of (tensor, label) tuples from a single episode
    data = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return data, labels

"""
PyTorch dataset classes for preprocessed hand-landmark sequences.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import compute_pairwise_distances


class LandmarkDataset(Dataset):
    """Dataset that loads ``.npy`` hand-landmark files.

    Folder layout::

        root/
            class_0/
                sample1.npy   # shape (T, 21, 3)
                sample2.npy
            class_1/
                ...

    Args:
        root: Root directory of preprocessed ``.npy`` files.
        representation: One of ``'raw'``, ``'pairwise'``, ``'graph'``.
        sequence_length: Expected temporal length.
        transform: Optional callable applied to the tensor.
    """

    def __init__(
        self,
        root: str,
        representation: str = "raw",
        sequence_length: int = 32,
        transform=None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.representation = representation
        self.sequence_length = sequence_length
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes: List[str] = []

        self._scan()

    def _scan(self) -> None:
        """Discover classes and samples from the directory tree."""
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        for idx, d in enumerate(class_dirs):
            cls_name = d.name
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = idx
            self.idx_to_class[idx] = cls_name
            for npy in sorted(d.glob("*.npy")):
                self.samples.append((str(npy), idx))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        lm = np.load(path).astype(np.float32)  # (T, 21, 3)

        # Ensure fixed length
        if lm.shape[0] > self.sequence_length:
            lm = lm[: self.sequence_length]
        elif lm.shape[0] < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - lm.shape[0], lm.shape[1], lm.shape[2]),
                dtype=np.float32,
            )
            lm = np.concatenate([lm, pad], axis=0)

        if self.representation == "pairwise":
            lm = compute_pairwise_distances(lm)  # (T, 210)
        elif self.representation == "graph":
            pass  # kept as (T, 21, 3) — GCN handles adjacency
        else:
            pass  # raw: (T, 21, 3)

        tensor = torch.from_numpy(lm)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label


class SyntheticLandmarkDataset(Dataset):
    """Synthetic dataset for testing and demo purposes.

    Generates random landmark sequences for *num_classes* classes, where
    each class is centred at a distinct random prototype.

    Args:
        num_classes: Number of classes.
        samples_per_class: Samples generated per class.
        sequence_length: Temporal length.
        representation: ``'raw'`` | ``'pairwise'`` | ``'graph'``.
        noise_std: Standard deviation of Gaussian noise around prototypes.
    """

    def __init__(
        self,
        num_classes: int = 100,
        samples_per_class: int = 20,
        sequence_length: int = 32,
        representation: str = "raw",
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.sequence_length = sequence_length
        self.representation = representation
        self.noise_std = noise_std

        # Create class prototypes
        rng = np.random.RandomState(0)
        self.prototypes = [
            rng.randn(sequence_length, 21, 3).astype(np.float32)
            for _ in range(num_classes)
        ]

        # Build sample list
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.classes = [str(i) for i in range(num_classes)]
        self.class_to_idx = {str(i): i for i in range(num_classes)}
        self.idx_to_class = {i: str(i) for i in range(num_classes)}

        for cls_idx in range(num_classes):
            proto = self.prototypes[cls_idx]
            for _ in range(samples_per_class):
                sample = proto + rng.randn(*proto.shape).astype(np.float32) * noise_std
                self.samples.append((sample, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        lm, label = self.samples[index]
        lm = lm.copy()

        if self.representation == "pairwise":
            lm = compute_pairwise_distances(lm)  # (T, 210)

        tensor = torch.from_numpy(lm)
        return tensor, label

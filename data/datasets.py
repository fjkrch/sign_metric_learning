"""
PyTorch dataset classes for preprocessed hand-landmark data (static images).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import compute_pairwise_distances, compute_joint_angles, compute_raw_angle


class LandmarkDataset(Dataset):
    """Dataset that loads ``.npy`` hand-landmark files (static images).

    Folder layout::

        root/
            class_0/
                sample1.npy   # shape (21, 3)
                sample2.npy
            class_1/
                ...

    Args:
        root: Root directory of preprocessed ``.npy`` files.
        representation: One of ``'raw'``, ``'pairwise'``, ``'graph'``.
        transform: Optional callable applied to the tensor.
    """

    def __init__(
        self,
        root: str,
        representation: str = "raw",
        transform=None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.representation = representation
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
        lm = np.load(path).astype(np.float32)  # (21, 3)

        # Handle legacy (T, 21, 3) files — take first frame
        if lm.ndim == 3:
            lm = lm[0]

        if self.representation == "pairwise":
            lm = compute_pairwise_distances(lm)  # (210,)
        elif self.representation == "angle":
            lm = compute_joint_angles(lm)          # (20,)
        elif self.representation == "raw_angle":
            lm = compute_raw_angle(lm)              # (83,)
        # raw / graph: kept as (21, 3)

        tensor = torch.from_numpy(lm)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label


class SyntheticLandmarkDataset(Dataset):
    """Synthetic dataset for testing and demo purposes.

    Generates random landmark data for *num_classes* classes, where
    each class is centred at a distinct random prototype (single static image).

    Args:
        num_classes: Number of classes.
        samples_per_class: Samples generated per class.
        representation: ``'raw'`` | ``'pairwise'`` | ``'graph'``.
        noise_std: Standard deviation of Gaussian noise around prototypes.
    """

    def __init__(
        self,
        num_classes: int = 100,
        samples_per_class: int = 20,
        representation: str = "raw",
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.representation = representation
        self.noise_std = noise_std

        # Create class prototypes — single frame (21, 3)
        rng = np.random.RandomState(0)
        self.prototypes = [
            rng.randn(21, 3).astype(np.float32)
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
            lm = compute_pairwise_distances(lm)  # (210,)
        elif self.representation == "angle":
            lm = compute_joint_angles(lm)          # (20,)
        elif self.representation == "raw_angle":
            lm = compute_raw_angle(lm)              # (83,)

        tensor = torch.from_numpy(lm)
        return tensor, label

"""Dataset loading, label alignment, and DataLoader construction.

Wraps torchxrayvision dataset classes with automatic label alignment
to the 7 common pathologies shared between CheXpert and NIH.
"""

from __future__ import annotations

import torch
import numpy as np
import torchxrayvision as xrv
from torch.utils.data import DataLoader

from .pathologies import COMMON_PATHOLOGIES


def load_and_align_dataset(
    dataset_name: str,
    imgpath: str,
    views: list[str] | None = None,
    unique_patients: bool = True,
) -> xrv.datasets.Dataset:
    """Load a torchxrayvision dataset and realign labels to COMMON_PATHOLOGIES.

    After this call:
      - dataset.pathologies == COMMON_PATHOLOGIES
      - dataset.labels has shape [N, 7] with columns in COMMON_PATHOLOGIES order
      - NaN entries indicate missing/uncertain labels

    Args:
        dataset_name: "chexpert" or "nih"
        imgpath: Path to the image directory
        views: X-ray view positions to include (default: ["PA", "AP"])
        unique_patients: Keep only one image per patient (default: True)
    """
    if views is None:
        views = ["PA", "AP"]

    if dataset_name == "chexpert":
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=imgpath,
            views=views,
            unique_patients=unique_patients,
        )
    elif dataset_name == "nih":
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=imgpath,
            views=views,
            unique_patients=unique_patients,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Use 'chexpert' or 'nih'.")

    xrv.datasets.relabel_dataset(COMMON_PATHOLOGIES, dataset, silent=False)
    return dataset


def collate_xrv(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate for torchxrayvision dict-based samples.

    Returns:
        imgs: [B, 1, H, W] float tensor
        labels: [B, 7] float tensor (may contain NaN)
        indices: [B] long tensor of dataset indices
    """
    imgs = torch.stack([torch.from_numpy(s["img"]).float() for s in batch])
    labels = torch.stack([torch.from_numpy(s["lab"]).float() for s in batch])
    indices = torch.tensor([s["idx"] for s in batch], dtype=torch.long)
    return imgs, labels, indices


def make_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader with the xrv collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_xrv,
        pin_memory=True,
    )

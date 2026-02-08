"""Feature extraction and caching.

Extracts 1024-dim feature vectors from a frozen DenseNet121 backbone
(avgpool layer) and saves them as compressed .npz files for fast
downstream experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import make_dataloader
from .pathologies import COMMON_PATHOLOGIES


@dataclass
class ExtractedFeatures:
    """Container for extracted features and metadata."""

    features: np.ndarray     # [N, 1024]
    labels: np.ndarray       # [N, num_pathologies]
    indices: np.ndarray      # [N]
    pathologies: list[str]
    dataset_name: str
    model_weights: str

    def save(self, path: Path) -> None:
        """Save to a compressed .npz file."""
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            indices=self.indices,
            pathologies=np.array(self.pathologies),
            dataset_name=np.array(self.dataset_name),
            model_weights=np.array(self.model_weights),
        )

    @classmethod
    def load(cls, path: Path) -> ExtractedFeatures:
        """Load from a .npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            features=data["features"],
            labels=data["labels"],
            indices=data["indices"],
            pathologies=list(data["pathologies"]),
            dataset_name=str(data["dataset_name"]),
            model_weights=str(data["model_weights"]),
        )


def load_model(weights: str = "densenet121-res224-chex", device: str = "mps") -> xrv.models.DenseNet:
    """Load a pretrained DenseNet and freeze it.

    Args:
        weights: Model weight identifier. Use "densenet121-res224-chex" for
                 the source-domain (CheXpert-trained) model.
        device: Target device ("mps", "cuda", or "cpu").
    """
    model = xrv.models.DenseNet(weights=weights)
    model = model.to(device)
    model.eval()
    return model


def extract_features(
    model: xrv.models.DenseNet,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 1024-dim features from model.features2() for all samples.

    Returns:
        features: [N, 1024] float32
        labels: [N, num_pathologies] float32 (may contain NaN)
        indices: [N] int64
    """
    all_features = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for imgs, labels, indices in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model.features2(imgs)  # [B, 1024]

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_indices, axis=0),
    )


def extract_and_save(
    dataset_name: str,
    dataset,
    model: xrv.models.DenseNet,
    model_weights: str,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "mps",
) -> ExtractedFeatures:
    """Full pipeline: DataLoader → extract → save.

    Args:
        dataset_name: "chexpert" or "nih"
        dataset: A torchxrayvision dataset (already label-aligned)
        model: Frozen DenseNet model
        model_weights: Weight identifier string (for metadata)
        output_dir: Directory to save .npz file
        batch_size: Batch size for extraction
        num_workers: DataLoader worker count
        device: Target device
    """
    dataloader = make_dataloader(dataset, batch_size, num_workers, shuffle=False)
    features, labels, indices = extract_features(model, dataloader, device)

    result = ExtractedFeatures(
        features=features,
        labels=labels,
        indices=indices,
        pathologies=list(COMMON_PATHOLOGIES),
        dataset_name=dataset_name,
        model_weights=model_weights,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{dataset_name}_{model_weights}_features.npz"
    result.save(save_path)
    print(f"Saved {len(features)} features to {save_path}")

    return result

#!/usr/bin/env python3
"""Feature extraction CLI for WCP-L2D.

Usage:
    uv run python scripts/extract_features.py --dataset chexpert --imgpath data/CheXpert-v1.0-small
    uv run python scripts/extract_features.py --dataset nih --imgpath data/NIH/images
    uv run python scripts/extract_features.py --dataset nih --imgpath data/NIH/images --device cpu --batch-size 16
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from wcp_l2d.data import load_and_align_dataset
from wcp_l2d.features import load_model, extract_and_save


def main():
    parser = argparse.ArgumentParser(description="Extract DenseNet121 features from chest X-ray datasets")
    parser.add_argument("--dataset", required=True, choices=["chexpert", "nih"],
                        help="Dataset to extract features from")
    parser.add_argument("--imgpath", required=True, type=str,
                        help="Path to image directory")
    parser.add_argument("--weights", default="densenet121-res224-chex",
                        help="Model weight identifier (default: densenet121-res224-chex)")
    parser.add_argument("--output-dir", default="data/features", type=str,
                        help="Directory to save extracted features")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="mps", type=str,
                        help="Device: mps, cuda, or cpu")
    parser.add_argument("--views", nargs="+", default=["PA", "AP"],
                        help="X-ray view positions to include")
    args = parser.parse_args()

    # Verify device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load dataset
    print(f"Loading {args.dataset} dataset from {args.imgpath}...")
    dataset = load_and_align_dataset(
        dataset_name=args.dataset,
        imgpath=args.imgpath,
        views=args.views,
    )
    print(f"  Loaded {len(dataset)} samples")
    print(f"  Pathologies: {dataset.pathologies}")

    # Load model
    print(f"Loading model: {args.weights}...")
    model = load_model(args.weights, device)

    # Extract and save
    result = extract_and_save(
        dataset_name=args.dataset,
        dataset=dataset,
        model=model,
        model_weights=args.weights,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Summary
    print(f"\nExtraction complete:")
    print(f"  Features shape: {result.features.shape}")
    print(f"  Labels shape:   {result.labels.shape}")
    print(f"  NaN label fractions per pathology:")
    for i, p in enumerate(result.pathologies):
        nan_frac = np.isnan(result.labels[:, i]).mean()
        pos_frac = (result.labels[:, i] == 1.0).mean()
        print(f"    {p:20s}  NaN: {nan_frac:5.1%}  Positive: {pos_frac:5.1%}")


if __name__ == "__main__":
    main()

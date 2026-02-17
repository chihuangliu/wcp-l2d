"""Multi-label to single-label conversion utilities.

Converts the multi-label [N, 7] label matrix (with possible NaN) into
single-label integer vectors suitable for multi-class conformal prediction.
"""

from __future__ import annotations

import numpy as np


def compute_pathology_prevalence(
    labels: np.ndarray,
    pathologies: list[str],
) -> dict[str, float]:
    """Compute prevalence (fraction positive among non-NaN) per pathology.

    Args:
        labels: [N, 7] float array with 0, 1, NaN.
        pathologies: list of 7 pathology names.

    Returns:
        Dict mapping pathology name to prevalence float.
    """
    prevalence = {}
    for i, p in enumerate(pathologies):
        col = labels[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            prevalence[p] = float((col[valid] == 1.0).mean())
        else:
            prevalence[p] = 0.0
    return prevalence


def multilabel_to_singlelabel(
    features: np.ndarray,
    labels: np.ndarray,
    pathologies: list[str],
    prevalence: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter and convert multi-label to single-label.

    Steps:
        1. Keep rows with at least one positive (==1.0) column.
           NaN columns are ignored (only explicit positives count).
        2. For multi-positive rows, assign to the rarest pathology
           (lowest prevalence) to preserve class balance.

    Args:
        features: [N, 1024] feature matrix.
        labels: [N, 7] multi-label matrix (0, 1, NaN).
        pathologies: ordered pathology list.
        prevalence: optional pre-computed prevalence dict. If None,
            computed from the provided labels.

    Returns:
        filtered_features: [M, 1024] where M <= N.
        single_labels: [M] integer in {0, ..., 6}.
        valid_mask: [N] boolean mask of kept rows.
    """
    if prevalence is None:
        prevalence = compute_pathology_prevalence(labels, pathologies)

    N, K = labels.shape
    assert K == len(pathologies)

    # Prevalence array aligned with column order
    prev_arr = np.array([prevalence[p] for p in pathologies])

    # Keep rows with at least one explicit positive (NaN columns ignored)
    valid_mask = (labels == 1.0).any(axis=1)
    valid_idx = np.where(valid_mask)[0]

    filtered_features = features[valid_mask]
    filtered_labels = labels[valid_mask]

    # Assign single label: rarest positive pathology
    single_labels = np.empty(len(valid_idx), dtype=np.int64)
    for i in range(len(valid_idx)):
        pos_cols = np.where(filtered_labels[i] == 1.0)[0]
        rarest = pos_cols[np.argmin(prev_arr[pos_cols])]
        single_labels[i] = rarest

    return filtered_features, single_labels, valid_mask


def extract_binary_labels(
    features: np.ndarray,
    labels: np.ndarray,
    pathologies: list[str],
    target_pathology: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract binary labels for a single pathology.

    Keeps only samples where the target pathology label is non-NaN.
    Returns binary labels: 0 = negative, 1 = positive.

    Args:
        features: [N, D] feature matrix.
        labels: [N, K] multi-label matrix (0, 1, NaN).
        pathologies: ordered pathology list.
        target_pathology: name of the pathology to extract.

    Returns:
        filtered_features: [M, D] where M <= N.
        binary_labels: [M] integer in {0, 1}.
        valid_mask: [N] boolean mask of kept rows.
    """
    col_idx = pathologies.index(target_pathology)
    col = labels[:, col_idx]
    valid_mask = ~np.isnan(col)

    filtered_features = features[valid_mask]
    binary_labels = col[valid_mask].astype(np.int64)

    return filtered_features, binary_labels, valid_mask


def extract_multilabel_valid_samples(
    features: np.ndarray,
    labels: np.ndarray,
    min_valid_pathologies: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter samples that have at least ``min_valid_pathologies`` non-NaN labels.

    Unlike ``extract_binary_labels`` (single pathology) or
    ``multilabel_to_singlelabel`` (lossy conversion), this keeps the
    original multi-label structure intact while filtering out samples
    with too many missing labels.

    Args:
        features: [N, D] feature matrix.
        labels: [N, K] multi-label matrix (0, 1, NaN).
        min_valid_pathologies: minimum number of non-NaN pathology labels
            required to keep a sample.

    Returns:
        filtered_features: [M, D] where M <= N.
        filtered_labels: [M, K] (still contains NaN for some pathologies).
        sample_mask: [N] boolean mask of kept rows.
        valid_pathology_mask: [M, K] boolean (True where label is non-NaN).
    """
    valid_per_sample = ~np.isnan(labels)
    n_valid = valid_per_sample.sum(axis=1)
    sample_mask = n_valid >= min_valid_pathologies

    filtered_features = features[sample_mask]
    filtered_labels = labels[sample_mask]
    valid_pathology_mask = valid_per_sample[sample_mask]

    return filtered_features, filtered_labels, sample_mask, valid_pathology_mask

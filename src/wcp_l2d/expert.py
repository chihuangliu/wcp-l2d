"""Simulated radiologist expert for Learning to Defer experiments.

Generates expert predictions with configurable per-pathology sensitivity
and specificity rates. Default values reflect realistic radiologist
performance from the literature.
"""

from __future__ import annotations

import numpy as np

from .pathologies import COMMON_PATHOLOGIES

# Default per-pathology accuracy rates.
# Sensitivity = P(expert predicts positive | truly positive)
# Specificity = P(expert predicts negative | truly negative)
DEFAULT_SENSITIVITY = {
    "Atelectasis": 0.75,
    "Cardiomegaly": 0.90,
    "Consolidation": 0.70,
    "Edema": 0.80,
    "Effusion": 0.85,
    "Pneumonia": 0.65,
    "Pneumothorax": 0.85,
}

DEFAULT_SPECIFICITY = {
    "Atelectasis": 0.85,
    "Cardiomegaly": 0.95,
    "Consolidation": 0.80,
    "Edema": 0.90,
    "Effusion": 0.90,
    "Pneumonia": 0.75,
    "Pneumothorax": 0.90,
}


class SimulatedExpert:
    """Simulated radiologist for L2D experiments.

    For each sample, the expert produces a binary prediction per pathology.
    Accuracy is governed by per-pathology sensitivity and specificity.
    Samples with NaN ground-truth labels get NaN expert predictions.

    Args:
        sensitivity: Per-pathology true positive rates.
        specificity: Per-pathology true negative rates.
        pathologies: Ordered pathology list (default: COMMON_PATHOLOGIES).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sensitivity: dict[str, float] | None = None,
        specificity: dict[str, float] | None = None,
        pathologies: list[str] | None = None,
        seed: int = 42,
    ):
        self.pathologies = pathologies or list(COMMON_PATHOLOGIES)
        self.sensitivity = sensitivity or DEFAULT_SENSITIVITY
        self.specificity = specificity or DEFAULT_SPECIFICITY
        self.rng = np.random.RandomState(seed)

        # Vectorized rates in pathology order
        self._sens = np.array([self.sensitivity[p] for p in self.pathologies])
        self._spec = np.array([self.specificity[p] for p in self.pathologies])

    def predict(self, labels: np.ndarray) -> np.ndarray:
        """Generate expert predictions given ground-truth labels.

        Args:
            labels: [N, num_pathologies] with values 0, 1, or NaN.

        Returns:
            predictions: [N, num_pathologies] with values 0, 1, or NaN.
        """
        N, K = labels.shape
        assert K == len(self.pathologies), f"Expected {len(self.pathologies)} pathologies, got {K}"

        predictions = np.full_like(labels, np.nan, dtype=np.float32)
        random_draw = self.rng.random((N, K))

        sens = np.broadcast_to(self._sens, (N, K))
        spec = np.broadcast_to(self._spec, (N, K))

        # Positive ground truth: expert correct with P = sensitivity
        pos_mask = labels == 1.0
        predictions[pos_mask] = (random_draw[pos_mask] < sens[pos_mask]).astype(np.float32)

        # Negative ground truth: expert correct with P = specificity
        # Correct means predicting 0, so predict 1 with P = (1 - specificity)
        neg_mask = labels == 0.0
        predictions[neg_mask] = (random_draw[neg_mask] >= spec[neg_mask]).astype(np.float32)

        return predictions

    def correctness(self, labels: np.ndarray) -> np.ndarray:
        """Generate binary correctness indicators.

        This is the signal used by L2D frameworks to decide when to defer.

        Returns:
            correct: [N, num_pathologies] where 1 = expert correct,
                     0 = expert incorrect, NaN = unknown.
        """
        predictions = self.predict(labels)
        correct = np.full_like(labels, np.nan, dtype=np.float32)
        valid = ~np.isnan(labels)
        correct[valid] = (predictions[valid] == labels[valid]).astype(np.float32)
        return correct

    def accuracy_report(self, labels: np.ndarray) -> dict[str, dict]:
        """Compute per-pathology accuracy for verification.

        Returns:
            Dict mapping pathology name to {accuracy, sensitivity, specificity, n_valid}.
        """
        predictions = self.predict(labels)
        report = {}
        for i, p in enumerate(self.pathologies):
            valid = ~np.isnan(labels[:, i])
            if valid.sum() == 0:
                continue
            acc = float((predictions[valid, i] == labels[valid, i]).mean())

            pos = labels[:, i] == 1.0
            neg = labels[:, i] == 0.0
            sens = float((predictions[pos, i] == 1.0).mean()) if pos.sum() > 0 else float("nan")
            spec = float((predictions[neg, i] == 0.0).mean()) if neg.sum() > 0 else float("nan")

            report[p] = {
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "n_valid": int(valid.sum()),
            }
        return report

"""Conformal prediction with RAPS scoring.

Implements standard (unweighted) split conformal prediction and
weighted conformal prediction (Tibshirani et al., 2019) using
torchcp's RAPS score function.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torchcp.classification.score import RAPS


class ConformalPredictor:
    """Standard split conformal prediction using RAPS.

    Operates directly on numpy logit arrays. Uses torchcp's RAPS
    for non-conformity scoring and implements the standard quantile
    threshold from Vovk et al. (2005).

    Args:
        penalty: RAPS regularization weight.
        kreg: RAPS rank of regularization.
        randomized: Whether to use randomized RAPS scores.
    """

    def __init__(
        self,
        penalty: float = 0.1,
        kreg: int = 1,
        randomized: bool = True,
    ):
        self.score_fn = RAPS(
            penalty=penalty, kreg=kreg, randomized=randomized,
        )
        self.q_hat: float | None = None
        self.cal_scores: np.ndarray | None = None

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.1,
    ) -> float:
        """Calibrate on the source calibration set.

        Args:
            logits: [N_cal, K] raw logits (pre-softmax).
            labels: [N_cal] integer class labels.
            alpha: significance level (0.1 = 90% coverage target).

        Returns:
            q_hat: the calibrated threshold.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        scores = self.score_fn(logits_t, labels_t).numpy()  # [N_cal]
        self.cal_scores = scores

        n = len(scores)
        k = math.ceil((n + 1) * (1 - alpha))
        sorted_scores = np.sort(scores)

        if k > n:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(sorted_scores[k - 1])  # 0-indexed

        return self.q_hat

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Generate prediction sets.

        Args:
            logits: [N_test, K] raw logits.

        Returns:
            prediction_sets: [N_test, K] binary matrix (1 = class in set).
        """
        assert self.q_hat is not None, "Call calibrate() first."

        logits_t = torch.tensor(logits, dtype=torch.float32)
        all_scores = self.score_fn(logits_t).numpy()  # [N_test, K]

        return (all_scores <= self.q_hat).astype(np.int32)


class WeightedConformalPredictor:
    """Weighted conformal prediction using RAPS + importance weights.

    Implements the weighted quantile from Tibshirani et al. (2019).
    For each test point, the quantile threshold is computed using
    importance-weighted calibration scores, with the test point
    appended with score = infinity.

    Args:
        penalty: RAPS regularization weight.
        kreg: RAPS rank of regularization.
        randomized: Whether to use randomized RAPS scores.
    """

    def __init__(
        self,
        penalty: float = 0.1,
        kreg: int = 1,
        randomized: bool = True,
    ):
        self.score_fn = RAPS(
            penalty=penalty, kreg=kreg, randomized=randomized,
        )
        self.cal_scores_sorted: np.ndarray | None = None
        self.cal_weights_sorted: np.ndarray | None = None

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Calibrate with weighted calibration data.

        Args:
            logits: [N_cal, K] raw logits.
            labels: [N_cal] integer class labels.
            weights: [N_cal] importance weights from DRE.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        scores = self.score_fn(logits_t, labels_t).numpy()  # [N_cal]

        # Sort by scores; keep weights in corresponding order
        sort_idx = np.argsort(scores)
        self.cal_scores_sorted = scores[sort_idx]
        self.cal_weights_sorted = weights[sort_idx]

    def predict(
        self,
        logits: np.ndarray,
        test_weights: np.ndarray,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """Generate prediction sets with per-point weighted quantiles.

        Args:
            logits: [N_test, K] raw logits.
            test_weights: [N_test] importance weights for test points.
            alpha: significance level.

        Returns:
            prediction_sets: [N_test, K] binary matrix.
        """
        assert self.cal_scores_sorted is not None, "Call calibrate() first."

        logits_t = torch.tensor(logits, dtype=torch.float32)
        all_scores = self.score_fn(logits_t).numpy()  # [N_test, K]

        N_test, K = all_scores.shape
        prediction_sets = np.zeros((N_test, K), dtype=np.int32)

        # Vectorized weighted quantile computation
        n_cal = len(self.cal_scores_sorted)

        # Broadcast calibration weights: [1, n_cal]
        cal_w = self.cal_weights_sorted[np.newaxis, :]  # [1, n_cal]
        test_w = test_weights[:, np.newaxis]  # [N_test, 1]

        # Combined weights: [N_test, n_cal + 1]
        # Last position is the test point weight (for score = inf)
        all_w = np.concatenate(
            [np.broadcast_to(cal_w, (N_test, n_cal)), test_w],
            axis=1,
        )

        # Normalize to probability distribution per row
        p = all_w / all_w.sum(axis=1, keepdims=True)

        # Cumulative probability (only over cal scores, last is inf)
        cumprob = np.cumsum(p[:, :n_cal], axis=1)  # [N_test, n_cal]

        # For each test point, find the first cal index where cumprob >= 1-alpha
        target = 1 - alpha
        # Use broadcasting: [N_test, n_cal] >= target
        reached = cumprob >= target
        # First index where reached (or n_cal if never reached → inf threshold)
        # argmax on boolean gives first True; if no True, gives 0 (wrong)
        has_any = reached.any(axis=1)
        first_idx = np.argmax(reached, axis=1)  # [N_test]

        # Compute per-point thresholds
        q_hat = np.where(
            has_any,
            self.cal_scores_sorted[first_idx],
            np.inf,
        )

        # Build prediction sets
        prediction_sets = (all_scores <= q_hat[:, np.newaxis]).astype(np.int32)

        return prediction_sets

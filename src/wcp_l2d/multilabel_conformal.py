"""Multi-label conformal prediction with joint aggregation.

Runs K independent binary conformal predictors (one per pathology) and
aggregates their non-conformity scores into a joint score for deferral.

Supports multiple aggregation strategies:
- "independent": each binary CP runs at α independently
- "bonferroni": each binary CP runs at α/K (FWER control)
- "max": joint score = max over per-pathology scores, single threshold
- "mean": joint score = mean over per-pathology scores, single threshold
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torchcp.classification.score import RAPS


AggregationStrategy = Literal["independent", "bonferroni", "max", "mean"]


class MultilabelConformalPredictor:
    """Multi-label CP using independent binary CPs with joint aggregation.

    Args:
        n_pathologies: number of pathologies (K).
        penalty: RAPS regularization weight per binary CP.
        kreg: RAPS rank of regularization.
        randomized: whether to use randomized RAPS scores.
    """

    def __init__(
        self,
        n_pathologies: int = 7,
        penalty: float = 0.1,
        kreg: int = 1,
        randomized: bool = False,
    ):
        self.n_pathologies = n_pathologies
        self.penalty = penalty
        self.kreg = kreg
        self.randomized = randomized
        self.score_fn = RAPS(
            penalty=penalty, kreg=kreg, randomized=randomized,
        )

        # Per-pathology calibration scores (list of 1-D arrays)
        self.cal_scores_per_path: list[np.ndarray | None] = [None] * n_pathologies
        # Per-pathology quantiles (for independent/bonferroni)
        self.q_hats: list[float | None] = [None] * n_pathologies
        # Joint calibration scores and quantile (for max/mean)
        self.joint_cal_scores: np.ndarray | None = None
        self.joint_q_hat: float | None = None

    def _compute_binary_scores(
        self,
        logits: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute RAPS scores for a binary classification task.

        Args:
            logits: [N, 2] binary logits.
            labels: [N] binary labels (0 or 1). If None, returns [N, 2]
                scores for all possible labels.

        Returns:
            If labels given: [N] scores. If labels None: [N, 2] scores.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        if labels is not None:
            labels_t = torch.tensor(labels, dtype=torch.long)
            return self.score_fn(logits_t, labels_t).numpy()
        else:
            return self.score_fn(logits_t).numpy()

    def calibrate(
        self,
        logits_list: list[np.ndarray],
        labels: np.ndarray,
        alpha: float = 0.1,
        aggregation: AggregationStrategy = "independent",
    ) -> dict:
        """Calibrate on source calibration set.

        Args:
            logits_list: list of K arrays, each [N, 2] binary logits per
                pathology. N is the same for all (full calibration set).
            labels: [N, K] multi-label matrix with 0, 1, NaN.
            alpha: significance level.
            aggregation: aggregation strategy.

        Returns:
            Dict with calibration diagnostics.
        """
        K = self.n_pathologies
        N = labels.shape[0]
        assert len(logits_list) == K

        effective_alpha = alpha / K if aggregation == "bonferroni" else alpha

        # Calibrate each binary CP independently
        for k in range(K):
            valid_mask = ~np.isnan(labels[:, k])
            if valid_mask.sum() == 0:
                continue

            scores = self._compute_binary_scores(
                logits_list[k][valid_mask],
                labels[valid_mask, k].astype(np.int64),
            )
            self.cal_scores_per_path[k] = scores

            # Compute per-pathology quantile
            n = len(scores)
            kk = math.ceil((n + 1) * (1 - effective_alpha))
            sorted_scores = np.sort(scores)
            if kk > n:
                self.q_hats[k] = float("inf")
            else:
                self.q_hats[k] = float(sorted_scores[kk - 1])

        # For max/mean aggregation, compute joint scores on intersection
        if aggregation in ("max", "mean"):
            all_valid = np.all(~np.isnan(labels), axis=1)
            n_joint = int(all_valid.sum())

            if n_joint > 0:
                joint_scores_list = []
                for k in range(K):
                    s = self._compute_binary_scores(
                        logits_list[k][all_valid],
                        labels[all_valid, k].astype(np.int64),
                    )
                    joint_scores_list.append(s)

                joint_scores_matrix = np.stack(joint_scores_list, axis=1)  # [M, K]
                if aggregation == "max":
                    self.joint_cal_scores = joint_scores_matrix.max(axis=1)
                else:  # mean
                    self.joint_cal_scores = joint_scores_matrix.mean(axis=1)

                n = len(self.joint_cal_scores)
                kk = math.ceil((n + 1) * (1 - alpha))
                sorted_joint = np.sort(self.joint_cal_scores)
                if kk > n:
                    self.joint_q_hat = float("inf")
                else:
                    self.joint_q_hat = float(sorted_joint[kk - 1])
            else:
                self.joint_cal_scores = np.array([])
                self.joint_q_hat = float("inf")

            return {
                "n_joint_cal": n_joint,
                "joint_q_hat": self.joint_q_hat,
                "per_pathology_q_hats": list(self.q_hats),
            }

        return {
            "per_pathology_q_hats": list(self.q_hats),
            "effective_alpha": effective_alpha,
        }

    def predict(
        self,
        logits_list: list[np.ndarray],
        aggregation: AggregationStrategy = "independent",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate per-pathology prediction sets.

        Args:
            logits_list: list of K arrays, each [N, 2] binary logits.
            aggregation: must match the strategy used in calibrate().

        Returns:
            per_pathology_sets: [N, K, 2] binary array. Entry [i, k, c] = 1
                means class c is in the prediction set for pathology k.
            set_sizes: [N, K] number of classes in each per-pathology set.
        """
        K = self.n_pathologies
        N = logits_list[0].shape[0]
        per_pathology_sets = np.zeros((N, K, 2), dtype=np.int32)

        if aggregation in ("independent", "bonferroni"):
            for k in range(K):
                q = self.q_hats[k]
                if q is None:
                    continue
                all_scores = self._compute_binary_scores(logits_list[k])  # [N, 2]
                per_pathology_sets[:, k, :] = (all_scores <= q).astype(np.int32)

        elif aggregation in ("max", "mean"):
            if self.joint_q_hat is None:
                raise ValueError("Joint quantile not calibrated. Call calibrate() with max/mean aggregation first.")

            # For max/mean, we need to determine which classes to include.
            # Strategy: include class c for pathology k if the score
            # s_k(x, c) is consistent with the joint threshold.
            # For max: include c if s_k(x, c) <= joint_q_hat
            # (since max >= each individual, this is a necessary condition)
            for k in range(K):
                all_scores = self._compute_binary_scores(logits_list[k])  # [N, 2]
                per_pathology_sets[:, k, :] = (all_scores <= self.joint_q_hat).astype(np.int32)

        set_sizes = per_pathology_sets.sum(axis=2)  # [N, K]
        return per_pathology_sets, set_sizes


class MultilabelWeightedConformalPredictor:
    """Multi-label WCP using independent binary CPs with DRE weights.

    Extends MultilabelConformalPredictor with importance-weighted quantiles
    from Tibshirani et al. (2019).

    Args:
        n_pathologies: number of pathologies (K).
        penalty: RAPS regularization weight.
        kreg: RAPS rank of regularization.
        randomized: whether to use randomized RAPS scores.
    """

    def __init__(
        self,
        n_pathologies: int = 7,
        penalty: float = 0.1,
        kreg: int = 1,
        randomized: bool = False,
    ):
        self.n_pathologies = n_pathologies
        self.score_fn = RAPS(
            penalty=penalty, kreg=kreg, randomized=randomized,
        )

        # Per-pathology sorted scores and weights
        self.cal_scores_sorted: list[np.ndarray | None] = [None] * n_pathologies
        self.cal_weights_sorted: list[np.ndarray | None] = [None] * n_pathologies

        # Joint sorted scores and weights (for max/mean)
        self.joint_cal_scores_sorted: np.ndarray | None = None
        self.joint_cal_weights_sorted: np.ndarray | None = None

    def _compute_binary_scores(
        self,
        logits: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        if labels is not None:
            labels_t = torch.tensor(labels, dtype=torch.long)
            return self.score_fn(logits_t, labels_t).numpy()
        else:
            return self.score_fn(logits_t).numpy()

    def calibrate(
        self,
        logits_list: list[np.ndarray],
        labels: np.ndarray,
        weights: np.ndarray,
        aggregation: AggregationStrategy = "independent",
    ) -> dict:
        """Calibrate with weighted calibration data.

        Args:
            logits_list: list of K arrays, each [N, 2] binary logits.
            labels: [N, K] multi-label matrix with 0, 1, NaN.
            weights: [N] importance weights from DRE.
            aggregation: aggregation strategy.

        Returns:
            Dict with calibration diagnostics.
        """
        K = self.n_pathologies
        N = labels.shape[0]
        assert len(logits_list) == K

        # Calibrate each binary WCP
        for k in range(K):
            valid_mask = ~np.isnan(labels[:, k])
            if valid_mask.sum() == 0:
                continue

            scores = self._compute_binary_scores(
                logits_list[k][valid_mask],
                labels[valid_mask, k].astype(np.int64),
            )
            w = weights[valid_mask]

            sort_idx = np.argsort(scores)
            self.cal_scores_sorted[k] = scores[sort_idx]
            self.cal_weights_sorted[k] = w[sort_idx]

        # Joint calibration for max/mean
        if aggregation in ("max", "mean"):
            all_valid = np.all(~np.isnan(labels), axis=1)
            n_joint = int(all_valid.sum())

            if n_joint > 0:
                joint_scores_list = []
                for k in range(K):
                    s = self._compute_binary_scores(
                        logits_list[k][all_valid],
                        labels[all_valid, k].astype(np.int64),
                    )
                    joint_scores_list.append(s)

                joint_matrix = np.stack(joint_scores_list, axis=1)  # [M, K]
                if aggregation == "max":
                    joint_scores = joint_matrix.max(axis=1)
                else:
                    joint_scores = joint_matrix.mean(axis=1)

                w_joint = weights[all_valid]
                sort_idx = np.argsort(joint_scores)
                self.joint_cal_scores_sorted = joint_scores[sort_idx]
                self.joint_cal_weights_sorted = w_joint[sort_idx]

            return {"n_joint_cal": n_joint}

        return {}

    def _weighted_quantile_per_point(
        self,
        cal_scores_sorted: np.ndarray,
        cal_weights_sorted: np.ndarray,
        test_weights: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Compute per-test-point weighted quantiles.

        Args:
            cal_scores_sorted: [N_cal] sorted calibration scores.
            cal_weights_sorted: [N_cal] weights in sorted order.
            test_weights: [N_test] test point weights.
            alpha: significance level.

        Returns:
            q_hat: [N_test] per-point thresholds.
        """
        N_test = len(test_weights)
        n_cal = len(cal_scores_sorted)

        cal_w = cal_weights_sorted[np.newaxis, :]  # [1, n_cal]
        test_w = test_weights[:, np.newaxis]  # [N_test, 1]

        all_w = np.concatenate(
            [np.broadcast_to(cal_w, (N_test, n_cal)), test_w],
            axis=1,
        )
        p = all_w / all_w.sum(axis=1, keepdims=True)
        cumprob = np.cumsum(p[:, :n_cal], axis=1)

        target = 1 - alpha
        reached = cumprob >= target
        has_any = reached.any(axis=1)
        first_idx = np.argmax(reached, axis=1)

        q_hat = np.where(
            has_any,
            cal_scores_sorted[first_idx],
            np.inf,
        )
        return q_hat

    def predict(
        self,
        logits_list: list[np.ndarray],
        test_weights: np.ndarray,
        alpha: float = 0.1,
        aggregation: AggregationStrategy = "independent",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate per-pathology prediction sets with weighted quantiles.

        Args:
            logits_list: list of K arrays, each [N, 2] binary logits.
            test_weights: [N] importance weights for test points.
            alpha: significance level.
            aggregation: must match calibrate().

        Returns:
            per_pathology_sets: [N, K, 2] binary array.
            set_sizes: [N, K] per-pathology set sizes.
        """
        K = self.n_pathologies
        N = logits_list[0].shape[0]
        per_pathology_sets = np.zeros((N, K, 2), dtype=np.int32)

        effective_alpha = alpha / K if aggregation == "bonferroni" else alpha

        if aggregation in ("independent", "bonferroni"):
            for k in range(K):
                if self.cal_scores_sorted[k] is None:
                    continue

                all_scores = self._compute_binary_scores(logits_list[k])  # [N, 2]
                q_hats = self._weighted_quantile_per_point(
                    self.cal_scores_sorted[k],
                    self.cal_weights_sorted[k],
                    test_weights,
                    effective_alpha,
                )
                # q_hats is [N], all_scores is [N, 2]
                per_pathology_sets[:, k, :] = (
                    all_scores <= q_hats[:, np.newaxis]
                ).astype(np.int32)

        elif aggregation in ("max", "mean"):
            if self.joint_cal_scores_sorted is None:
                raise ValueError("Joint calibration not done.")

            # Compute per-point joint thresholds
            q_hats = self._weighted_quantile_per_point(
                self.joint_cal_scores_sorted,
                self.joint_cal_weights_sorted,
                test_weights,
                alpha,
            )

            for k in range(K):
                all_scores = self._compute_binary_scores(logits_list[k])  # [N, 2]
                per_pathology_sets[:, k, :] = (
                    all_scores <= q_hats[:, np.newaxis]
                ).astype(np.int32)

        set_sizes = per_pathology_sets.sum(axis=2)  # [N, K]
        return per_pathology_sets, set_sizes

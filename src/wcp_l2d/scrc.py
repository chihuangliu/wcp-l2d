"""Selective Conformal Risk Control (SCRC) for multi-label learning to defer.

Two-stage pipeline:
  Stage 1 (Selection): Budget-constrained deferral via multi-label entropy.
      Guarantees deferral rate <= beta by construction.
  Stage 2 (CRC): Weighted Conformal Risk Control for FNR on non-deferred samples.
      Finds threshold lambda* on per-pathology probabilities to control
      the weighted False Negative Rate at level alpha.

References:
  - Angelopoulos et al. (2022), "Conformal Risk Control"
  - Tibshirani et al. (2019), "Conformal Prediction Under Covariate Shift"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

def multilabel_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute multi-label binary entropy as an uncertainty score.

    For K independent binary predictions with probabilities p_k(x):
        H(x) = -sum_k [p_k log(p_k) + (1 - p_k) log(1 - p_k)]

    Args:
        probs: [N, K] per-pathology positive-class probabilities.

    Returns:
        entropy: [N] entropy values. Higher = more uncertain.
    """
    eps = 1e-10
    p = np.clip(probs, eps, 1 - eps)
    per_label = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return per_label.sum(axis=1)


# ---------------------------------------------------------------------------
# Stage 1: Budget-constrained selection
# ---------------------------------------------------------------------------

def select_for_deferral(
    uncertainty_scores: np.ndarray,
    beta: float,
    seed: int = 42,
) -> np.ndarray:
    """Select top-beta fraction of samples for deferral (most uncertain).

    Guarantees deferral rate <= beta by construction.

    Args:
        uncertainty_scores: [N] uncertainty scores (higher = more uncertain).
        beta: maximum deferral rate budget in (0, 1).
        seed: random seed for deterministic tie-breaking.

    Returns:
        defer_mask: [N] boolean array. True = defer to expert.
    """
    N = len(uncertainty_scores)
    n_defer = math.floor(N * beta)

    if n_defer == 0:
        return np.zeros(N, dtype=bool)
    if n_defer >= N:
        return np.ones(N, dtype=bool)

    # Find the threshold: the n_defer-th largest score
    # np.partition is O(N)
    threshold = np.partition(uncertainty_scores, -n_defer)[-n_defer]

    # Samples strictly above threshold are always deferred
    above = uncertainty_scores > threshold
    n_above = int(above.sum())

    if n_above == n_defer:
        return above

    # Handle ties: need (n_defer - n_above) more from those exactly at threshold
    at_threshold = uncertainty_scores == threshold
    n_needed = n_defer - n_above

    rng = np.random.RandomState(seed)
    tie_indices = np.where(at_threshold)[0]
    chosen = rng.choice(tie_indices, size=n_needed, replace=False)

    defer_mask = above.copy()
    defer_mask[chosen] = True
    return defer_mask


# ---------------------------------------------------------------------------
# Per-sample FNR computation
# ---------------------------------------------------------------------------

def compute_sample_fnr(
    probs: np.ndarray,
    labels: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Compute per-sample False Negative Rate at threshold lambda.

    For sample i:
        predicted_positive_k = 1[p_k(x_i) >= lambda]
        FNR_i = sum_k 1[y_ik=1 AND p_k < lambda] / max(1, sum_k 1[y_ik=1])

    Pathologies with NaN labels are excluded. Samples with no true
    positives (among valid labels) get FNR = 0.

    Args:
        probs: [N, K] per-pathology probabilities.
        labels: [N, K] multi-label ground truth (0, 1, NaN).
        lam: threshold in [0, 1].

    Returns:
        fnr: [N] per-sample false negative rates.
    """
    valid = ~np.isnan(labels)
    true_pos = (labels == 1) & valid  # [N, K]
    missed = true_pos & (probs < lam)  # [N, K]

    n_true_pos = true_pos.sum(axis=1)  # [N]
    n_missed = missed.sum(axis=1)  # [N]

    return n_missed / np.maximum(1, n_true_pos)


# ---------------------------------------------------------------------------
# Stage 2: Weighted CRC calibration for FNR
# ---------------------------------------------------------------------------

@dataclass
class CRCResult:
    """Result from Conformal Risk Control calibration."""

    lambda_hat: float
    weighted_fnr_at_lambda: float
    n_calibration: int
    ess: float
    ess_fraction: float
    lambda_path: np.ndarray = field(repr=False)
    fnr_path: np.ndarray = field(repr=False)


def calibrate_crc_fnr(
    probs: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.10,
    n_grid: int = 1000,
) -> CRCResult:
    """Find threshold lambda* via weighted Conformal Risk Control for FNR.

    Finds lambda* = sup{lambda : weighted_FNR(lambda) <= alpha}.

    As lambda increases (stricter threshold), fewer labels are predicted
    positive and FNR monotonically increases. We sweep from low to high
    and find the largest lambda satisfying the risk bound.

    Args:
        probs: [N_cal, K] calibration probabilities (non-deferred portion).
        labels: [N_cal, K] calibration labels (0, 1, NaN).
        weights: [N_cal] importance weights from DRE.
        alpha: target FNR level (e.g., 0.10 = at most 10% false negatives).
        n_grid: number of grid points for lambda search.

    Returns:
        CRCResult with calibrated lambda and diagnostics.
    """
    # Collect candidate thresholds: uniform grid + unique probability values
    grid = np.linspace(0.0, 1.0, n_grid)
    unique_probs = np.unique(probs[~np.isnan(probs)])
    candidates = np.unique(np.concatenate([grid, unique_probs]))
    candidates.sort()

    # Compute weighted FNR at each candidate
    w_sum = weights.sum()
    fnr_path = np.empty(len(candidates))

    for i, lam in enumerate(candidates):
        sample_fnr = compute_sample_fnr(probs, labels, lam)
        fnr_path[i] = (weights * sample_fnr).sum() / w_sum

    # Find lambda* = largest lambda where weighted_FNR <= alpha
    valid_mask = fnr_path <= alpha
    if valid_mask.any():
        best_idx = np.where(valid_mask)[0][-1]
        lambda_hat = float(candidates[best_idx])
        fnr_at_lambda = float(fnr_path[best_idx])
    else:
        # No lambda satisfies the bound; use lambda=0 (predict everything positive)
        lambda_hat = 0.0
        fnr_at_lambda = float(fnr_path[0])

    # ESS of weights used
    ess = float(w_sum**2 / (weights**2).sum())

    return CRCResult(
        lambda_hat=lambda_hat,
        weighted_fnr_at_lambda=fnr_at_lambda,
        n_calibration=len(probs),
        ess=ess,
        ess_fraction=ess / len(probs),
        lambda_path=candidates,
        fnr_path=fnr_path,
    )


# ---------------------------------------------------------------------------
# Full SCRC pipeline
# ---------------------------------------------------------------------------

@dataclass
class SCRCResult:
    """Full result from the two-stage SCRC pipeline."""

    # Stage 1: Selection
    defer_mask: np.ndarray  # [N_test] boolean
    uncertainty_scores: np.ndarray  # [N_test]
    deferral_rate: float
    beta: float

    # Stage 2: CRC
    lambda_hat: float
    crc_result: CRCResult

    # Final predictions (for deferred samples, all zeros)
    prediction_sets: np.ndarray  # [N_test, K] binary

    # Parameters
    target_fnr_alpha: float


class SCRCPredictor:
    """Two-stage Selective Conformal Risk Control predictor.

    Stage 1: Budget-constrained deferral via multi-label entropy.
    Stage 2: Weighted CRC for FNR control on non-deferred samples.

    Args:
        beta: maximum deferral rate budget in (0, 1).
        alpha: target FNR level for CRC.
        n_grid: lambda search grid size.
        seed: random seed for tie-breaking in selection.
    """

    def __init__(
        self,
        beta: float = 0.15,
        alpha: float = 0.10,
        n_grid: int = 1000,
        seed: int = 42,
    ):
        self.beta = beta
        self.alpha = alpha
        self.n_grid = n_grid
        self.seed = seed
        self._crc_result: CRCResult | None = None

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        cal_weights: np.ndarray,
    ) -> CRCResult:
        """Calibrate the CRC threshold on non-deferred calibration samples.

        Stage 1 selection is applied to calibration data first, then
        Stage 2 CRC is calibrated on the remaining samples.

        Args:
            cal_probs: [N_cal, K] per-pathology probabilities.
            cal_labels: [N_cal, K] multi-label ground truth.
            cal_weights: [N_cal] importance weights.

        Returns:
            CRCResult from calibration.
        """
        # Stage 1: select calibration samples for deferral
        entropy = multilabel_entropy(cal_probs)
        cal_defer_mask = select_for_deferral(entropy, self.beta, seed=self.seed)

        # Stage 2: calibrate CRC on non-deferred subset
        kept = ~cal_defer_mask
        self._crc_result = calibrate_crc_fnr(
            probs=cal_probs[kept],
            labels=cal_labels[kept],
            weights=cal_weights[kept],
            alpha=self.alpha,
            n_grid=self.n_grid,
        )
        return self._crc_result

    def predict(
        self,
        test_probs: np.ndarray,
        test_weights: np.ndarray | None = None,
    ) -> SCRCResult:
        """Apply the full SCRC pipeline to test data.

        Args:
            test_probs: [N_test, K] per-pathology probabilities.
            test_weights: [N_test] importance weights (used only for
                potential diagnostics; lambda is fixed from calibration).

        Returns:
            SCRCResult with deferral decisions and prediction sets.
        """
        if self._crc_result is None:
            raise RuntimeError("Call calibrate() first.")

        N, K = test_probs.shape

        # Stage 1: budget-constrained deferral
        entropy = multilabel_entropy(test_probs)
        defer_mask = select_for_deferral(entropy, self.beta, seed=self.seed)

        # Stage 2: apply threshold to non-deferred samples
        prediction_sets = np.zeros((N, K), dtype=np.int32)
        kept = ~defer_mask
        if kept.any():
            prediction_sets[kept] = (
                test_probs[kept] >= self._crc_result.lambda_hat
            ).astype(np.int32)

        deferral_rate = float(defer_mask.sum()) / N if N > 0 else 0.0

        return SCRCResult(
            defer_mask=defer_mask,
            uncertainty_scores=entropy,
            deferral_rate=deferral_rate,
            beta=self.beta,
            lambda_hat=self._crc_result.lambda_hat,
            crc_result=self._crc_result,
            prediction_sets=prediction_sets,
            target_fnr_alpha=self.alpha,
        )


# ---------------------------------------------------------------------------
# Per-pathology CRC calibration
# ---------------------------------------------------------------------------

@dataclass
class PerPathologyCRCResult:
    """Result from per-pathology Conformal Risk Control calibration."""

    lambda_hats: np.ndarray            # [K] per-pathology thresholds
    weighted_fnr_at_lambda: np.ndarray # [K] weighted FNR at each lambda_k*
    n_positives: np.ndarray            # [K] number of positives per pathology
    n_valid: np.ndarray                # [K] number of valid (non-NaN) samples per pathology
    ess: float
    ess_fraction: float
    n_calibration: int
    pathology_names: list              # [K]
    lambda_paths: list                 # list of [M_k] arrays (one per pathology)
    fnr_paths: list                    # list of [M_k] arrays (one per pathology)


def calibrate_per_pathology_crc_fnr(
    probs: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    alpha: float | np.ndarray = 0.10,
    n_grid: int = 1000,
    pathology_names: list | None = None,
) -> PerPathologyCRCResult:
    """Find per-pathology thresholds via weighted CRC for FNR.

    For each pathology k independently:
        FNR_k(lambda) = sum_{i: y_ik=1, valid} w_i * 1[p_k(x_i) < lambda]
                        / sum_{i: y_ik=1, valid} w_i
        lambda_k* = sup{lambda : FNR_k(lambda) <= alpha_k}

    Args:
        probs: [N_cal, K] calibration probabilities (non-deferred portion).
        labels: [N_cal, K] calibration labels (0, 1, NaN).
        weights: [N_cal] importance weights from DRE.
        alpha: target FNR level — scalar (same for all) or [K] per-pathology.
        n_grid: number of grid points for lambda search.
        pathology_names: optional list of K pathology name strings.

    Returns:
        PerPathologyCRCResult with per-pathology lambda_hats and diagnostics.
    """
    N, K = probs.shape

    if pathology_names is None:
        pathology_names = [f"path_{k}" for k in range(K)]

    # Broadcast alpha to [K]
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (K,)).copy()

    # ESS of the weight vector
    w_sum = weights.sum()
    ess = float(w_sum ** 2 / (weights ** 2).sum())

    lambda_hats = np.zeros(K)
    weighted_fnr_at_lambda = np.zeros(K)
    n_positives = np.zeros(K, dtype=int)
    n_valid = np.zeros(K, dtype=int)
    lambda_paths = []
    fnr_paths = []

    for k in range(K):
        # Valid (non-NaN) mask for this pathology
        valid_k = ~np.isnan(labels[:, k])
        pos_k = valid_k & (labels[:, k] == 1)

        n_valid[k] = int(valid_k.sum())
        n_positives[k] = int(pos_k.sum())

        if n_positives[k] == 0:
            # No positives: any lambda satisfies FNR=0 <= alpha; use lambda=1
            lambda_hats[k] = 1.0
            weighted_fnr_at_lambda[k] = 0.0
            lambda_paths.append(np.array([0.0, 1.0]))
            fnr_paths.append(np.array([0.0, 0.0]))
            continue

        # Build candidate grid from uniform points + unique prob values for this pathology
        grid = np.linspace(0.0, 1.0, n_grid)
        unique_p = np.unique(probs[pos_k, k])
        candidates = np.unique(np.concatenate([grid, unique_p]))
        candidates.sort()

        # Vectorised computation:
        # missed_matrix[i, j] = 1 if p_k(x_i) < candidates[j]   (i indexes pos_k samples)
        p_pos = probs[pos_k, k]        # [n_pos]
        w_pos = weights[pos_k]         # [n_pos]
        w_pos_sum = w_pos.sum()

        # [n_grid_pts, n_pos]  broadcasted comparison
        missed_matrix = p_pos[np.newaxis, :] < candidates[:, np.newaxis]  # [M, n_pos]
        fnr_path_k = (missed_matrix * w_pos[np.newaxis, :]).sum(axis=1) / w_pos_sum

        # Find lambda_k* = largest candidate where FNR <= alpha_k
        valid_mask = fnr_path_k <= alpha_arr[k]
        if valid_mask.any():
            best_idx = int(np.where(valid_mask)[0][-1])
            lambda_hats[k] = float(candidates[best_idx])
            weighted_fnr_at_lambda[k] = float(fnr_path_k[best_idx])
        else:
            # No candidate satisfies the bound; use lambda=0 (predict all positive)
            lambda_hats[k] = 0.0
            weighted_fnr_at_lambda[k] = float(fnr_path_k[0])

        lambda_paths.append(candidates)
        fnr_paths.append(fnr_path_k)

    return PerPathologyCRCResult(
        lambda_hats=lambda_hats,
        weighted_fnr_at_lambda=weighted_fnr_at_lambda,
        n_positives=n_positives,
        n_valid=n_valid,
        ess=ess,
        ess_fraction=ess / N,
        n_calibration=N,
        pathology_names=list(pathology_names),
        lambda_paths=lambda_paths,
        fnr_paths=fnr_paths,
    )


# ---------------------------------------------------------------------------
# Full per-pathology SCRC pipeline
# ---------------------------------------------------------------------------

@dataclass
class PerPathologySCRCResult:
    """Full result from the two-stage PP-SCRC pipeline."""

    # Stage 1: Selection
    defer_mask: np.ndarray          # [N_test] boolean
    uncertainty_scores: np.ndarray  # [N_test]
    deferral_rate: float
    beta: float

    # Stage 2: per-pathology CRC
    lambda_hats: np.ndarray         # [K]
    crc_result: PerPathologyCRCResult

    # Final predictions (for deferred samples, all zeros)
    prediction_sets: np.ndarray     # [N_test, K] binary

    # Parameters
    target_fnr_alpha: np.ndarray    # [K]
    pathology_names: list           # [K]


class PerPathologySCRCPredictor:
    """Two-stage per-pathology Selective Conformal Risk Control predictor.

    Stage 1: Budget-constrained deferral via multi-label entropy (same as SCRCPredictor).
    Stage 2: Weighted CRC applied independently per pathology, each with its
             own threshold lambda_k* controlling FNR_k <= alpha_k.

    Args:
        beta: maximum deferral rate budget in (0, 1).
        alpha: target FNR level — scalar or [K] array for per-pathology targets.
        n_grid: lambda search grid size.
        seed: random seed for tie-breaking in selection.
    """

    def __init__(
        self,
        beta: float = 0.15,
        alpha: float | np.ndarray = 0.10,
        n_grid: int = 1000,
        seed: int = 42,
    ):
        self.beta = beta
        self.alpha = alpha
        self.n_grid = n_grid
        self.seed = seed
        self._crc_result: PerPathologyCRCResult | None = None

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        cal_weights: np.ndarray,
        pathology_names: list | None = None,
    ) -> PerPathologyCRCResult:
        """Calibrate per-pathology CRC thresholds on non-deferred calibration samples.

        Stage 1 selection is applied to calibration data first, then
        Stage 2 per-pathology CRC is calibrated on the remaining samples.

        Args:
            cal_probs: [N_cal, K] per-pathology probabilities.
            cal_labels: [N_cal, K] multi-label ground truth.
            cal_weights: [N_cal] importance weights.
            pathology_names: optional list of K name strings.

        Returns:
            PerPathologyCRCResult from calibration.
        """
        # Stage 1: select calibration samples for deferral
        entropy = multilabel_entropy(cal_probs)
        cal_defer_mask = select_for_deferral(entropy, self.beta, seed=self.seed)

        # Stage 2: calibrate per-pathology CRC on non-deferred subset
        kept = ~cal_defer_mask
        self._crc_result = calibrate_per_pathology_crc_fnr(
            probs=cal_probs[kept],
            labels=cal_labels[kept],
            weights=cal_weights[kept],
            alpha=self.alpha,
            n_grid=self.n_grid,
            pathology_names=pathology_names,
        )
        return self._crc_result

    def predict(
        self,
        test_probs: np.ndarray,
    ) -> PerPathologySCRCResult:
        """Apply the full PP-SCRC pipeline to test data.

        Args:
            test_probs: [N_test, K] per-pathology probabilities.

        Returns:
            PerPathologySCRCResult with deferral decisions and prediction sets.
        """
        if self._crc_result is None:
            raise RuntimeError("Call calibrate() first.")

        N, K = test_probs.shape

        # Stage 1: budget-constrained deferral
        entropy = multilabel_entropy(test_probs)
        defer_mask = select_for_deferral(entropy, self.beta, seed=self.seed)

        # Stage 2: apply per-pathology thresholds to non-deferred samples
        prediction_sets = np.zeros((N, K), dtype=np.int32)
        kept = ~defer_mask
        if kept.any():
            # lambda_hats is [K]; compare probs[kept] [n_kept, K] >= lambda_hats [K]
            prediction_sets[kept] = (
                test_probs[kept] >= self._crc_result.lambda_hats[np.newaxis, :]
            ).astype(np.int32)

        deferral_rate = float(defer_mask.sum()) / N if N > 0 else 0.0

        # Broadcast alpha to [K]
        K_size = len(self._crc_result.lambda_hats)
        alpha_arr = np.broadcast_to(
            np.asarray(self.alpha, dtype=float), (K_size,)
        ).copy()

        return PerPathologySCRCResult(
            defer_mask=defer_mask,
            uncertainty_scores=entropy,
            deferral_rate=deferral_rate,
            beta=self.beta,
            lambda_hats=self._crc_result.lambda_hats,
            crc_result=self._crc_result,
            prediction_sets=prediction_sets,
            target_fnr_alpha=alpha_arr,
            pathology_names=self._crc_result.pathology_names,
        )


# ---------------------------------------------------------------------------
# Fix 7.2: Post-selection DRE refit
# ---------------------------------------------------------------------------

def refit_dre_post_selection(
    cal_features_kept: np.ndarray,
    target_features: np.ndarray,
    n_components: int = 4,
    weight_clip: float = 20.0,
    random_state: int = 42,
) -> "AdaptiveDRE":
    """Refit a fresh DRE using only the post-Stage1 kept calibration features.

    Fix 7.2: After Stage 1 deferral, the distribution of calibration samples
    shifts (most uncertain samples are removed). The original DRE was fitted on
    the full calibration set, so weights computed on the kept subset are biased.
    Refitting on just the kept features ensures the importance sampling identity
    holds on the actual distribution used for CRC calibration.

    Args:
        cal_features_kept: [N_kept, D] features of non-deferred cal samples.
        target_features: [N_target, D] target domain features (NIH pool).
        n_components: PCA output dimensionality.
        weight_clip: maximum importance weight.
        random_state: seed for reproducibility.

    Returns:
        Fitted AdaptiveDRE instance.
    """
    from .dre import AdaptiveDRE

    dre = AdaptiveDRE(
        n_components=n_components,
        weight_clip=weight_clip,
        random_state=random_state,
    )
    dre.fit(source_features=cal_features_kept, target_features=target_features)
    return dre


# ---------------------------------------------------------------------------
# Fix 7.3: AUC-based capability alpha allocation
# ---------------------------------------------------------------------------

def compute_capability_alpha(
    nih_aucs: np.ndarray,
    alpha_global: float,
    pathology_names: list | None = None,
) -> np.ndarray:
    """Compute per-pathology alpha inversely proportional to classifier discriminability.

    Fix 7.3: Instead of using the same alpha_global for all pathologies, allocate
    tighter budgets to high-AUC pathologies and looser budgets to low-AUC ones.
    Low-AUC pathologies (AUC ~ 0.5) are near-random, so a loose alpha_k allows
    the CRC calibration to find a very low lambda_k*, catching more true positives.

    Formula:
        excess_k = max(AUC_k - 0.5, 1e-3)
        alpha_k = alpha_global * K * (1/excess_k) / sum_j(1/excess_j)
        alpha_k = min(alpha_k, 1.0)

    Example (K=7, alpha=0.10, Edema AUC=0.825, Pneumothorax AUC=0.531):
        Edema:        excess=0.325, alpha_k ≈ 0.035
        Pneumothorax: excess=0.031, alpha_k ≈ 0.363

    Args:
        nih_aucs: [K] AUC values on target domain per pathology.
        alpha_global: scalar alpha budget to distribute.
        pathology_names: optional K pathology names (for debugging).

    Returns:
        alpha_k: [K] per-pathology alpha values.
    """
    nih_aucs = np.asarray(nih_aucs, dtype=float)
    K = len(nih_aucs)

    excess = np.maximum(nih_aucs - 0.5, 1e-3)  # [K]
    inv_excess = 1.0 / excess                   # [K]
    alpha_k = alpha_global * K * inv_excess / inv_excess.sum()
    alpha_k = np.minimum(alpha_k, 1.0)
    return alpha_k

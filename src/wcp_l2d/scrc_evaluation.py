"""Evaluation metrics and plots for SCRC (Selective Conformal Risk Control).

Provides FNR-specific metrics, system accuracy computation, grid evaluation,
and plotting functions for the two-stage SCRC pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .expert import SimulatedExpert
from .scrc import CRCResult, SCRCPredictor


@dataclass
class SCRCDeferralResult:
    """Results from an SCRC experiment at one operating point."""

    method: str
    alpha: float
    beta: float
    # Deferral
    deferral_rate: float
    n_total: int
    n_deferred: int
    # FNR metrics on non-deferred (kept) samples
    empirical_fnr_on_kept: float
    weighted_fnr_on_kept: float
    per_pathology_fnr: dict[str, float] = field(default_factory=dict)
    per_pathology_fpr: dict[str, float] = field(default_factory=dict)
    # System metrics
    system_accuracy: float = 0.0
    model_accuracy_on_kept: float = 0.0
    lambda_hat: float = 0.0
    ess_fraction: float = 0.0


# ---------------------------------------------------------------------------
# FNR / FPR metrics
# ---------------------------------------------------------------------------

def compute_fnr_metrics(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    pathology_names: list[str],
    weights: np.ndarray | None = None,
) -> dict:
    """Compute FNR/FPR metrics for multi-label prediction sets.

    Args:
        prediction_sets: [N, K] binary (1 = predicted positive).
        labels: [N, K] multi-label (0, 1, NaN).
        pathology_names: K pathology names.
        weights: [N] optional importance weights for weighted FNR.

    Returns:
        Dict with overall_fnr, weighted_fnr, per_pathology_fnr,
        per_pathology_fpr, overall_accuracy.
    """
    N, K = labels.shape
    valid = ~np.isnan(labels)

    if weights is None:
        weights = np.ones(N)

    # Per-pathology FNR and FPR
    per_path_fnr = {}
    per_path_fpr = {}

    for k, name in enumerate(pathology_names):
        v = valid[:, k]
        if v.sum() == 0:
            per_path_fnr[name] = float("nan")
            per_path_fpr[name] = float("nan")
            continue

        y = labels[v, k]
        pred = prediction_sets[v, k]

        # FNR: among true positives, fraction missed
        pos = y == 1
        if pos.sum() > 0:
            missed = pos & (pred == 0)
            per_path_fnr[name] = float(missed.sum() / pos.sum())
        else:
            per_path_fnr[name] = 0.0

        # FPR: among true negatives, fraction falsely predicted
        neg = y == 0
        if neg.sum() > 0:
            false_pos = neg & (pred == 1)
            per_path_fpr[name] = float(false_pos.sum() / neg.sum())
        else:
            per_path_fpr[name] = 0.0

    # Overall sample-level FNR
    true_pos = (labels == 1) & valid
    missed = true_pos & (prediction_sets == 0)
    n_true_pos_per_sample = true_pos.sum(axis=1)
    n_missed_per_sample = missed.sum(axis=1)
    sample_fnr = n_missed_per_sample / np.maximum(1, n_true_pos_per_sample)

    overall_fnr = float(sample_fnr.mean())
    w_sum = weights.sum()
    weighted_fnr = float((weights * sample_fnr).sum() / w_sum) if w_sum > 0 else 0.0

    # Overall accuracy (fraction of valid label-pathology pairs correct)
    correct = (prediction_sets == labels) & valid
    total_valid = valid.sum()
    overall_accuracy = float(correct.sum() / total_valid) if total_valid > 0 else 0.0

    return {
        "overall_fnr": overall_fnr,
        "weighted_fnr": weighted_fnr,
        "per_pathology_fnr": per_path_fnr,
        "per_pathology_fpr": per_path_fpr,
        "overall_accuracy": overall_accuracy,
        "sample_fnr": sample_fnr,
    }


# ---------------------------------------------------------------------------
# System accuracy
# ---------------------------------------------------------------------------

def compute_scrc_system_accuracy(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    defer_mask: np.ndarray,
    expert: SimulatedExpert,
) -> dict:
    """Compute system accuracy for SCRC deferral.

    For non-deferred samples: prediction_sets are used as the model output.
    For deferred samples: expert predictions are used.
    System accuracy = fraction of valid (sample, pathology) pairs correct.

    Args:
        prediction_sets: [N, K] binary (1 = predicted positive).
        labels: [N, K] multi-label (0, 1, NaN).
        defer_mask: [N] boolean, True = deferred to expert.
        expert: SimulatedExpert instance.

    Returns:
        Dict with system_accuracy, deferral_rate, model_accuracy_on_kept.
    """
    N, K = labels.shape
    n_deferred = int(defer_mask.sum())
    n_kept = N - n_deferred

    # Expert predictions for deferred samples
    expert_preds = expert.predict(labels)  # [N, K]

    # Combine: model for kept, expert for deferred
    combined = prediction_sets.astype(float).copy()
    combined[defer_mask] = expert_preds[defer_mask]

    valid = ~np.isnan(labels)
    correct = (combined == labels) & valid
    total_valid = valid.sum()
    system_accuracy = float(correct.sum() / total_valid) if total_valid > 0 else 0.0

    # Model accuracy on kept samples only
    if n_kept > 0:
        kept = ~defer_mask
        kept_valid = valid[kept]
        kept_correct = (prediction_sets[kept] == labels[kept]) & kept_valid
        model_acc = (
            float(kept_correct.sum() / kept_valid.sum())
            if kept_valid.sum() > 0
            else 0.0
        )
    else:
        model_acc = 0.0

    return {
        "system_accuracy": system_accuracy,
        "deferral_rate": n_deferred / N if N > 0 else 0.0,
        "model_accuracy_on_kept": model_acc,
        "n_deferred": n_deferred,
        "n_kept": n_kept,
    }


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def evaluate_scrc(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_weights: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    pathology_names: list[str],
    expert: SimulatedExpert,
    alphas: np.ndarray,
    betas: np.ndarray,
    method_name: str = "SCRC (weighted)",
    seed: int = 42,
) -> list[SCRCDeferralResult]:
    """Evaluate SCRC across alpha x beta grid.

    Args:
        cal_probs: [N_cal, K] calibration probabilities.
        cal_labels: [N_cal, K] calibration labels.
        cal_weights: [N_cal] importance weights.
        test_probs: [N_test, K] test probabilities.
        test_labels: [N_test, K] test labels.
        test_weights: [N_test] importance weights.
        pathology_names: K pathology names.
        expert: SimulatedExpert instance.
        alphas: array of target FNR levels.
        betas: array of deferral budgets.
        method_name: label for results.
        seed: random seed.

    Returns:
        List of SCRCDeferralResult, one per (alpha, beta) pair.
    """
    results = []

    for beta in betas:
        for alpha in alphas:
            predictor = SCRCPredictor(
                beta=float(beta),
                alpha=float(alpha),
                seed=seed,
            )
            crc_result = predictor.calibrate(cal_probs, cal_labels, cal_weights)
            scrc_result = predictor.predict(test_probs, test_weights)

            # FNR metrics on kept samples
            kept = ~scrc_result.defer_mask
            if kept.any():
                fnr_metrics = compute_fnr_metrics(
                    scrc_result.prediction_sets[kept],
                    test_labels[kept],
                    pathology_names,
                    weights=test_weights[kept],
                )
            else:
                fnr_metrics = {
                    "overall_fnr": 0.0,
                    "weighted_fnr": 0.0,
                    "per_pathology_fnr": {p: 0.0 for p in pathology_names},
                    "per_pathology_fpr": {p: 0.0 for p in pathology_names},
                }

            # System accuracy
            sys = compute_scrc_system_accuracy(
                scrc_result.prediction_sets,
                test_labels,
                scrc_result.defer_mask,
                expert,
            )

            results.append(SCRCDeferralResult(
                method=method_name,
                alpha=float(alpha),
                beta=float(beta),
                deferral_rate=scrc_result.deferral_rate,
                n_total=len(test_labels),
                n_deferred=int(scrc_result.defer_mask.sum()),
                empirical_fnr_on_kept=fnr_metrics["overall_fnr"],
                weighted_fnr_on_kept=fnr_metrics["weighted_fnr"],
                per_pathology_fnr=fnr_metrics["per_pathology_fnr"],
                per_pathology_fpr=fnr_metrics["per_pathology_fpr"],
                system_accuracy=sys["system_accuracy"],
                model_accuracy_on_kept=sys["model_accuracy_on_kept"],
                lambda_hat=scrc_result.lambda_hat,
                ess_fraction=crc_result.ess_fraction,
            ))

    return results


def evaluate_scrc_unweighted(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    pathology_names: list[str],
    expert: SimulatedExpert,
    alphas: np.ndarray,
    betas: np.ndarray,
    seed: int = 42,
) -> list[SCRCDeferralResult]:
    """Evaluate unweighted SCRC (standard CRC, no DRE) as a baseline."""
    N_cal = len(cal_probs)
    N_test = len(test_probs)
    return evaluate_scrc(
        cal_probs=cal_probs,
        cal_labels=cal_labels,
        cal_weights=np.ones(N_cal),
        test_probs=test_probs,
        test_labels=test_labels,
        test_weights=np.ones(N_test),
        pathology_names=pathology_names,
        expert=expert,
        alphas=alphas,
        betas=betas,
        method_name="SCRC (unweighted)",
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fnr_vs_deferral(
    results: dict[str, list[SCRCDeferralResult]],
    title: str = "FNR vs Deferral Rate",
) -> plt.Figure:
    """Plot FNR-deferral trade-off curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, res_list in results.items():
        dr = [r.deferral_rate for r in res_list]
        fnr = [r.empirical_fnr_on_kept for r in res_list]

        order = np.argsort(dr)
        ax.plot(
            np.array(dr)[order],
            np.array(fnr)[order],
            label=method_name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Deferral Rate")
    ax.set_ylabel("FNR (on non-deferred samples)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scrc_accuracy_rejection(
    results: dict[str, list[SCRCDeferralResult]],
    title: str = "SCRC: System Accuracy vs Deferral Rate",
) -> plt.Figure:
    """Accuracy-rejection curve for SCRC methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, res_list in results.items():
        dr = [r.deferral_rate for r in res_list]
        sa = [r.system_accuracy for r in res_list]

        order = np.argsort(dr)
        ax.plot(
            np.array(dr)[order],
            np.array(sa)[order],
            label=method_name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Deferral Rate")
    ax.set_ylabel("System Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lambda_calibration(
    crc_result: CRCResult,
    alpha: float,
    title: str = "CRC Lambda Calibration",
) -> plt.Figure:
    """Plot FNR vs lambda curve from calibration, showing lambda* cutoff."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        crc_result.lambda_path,
        crc_result.fnr_path,
        color="#1f77b4",
        linewidth=1.5,
        label="Weighted FNR",
    )
    ax.axhline(
        y=alpha,
        color="red",
        linestyle="--",
        linewidth=1,
        label=rf"$\alpha = {alpha}$",
    )
    ax.axvline(
        x=crc_result.lambda_hat,
        color="green",
        linestyle="--",
        linewidth=1,
        label=rf"$\lambda^* = {crc_result.lambda_hat:.4f}$",
    )

    ax.set_xlabel(r"$\lambda$ (threshold)")
    ax.set_ylabel("Weighted FNR")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_pathology_fnr(
    results_list: list[SCRCDeferralResult],
    pathology_names: list[str],
    title: str = "Per-Pathology FNR vs Beta",
) -> plt.Figure:
    """Plot per-pathology FNR across operating points."""
    fig, ax = plt.subplots(figsize=(10, 5))

    betas = [r.beta for r in results_list]

    for p in pathology_names:
        fnrs = [r.per_pathology_fnr.get(p, float("nan")) for r in results_list]
        ax.plot(betas, fnrs, marker="o", markersize=3, linewidth=1.5, label=p)

    ax.set_xlabel(r"$\beta$ (Deferral Budget)")
    ax.set_ylabel("FNR")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def scrc_summary_table(
    results: dict[str, list[SCRCDeferralResult]],
    alpha: float = 0.10,
    beta: float = 0.15,
) -> pd.DataFrame:
    """Summary table at a fixed (alpha, beta) operating point."""
    rows = []
    for method_name, res_list in results.items():
        closest = min(
            res_list,
            key=lambda r: abs(r.alpha - alpha) + abs(r.beta - beta),
        )
        rows.append({
            "Method": method_name,
            "Alpha": f"{closest.alpha:.2f}",
            "Beta": f"{closest.beta:.2f}",
            "Deferral": f"{closest.deferral_rate:.3f}",
            "FNR (kept)": f"{closest.empirical_fnr_on_kept:.3f}",
            "W-FNR (kept)": f"{closest.weighted_fnr_on_kept:.3f}",
            "System Acc": f"{closest.system_accuracy:.3f}",
            "Model Acc (kept)": f"{closest.model_accuracy_on_kept:.3f}",
            "Lambda": f"{closest.lambda_hat:.4f}",
        })

    return pd.DataFrame(rows)

"""Evaluation metrics and plots for SCRC (Selective Conformal Risk Control).

Provides FNR-specific metrics, system accuracy computation, grid evaluation,
and plotting functions for the two-stage SCRC pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .expert import SimulatedExpert
from .scrc import (
    CRCResult,
    SCRCPredictor,
    PerPathologyCRCResult,
    PerPathologySCRCPredictor,
    calibrate_per_pathology_crc_fnr,
    multilabel_entropy,
    select_for_deferral,
    refit_dre_post_selection,
    compute_capability_alpha,
)


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


# ---------------------------------------------------------------------------
# Per-pathology SCRC evaluation
# ---------------------------------------------------------------------------

@dataclass
class PerPathologySCRCDeferralResult:
    """Results from a PP-SCRC experiment at one operating point."""

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
    per_pathology_fnr: dict = None
    per_pathology_fpr: dict = None
    # System metrics
    system_accuracy: float = 0.0
    model_accuracy_on_kept: float = 0.0
    lambda_hats: dict = None          # {pathology: lambda_k*}
    ess_fraction: float = 0.0

    def __post_init__(self):
        if self.per_pathology_fnr is None:
            self.per_pathology_fnr = {}
        if self.per_pathology_fpr is None:
            self.per_pathology_fpr = {}
        if self.lambda_hats is None:
            self.lambda_hats = {}


def evaluate_per_pathology_scrc(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_weights: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    pathology_names: list,
    expert: "SimulatedExpert",
    alphas: np.ndarray,
    betas: np.ndarray,
    method_name: str = "PP-SCRC (weighted)",
    seed: int = 42,
) -> list:
    """Evaluate per-pathology SCRC across alpha x beta grid.

    Args:
        cal_probs: [N_cal, K] calibration probabilities.
        cal_labels: [N_cal, K] calibration labels.
        cal_weights: [N_cal] importance weights.
        test_probs: [N_test, K] test probabilities.
        test_labels: [N_test, K] test labels.
        test_weights: [N_test] importance weights.
        pathology_names: K pathology names.
        expert: SimulatedExpert instance.
        alphas: array of target FNR levels (scalar per experiment; same alpha for all K).
        betas: array of deferral budgets.
        method_name: label for results.
        seed: random seed.

    Returns:
        List of PerPathologySCRCDeferralResult, one per (alpha, beta) pair.
    """
    results = []

    for beta in betas:
        for alpha in alphas:
            predictor = PerPathologySCRCPredictor(
                beta=float(beta),
                alpha=float(alpha),
                seed=seed,
            )
            crc_result = predictor.calibrate(
                cal_probs, cal_labels, cal_weights, pathology_names=pathology_names
            )
            pp_result = predictor.predict(test_probs)

            # FNR metrics on kept samples
            kept = ~pp_result.defer_mask
            if kept.any():
                fnr_metrics = compute_fnr_metrics(
                    pp_result.prediction_sets[kept],
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
                pp_result.prediction_sets,
                test_labels,
                pp_result.defer_mask,
                expert,
            )

            lambda_hats_dict = {
                p: float(crc_result.lambda_hats[k])
                for k, p in enumerate(pathology_names)
            }

            results.append(PerPathologySCRCDeferralResult(
                method=method_name,
                alpha=float(alpha),
                beta=float(beta),
                deferral_rate=pp_result.deferral_rate,
                n_total=len(test_labels),
                n_deferred=int(pp_result.defer_mask.sum()),
                empirical_fnr_on_kept=fnr_metrics["overall_fnr"],
                weighted_fnr_on_kept=fnr_metrics["weighted_fnr"],
                per_pathology_fnr=fnr_metrics["per_pathology_fnr"],
                per_pathology_fpr=fnr_metrics["per_pathology_fpr"],
                system_accuracy=sys["system_accuracy"],
                model_accuracy_on_kept=sys["model_accuracy_on_kept"],
                lambda_hats=lambda_hats_dict,
                ess_fraction=crc_result.ess_fraction,
            ))

    return results


def evaluate_per_pathology_scrc_unweighted(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    pathology_names: list,
    expert: "SimulatedExpert",
    alphas: np.ndarray,
    betas: np.ndarray,
    seed: int = 42,
) -> list:
    """Evaluate unweighted PP-SCRC (standard CRC, no DRE) as a baseline."""
    N_cal = len(cal_probs)
    N_test = len(test_probs)
    return evaluate_per_pathology_scrc(
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
        method_name="PP-SCRC (unweighted)",
        seed=seed,
    )


# ---------------------------------------------------------------------------
# PP-SCRC plotting
# ---------------------------------------------------------------------------

def plot_per_pathology_thresholds(
    results_list: list,
    pathology_names: list,
    reference_results: list | None = None,
    title: str = "Per-Pathology λ* vs β",
) -> plt.Figure:
    """Line plot of per-pathology lambda_k* across beta values.

    Args:
        results_list: list of PerPathologySCRCDeferralResult (one alpha, varying betas).
        pathology_names: K pathology names.
        reference_results: optional list of SCRCDeferralResult (global SCRC) for dashed
            reference line.
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    betas = [r.beta for r in results_list]

    for p in pathology_names:
        lambdas = [r.lambda_hats.get(p, float("nan")) for r in results_list]
        ax.plot(betas, lambdas, marker="o", markersize=4, linewidth=1.5, label=p)

    if reference_results is not None:
        ref_betas = [r.beta for r in reference_results]
        ref_lambdas = [r.lambda_hat for r in reference_results]
        order = np.argsort(ref_betas)
        ax.plot(
            np.array(ref_betas)[order],
            np.array(ref_lambdas)[order],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Global λ* (SCRC)",
        )

    ax.set_xlabel(r"$\beta$ (Deferral Budget)")
    ax.set_ylabel(r"$\lambda_k^*$ (threshold)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lambda_comparison_bar(
    global_result: "SCRCDeferralResult",
    pp_result: PerPathologySCRCDeferralResult,
    pathology_names: list,
    title: str = "Global vs Per-Pathology λ* Comparison",
) -> plt.Figure:
    """Grouped bar chart comparing global vs per-pathology lambda* at one operating point.

    Args:
        global_result: single SCRCDeferralResult at the operating point.
        pp_result: single PerPathologySCRCDeferralResult at the operating point.
        pathology_names: K pathology names.
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    K = len(pathology_names)
    x = np.arange(K)
    width = 0.35

    global_lambda = global_result.lambda_hat
    pp_lambdas = [pp_result.lambda_hats.get(p, float("nan")) for p in pathology_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [global_lambda] * K,
        width,
        label=f"Global λ* = {global_lambda:.4f}",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.bar(x + width / 2, pp_lambdas, width, label="Per-pathology λ_k*", color="#ff7f0e", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pathology_names, rotation=30, ha="right")
    ax.set_ylabel(r"$\lambda^*$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_per_pathology_calibration_curves(
    pp_crc_result: PerPathologyCRCResult,
    alpha: float,
    pathology_names: list | None = None,
    n_cols: int = 4,
) -> plt.Figure:
    """K-panel subplots showing FNR vs lambda calibration curve per pathology.

    Args:
        pp_crc_result: PerPathologyCRCResult from calibration.
        alpha: target FNR level (drawn as horizontal dashed line).
        pathology_names: K pathology names (falls back to pp_crc_result.pathology_names).
        n_cols: number of subplot columns.

    Returns:
        matplotlib Figure.
    """
    if pathology_names is None:
        pathology_names = pp_crc_result.pathology_names

    K = len(pathology_names)
    n_rows = math.ceil(K / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for k, name in enumerate(pathology_names):
        ax = axes_flat[k]
        lam_path = pp_crc_result.lambda_paths[k]
        fnr_path = pp_crc_result.fnr_paths[k]
        lam_hat = float(pp_crc_result.lambda_hats[k])
        fnr_at_hat = float(pp_crc_result.weighted_fnr_at_lambda[k])

        ax.plot(lam_path, fnr_path, color="#1f77b4", linewidth=1.5)
        ax.axhline(y=alpha, color="red", linestyle="--", linewidth=1,
                   label=rf"$\alpha={alpha}$")
        ax.axvline(x=lam_hat, color="green", linestyle="--", linewidth=1,
                   label=rf"$\lambda^*={lam_hat:.3f}$")

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Weighted FNR")
        ax.set_title(f"{name}\n(n_pos={pp_crc_result.n_positives[k]})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)

    # Hide unused axes
    for k in range(K, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        rf"Per-Pathology CRC Calibration Curves ($\alpha={alpha}$)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PP-SCRC table functions
# ---------------------------------------------------------------------------

def per_pathology_threshold_table(
    global_results: list,
    pp_results: list,
    alpha: float,
    beta: float,
    pathology_names: list,
) -> pd.DataFrame:
    """Return DataFrame comparing global vs per-pathology thresholds.

    Columns: Pathology | Global λ* | PP λ* | Global FNR | PP FNR | ΔFNR | Global FPR | PP FPR

    Args:
        global_results: list of SCRCDeferralResult.
        pp_results: list of PerPathologySCRCDeferralResult.
        alpha: target FNR level to filter by.
        beta: deferral budget to filter by.
        pathology_names: K pathology names.

    Returns:
        pandas DataFrame.
    """
    # Find closest result to requested (alpha, beta)
    global_r = min(
        global_results,
        key=lambda r: abs(r.alpha - alpha) + abs(r.beta - beta),
    )
    pp_r = min(
        pp_results,
        key=lambda r: abs(r.alpha - alpha) + abs(r.beta - beta),
    )

    rows = []
    for p in pathology_names:
        g_lam = global_r.lambda_hat
        pp_lam = pp_r.lambda_hats.get(p, float("nan"))
        g_fnr = global_r.per_pathology_fnr.get(p, float("nan"))
        pp_fnr = pp_r.per_pathology_fnr.get(p, float("nan"))
        delta_fnr = pp_fnr - g_fnr if not (np.isnan(g_fnr) or np.isnan(pp_fnr)) else float("nan")
        g_fpr = global_r.per_pathology_fpr.get(p, float("nan"))
        pp_fpr = pp_r.per_pathology_fpr.get(p, float("nan"))

        rows.append({
            "Pathology": p,
            "Global λ*": f"{g_lam:.4f}",
            "PP λ*": f"{pp_lam:.4f}",
            "Global FNR": f"{g_fnr:.3f}",
            "PP FNR": f"{pp_fnr:.3f}",
            "ΔFNR": f"{delta_fnr:+.3f}" if not np.isnan(delta_fnr) else "nan",
            "Global FPR": f"{g_fpr:.3f}",
            "PP FPR": f"{pp_fpr:.3f}",
        })

    return pd.DataFrame(rows)


def pp_scrc_summary_table(
    results: dict,
    alpha: float = 0.10,
    beta: float = 0.15,
) -> pd.DataFrame:
    """Summary table at a fixed (alpha, beta) operating point for PP-SCRC results.

    Args:
        results: dict mapping method name -> list of PerPathologySCRCDeferralResult.
        alpha: target FNR level.
        beta: deferral budget.

    Returns:
        pandas DataFrame (same columns as scrc_summary_table).
    """
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
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed PP-SCRC evaluation (Fixes 7.2 and 7.3)
# ---------------------------------------------------------------------------

def evaluate_fixed_pp_scrc(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_features: np.ndarray,
    cal_weights: np.ndarray,
    target_features_pool: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    pathology_names: list,
    expert: "SimulatedExpert",
    alphas: np.ndarray,
    betas: np.ndarray,
    nih_aucs: np.ndarray | None = None,
    use_post_selection_dre: bool = True,
    use_capability_alpha: bool = True,
    n_components: int = 4,
    weight_clip: float = 20.0,
    method_name: str = "Fixed PP-SCRC",
    seed: int = 42,
) -> list:
    """Evaluate fixed PP-SCRC across alpha x beta grid.

    Implements Fix 7.2 (post-selection DRE refit) and Fix 7.3 (AUC-based
    capability alpha allocation) as optional ablations, controlled by
    ``use_post_selection_dre`` and ``use_capability_alpha`` flags.

    Fix 7.2 (post-selection DRE refit): After Stage 1, refit a fresh DRE using
    only the kept calibration features as source. This corrects the bias in
    importance weights caused by the distributional shift between the full cal
    set and the post-Stage1 kept subset.

    Fix 7.3 (capability alpha): Allocate per-pathology alpha_k inversely
    proportional to AUC-0.5 excess. Low-AUC pathologies receive looser alpha_k,
    allowing lower lambda_k* thresholds that catch more true positives.

    Args:
        cal_probs: [N_cal, K] calibration probabilities.
        cal_labels: [N_cal, K] calibration labels.
        cal_features: [N_cal, D] calibration raw features (for DRE refit).
        cal_weights: [N_cal] pre-computed importance weights (full cal set DRE).
        target_features_pool: [N_pool, D] target domain features for DRE.
        test_probs: [N_test, K] test probabilities.
        test_labels: [N_test, K] test labels.
        test_weights: [N_test] importance weights (for metric computation).
        pathology_names: K pathology name strings.
        expert: SimulatedExpert instance.
        alphas: array of target FNR levels.
        betas: array of deferral budgets.
        nih_aucs: [K] AUC values on target domain (required if use_capability_alpha).
        use_post_selection_dre: if True, refit DRE on kept cal features (Fix 7.2).
        use_capability_alpha: if True, allocate alpha per pathology via AUC (Fix 7.3).
        n_components: PCA dimensions for DRE refit.
        weight_clip: weight clipping for DRE refit.
        method_name: label for results.
        seed: random seed.

    Returns:
        List of PerPathologySCRCDeferralResult, one per (alpha, beta) pair.
    """
    if use_capability_alpha and nih_aucs is None:
        raise ValueError("nih_aucs must be provided when use_capability_alpha=True.")

    results = []

    for beta in betas:
        # --- Stage 1: deferral on calibration set ---
        cal_entropy = multilabel_entropy(cal_probs)
        cal_defer_mask = select_for_deferral(cal_entropy, float(beta), seed)
        kept_cal = ~cal_defer_mask

        # --- Fix 7.2: post-selection DRE refit ---
        if use_post_selection_dre:
            dre_post = refit_dre_post_selection(
                cal_features_kept=cal_features[kept_cal],
                target_features=target_features_pool,
                n_components=n_components,
                weight_clip=weight_clip,
                random_state=seed,
            )
            w_cal = dre_post.compute_weights(cal_features[kept_cal])
        else:
            w_cal = cal_weights[kept_cal]

        # --- Stage 1: deferral on test set ---
        test_entropy = multilabel_entropy(test_probs)
        test_defer_mask = select_for_deferral(test_entropy, float(beta), seed)
        kept_test = ~test_defer_mask

        for alpha in alphas:
            # --- Fix 7.3: capability alpha ---
            if use_capability_alpha:
                alpha_k = compute_capability_alpha(nih_aucs, float(alpha))
            else:
                alpha_k = float(alpha)

            # --- Stage 2: per-pathology CRC calibration ---
            crc_result = calibrate_per_pathology_crc_fnr(
                probs=cal_probs[kept_cal],
                labels=cal_labels[kept_cal],
                weights=w_cal,
                alpha=alpha_k,
                pathology_names=pathology_names,
            )

            # --- Apply per-pathology thresholds to test ---
            N_test, K = test_probs.shape
            prediction_sets = np.zeros((N_test, K), dtype=np.int32)
            if kept_test.any():
                prediction_sets[kept_test] = (
                    test_probs[kept_test] >= crc_result.lambda_hats[np.newaxis, :]
                ).astype(np.int32)

            deferral_rate = float(test_defer_mask.sum()) / N_test if N_test > 0 else 0.0

            # --- FNR metrics on kept test samples ---
            if kept_test.any():
                fnr_metrics = compute_fnr_metrics(
                    prediction_sets[kept_test],
                    test_labels[kept_test],
                    pathology_names,
                    weights=test_weights[kept_test],
                )
            else:
                fnr_metrics = {
                    "overall_fnr": 0.0,
                    "weighted_fnr": 0.0,
                    "per_pathology_fnr": {p: 0.0 for p in pathology_names},
                    "per_pathology_fpr": {p: 0.0 for p in pathology_names},
                }

            # --- System accuracy ---
            sys = compute_scrc_system_accuracy(
                prediction_sets,
                test_labels,
                test_defer_mask,
                expert,
            )

            lambda_hats_dict = {
                p: float(crc_result.lambda_hats[k])
                for k, p in enumerate(pathology_names)
            }

            results.append(PerPathologySCRCDeferralResult(
                method=method_name,
                alpha=float(alpha),
                beta=float(beta),
                deferral_rate=deferral_rate,
                n_total=N_test,
                n_deferred=int(test_defer_mask.sum()),
                empirical_fnr_on_kept=fnr_metrics["overall_fnr"],
                weighted_fnr_on_kept=fnr_metrics["weighted_fnr"],
                per_pathology_fnr=fnr_metrics["per_pathology_fnr"],
                per_pathology_fpr=fnr_metrics["per_pathology_fpr"],
                system_accuracy=sys["system_accuracy"],
                model_accuracy_on_kept=sys["model_accuracy_on_kept"],
                lambda_hats=lambda_hats_dict,
                ess_fraction=crc_result.ess_fraction,
            ))

    return results

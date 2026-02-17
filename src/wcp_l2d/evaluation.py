"""Evaluation metrics, deferral logic, and comparison plots.

Computes system accuracy under deferral, coverage metrics, and
generates accuracy-rejection curves comparing multiple methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

from .conformal import ConformalPredictor, WeightedConformalPredictor


@dataclass
class DeferralResult:
    """Results from a single deferral experiment at one operating point."""

    method: str
    alpha_or_threshold: float
    system_accuracy: float
    deferral_rate: float
    coverage_rate: float
    average_set_size: float
    model_accuracy_on_kept: float
    n_total: int
    n_deferred: int


def compute_system_accuracy(
    model_predictions: np.ndarray,
    true_labels: np.ndarray,
    defer_mask: np.ndarray,
    expert_accuracy: float = 0.85,
    seed: int = 42,
) -> dict:
    """Compute system accuracy under a deferral policy.

    Non-deferred samples: model prediction vs true label.
    Deferred samples: each correct with probability ``expert_accuracy``.

    Args:
        model_predictions: [N] predicted class labels.
        true_labels: [N] ground truth class labels.
        defer_mask: [N] boolean, True = defer to expert.
        expert_accuracy: fixed expert accuracy.
        seed: random seed for expert simulation.

    Returns:
        Dict with system_accuracy, deferral_rate, model_accuracy_on_kept,
        n_deferred, n_kept.
    """
    N = len(true_labels)
    n_deferred = int(defer_mask.sum())
    n_kept = N - n_deferred

    # Model accuracy on kept samples
    if n_kept > 0:
        model_correct = int((model_predictions[~defer_mask] == true_labels[~defer_mask]).sum())
    else:
        model_correct = 0

    # Expert accuracy on deferred samples (Bernoulli simulation)
    rng = np.random.RandomState(seed)
    expert_correct = int(rng.binomial(n_deferred, expert_accuracy))

    system_accuracy = (model_correct + expert_correct) / N if N > 0 else 0.0
    model_acc_on_kept = model_correct / n_kept if n_kept > 0 else 0.0

    return {
        "system_accuracy": system_accuracy,
        "deferral_rate": n_deferred / N if N > 0 else 0.0,
        "model_accuracy_on_kept": model_acc_on_kept,
        "n_deferred": n_deferred,
        "n_kept": n_kept,
    }


def compute_coverage(
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Compute conformal prediction coverage metrics.

    Args:
        prediction_sets: [N, K] binary matrix.
        true_labels: [N] integer labels.

    Returns:
        Dict with coverage_rate, average_set_size.
    """
    N = len(true_labels)
    covered = prediction_sets[np.arange(N), true_labels].astype(bool)
    set_sizes = prediction_sets.sum(axis=1)

    return {
        "coverage_rate": float(covered.mean()),
        "average_set_size": float(set_sizes.mean()),
    }


def _predictions_from_sets(
    prediction_sets: np.ndarray,
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract model predictions and defer mask from prediction sets.

    For singleton sets: predicted class = the single included class.
    For empty or multi-class sets: defer.

    Args:
        prediction_sets: [N, K] binary matrix.
        logits: [N, K] raw logits (used for argmax when set is singleton).

    Returns:
        model_predictions: [N] predicted class (argmax of logits for all).
        defer_mask: [N] boolean, True if |C(x)| != 1.
    """
    set_sizes = prediction_sets.sum(axis=1)
    defer_mask = set_sizes != 1

    # For non-deferred (singleton sets), the prediction is the single class
    # For deferred, use argmax as placeholder (won't be evaluated)
    model_predictions = np.argmax(logits, axis=1)
    # Override with the singleton class for non-deferred
    singleton_idx = np.where(~defer_mask)[0]
    if len(singleton_idx) > 0:
        model_predictions[singleton_idx] = np.argmax(
            prediction_sets[singleton_idx], axis=1,
        )

    return model_predictions, defer_mask


def evaluate_max_logit(
    logits: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray,
    expert_accuracy: float = 0.85,
) -> list[DeferralResult]:
    """Baseline 1: defer if max(softmax(logits)) < threshold.

    Args:
        logits: [N, K] raw logits.
        true_labels: [N] integer labels.
        thresholds: array of confidence thresholds to sweep.
        expert_accuracy: fixed expert accuracy.

    Returns:
        List of DeferralResult, one per threshold.
    """
    probs = softmax(logits, axis=1)
    max_probs = probs.max(axis=1)
    model_predictions = np.argmax(logits, axis=1)

    results = []
    for t in thresholds:
        defer_mask = max_probs < t
        metrics = compute_system_accuracy(
            model_predictions, true_labels, defer_mask,
            expert_accuracy=expert_accuracy,
        )
        results.append(DeferralResult(
            method="Max Logit",
            alpha_or_threshold=float(t),
            system_accuracy=metrics["system_accuracy"],
            deferral_rate=metrics["deferral_rate"],
            coverage_rate=np.nan,  # N/A for this baseline
            average_set_size=np.nan,
            model_accuracy_on_kept=metrics["model_accuracy_on_kept"],
            n_total=len(true_labels),
            n_deferred=metrics["n_deferred"],
        ))

    return results


def evaluate_standard_cp(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    alphas: np.ndarray,
    expert_accuracy: float = 0.85,
    raps_penalty: float = 0.1,
    kreg: int = 1,
) -> list[DeferralResult]:
    """Baseline 2: standard CP with RAPS.

    Args:
        cal_logits: [N_cal, K] calibration logits.
        cal_labels: [N_cal] calibration labels.
        test_logits: [N_test, K] test logits.
        test_labels: [N_test] test labels.
        alphas: array of significance levels to sweep.
        expert_accuracy: fixed expert accuracy.
        raps_penalty: RAPS penalty parameter.
        kreg: RAPS rank of regularization.

    Returns:
        List of DeferralResult, one per alpha.
    """
    results = []
    for alpha in alphas:
        cp = ConformalPredictor(
            penalty=raps_penalty, kreg=kreg, randomized=False,
        )
        cp.calibrate(cal_logits, cal_labels, alpha=alpha)
        pred_sets = cp.predict(test_logits)

        cov = compute_coverage(pred_sets, test_labels)
        preds, defer_mask = _predictions_from_sets(pred_sets, test_logits)
        sys_metrics = compute_system_accuracy(
            preds, test_labels, defer_mask,
            expert_accuracy=expert_accuracy,
        )

        results.append(DeferralResult(
            method="Standard CP",
            alpha_or_threshold=float(alpha),
            system_accuracy=sys_metrics["system_accuracy"],
            deferral_rate=sys_metrics["deferral_rate"],
            coverage_rate=cov["coverage_rate"],
            average_set_size=cov["average_set_size"],
            model_accuracy_on_kept=sys_metrics["model_accuracy_on_kept"],
            n_total=len(test_labels),
            n_deferred=sys_metrics["n_deferred"],
        ))

    return results


def evaluate_wcp(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    cal_weights: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    alphas: np.ndarray,
    expert_accuracy: float = 0.85,
    raps_penalty: float = 0.1,
    kreg: int = 1,
) -> list[DeferralResult]:
    """Proposed: WCP with RAPS + DRE weights.

    Args:
        cal_logits: [N_cal, K] calibration logits.
        cal_labels: [N_cal] calibration labels.
        cal_weights: [N_cal] importance weights for calibration samples.
        test_logits: [N_test, K] test logits.
        test_labels: [N_test] test labels.
        test_weights: [N_test] importance weights for test samples.
        alphas: array of significance levels to sweep.
        expert_accuracy: fixed expert accuracy.
        raps_penalty: RAPS penalty parameter.
        kreg: RAPS rank of regularization.

    Returns:
        List of DeferralResult, one per alpha.
    """
    results = []
    for alpha in alphas:
        wcp = WeightedConformalPredictor(
            penalty=raps_penalty, kreg=kreg, randomized=False,
        )
        wcp.calibrate(cal_logits, cal_labels, cal_weights)
        pred_sets = wcp.predict(test_logits, test_weights, alpha=alpha)

        cov = compute_coverage(pred_sets, test_labels)
        preds, defer_mask = _predictions_from_sets(pred_sets, test_logits)
        sys_metrics = compute_system_accuracy(
            preds, test_labels, defer_mask,
            expert_accuracy=expert_accuracy,
        )

        results.append(DeferralResult(
            method="WCP",
            alpha_or_threshold=float(alpha),
            system_accuracy=sys_metrics["system_accuracy"],
            deferral_rate=sys_metrics["deferral_rate"],
            coverage_rate=cov["coverage_rate"],
            average_set_size=cov["average_set_size"],
            model_accuracy_on_kept=sys_metrics["model_accuracy_on_kept"],
            n_total=len(test_labels),
            n_deferred=sys_metrics["n_deferred"],
        ))

    return results


def evaluate_continuous_deferral(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    alphas: np.ndarray,
    cal_weights: np.ndarray | None = None,
    expert_accuracy: float = 0.85,
) -> list[DeferralResult]:
    """Continuous deferral with calibrated uncertainty threshold.

    Uses uncertainty score u(x) = 1 - max(softmax(logits)) and sets the
    deferral threshold via a (possibly weighted) quantile of calibration
    scores.  This is the method from Section 4 of the continuous-deferral
    report.

    Args:
        cal_logits: [N_cal, K] calibration logits.
        cal_labels: [N_cal] calibration labels (unused for threshold, kept
            for API consistency).
        test_logits: [N_test, K] test logits.
        test_labels: [N_test] test labels.
        alphas: array of significance levels to sweep.
        cal_weights: [N_cal] importance weights from DRE.  If None, uses
            the unweighted (source-calibrated) quantile.
        expert_accuracy: fixed expert accuracy for deferred samples.

    Returns:
        List of DeferralResult, one per alpha.
    """
    cal_probs = softmax(cal_logits, axis=1)
    u_cal = 1.0 - cal_probs.max(axis=1)

    test_probs = softmax(test_logits, axis=1)
    u_test = 1.0 - test_probs.max(axis=1)
    model_predictions = np.argmax(test_logits, axis=1)

    # Pre-sort calibration scores (and weights) for weighted quantile
    sort_idx = np.argsort(u_cal)
    u_cal_sorted = u_cal[sort_idx]
    if cal_weights is not None:
        w_sorted = cal_weights[sort_idx]
        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]

    results = []
    for alpha in alphas:
        # Compute threshold tau
        if cal_weights is None:
            tau = float(np.quantile(u_cal, 1.0 - alpha))
        else:
            # Weighted quantile: smallest tau s.t.
            # sum(w_i * 1[u_i <= tau]) / sum(w_i) >= 1 - alpha
            target = (1.0 - alpha) * total_w
            idx = np.searchsorted(cum_w, target, side="left")
            idx = min(idx, len(u_cal_sorted) - 1)
            tau = float(u_cal_sorted[idx])

        defer_mask = u_test > tau
        metrics = compute_system_accuracy(
            model_predictions, test_labels, defer_mask,
            expert_accuracy=expert_accuracy,
        )

        method = "Continuous (DRE)" if cal_weights is not None else "Continuous (source)"
        results.append(DeferralResult(
            method=method,
            alpha_or_threshold=float(alpha),
            system_accuracy=metrics["system_accuracy"],
            deferral_rate=metrics["deferral_rate"],
            coverage_rate=np.nan,  # N/A for continuous deferral
            average_set_size=np.nan,
            model_accuracy_on_kept=metrics["model_accuracy_on_kept"],
            n_total=len(test_labels),
            n_deferred=metrics["n_deferred"],
        ))

    return results


def plot_accuracy_rejection_curve(
    results: dict[str, list[DeferralResult]],
    title: str = "System Accuracy vs Deferral Rate",
) -> plt.Figure:
    """Plot accuracy-rejection curves for all methods.

    Args:
        results: mapping from method name to list of DeferralResult.
        title: plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Max Logit": "#1f77b4", "Standard CP": "#ff7f0e", "WCP (Ours)": "#2ca02c"}
    markers = {"Max Logit": "s", "Standard CP": "^", "WCP (Ours)": "o"}

    for method_name, res_list in results.items():
        deferral_rates = [r.deferral_rate for r in res_list]
        sys_accs = [r.system_accuracy for r in res_list]

        # Sort by deferral rate for a clean curve
        order = np.argsort(deferral_rates)
        dr = np.array(deferral_rates)[order]
        sa = np.array(sys_accs)[order]

        ax.plot(
            dr, sa,
            label=method_name,
            color=colors.get(method_name, None),
            marker=markers.get(method_name, "o"),
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


def plot_coverage_comparison(
    results: dict[str, list[DeferralResult]],
    target_coverage: float = 0.9,
) -> plt.Figure:
    """Plot coverage rate vs alpha for CP methods.

    Args:
        results: mapping from method name to list of DeferralResult.
        target_coverage: target coverage line.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, res_list in results.items():
        alphas = [r.alpha_or_threshold for r in res_list]
        coverages = [r.coverage_rate for r in res_list]
        if np.isnan(coverages[0]):
            continue  # skip Max Logit (no coverage)
        ax.plot(alphas, coverages, label=method_name, linewidth=1.5, marker="o", markersize=3)

    # Ideal coverage line: y = 1 - alpha
    all_alphas = sorted({r.alpha_or_threshold for res_list in results.values() for r in res_list})
    ax.plot(all_alphas, [1 - a for a in all_alphas], color="black", linestyle="--", alpha=0.5, linewidth=1.5, label=r"Ideal ($1 - \alpha$)")
    ax.axhline(y=target_coverage, color="red", linestyle="--", alpha=0.5, label=f"Target ({target_coverage:.0%})")
    ax.set_xlabel(r"$\alpha$ (Significance Level)")
    ax.set_ylabel("Coverage Rate")
    ax.set_title(r"Coverage Rate vs $\alpha$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def summary_table(
    results: dict[str, list[DeferralResult]],
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Generate summary table at a fixed alpha / closest operating point.

    For Max Logit, picks the threshold closest to the CP methods' deferral rate.

    Args:
        results: mapping from method name to list of DeferralResult.
        alpha: target significance level.

    Returns:
        DataFrame with one row per method.
    """
    rows = []

    # Find the CP/WCP result closest to the target alpha
    cp_deferral_rate = None
    for method_name, res_list in results.items():
        if method_name == "Max Logit":
            continue
        closest = min(res_list, key=lambda r: abs(r.alpha_or_threshold - alpha))
        if cp_deferral_rate is None:
            cp_deferral_rate = closest.deferral_rate
        rows.append({
            "Method": method_name,
            "System Accuracy": f"{closest.system_accuracy:.4f}",
            "Deferral Rate": f"{closest.deferral_rate:.4f}",
            "Coverage": f"{closest.coverage_rate:.4f}",
            "Avg Set Size": f"{closest.average_set_size:.2f}",
        })

    # For Max Logit, match deferral rate to CP methods
    if "Max Logit" in results and cp_deferral_rate is not None:
        ml_results = results["Max Logit"]
        closest_ml = min(ml_results, key=lambda r: abs(r.deferral_rate - cp_deferral_rate))
        rows.insert(0, {
            "Method": "Max Logit",
            "System Accuracy": f"{closest_ml.system_accuracy:.4f}",
            "Deferral Rate": f"{closest_ml.deferral_rate:.4f}",
            "Coverage": "N/A",
            "Avg Set Size": "N/A",
        })

    return pd.DataFrame(rows)

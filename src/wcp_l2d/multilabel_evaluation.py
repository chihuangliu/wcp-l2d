"""Evaluation metrics and plots for multi-label conformal prediction.

Extends the single-label evaluation module with metrics appropriate for
multi-label prediction sets: joint coverage, per-pathology coverage,
count-based deferral, and multi-label system accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .expert import SimulatedExpert
from .multilabel_conformal import (
    AggregationStrategy,
    MultilabelConformalPredictor,
    MultilabelWeightedConformalPredictor,
)


@dataclass
class MultilabelDeferralResult:
    """Results from a multi-label deferral experiment at one operating point."""

    method: str
    alpha_or_threshold: float
    aggregation: str
    # Per-pathology metrics
    per_pathology_coverage: dict[str, float] = field(default_factory=dict)
    per_pathology_set_size: dict[str, float] = field(default_factory=dict)
    # Joint metrics
    joint_coverage: float = 0.0
    average_label_coverage: float = 0.0
    mean_uncertain_pathologies: float = 0.0
    # Deferral metrics
    deferral_rate: float = 0.0
    system_accuracy: float = 0.0
    model_accuracy_on_kept: float = 0.0
    n_total: int = 0
    n_deferred: int = 0


def compute_multilabel_coverage(
    per_pathology_sets: np.ndarray,
    labels: np.ndarray,
    pathology_names: list[str],
) -> dict:
    """Compute coverage metrics for multi-label prediction sets.

    Args:
        per_pathology_sets: [N, K, 2] binary prediction sets.
        labels: [N, K] multi-label matrix (0, 1, NaN).
        pathology_names: list of K pathology names.

    Returns:
        Dict with joint_coverage, average_label_coverage,
        per_pathology_coverage, per_pathology_set_size,
        mean_uncertain_pathologies.
    """
    N, K, _ = per_pathology_sets.shape
    set_sizes = per_pathology_sets.sum(axis=2)  # [N, K]

    # Per-pathology coverage: fraction of non-NaN samples where true label
    # is in prediction set
    per_path_cov = {}
    per_path_size = {}
    per_path_covered = np.full((N, K), np.nan)

    for k in range(K):
        valid = ~np.isnan(labels[:, k])
        if valid.sum() == 0:
            per_path_cov[pathology_names[k]] = float("nan")
            per_path_size[pathology_names[k]] = float("nan")
            continue

        true_labels_k = labels[valid, k].astype(int)
        sets_k = per_pathology_sets[valid, k, :]  # [M, 2]
        covered = sets_k[np.arange(valid.sum()), true_labels_k].astype(bool)
        per_path_covered[valid, k] = covered.astype(float)

        per_path_cov[pathology_names[k]] = float(covered.mean())
        per_path_size[pathology_names[k]] = float(set_sizes[valid, k].mean())

    # Joint coverage: fraction of samples where ALL valid pathologies are covered
    all_valid = np.all(~np.isnan(labels), axis=1)
    if all_valid.sum() > 0:
        joint_covered = np.all(
            np.nan_to_num(per_path_covered[all_valid], nan=1.0) == 1.0,
            axis=1,
        )
        joint_coverage = float(joint_covered.mean())
    else:
        joint_coverage = float("nan")

    # Average label coverage
    valid_coverages = [v for v in per_path_cov.values() if not np.isnan(v)]
    avg_label_cov = float(np.mean(valid_coverages)) if valid_coverages else float("nan")

    # Mean number of uncertain pathologies (set size > 1)
    uncertain = (set_sizes > 1).sum(axis=1)  # [N]
    mean_uncertain = float(uncertain.mean())

    return {
        "joint_coverage": joint_coverage,
        "average_label_coverage": avg_label_cov,
        "per_pathology_coverage": per_path_cov,
        "per_pathology_set_size": per_path_size,
        "mean_uncertain_pathologies": mean_uncertain,
        "uncertain_counts": uncertain,
        "set_sizes": set_sizes,
    }


def compute_multilabel_system_accuracy(
    per_pathology_sets: np.ndarray,
    logits_list: list[np.ndarray],
    labels: np.ndarray,
    defer_mask: np.ndarray,
    expert: SimulatedExpert,
) -> dict:
    """Compute system accuracy for multi-label deferral.

    For non-deferred samples: per-pathology argmax predictions are evaluated.
    For deferred samples: the expert's per-pathology predictions are used.
    System accuracy = fraction of (sample, pathology) pairs correct,
    averaged over non-NaN pathologies.

    Args:
        per_pathology_sets: [N, K, 2] binary prediction sets.
        logits_list: list of K arrays, each [N, 2].
        labels: [N, K] multi-label matrix (0, 1, NaN).
        defer_mask: [N] boolean, True = defer to expert.
        expert: SimulatedExpert instance.

    Returns:
        Dict with system_accuracy, deferral_rate, model_accuracy_on_kept.
    """
    N, K, _ = per_pathology_sets.shape
    n_deferred = int(defer_mask.sum())
    n_kept = N - n_deferred

    # Model predictions: argmax of logits per pathology
    model_preds = np.stack(
        [np.argmax(logits_list[k], axis=1) for k in range(K)], axis=1,
    )  # [N, K]

    # For non-deferred samples with singleton prediction sets,
    # use the singleton class instead of argmax
    for k in range(K):
        singleton_mask = (~defer_mask) & (per_pathology_sets[:, k, :].sum(axis=1) == 1)
        if singleton_mask.any():
            model_preds[singleton_mask, k] = np.argmax(
                per_pathology_sets[singleton_mask, k, :], axis=1,
            )

    # Expert predictions for deferred samples
    expert_preds = expert.predict(labels)  # [N, K]

    # Combine: model for kept, expert for deferred
    combined_preds = np.copy(model_preds).astype(float)
    combined_preds[defer_mask] = expert_preds[defer_mask]

    # Compute accuracy: fraction of valid (sample, pathology) pairs correct
    valid = ~np.isnan(labels)
    correct = (combined_preds == labels) & valid
    total_valid = valid.sum()
    system_accuracy = float(correct.sum()) / total_valid if total_valid > 0 else 0.0

    # Model accuracy on kept samples only
    if n_kept > 0:
        kept_valid = valid[~defer_mask]
        kept_correct = (model_preds[~defer_mask] == labels[~defer_mask]) & kept_valid
        model_acc = float(kept_correct.sum()) / kept_valid.sum() if kept_valid.sum() > 0 else 0.0
    else:
        model_acc = 0.0

    return {
        "system_accuracy": system_accuracy,
        "deferral_rate": n_deferred / N if N > 0 else 0.0,
        "model_accuracy_on_kept": model_acc,
        "n_deferred": n_deferred,
        "n_kept": n_kept,
    }


def evaluate_multilabel_cp(
    cal_logits_list: list[np.ndarray],
    cal_labels: np.ndarray,
    test_logits_list: list[np.ndarray],
    test_labels: np.ndarray,
    alphas: np.ndarray,
    pathology_names: list[str],
    expert: SimulatedExpert,
    aggregation: AggregationStrategy = "independent",
    defer_threshold: int = 1,
    penalty: float = 0.1,
    kreg: int = 1,
) -> list[MultilabelDeferralResult]:
    """Evaluate standard multi-label CP across alpha values.

    Args:
        cal_logits_list: list of K [N_cal, 2] arrays.
        cal_labels: [N_cal, K] multi-label matrix.
        test_logits_list: list of K [N_test, 2] arrays.
        test_labels: [N_test, K] multi-label matrix.
        alphas: array of significance levels.
        pathology_names: list of K pathology names.
        expert: SimulatedExpert instance.
        aggregation: aggregation strategy.
        defer_threshold: defer if number of uncertain pathologies >= this.
        penalty: RAPS penalty.
        kreg: RAPS kreg.

    Returns:
        List of MultilabelDeferralResult, one per alpha.
    """
    K = len(pathology_names)
    results = []

    for alpha in alphas:
        cp = MultilabelConformalPredictor(
            n_pathologies=K, penalty=penalty, kreg=kreg,
        )
        cp.calibrate(cal_logits_list, cal_labels, alpha=alpha, aggregation=aggregation)
        pred_sets, set_sizes = cp.predict(test_logits_list, aggregation=aggregation)

        cov = compute_multilabel_coverage(pred_sets, test_labels, pathology_names)

        # Defer if >= defer_threshold pathologies are uncertain
        uncertain_counts = cov["uncertain_counts"]
        defer_mask = uncertain_counts >= defer_threshold

        sys = compute_multilabel_system_accuracy(
            pred_sets, test_logits_list, test_labels, defer_mask, expert,
        )

        results.append(MultilabelDeferralResult(
            method=f"Std CP ({aggregation})",
            alpha_or_threshold=float(alpha),
            aggregation=aggregation,
            per_pathology_coverage=cov["per_pathology_coverage"],
            per_pathology_set_size=cov["per_pathology_set_size"],
            joint_coverage=cov["joint_coverage"],
            average_label_coverage=cov["average_label_coverage"],
            mean_uncertain_pathologies=cov["mean_uncertain_pathologies"],
            deferral_rate=sys["deferral_rate"],
            system_accuracy=sys["system_accuracy"],
            model_accuracy_on_kept=sys["model_accuracy_on_kept"],
            n_total=len(test_labels),
            n_deferred=sys["n_deferred"],
        ))

    return results


def evaluate_multilabel_wcp(
    cal_logits_list: list[np.ndarray],
    cal_labels: np.ndarray,
    cal_weights: np.ndarray,
    test_logits_list: list[np.ndarray],
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    alphas: np.ndarray,
    pathology_names: list[str],
    expert: SimulatedExpert,
    aggregation: AggregationStrategy = "independent",
    defer_threshold: int = 1,
    penalty: float = 0.1,
    kreg: int = 1,
) -> list[MultilabelDeferralResult]:
    """Evaluate weighted multi-label CP across alpha values.

    Args:
        cal_logits_list: list of K [N_cal, 2] arrays.
        cal_labels: [N_cal, K] multi-label matrix.
        cal_weights: [N_cal] importance weights.
        test_logits_list: list of K [N_test, 2] arrays.
        test_labels: [N_test, K] multi-label matrix.
        test_weights: [N_test] importance weights.
        alphas: array of significance levels.
        pathology_names: list of K pathology names.
        expert: SimulatedExpert instance.
        aggregation: aggregation strategy.
        defer_threshold: defer if uncertain pathologies >= this.
        penalty: RAPS penalty.
        kreg: RAPS kreg.

    Returns:
        List of MultilabelDeferralResult, one per alpha.
    """
    K = len(pathology_names)
    results = []

    for alpha in alphas:
        wcp = MultilabelWeightedConformalPredictor(
            n_pathologies=K, penalty=penalty, kreg=kreg,
        )
        wcp.calibrate(cal_logits_list, cal_labels, cal_weights, aggregation=aggregation)
        pred_sets, set_sizes = wcp.predict(
            test_logits_list, test_weights, alpha=alpha, aggregation=aggregation,
        )

        cov = compute_multilabel_coverage(pred_sets, test_labels, pathology_names)

        uncertain_counts = cov["uncertain_counts"]
        defer_mask = uncertain_counts >= defer_threshold

        sys = compute_multilabel_system_accuracy(
            pred_sets, test_logits_list, test_labels, defer_mask, expert,
        )

        results.append(MultilabelDeferralResult(
            method=f"WCP ({aggregation})",
            alpha_or_threshold=float(alpha),
            aggregation=aggregation,
            per_pathology_coverage=cov["per_pathology_coverage"],
            per_pathology_set_size=cov["per_pathology_set_size"],
            joint_coverage=cov["joint_coverage"],
            average_label_coverage=cov["average_label_coverage"],
            mean_uncertain_pathologies=cov["mean_uncertain_pathologies"],
            deferral_rate=sys["deferral_rate"],
            system_accuracy=sys["system_accuracy"],
            model_accuracy_on_kept=sys["model_accuracy_on_kept"],
            n_total=len(test_labels),
            n_deferred=sys["n_deferred"],
        ))

    return results


def evaluate_count_deferral(
    per_pathology_sets: np.ndarray,
    logits_list: list[np.ndarray],
    labels: np.ndarray,
    pathology_names: list[str],
    expert: SimulatedExpert,
    method_name: str = "Count deferral",
    alpha: float = 0.1,
) -> list[MultilabelDeferralResult]:
    """Sweep deferral threshold τ: defer if ≥τ pathologies are uncertain.

    Args:
        per_pathology_sets: [N, K, 2] binary prediction sets.
        logits_list: list of K [N, 2] arrays.
        labels: [N, K] multi-label matrix.
        pathology_names: list of K pathology names.
        expert: SimulatedExpert instance.
        method_name: method label for results.
        alpha: the alpha used to generate the prediction sets.

    Returns:
        List of MultilabelDeferralResult, one per threshold τ ∈ {1, ..., K+1}.
    """
    K = len(pathology_names)
    cov = compute_multilabel_coverage(per_pathology_sets, labels, pathology_names)
    uncertain_counts = cov["uncertain_counts"]

    results = []
    for tau in range(0, K + 2):
        defer_mask = uncertain_counts >= tau

        sys = compute_multilabel_system_accuracy(
            per_pathology_sets, logits_list, labels, defer_mask, expert,
        )

        results.append(MultilabelDeferralResult(
            method=method_name,
            alpha_or_threshold=float(tau),
            aggregation="count",
            per_pathology_coverage=cov["per_pathology_coverage"],
            per_pathology_set_size=cov["per_pathology_set_size"],
            joint_coverage=cov["joint_coverage"],
            average_label_coverage=cov["average_label_coverage"],
            mean_uncertain_pathologies=cov["mean_uncertain_pathologies"],
            deferral_rate=sys["deferral_rate"],
            system_accuracy=sys["system_accuracy"],
            model_accuracy_on_kept=sys["model_accuracy_on_kept"],
            n_total=len(labels),
            n_deferred=sys["n_deferred"],
        ))

    return results


def plot_multilabel_accuracy_rejection(
    results: dict[str, list[MultilabelDeferralResult]],
    title: str = "Multi-Label: System Accuracy vs Deferral Rate",
) -> plt.Figure:
    """Plot accuracy-rejection curves for multi-label methods.

    Args:
        results: mapping from method name to list of results.
        title: plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, res_list in results.items():
        deferral_rates = [r.deferral_rate for r in res_list]
        sys_accs = [r.system_accuracy for r in res_list]

        order = np.argsort(deferral_rates)
        dr = np.array(deferral_rates)[order]
        sa = np.array(sys_accs)[order]

        ax.plot(dr, sa, label=method_name, marker="o", markersize=3, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Deferral Rate")
    ax.set_ylabel("System Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_multilabel_coverage(
    results: dict[str, list[MultilabelDeferralResult]],
    coverage_type: str = "average_label",
    title: str | None = None,
) -> plt.Figure:
    """Plot coverage vs alpha for multi-label CP methods.

    Args:
        results: mapping from method name to list of results.
        coverage_type: "average_label" or "joint".
        title: plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, res_list in results.items():
        alphas = [r.alpha_or_threshold for r in res_list]
        if coverage_type == "joint":
            coverages = [r.joint_coverage for r in res_list]
        else:
            coverages = [r.average_label_coverage for r in res_list]

        ax.plot(alphas, coverages, label=method_name, linewidth=1.5, marker="o", markersize=3)

    all_alphas = sorted({r.alpha_or_threshold for rlist in results.values() for r in rlist})
    ax.plot(all_alphas, [1 - a for a in all_alphas], color="black",
            linestyle="--", alpha=0.5, linewidth=1.5, label=r"Ideal ($1-\alpha$)")

    cov_label = "Joint Coverage" if coverage_type == "joint" else "Average Label Coverage"
    ax.set_xlabel(r"$\alpha$ (Significance Level)")
    ax.set_ylabel(cov_label)
    ax.set_title(title or f"{cov_label} vs " + r"$\alpha$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_pathology_coverage_heatmap(
    results_list: list[MultilabelDeferralResult],
    pathology_names: list[str],
    title: str = "Per-Pathology Coverage vs Alpha",
) -> plt.Figure:
    """Heatmap of per-pathology coverage across alpha values.

    Args:
        results_list: list of results at different alpha values.
        pathology_names: list of K pathology names.
        title: plot title.

    Returns:
        Matplotlib Figure.
    """
    alphas = [r.alpha_or_threshold for r in results_list]
    data = np.zeros((len(pathology_names), len(alphas)))

    for j, r in enumerate(results_list):
        for i, p in enumerate(pathology_names):
            data[i, j] = r.per_pathology_coverage.get(p, float("nan"))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45)
    ax.set_yticks(range(len(pathology_names)))
    ax.set_yticklabels(pathology_names)
    ax.set_xlabel(r"$\alpha$")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(pathology_names)):
        for j in range(len(alphas)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Coverage")
    fig.tight_layout()
    return fig


def multilabel_summary_table(
    results: dict[str, list[MultilabelDeferralResult]],
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Generate summary table at a fixed alpha.

    Args:
        results: mapping from method name to list of results.
        alpha: target significance level.

    Returns:
        DataFrame with one row per method.
    """
    rows = []
    for method_name, res_list in results.items():
        closest = min(res_list, key=lambda r: abs(r.alpha_or_threshold - alpha))
        rows.append({
            "Method": method_name,
            "Avg Label Cov": f"{closest.average_label_coverage:.3f}",
            "Joint Cov": f"{closest.joint_coverage:.3f}",
            "Mean Uncertain": f"{closest.mean_uncertain_pathologies:.2f}",
            "Deferral": f"{closest.deferral_rate:.3f}",
            "System Acc": f"{closest.system_accuracy:.3f}",
            "Model Acc (kept)": f"{closest.model_accuracy_on_kept:.3f}",
        })

    return pd.DataFrame(rows)

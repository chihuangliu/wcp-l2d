"""Temperature scaling for logit calibration.

Finds a scalar temperature T > 0 that minimizes NLL on a
held-out set, then scales logits as z / T before softmax.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    T_bounds: tuple[float, float] = (0.01, 100.0),
) -> float:
    """Find optimal temperature T by minimizing NLL on a held-out set.

    Args:
        logits: [N, K] raw logits (decision_function output).
        labels: [N] integer class labels.
        T_bounds: (T_min, T_max) search range for bounded scalar minimization.

    Returns:
        T_opt: scalar temperature >= T_min.
    """

    def nll(T: float) -> float:
        probs = softmax(logits / T, axis=1)
        p_true = probs[np.arange(len(labels)), labels]
        return -np.log(np.clip(p_true, 1e-12, 1.0)).mean()

    result = minimize_scalar(nll, bounds=T_bounds, method="bounded")
    return float(result.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Scale logits by temperature.

    Args:
        logits: [N, K] raw logits.
        T: temperature scalar (> 0).

    Returns:
        scaled_logits: [N, K] = logits / T.
    """
    assert T > 0, f"Temperature must be positive, got {T}"
    return logits / T

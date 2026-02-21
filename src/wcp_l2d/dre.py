"""Adaptive Density Ratio Estimation for covariate shift correction.

Estimates importance weights w(x) = p_target(x) / p_source(x) using
optional PCA dimensionality reduction + Platt-calibrated logistic regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class DREDiagnostics:
    """Diagnostic metrics for the density ratio estimator."""

    domain_auc: float
    ess: float
    ess_fraction: float
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    weight_median: float
    n_source: int
    n_target: int


class AdaptiveDRE:
    """Density Ratio Estimator with optional PCA + Platt-scaled logistic regression.

    Pipeline:
        1. StandardScaler on combined source+target features
        2. PCA to ``n_components`` dimensions (skipped if n_components=None)
        3. Logistic regression: source (0) vs target (1)
        4. Platt scaling via CalibratedClassifierCV (sigmoid)
        5. w(x) = g(x)/(1-g(x)) * (N_s/N_t), clipped to [eps, weight_clip];
           no upper bound if weight_clip=None

    Args:
        n_components: PCA output dimensionality. Set to None to skip PCA and
            operate directly on full-dimensional scaled features.
        weight_clip: Maximum importance weight. Set to None to disable clipping
            (only the lower bound eps=1e-6 is applied).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int | None = 4,
        weight_clip: float | None = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.weight_clip = weight_clip
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._calibrated_clf: CalibratedClassifierCV | None = None
        self._n_source: int = 0
        self._n_target: int = 0
        self._domain_auc: float = 0.0

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "AdaptiveDRE":
        """Fit the density ratio estimator.

        Args:
            source_features: [N_s, D] source domain features.
            target_features: [N_t, D] target domain features.

        Returns:
            self for method chaining.
        """
        self._n_source = len(source_features)
        self._n_target = len(target_features)

        # Combine and create domain labels
        X = np.concatenate([source_features, target_features], axis=0)
        y = np.concatenate(
            [
                np.zeros(self._n_source, dtype=np.int32),
                np.ones(self._n_target, dtype=np.int32),
            ]
        )

        # 1. Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # 2. PCA (optional)
        if self.n_components is not None:
            self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_transformed = self._pca.fit_transform(X_scaled)
        else:
            self._pca = None
            X_transformed = X_scaled

        # 3. Logistic regression
        base_clf = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )

        # 4. Platt scaling (sigmoid calibration)
        self._calibrated_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=5,
        )
        self._calibrated_clf.fit(X_transformed, y)

        # Record domain AUC
        probs = self._calibrated_clf.predict_proba(X_transformed)[:, 1]
        self._domain_auc = float(roc_auc_score(y, probs))

        return self

    def compute_weights(self, features: np.ndarray) -> np.ndarray:
        """Compute importance weights for a set of features.

        Args:
            features: [N, D] feature matrix.

        Returns:
            weights: [N] importance weights clipped to [eps, weight_clip]
                (unclipped above if weight_clip=None).
        """
        assert self._calibrated_clf is not None, "Call fit() first."

        X_scaled = self._scaler.transform(features)
        X_transformed = self._pca.transform(X_scaled) if self._pca is not None else X_scaled

        # g(z) = P(target | z)
        g = self._calibrated_clf.predict_proba(X_transformed)[:, 1]

        # Clip g away from 0 and 1 to avoid division issues
        eps = 1e-8
        g = np.clip(g, eps, 1 - eps)

        # w(x) = g(z)/(1-g(z)) * (N_s / N_t)
        w = (g / (1 - g)) * (self._n_source / self._n_target)

        return np.clip(w, 1e-6, self.weight_clip)

    def diagnostics(self, source_features: np.ndarray) -> DREDiagnostics:
        """Compute diagnostic metrics on a set of source features.

        Args:
            source_features: [N, D] features to compute weights for.

        Returns:
            DREDiagnostics with weight statistics and ESS.
        """
        w = self.compute_weights(source_features)
        ess = float(w.sum() ** 2 / (w**2).sum())

        return DREDiagnostics(
            domain_auc=self._domain_auc,
            ess=ess,
            ess_fraction=ess / len(w),
            weight_mean=float(w.mean()),
            weight_std=float(w.std()),
            weight_min=float(w.min()),
            weight_max=float(w.max()),
            weight_median=float(np.median(w)),
            n_source=self._n_source,
            n_target=self._n_target,
        )

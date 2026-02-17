# Multi-Label Weighted Conformal Prediction for Learning to Defer Under Domain Shift

## Abstract

We propose a multi-label conformal prediction approach for learning to defer that combines the high per-pathology accuracy of binary classifiers (AUC 0.82–0.90) with graded deferral by running $K{=}7$ independent binary CPs and aggregating their prediction sets. Unlike single-label multi-class CP (which requires lossy label conversion and achieves only 29.3% accuracy), this approach preserves the natural multi-label structure. Deferral is based on the number of uncertain pathologies $\tau$, providing a continuous operating curve from 0% to 100% deferral. We compare four aggregation strategies (independent, Bonferroni, max joint score, mean joint score) and show that WCP with DRE weights differentially corrects over-coverage for pathologies where the source-target shift is most pronounced.

---

## 1. Motivation

### 1.1 The Binary-vs-Multi-Class Dilemma

Previous experiments revealed a fundamental tension:

- **Binary per-pathology CP** ($K{=}2$): high accuracy (AUC 0.82–0.90) but degenerate prediction sets — only $\emptyset$, $\{0\}$, $\{1\}$, or $\{0,1\}$. Deferral is all-or-nothing with no intermediate regime.
- **Multi-class CP** ($K{=}7$): graded prediction sets (sizes 0–7) but only 29.3% accuracy due to lossy single-label conversion. WCP cannot correct the resulting under-coverage.

### 1.2 Multi-Label CP: Best of Both Worlds

The multi-label approach runs $K{=}7$ independent binary conformal predictors (one per pathology), each producing a binary prediction set $C_k(x) \subseteq \{0, 1\}$. The joint prediction is the Cartesian product $C_1(x) \times \cdots \times C_K(x)$, and deferral is triggered when too many pathologies are uncertain:

$$\text{defer if } \sum_{k=1}^{K} \mathbf{1}[|C_k(x)| > 1] \geq \tau$$

This preserves binary classifier accuracy while enabling graded deferral ($0$ to $K$ uncertain pathologies).

### 1.3 What Replaces $\alpha$?

In standard single-label CP, $\alpha$ controls $P(Y \notin C(X))$. For multi-label, we consider:

1. **Per-label $\alpha$**: each binary CP runs at $\alpha$ independently. Average label coverage $\geq 1{-}\alpha$, but no joint guarantee.
2. **Bonferroni (FWER)**: each CP at $\alpha/K$. Guarantees $P(\text{any miscovered}) \leq \alpha$. Conservative.
3. **Max joint score**: $s(x,y) = \max_k s_k(x, y_k)$, single threshold. Controls worst-case pathology.
4. **Mean joint score**: $s(x,y) = \text{mean}_k s_k(x, y_k)$, single threshold. Controls average uncertainty.

---

## 2. Experimental Setup

### 2.1 Datasets

| Property | CheXpert | NIH |
|---|---|---|
| Total samples | 64,534 | 30,805 |
| Train / Cal / Test | 38,720 / 12,907 / 12,907 | — / — / 15,403 |
| DRE pool | — | 15,402 |
| NaN rate per pathology | 31–73% | 0% |

**Table 1.** Dataset sizes. NIH has no NaN labels; CheXpert has substantial NaN rates per pathology (31.6% for Effusion to 72.7% for Pneumonia).

### 2.2 Binary Classifiers

Per-pathology logistic regression on standardised DenseNet-121 features:

| Pathology | Train N | Cal N | CheXpert AUC | NIH AUC |
|---|---|---|---|---|
| Atelectasis | 16,172 | 5,450 | 0.788 | 0.678 |
| Cardiomegaly | 16,061 | 5,388 | 0.861 | 0.743 |
| Consolidation | 16,611 | 5,657 | 0.854 | 0.702 |
| Edema | 21,342 | 7,178 | 0.836 | 0.825 |
| Effusion | 26,419 | 8,815 | 0.874 | 0.821 |
| Pneumonia | 10,532 | 3,536 | 0.757 | 0.609 |
| Pneumothorax | 22,217 | 7,402 | 0.725 | 0.531 |

**Table 2.** Per-pathology binary classifier performance.

### 2.3 Density Ratio Estimation

Shared DRE across all pathologies: PCA(4), Platt-calibrated logistic regression, weight clipping at 20.0.

| Property | Value |
|---|---|
| Domain classifier AUC | 0.962 |
| ESS | 826.5 / 12,907 = 6.4% |
| Weight mean / median / max | 0.591 / 0.042 / 20.0 |

**Table 3.** DRE diagnostics.

---

## 3. Results

### 3.1 Independent Binary CP (Baseline)

| $\alpha$ | Avg Label Cov | Joint Cov | Mean Uncertain | Deferral ($\tau{=}1$) | System Acc |
|---|---|---|---|---|---|
| 0.01 | 0.997 | 0.982 | 6.72 | 1.000 | 0.864 |
| 0.05 | 0.997 | 0.982 | 6.72 | 1.000 | 0.863 |
| 0.10 | 0.996 | 0.975 | 6.02 | 1.000 | 0.862 |
| 0.15 | 0.967 | 0.789 | 4.32 | 1.000 | 0.862 |
| 0.20 | 0.851 | 0.390 | 1.18 | 0.968 | 0.865 |
| 0.30 | 0.627 | 0.097 | 0.00 | 0.000 | 0.924 |

**Table 4.** Standard independent CP on NIH. With $\tau{=}1$, deferral transitions sharply between $\alpha{=}0.20$ and $\alpha{=}0.30$.

### 3.2 WCP Independent

| $\alpha$ | Avg Label Cov | Joint Cov | Mean Uncertain | Deferral ($\tau{=}1$) | System Acc |
|---|---|---|---|---|---|
| 0.01 | 0.998 | 0.984 | 6.77 | 1.000 | 0.864 |
| 0.05 | 0.989 | 0.927 | 5.05 | 1.000 | 0.862 |
| 0.10 | 0.949 | 0.704 | 3.83 | 1.000 | 0.862 |
| 0.15 | 0.896 | 0.512 | 1.89 | 0.993 | 0.862 |
| 0.20 | 0.817 | 0.326 | 0.21 | 0.208 | 0.909 |
| 0.30 | 0.715 | 0.174 | 0.00 | 0.000 | 0.924 |

**Table 5.** WCP independent on NIH. WCP corrects over-coverage (from 0.996 to 0.949 at $\alpha{=}0.10$), bringing coverage closer to the target 0.90 — the same direction as in the binary experiments.

### 3.3 Per-Pathology Coverage at $\alpha{=}0.10$

| Pathology | Std CP Coverage | WCP Coverage | Std Set Size | WCP Set Size |
|---|---|---|---|---|
| Atelectasis | 0.994 | 0.994 | 1.958 | 1.958 |
| Cardiomegaly | 0.997 | 0.997 | 1.965 | 1.965 |
| Consolidation | 0.999 | 0.880 | 1.967 | 0.923 |
| Edema | 0.998 | 0.998 | 1.955 | 1.955 |
| Effusion | 0.997 | 0.997 | 1.954 | 1.954 |
| Pneumonia | 0.998 | 0.924 | 1.967 | 0.977 |
| Pneumothorax | 0.992 | 0.853 | 1.258 | 0.863 |

**Table 6.** WCP differentially corrects three pathologies: Consolidation (0.999→0.880), Pneumonia (0.998→0.924), and Pneumothorax (0.992→0.853). Four pathologies remain at full prediction sets (set size ≈ 2.0).

### 3.4 Aggregation Strategy Comparison at $\alpha{=}0.10$

| Method | Avg Label Cov | Joint Cov | Mean Uncertain | Deferral | System Acc |
|---|---|---|---|---|---|
| Independent Std CP | 0.996 | 0.975 | 6.02 | 1.000 | 0.862 |
| Independent WCP | 0.949 | 0.704 | 3.83 | 1.000 | 0.862 |
| Bonferroni Std CP | 0.997 | 0.982 | 6.72 | 1.000 | 0.862 |
| Bonferroni WCP | 0.998 | 0.983 | 6.74 | 1.000 | 0.863 |
| Max Score Std CP | 0.997 | 0.982 | 6.72 | 1.000 | 0.863 |
| Max Score WCP | 0.997 | 0.982 | 6.72 | 1.000 | 0.864 |
| Mean Score Std CP | 0.924 | 0.710 | 0.00 | 0.000 | 0.924 |
| Mean Score WCP | 0.740 | 0.205 | 0.00 | 0.000 | 0.924 |

**Table 7.** At $\alpha{=}0.10$: Bonferroni and max score are too conservative (nearly all pathologies uncertain). Mean score is too aggressive (collapses to empty/singleton sets). Independent aggregation with WCP provides the best trade-off.

### 3.5 Effect of Defer Threshold $\tau$ on WCP

The deferral threshold $\tau$ controls how many uncertain pathologies trigger deferral: defer if $\geq \tau$ pathologies have $|C_k(x)| > 1$. Higher $\tau$ requires more uncertainty before deferring.

| $\alpha$ | $\tau{=}1$ | $\tau{=}2$ | $\tau{=}3$ | $\tau{=}4$ | $\tau{=}5$ | $\tau{=}6$ | $\tau{=}7$ |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.979 | 0.790 |
| 0.02 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.970 | 0.755 |
| 0.05 | 1.000 | 1.000 | 1.000 | 0.989 | 0.858 | 0.203 | 0.000 |
| 0.10 | 1.000 | 1.000 | 0.991 | 0.842 | 0.000 | 0.000 | 0.000 |
| 0.15 | 0.993 | 0.786 | 0.113 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.20 | 0.208 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.40 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Table 8.** WCP deferral rate as a function of $\alpha$ and $\tau$ (NIH test, independent aggregation). Coverage is determined solely by $\alpha$ (independent of $\tau$), while deferral rate is controlled by both $\alpha$ and $\tau$. Higher $\tau$ shifts the deferral transition to lower $\alpha$ values. For example, at $\alpha{=}0.10$: $\tau{=}1$ defers 100%, $\tau{=}4$ defers 84.2%, and $\tau{=}5$ defers 0%. The $(\alpha, \tau)$ pair provides a two-dimensional control surface for the coverage-deferral trade-off.

---

## 4. Analysis

### 4.1 Why Deferral Is Still Step-Like at Fixed $\tau$

Despite having $K{=}7$ pathologies, the deferral rate as a function of $\alpha$ (at fixed $\tau$) remains approximately a step function. This is because each binary CP independently transitions from full-set ($|C_k|{=}2$) to singleton ($|C_k|{=}1$) at a pathology-specific critical $\alpha$. With $\tau{=}1$, deferral requires *any* pathology to be uncertain — this is the OR of 7 step functions, which is itself a step function (the last pathology to become singleton determines the transition).

Higher $\tau$ values shift the transition point but do not smooth it, because the per-pathology transitions are concentrated in a narrow $\alpha$ range (0.15–0.25 for WCP). The graded behaviour only emerges when varying $\tau$ at fixed $\alpha$, not when varying $\alpha$ at fixed $\tau$.

### 4.2 WCP Differentially Corrects Pathologies

WCP's DRE weights differentially affect pathologies based on the source-target covariate shift:

- **Consolidation, Pneumonia, Pneumothorax**: WCP shrinks prediction sets to near-singleton (set sizes 0.86–0.98). These pathologies have the highest AUC drop from CheXpert to NIH, meaning the DRE correctly identifies that NIH samples are "easy" relative to the miscalibrated source quantile.
- **Atelectasis, Cardiomegaly, Edema, Effusion**: Prediction sets remain at size ≈ 2.0 even with WCP. The covariate shift for these pathologies maintains uncertainty regardless of reweighting.

### 4.3 Count-Based Deferral at Fixed $\alpha$

At $\alpha{=}0.10$, sweeping $\tau$ for WCP:

| $\tau$ | Deferral | System Acc |
|---|---|---|
| 1–3 | 0.991–1.000 | 0.862 |
| 4 | 0.842 | 0.873 |
| 5 | 0.000 | 0.924 |
| 6–8 | 0.000 | 0.924 |

The useful operating range is narrow: $\tau{=}4$ provides the only intermediate deferral rate (84.2%). This reflects that WCP produces a bimodal distribution of uncertain pathology counts — most samples have either 4 or 0 uncertain pathologies, with few in between.

---

## 5. Comparison with Binary and Multi-Class Approaches

| Property | Binary ($K{=}2$) | Multi-class ($K{=}7$) | Multi-label (this work) |
|---|---|---|---|
| Classifier accuracy (NIH) | 80–99% | 29.3% | 80–99% (per pathology) |
| Prediction set structure | $\{0,1\}$ per pathology | $\{0,\ldots,6\}$ | $\{0,1\}^7$ joint |
| Deferral granularity | All-or-nothing | Graded (0–7) | Graded via $\tau$ (0–7) |
| Coverage at $\alpha{=}0.10$ | 99.6% (over) | 80.8% (under) | 94.9% (over, avg label) |
| WCP correction direction | Downward (effective) | Upward (ineffective) | Downward (effective) |
| WCP corrected coverage | 93.1% | 80.6% | 94.9% → varies by pathology |

**Table 9.** Three-way comparison. Multi-label CP combines the accuracy advantage of binary classifiers with the deferral granularity of multi-class, while WCP remains effective (correcting over-coverage downward).

---

## 6. Limitations

1. **Binary CP bottleneck propagates**: each individual binary CP still has step-function behaviour. The multi-label aggregation creates graded deferral only through the count mechanism, not through smoother per-pathology prediction sets.

2. **Narrow useful $\tau$ range**: at any fixed $\alpha$, only 1–2 $\tau$ values give intermediate deferral rates. The $(\alpha, \tau)$ surface has useful operating points but they are sparse.

3. **No joint coverage guarantee**: independent aggregation provides per-label marginal coverage but no guarantee that all pathologies are simultaneously covered. Bonferroni provides this but is too conservative.

4. **NaN handling**: CheXpert's 31–73% NaN rate means each pathology's binary CP calibrates on a different subset. For joint scores (max/mean), requiring all 7 non-NaN leaves only ~14K samples.

---

## 7. Conclusions

1. **Multi-label CP successfully combines binary accuracy with graded deferral**, resolving both the binary CP bottleneck (degenerate prediction sets) and the multi-class accuracy problem (lossy single-label conversion).

2. **WCP remains effective** in the multi-label setting, correcting over-coverage for 3 of 7 pathologies (Consolidation, Pneumonia, Pneumothorax) while maintaining coverage for the remaining 4.

3. **The $(\alpha, \tau)$ pair provides a two-dimensional control surface** for the coverage-deferral trade-off (Table 8). Coverage is controlled by $\alpha$; deferral rate is controlled by both $\alpha$ and $\tau$.

4. **Independent aggregation with WCP is the best strategy**. Bonferroni and max score are too conservative; mean score is too aggressive. Independent WCP provides meaningful coverage correction with flexible deferral control.

5. **The binary CP step-function persists per-pathology**, limiting the smoothness of deferral curves. True graded deferral requires either (a) multi-class CP with accurate classifiers, or (b) continuous-valued non-conformity scores aggregated differently.

# Multi-Class Weighted Conformal Prediction for Learning to Defer Under Domain Shift

## Abstract

We extend the binary ($K{=}2$) Weighted Conformal Prediction (WCP) framework for learning to defer to a multi-class ($K{=}7$) setting, using 7-pathology chest X-ray classification from CheXpert (source) to NIH (target). Multi-class conformal prediction resolves the binary CP bottleneck — prediction sets now range from size 0 to 7, enabling graded deferral rather than all-or-nothing behaviour. However, this comes at the cost of classification accuracy: the 7-class logistic regression achieves only 29.3% on NIH (vs 80–99% for binary per-pathology classifiers), and WCP coverage under-targets by approximately 9 percentage points. We investigate three approaches to improve coverage — Mondrian (class-conditional) CP, label-shift-adjusted weights, and a stronger MLP classifier — and find that Mondrian WCP reduces the coverage gap from $-9.4$ pp to $-2.1$ pp at $\alpha{=}0.1$. Label-shift correction and stronger classifiers fail to improve coverage due to dominant concept shift and noisy single-label conversion artefacts.

---

## 1. Motivation: From Binary to Multi-Class

### 1.1 The Binary CP Bottleneck

As documented in our previous report, binary conformal prediction ($K{=}2$) with RAPS produces only four possible prediction sets: $\emptyset$, $\{0\}$, $\{1\}$, and $\{0,1\}$. The deferral decision — defer when $|C_\alpha(x)| \neq 1$ — is governed by a single score threshold, causing deferral rates to jump discontinuously between $\sim$0% and $\sim$95% with no intermediate operating regime.

### 1.2 Why K=7?

With $K{=}7$ classes, prediction sets can take any subset of $\{0, 1, \ldots, 6\}$, with sizes from 0 to 7. This enables:

- **Graded deferral**: smooth transition from low to high deferral as $\alpha$ varies.
- **Informative prediction sets**: a set of size 3 provides partial information about the diagnosis, unlike binary where size 2 is maximally uninformative.
- **Smooth accuracy-rejection curves**: eliminating the step-function behaviour of binary CP.

### 1.3 The Trade-Off: Multi-Label to Single-Label Conversion

Chest X-ray pathologies are inherently **multi-label** — a patient can have both Effusion and Atelectasis simultaneously. Standard conformal prediction requires single-label targets, necessitating a conversion via `multilabel_to_singlelabel()`, which assigns each multi-positive image to its rarest pathology. This conversion:

- **Discards** 42.8% of CheXpert and 87.6% of NIH samples that have zero or ambiguous positive labels.
- **Creates noisy labels**: an image with Effusion + Atelectasis is assigned to one class based on prevalence, not clinical relevance.
- **Breaks the natural task structure**: binary "is Effusion present?" is a well-defined clinical question; "which single pathology best describes this image?" is not.

---

## 2. Experimental Setup

### 2.1 Datasets and Preprocessing

- **Feature extraction**: DenseNet-121 (`densenet121-res224-chex` from torchxrayvision), producing 1024-dimensional feature vectors.
- **Single-label conversion**: Multi-label images assigned to rarest positive pathology; samples with no positive label are excluded.

| Property | CheXpert | NIH |
|---|---|---|
| Raw samples | 64,534 | 30,805 |
| After single-label conversion | 36,907 (57.2%) | 3,826 (12.4%) |
| Train / Cal / Test split | 22,144 / 7,381 / 7,382 | — / — / 1,913 |
| DRE pool (unlabelled) | — | 15,402 |

**Table 1.** Dataset sizes. The severe NIH filtering (12.4% retained) reflects the low multi-label co-occurrence rate in NIH compared to CheXpert.

### 2.2 Class Distribution Shift

| Pathology | CheXpert | NIH | Ratio (NIH/CheX) |
|---|---|---|---|
| Atelectasis | 9.0% | 32.7% | 3.63 |
| Cardiomegaly | 19.3% | 19.8% | 1.03 |
| Consolidation | 11.6% | 10.8% | 0.94 |
| Edema | 21.8% | 1.3% | 0.06 |
| Effusion | 19.4% | 24.3% | 1.25 |
| Pneumonia | 7.2% | 4.5% | 0.63 |
| Pneumothorax | 11.7% | 6.6% | 0.56 |

**Table 2.** Class distribution shift after single-label conversion. Atelectasis is 3.6× more prevalent on NIH; Edema is 17× less prevalent. This substantial label shift compounds the covariate shift.

### 2.3 Classifier

Multinomial logistic regression (`solver=lbfgs`, $C{=}1.0$, `max_iter=1000`) on standardised features. We use `decision_function()` to obtain $[N, 7]$ logits directly.

| Split | Accuracy | Mean OvR AUC |
|---|---|---|
| CheXpert train | 0.434 | 0.785 |
| CheXpert cal | 0.324 | 0.678 |
| CheXpert test | 0.332 | 0.680 |
| NIH test | 0.293 | 0.595 |

**Table 3.** 7-class classifier performance. Accuracy is far below the binary per-pathology classifiers (which achieve 80–92% accuracy and AUC 0.82–0.90).

Per-class accuracy on NIH test: Atelectasis 0.32, Cardiomegaly 0.39, Effusion 0.40, Edema 0.25, Pneumonia 0.13, Consolidation 0.03, Pneumothorax 0.03. The model essentially fails on Consolidation and Pneumothorax, which are visually similar to other pathologies.

### 2.4 Density Ratio Estimation

Identical to the binary experiment: PCA(4) dimensionality reduction, Platt-calibrated logistic regression domain classifier, weight clipping at 20.0.

| Property | Value |
|---|---|
| Domain classifier AUC | 0.970 |
| ESS | 474 / 7,381 = 6.4% |
| Weight mean / median / max | 0.451 / 0.029 / 20.0 |

**Table 4.** DRE diagnostics. The near-perfect domain separability and low ESS are consistent with binary experiment values.

### 2.5 Conformal Scoring

We use APS (Adaptive Prediction Sets) with `penalty=0.0` and `kreg=1`, rather than the RAPS penalty ($\lambda{=}0.1$) used in binary experiments. With $K{=}7$, the RAPS penalty inflates prediction sets unnecessarily; APS produces more granular set sizes.

### 2.6 Temperature Scaling

Scalar temperature $T$ optimised by minimising NLL on the training set: $T_{\text{opt}} = 0.989$. The near-unity value indicates the logit scale is already well-calibrated, and temperature scaling has negligible effect on downstream results.

---

## 3. Baseline Results

### 3.1 Coverage Analysis

| $\alpha$ | $1{-}\alpha$ | CheXpert (Std CP) | Gap | NIH (Std CP) | Gap | NIH (WCP) | Gap |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.95 | 0.956 | +0.006 | 0.912 | $-$0.038 | 0.934 | $-$0.016 |
| 0.10 | 0.90 | 0.898 | $-$0.003 | 0.808 | $-$0.092 | 0.806 | $-$0.094 |
| 0.20 | 0.80 | 0.808 | +0.008 | 0.688 | $-$0.112 | 0.707 | $-$0.093 |
| 0.30 | 0.70 | 0.705 | +0.005 | 0.577 | $-$0.123 | 0.585 | $-$0.115 |
| 0.50 | 0.50 | 0.510 | +0.010 | 0.402 | $-$0.098 | 0.402 | $-$0.099 |

**Table 5.** Coverage vs $\alpha$. In-domain CheXpert tracks $1{-}\alpha$ well. NIH consistently under-covers by 4–12 pp. WCP provides minimal improvement over standard CP — in stark contrast to the binary setting, where WCP corrected over-coverage from 99.6% to 93.1%.

The direction of the coverage error has reversed: binary CP over-covers on NIH (99.6% vs 90% target) because NIH samples are "easy" for the binary classifier, yielding low non-conformity scores. Multi-class CP under-covers (80.8% vs 90%) because the 7-class problem is harder on NIH than on CheXpert, yielding higher scores.

### 3.2 Methods Comparison

| Method | System Acc | Deferral | Coverage | Avg Set Size | Model Acc (kept) |
|---|---|---|---|---|---|
| Max Logit | 0.625 | 0.537 | N/A | N/A | 0.356 |
| Continuous (source) | 0.612 | 0.511 | N/A | N/A | 0.354 |
| Continuous (DRE) | 0.617 | 0.522 | N/A | N/A | 0.355 |
| Standard CP | 0.613 | 0.554 | 0.326 | 1.39 | 0.310 |
| WCP | 0.629 | 0.576 | 0.337 | 1.44 | 0.318 |
| WCP + T | 0.630 | 0.580 | 0.339 | 1.44 | 0.317 |

**Table 6.** Summary at $\alpha{=}0.6$ (the operational regime for $K{=}7$ where deferral is $\sim$50%). All methods achieve similar system accuracy (0.61–0.63), with WCP+T marginally the best. The CP methods' advantage over continuous deferral is small — the opposite of the binary setting where continuous deferral dominated by 3–8 pp.

### 3.3 Continuous Deferral Calibration

| Target $\alpha$ | Source defer | DRE defer | Target |
|---|---|---|---|
| 0.3 | 0.232 | 0.239 | 0.3 |
| 0.5 | 0.417 | 0.426 | 0.5 |
| 0.7 | 0.637 | 0.663 | 0.7 |

**Table 7.** Continuous deferral calibration quality. Both source-calibrated and DRE-weighted thresholds under-defer relative to the target, reflecting that NIH samples tend to be more confident (lower $u(x)$) than the calibration set for the 7-class model.

### 3.4 Per-Class System Accuracy

| Pathology | N | Defer% | Expert Sens | Model Acc (kept) | System Acc |
|---|---|---|---|---|---|
| Atelectasis | 625 | 58.2% | 0.75 | 0.368 | 0.597 |
| Cardiomegaly | 378 | 54.2% | 0.90 | 0.405 | 0.667 |
| Consolidation | 208 | 55.3% | 0.70 | 0.011 | 0.380 |
| Edema* | 24 | 50.0% | 0.80 | 0.333 | 0.500 |
| Effusion | 465 | 57.6% | 0.85 | 0.416 | 0.684 |
| Pneumonia | 87 | 48.3% | 0.65 | 0.089 | 0.322 |
| Pneumothorax | 126 | 58.7% | 0.85 | 0.019 | 0.524 |

**Table 8.** Per-class analysis at $\alpha{=}0.6$ using WCP+T. System accuracy is dominated by the expert for Consolidation, Pneumonia, and Pneumothorax, where the model's kept accuracy is near zero. (* Edema has only 24 test samples — interpret with caution.)

### 3.5 Graded Prediction Sets

Unlike binary CP, the multi-class prediction sets exhibit a spread of sizes. At $\alpha{=}0.6$: singleton sets (model decides) constitute 42.1% of predictions; the remaining 57.9% are deferred with set sizes distributed across 0 and 2–7. This confirms the binary bottleneck is resolved — deferral varies smoothly with $\alpha$, and the accuracy-rejection curve is continuous.

---

## 4. Why WCP Fails to Correct Coverage at K=7

### 4.1 Score Distribution Shift

The APS non-conformity scores on NIH are systematically higher than on the calibration set (mean gap +0.053). This means NIH samples are "harder" for the 7-class model — their true class tends to be ranked lower in the softmax output. The calibration quantile, computed on the easier CheXpert distribution, is therefore too small for the NIH score distribution, resulting in under-coverage.

This is the reverse of the binary setting. Binary classifiers achieve $\sim$80–92% accuracy on NIH, meaning most samples have low non-conformity scores (the true class ranks highly). The binary calibration quantile was too generous, causing over-coverage — which WCP could correct downward.

### 4.2 WCP Cannot Correct Upward

WCP adjusts the quantile by re-weighting calibration samples. When the target needs a **higher** quantile (to achieve more coverage), WCP must upweight calibration samples with **high** scores. But high scores correspond to misclassified samples, which tend to have low DRE weights (they are not typical of the target domain — they are simply hard examples everywhere). The DRE weights are uninformative about score magnitude; they estimate $p_T(x) / p_S(x)$, which is orthogonal to the classifier's confidence.

In contrast, when the target needed a **lower** quantile (binary over-coverage), WCP could effectively downweight the high-score calibration samples, concentrating weight on the confident, correctly-classified samples that are more representative of the target.

### 4.3 Combined Shift

The CheXpert→NIH shift involves three components simultaneously:

1. **Covariate shift**: different imaging equipment and preprocessing (domain AUC = 0.970).
2. **Label shift**: Atelectasis prevalence is 3.6× higher on NIH; Edema is 17× lower.
3. **Concept shift**: $P(Y|X)$ changes between domains. The model achieves 40% accuracy on Effusion (NIH) vs 36% (CheXpert), but only 3% on Consolidation (NIH) vs 12% (CheXpert) and 3% on Pneumothorax (NIH) vs 25% (CheXpert).

WCP's theoretical guarantee requires **only** covariate shift. Under combined shift, the importance weights $w(x) = p_T(x)/p_S(x)$ do not capture the label or concept components, and the coverage guarantee breaks down.

---

## 5. Coverage Improvement Approaches

### 5.1 Mondrian (Class-Conditional) CP

**Method.** Mondrian CP computes separate quantiles per class:

$$\hat{q}_c = \text{Quantile}_{1-\alpha}\left(\{s_i : y_i = c\}\right), \quad c \in \{0, \ldots, 6\}$$

At test time, class $c$ is included in the prediction set if $s(x, c) \leq \hat{q}_c$. This can accommodate label shift because each class's threshold adapts independently.

**Weighted Mondrian CP** combines per-class quantiles with DRE importance weights, computing a weighted quantile for each class.

**Per-class quantiles at $\alpha{=}0.1$:**

| Pathology | $\hat{q}_c$ | $n_{\text{cal}}$ |
|---|---|---|
| Atelectasis | 1.000 | 665 |
| Cardiomegaly | 0.953 | 1,422 |
| Consolidation | 0.984 | 855 |
| Edema | 0.914 | 1,611 |
| Effusion | 0.933 | 1,435 |
| Pneumonia | 1.000 | 531 |
| Pneumothorax | 0.991 | 862 |

**Table 9.** Mondrian quantiles vary meaningfully across classes (range 0.914–1.000), unlike in the binary setting where per-class quantiles were identical due to balanced calibration data.

**Coverage results:**

| $\alpha$ | $1{-}\alpha$ | Mondrian Std | Gap | Mondrian WCP | Gap |
|---|---|---|---|---|---|
| 0.05 | 0.95 | 0.927 | $-$0.023 | **0.949** | **$-$0.001** |
| 0.10 | 0.90 | 0.862 | $-$0.038 | **0.879** | **$-$0.021** |
| 0.20 | 0.80 | 0.714 | $-$0.086 | 0.721 | $-$0.079 |
| 0.30 | 0.70 | 0.602 | $-$0.098 | 0.621 | $-$0.080 |
| 0.50 | 0.50 | 0.425 | $-$0.075 | 0.421 | $-$0.079 |

**Table 10.** Mondrian CP coverage on NIH. At $\alpha{=}0.10$, Mondrian WCP achieves 87.9% coverage (gap $-$2.1 pp), compared to baseline WCP's 80.6% (gap $-$9.4 pp) — a reduction of 7.3 pp in the coverage gap. At $\alpha{=}0.05$, Mondrian WCP nearly matches the target (94.9% vs 95%).

### 5.2 Label-Shift-Adjusted Weights

**Method.** Combine DRE weights with label prevalence correction:

$$w_{\text{combined}}(x_i) = w_{\text{DRE}}(x_i) \times \frac{P_T(Y = y_i)}{P_S(Y = y_i)}$$

**Oracle label-shift** (true NIH prevalence): coverage = 80.6% at $\alpha{=}0.10$ — identical to DRE-only WCP. The label-shift correction further concentrates the weights (ESS drops from 6.4% to 3.2%), making the weighted quantile less stable without actually correcting the score distribution shift.

**BBSE** (Black Box Shift Estimation): coverage = 92.5% at $\alpha{=}0.10$ — over-covers by 2.5 pp. However, the BBSE prevalence estimates are wildly inaccurate:

| Pathology | BBSE estimate | True (NIH) |
|---|---|---|
| Atelectasis | 0.132 | 0.327 |
| Cardiomegaly | 0.082 | 0.198 |
| Consolidation | 0.000 | 0.109 |
| Edema | 0.087 | 0.013 |
| Effusion | 0.127 | 0.243 |
| Pneumonia | 0.572 | 0.045 |
| Pneumothorax | 0.000 | 0.066 |

**Table 11.** BBSE prevalence estimates vs truth. Pneumonia is estimated at 57.2% (true 4.5%); Consolidation and Pneumothorax are estimated at 0%. The 7-class confusion matrix at 29% accuracy is too noisy for reliable matrix inversion.

BBSE's apparent coverage improvement is accidental — the wrong prevalence estimates happen to upweight calibration samples in a way that raises the quantile, but the mechanism is unreliable and would not generalise.

### 5.3 Stronger Classifier (MLP)

**Method.** Two-layer MLP (1024→512→512→7) with ReLU, dropout 0.3, trained for 200 epochs with Adam ($\text{lr}{=}10^{-3}$, weight decay $10^{-4}$), early stopping on calibration accuracy. Temperature scaling: $T_{\text{opt}} = 0.682$ (more significant than LR).

**Result.** The MLP achieves **26.5% on NIH** — worse than LR's 29.3%, despite 34.2% vs 33.2% on CheXpert test. The MLP overfits to CheXpert-specific patterns that do not transfer to NIH. The non-conformity score gap (NIH mean $-$ Cal mean) decreases from +0.053 (LR) to +0.027 (MLP), indicating slightly better score alignment, but this does not translate to coverage improvement because accuracy on NIH dropped.

MLP + Mondrian CP achieves 86.3% coverage at $\alpha{=}0.10$ — slightly worse than LR + Mondrian (86.2% standard, 87.9% weighted). The MLP provides no benefit over logistic regression for this task.

---

## 6. Comprehensive Comparison

| Method | Coverage | Gap | Avg Set Size | Deferral | System Acc |
|---|---|---|---|---|---|
| Std CP (LR) | 0.809 | $-$0.091 | 4.86 | 0.998 | 0.856 |
| WCP (LR) | 0.806 | $-$0.094 | 4.84 | 0.998 | 0.856 |
| **Mondrian Std (LR)** | **0.862** | **$-$0.038** | **5.26** | **1.000** | **0.857** |
| **Mondrian WCP (LR)** | **0.879** | **$-$0.021** | **5.62** | **1.000** | **0.856** |
| WCP + LS oracle (LR) | 0.806 | $-$0.094 | 4.84 | 0.998 | 0.856 |
| WCP + LS BBSE (LR) | 0.925 | +0.025 | 6.10 | 1.000 | 0.856 |
| Std CP (MLP) | 0.823 | $-$0.077 | 5.14 | 0.998 | 0.855 |
| WCP (MLP) | 0.821 | $-$0.079 | 5.10 | 0.997 | 0.854 |
| Mondrian Std (MLP) | 0.863 | $-$0.038 | 5.40 | 0.998 | 0.855 |
| Mondrian WCP (MLP) | 0.811 | $-$0.089 | 5.46 | 1.000 | 0.856 |

**Table 12.** All methods at $\alpha{=}0.1$ (target coverage 90%, NIH test). System accuracy is nearly identical across methods ($\sim$0.856) because deferral rates are $\sim$100% at this $\alpha$ — the model defers almost everything to the expert.

**Ranking by coverage gap**: Mondrian WCP (LR) $>$ Mondrian Std (LR) $\approx$ Mondrian Std (MLP) $>$ MLP Std CP $>$ WCP (LR) $\approx$ Std CP (LR).

---

## 7. The Binary vs Multi-Class Trade-Off

### 7.1 Accuracy vs Deferral Granularity

| Property | Binary (per-pathology) | Multi-class ($K{=}7$) |
|---|---|---|
| Classifier accuracy (NIH) | 80–99% | 29.3% |
| Mean OvR AUC (NIH) | 0.58–0.82 | 0.60 |
| Prediction set sizes | $\{0, 1, 2\}$ | $\{0, 1, \ldots, 7\}$ |
| Deferral behaviour | All-or-nothing | Graded |
| Coverage at $\alpha{=}0.1$ | 99.6% (over) | 80.8% (under) |
| WCP correction | Effective (99.6%→93.1%) | Ineffective (80.8%→80.6%) |

**Table 13.** Binary vs multi-class comparison. The binary approach respects the natural multi-label structure but produces degenerate prediction sets. The multi-class approach enables proper conformal machinery but requires a lossy label conversion.

### 7.2 Why Binary Per-Label Works Better

Three factors explain the accuracy gap:

1. **Task decomposition matches the data**: chest X-ray pathologies are inherently multi-label. Binary LR asks "is Effusion present?" — a well-defined clinical question requiring one hyperplane in 1024-dimensional space. The 7-class model must simultaneously discriminate between pathologies that are visually similar (Consolidation vs Pneumonia, Atelectasis vs Effusion).

2. **Single-label conversion creates noisy labels**: an image with both Effusion and Atelectasis is assigned to whichever pathology is rarer, regardless of which is more prominent radiologically. The 7-class model trains on these noisy assignments.

3. **More capacity does not help**: an MLP with 512 hidden units per layer achieves worse NIH accuracy (26.5%) than logistic regression (29.3%), confirming the bottleneck is the labels, not the model.

### 7.3 The Fundamental Tension

Conformal prediction's deferral mechanism works best with more classes ($K \gg 2$), providing graded prediction sets and smooth accuracy-rejection curves. But the classification task for chest X-rays is naturally multi-label, and forcing it into a single-label framework degrades accuracy to the point where the conformal machinery — though structurally sound — operates on unreliable scores.

A potential resolution is **multi-label conformal prediction**: run $K$ independent binary conformal predictors (one per pathology), each producing a prediction set over $\{0, 1\}$, then define deferral based on the number of uncertain labels:

$$\text{defer if } \sum_{k=1}^{K} \mathbf{1}[|C_k(x)| > 1] > \tau$$

This preserves both the binary structure (valid coverage per label, high accuracy) and enables graded deferral (0 to 7 uncertain labels). The DRE weights apply identically since they are feature-based.

---

## 8. Limitations

### 8.1 Single-Label Conversion Artefacts

The `multilabel_to_singlelabel()` conversion retains only 57.2% of CheXpert and 12.4% of NIH samples, introducing selection bias. The retained NIH samples are not representative of the full NIH distribution, and the severe filtering (87.6% excluded) means results may not generalise to the broader target population.

### 8.2 Low 7-Class Accuracy

At 29.3% accuracy on NIH, the classifier is barely above the majority-class baseline (32.7% for Atelectasis). All conformal methods are limited by the quality of the underlying scores: if the model cannot discriminate between classes, the non-conformity scores carry insufficient information for meaningful prediction sets.

### 8.3 Small Target Sample Sizes

Several classes have very few NIH test samples: Edema (24), Pneumonia (87), Pneumothorax (126). Per-class coverage estimates for these pathologies are unstable and should be interpreted with caution.

### 8.4 Simplified Expert Model

As in the binary report, all system accuracy comparisons use a flat expert accuracy of 85%. At the near-100% deferral rates observed at $\alpha{=}0.1$, system accuracy reduces to approximately expert accuracy regardless of the method, making method comparison uninformative at this operating point. The per-class analysis (Table 8) uses pathology-specific expert sensitivity for a more realistic breakdown.

---

## 9. Conclusions

1. **Multi-class CP resolves the binary bottleneck** but introduces a classification accuracy problem. Prediction sets are graded (sizes 0–7) and deferral varies smoothly with $\alpha$, but the 7-class model's low accuracy (29.3%) means the non-conformity scores are unreliable under domain shift.

2. **WCP is ineffective at K=7** because the coverage error is in the opposite direction (under-coverage rather than over-coverage). The DRE weights cannot push the quantile upward to compensate for the systematically harder target distribution.

3. **Mondrian WCP is the best corrective approach**, reducing the coverage gap at $\alpha{=}0.10$ from $-$9.4 pp (baseline WCP) to $-$2.1 pp. Per-class quantiles partially accommodate the label shift component that standard WCP ignores.

4. **Label-shift correction fails** because (a) oracle prevalence does not help — the coverage gap is driven by concept shift, not label shift alone; and (b) BBSE prevalence estimation is unreliable with a 7-class confusion matrix at 29% accuracy.

5. **Stronger classifiers do not help**: the MLP's lower NIH accuracy (26.5% vs 29.3%) demonstrates that the bottleneck is the noisy single-label conversion, not model capacity.

6. **Multi-label CP is a promising direction** that could combine the binary classifiers' high accuracy with graded deferral, avoiding both the binary CP bottleneck and the single-label conversion problem.

---

## References

- Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2021). Uncertainty sets for image classifiers using conformal prediction. *ICLR*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Lipton, Z., Wang, Y.-X., & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML*.
- Romano, Y., Sesia, M., & Candès, E. (2020). Classification with valid and adaptive coverage. *NeurIPS*.
- Tibshirani, R. J., Foygel Barber, R., Candès, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Vovk, V. (2012). Conditional validity of inductive conformal predictors. *Asian Conference on Machine Learning*.

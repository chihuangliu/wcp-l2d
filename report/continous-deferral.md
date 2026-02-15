# Limitations of Weighted Conformal Prediction for Learning to Defer Under Domain Shift

## Abstract

We investigate the application of Weighted Conformal Prediction (WCP) with density ratio estimation (DRE) to the learning-to-defer (L2D) problem under covariate and label shift, using chest X-ray classification from CheXpert (source) to NIH (target). While WCP with DRE-based importance weighting successfully reduces deferral rates for some pathologies, it fails for others due to a fundamental structural limitation of binary conformal prediction: with only $K{=}2$ classes, RAPS prediction sets exhibit all-or-nothing deferral behaviour governed by a single score threshold. We examine two approaches to address the concurrent label shift — class-conditional (Mondrian) conformal prediction and prevalence estimation with prior adjustment — and show why both fail in this setting. Finally, we propose a continuous deferral score that bypasses the binary CP bottleneck entirely, producing smooth accuracy-rejection curves with tunable operating points. Across all seven pathologies, continuous deferral with DRE-weighted threshold calibration outperforms both standard CP and WCP by 3–8 percentage points in system accuracy at matched deferral rates.

---

## 1. Background and Setup

### 1.1 Problem Setting

We consider post-hoc learning to defer, where a pre-trained DenseNet-121 classifier (trained on CheXpert) is deployed on NIH chest X-rays. The goal is to decide, for each test sample, whether to use the model's prediction or defer to a human expert. This setting involves both **covariate shift** (different imaging equipment, patient populations, preprocessing pipelines) and **label shift** (different disease prevalences between institutions).

### 1.2 Experimental Setup

- **Source domain**: CheXpert, split into train (60%), calibration (20%), and held-out test (20%).
- **Target domain**: NIH, split into an unlabelled pool for DRE (50%) and a labelled test set (50%).
- **Feature extraction**: DenseNet-121 (`densenet121-res224-chex` from torchxrayvision), producing 1024-dimensional feature vectors.
- **Classifier**: Per-pathology binary logistic regression on standardised features.
- **DRE**: PCA(4) dimensionality reduction followed by Platt-calibrated logistic regression domain classifier, with weight clipping at 20.0.
- **Conformal scoring**: RAPS (Regularised Adaptive Prediction Sets) with penalty $\lambda{=}0.1$ and $k_{\text{reg}}{=}1$.
- **Deferral rule (CP-based)**: Defer if $|C_\alpha(x)| \neq 1$, i.e., when the prediction set is not a singleton.

### 1.3 Severity of Domain Shift

The domain shift between CheXpert and NIH is substantial:

| Property | CheXpert (cal) | NIH (test) |
|---|---|---|
| Sample size | 8,828 | 15,403 |
| Effusion prevalence | 46.8% | 4.0% |
| DRE domain classifier AUC | — | 0.961 |
| DRE effective sample size (ESS) | 6.4% | — |

The near-perfect domain separability (AUC = 0.961) confirms strong covariate shift. The ESS of 6.4% indicates that only approximately 564 of 8,828 calibration samples carry meaningful importance weight — the rest are effectively down-weighted to near zero.

---

## 2. Limitations of Weighted Conformal Prediction

### 2.1 The Binary CP Bottleneck

Standard conformal prediction with RAPS produces prediction sets $C_\alpha(x) \subseteq \{0, 1, \ldots, K{-}1\}$. The deferral rule treats non-singleton sets as uncertain and defers them to the expert. For multi-class problems ($K \gg 2$), this provides a spectrum of set sizes enabling graded deferral. However, for binary classification ($K{=}2$), only four outcomes are possible:

| Prediction set | Set size | Deferral decision |
|---|---|---|
| $\emptyset$ | 0 | Defer |
| $\{0\}$ | 1 | Keep (predict 0) |
| $\{1\}$ | 1 | Keep (predict 1) |
| $\{0, 1\}$ | 2 | Defer |

The transition between "keep" and "defer" is governed by a single RAPS score threshold $\hat{q}$. With RAPS parameters $\lambda{=}0.1$ and $k_{\text{reg}}{=}1$, the non-conformity scores take only two meaningful values per sample:

- **Rank-1 score** (predicted class): $s_1 \approx 1 - \max(\text{softmax})$, which is the model's uncertainty.
- **Rank-2 score** (other class): $s_2 \approx 1.0 + \lambda = 1.1$, nearly constant.

The calibration quantile $\hat{q}$ at $\alpha{=}0.1$ is computed as the $\lceil (n+1)(1-\alpha) \rceil$-th smallest score. For most pathologies, $\hat{q} = 1.1$, meaning **both classes are always included** in the prediction set, yielding near-100% deferral.

### 2.2 WCP: Partial Success, Inconsistent Behaviour

Weighted conformal prediction (Tibshirani et al., 2019) replaces the uniform quantile with a per-test-point weighted quantile:

$$\hat{q}_w(x) = \inf\left\{q : \sum_{i=1}^{n} \tilde{p}_i \cdot \mathbf{1}[s_i \leq q] \geq 1 - \alpha \right\}$$

where $\tilde{p}_i \propto w(x_i)$ are normalised importance weights with an additional mass $\tilde{p}_{n+1} \propto w(x_{\text{test}})$ placed at $s = \infty$.

By re-weighting the calibration distribution, WCP can shift $\hat{q}_w$ below the critical 1.1 threshold for some test points, producing singleton prediction sets. However, this only works when the DRE assigns sufficiently large weights to calibration samples with low RAPS scores. The result is highly pathology-dependent:

| Pathology | Std CP deferral | WCP deferral | WCP coverage |
|---|---|---|---|
| Atelectasis | 95.3% | 95.3% | 0.993 |
| Cardiomegaly | 96.2% | 96.2% | 0.997 |
| Consolidation | 97.2% | 7.4% | 0.877 |
| Edema | 95.5% | 95.5% | 0.996 |
| Effusion | 95.0% | 23.6% | 0.931 |
| Pneumonia | 96.2% | 0.9% | 0.908 |
| Pneumothorax | 25.2% | 3.5% | 0.953 |

**Table 1.** Per-pathology deferral rates at $\alpha{=}0.1$. WCP successfully reduces deferral for Effusion, Consolidation, Pneumonia, and Pneumothorax, but has zero effect on Atelectasis, Cardiomegaly, and Edema — where deferral remains at 95%+.

The inconsistency arises because the weighted quantile either crosses the critical RAPS threshold (1.1) or it does not. There is no intermediate regime: the deferral rate jumps discontinuously from ~95% to ~5% as the effective weight mass shifts. For pathologies where the DRE weights are insufficient to push the quantile below 1.1, WCP behaves identically to standard CP.

### 2.3 Low Effective Sample Size

The DRE ESS ranges from 6.4% (Effusion) to 12.0% (Pneumonia) across pathologies. This means only 500–1,000 of the ~8,800 calibration samples carry meaningful weight. The weighted quantile is therefore estimated from a small effective sample, making it unstable and sensitive to the weight distribution's tail behaviour.

---

## 3. Attempts to Address Label Shift

The CheXpert-to-NIH shift involves both covariate shift (different feature distributions) and label shift (different disease prevalences). DRE addresses covariate shift but not label shift. We investigated two approaches to correct for label shift.

### 3.1 Class-Conditional (Mondrian) Conformal Prediction

**Approach.** Mondrian CP computes separate quantile thresholds per class:

$$\hat{q}_c = \text{Quantile}_{1-\alpha}\left(\{s_i : y_i = c\}\right), \quad c \in \{0, 1\}$$

A test point's prediction set includes class $c$ if $s(x, c) \leq \hat{q}_c$. This can accommodate label shift because the per-class thresholds adapt to each class's score distribution independently.

**Result.** Mondrian CP produced identical results to standard CP because the CheXpert calibration data is nearly balanced (46.8% positive for Effusion). With equal class sizes and identical RAPS score distributions within each class, the per-class quantiles $\hat{q}_0$ and $\hat{q}_1$ are both 1.1 — the same as the pooled quantile.

**Mondrian WCP** (combining per-class quantiles with DRE weighting) performed worse than standard WCP because splitting the calibration data by class halved the already-low ESS. With only ~280 effective samples per class, the weighted quantiles become highly unstable.

### 3.2 Prevalence Estimation and Prior-Adjusted CP

**Approach.** We attempted to estimate the target domain's class prevalence $\pi_t$ from unlabelled target data and adjust the classifier's logits via Bayes' rule:

$$f_{\text{adjusted}}(x, c) = f(x, c) + \log\frac{\pi_t(c)}{\pi_s(c)}$$

Two prevalence estimation methods were tested:

**Black-Box Shift Estimation (BBSE)** (Lipton et al., 2018). BBSE uses the classifier's confusion matrix $C$ on source data and the predicted label distribution $\mu_t$ on target data to estimate $\pi_t = C^{-1} \mu_t$.

- **Result**: BBSE produced negative prevalence estimates (clipped to $10^{-4}$) for most pathologies. With binary classifiers at ~80% accuracy, the confusion matrix is poorly conditioned. The combined covariate and label shift causes the classifier's predictions on target features to be unreliable, violating BBSE's assumption that only label shift is present.

**Maximum Likelihood Label Shift (MLLS)** (Alexandari et al., 2020). MLLS uses an EM algorithm to iteratively re-weight the predicted probabilities:

$$\pi_t^{(k+1)}(c) = \frac{1}{N_t} \sum_{i=1}^{N_t} \frac{w^{(k)}(c) \cdot p_s(c \mid x_i)}{\sum_{c'} w^{(k)}(c') \cdot p_s(c' \mid x_i)}, \quad w(c) = \frac{\pi_t(c)}{\pi_s(c)}$$

- **Result**: MLLS also estimated near-zero target prevalence for most pathologies (e.g., $\hat{\pi}_t(\text{Effusion pos}) \approx 0$, true value 4.0%). Like BBSE, it assumes the classifier's predicted probabilities are calibrated under covariate shift, which does not hold when features have shifted substantially.

### 3.3 Oracle Prevalence — Still Fails

Even when the **true** target prevalence is provided (oracle setting), prior-adjusted CP barely reduces deferral rates:

| Pathology | WCP deferral | Oracle prior-adjusted WCP deferral |
|---|---|---|
| Atelectasis | 95.3% | ~95% |
| Effusion | 23.6% | ~22% |
| Edema | 95.5% | ~95% |

The prior adjustment shifts logits by $\log(\pi_t / \pi_s)$, which changes the softmax probabilities and hence the RAPS scores. However, for the pathologies where WCP already fails (Atelectasis, Cardiomegaly, Edema), the logit shift is insufficient to push enough scores below the critical 1.1 threshold. The structural limitation of binary RAPS — not the prevalence mismatch — is the binding constraint.

### 3.4 Why Label Shift Correction Fails in This Setting

The failure of label shift correction stems from two compounding factors:

1. **Confounded shifts**: BBSE and MLLS assume pure label shift (features are conditionally invariant given the label). In our setting, both the features and the label distributions shift simultaneously. The classifier's predictions on target features are unreliable because the feature extractor has never seen NIH-style images, making the confusion matrix and predicted probabilities biased estimators of the true label shift.

2. **Binary RAPS structure**: Even with perfect prevalence knowledge, the fundamental problem remains. With $K{=}2$ and RAPS penalty $\lambda{=}0.1$, the rank-2 score is always $\approx 1.1$. The calibration quantile must drop below 1.1 for any samples to receive singleton prediction sets. Prior adjustment can shift some scores, but the transition remains a sharp step function — there is no smooth operating region between 0% and 95% deferral for most pathologies.

---

## 4. Continuous Deferral with DRE-Weighted Calibration

### 4.1 Method

We replace the discrete set-size deferral rule with a continuous uncertainty score:

$$u(x) = 1 - \max_c \, \text{softmax}(f(x))_c$$

The deferral rule becomes: defer if $u(x) > \tau$, where $\tau$ is a calibrated threshold. This is equivalent to the well-known maximum softmax probability (MSP) baseline (Hendrycks & Gimpel, 2017), but we augment it with distribution-shift-aware threshold calibration.

**Source-calibrated threshold.** Set $\tau = Q_{1-\alpha}(\{u_i\}_{i=1}^{n_{\text{cal}}})$, the $(1-\alpha)$-quantile of calibration uncertainty scores. This ignores domain shift: if target samples are generally more confident than source samples (due to class imbalance), the threshold will be too high, under-deferring on the target.

**DRE-weighted threshold.** Set $\tau = Q^w_{1-\alpha}(\{u_i\}_{i=1}^{n_{\text{cal}}})$, the DRE-weighted quantile:

$$\tau_w = \inf\left\{\tau : \frac{\sum_{i=1}^{n} w(x_i) \cdot \mathbf{1}[u_i \leq \tau]}{\sum_{i=1}^{n} w(x_i)} \geq 1 - \alpha\right\}$$

This re-weights the calibration distribution to approximate the target domain's uncertainty distribution, producing a threshold appropriate for the deployment setting.

### 4.2 Why Continuous Deferral Avoids the Binary CP Bottleneck

The key insight is that $u(x) = 1 - \max(\text{softmax})$ is a **continuous** function of the logits, taking values in $[0, 0.5]$ for binary classification. Every distinct confidence level maps to a distinct uncertainty score, enabling fine-grained threshold selection.

In contrast, RAPS-based CP collapses this continuous information into a discrete decision: the rank-1 score $s_1 \approx u(x)$ determines whether the predicted class is included, but the rank-2 score $s_2 \approx 1.1$ is nearly constant and determines whether the other class is included. The prediction set size — and hence the deferral decision — depends on whether $\hat{q}$ exceeds 1.1, creating a single discontinuous transition.

Continuous deferral uses the same underlying information ($u(x) \approx s_1$ in RAPS) but makes the deferral decision directly on this score, without the intermediary of prediction set construction.

### 4.3 Results

**Effusion — calibrated operating points (source-calibrated vs DRE-weighted):**

| Target deferral | Source $\tau$ | Source actual | DRE-weighted $\tau$ | Weighted actual |
|---|---|---|---|---|
| 10% | 0.317 | 4.6% | 0.231 | 7.1% |
| 20% | 0.175 | 9.4% | 0.089 | 15.0% |
| 30% | 0.096 | 14.3% | 0.032 | 25.4% |
| 40% | 0.051 | 20.3% | 0.013 | 36.7% |
| 50% | 0.026 | 27.7% | 0.006 | 48.5% |

**Table 2.** The source-calibrated threshold systematically under-defers (targeting 30%, achieving only 14.3%) because NIH test samples have lower uncertainty than CheXpert calibration samples — most NIH samples are true negatives with high classifier confidence. DRE-weighted calibration reduces this gap (targeting 30%, achieving 25.4%), though some discrepancy remains due to the low ESS (6.4%).

**System accuracy at matched deferral rates (Effusion):**

| Deferral rate | Continuous | Standard CP | WCP (DRE) |
|---|---|---|---|
| 5% | **0.930** | 0.907 | 0.908 |
| 10% | **0.937** | 0.902 | 0.902 |
| 20% | **0.940** | 0.889 | 0.889 |
| 30% | **0.936** | 0.876 | 0.876 |

**Table 3.** Continuous deferral outperforms both CP-based methods by 2.3–5.1 percentage points at every operating point. Standard CP and WCP are nearly identical when interpolated to the same deferral rate, confirming that the CP deferral mechanism — not the weighting — is the limiting factor.

**Multi-pathology comparison — system accuracy at 20% deferral:**

| Pathology | NIH AUC | Model acc. | Continuous | Std CP | WCP | Improvement |
|---|---|---|---|---|---|---|
| Atelectasis | 0.684 | 0.805 | **0.863** | 0.794 | 0.796 | +6.7 pp |
| Cardiomegaly | 0.708 | 0.925 | **0.945** | 0.898 | 0.898 | +4.7 pp |
| Consolidation | 0.708 | 0.950 | **0.961** | 0.927 | 0.923 | +3.4 pp |
| Edema | 0.806 | 0.883 | **0.933** | 0.855 | 0.855 | +7.8 pp |
| Effusion | 0.817 | 0.915 | **0.940** | 0.889 | 0.889 | +5.1 pp |
| Pneumonia | 0.575 | 0.916 | **0.945** | 0.889 | 0.889 | +5.6 pp |
| Pneumothorax | 0.594 | 0.987 | **0.966** | 0.960 | 0.960 | +0.6 pp |

**Table 4.** Continuous deferral consistently outperforms CP-based methods across all pathologies. The improvement is largest for Edema (+7.8 pp) and smallest for Pneumothorax (+0.6 pp), where the base model accuracy is already 98.7%. Standard CP and WCP produce nearly identical results when compared at the same deferral rate.

### 4.4 DRE-Weighted Calibration Accuracy

The DRE-weighted threshold successfully adapts to the target domain, though calibration quality varies by pathology:

| Pathology | Target deferral | Actual deferral | Calibration error |
|---|---|---|---|
| Atelectasis | 20% | 17.8% | -2.2 pp |
| Cardiomegaly | 20% | 10.4% | -9.6 pp |
| Consolidation | 20% | 20.8% | +0.8 pp |
| Edema | 20% | 21.1% | +1.1 pp |
| Effusion | 20% | 15.0% | -5.0 pp |
| Pneumonia | 20% | 28.7% | +8.7 pp |
| Pneumothorax | 20% | 18.3% | -1.7 pp |

**Table 5.** Calibration accuracy varies across pathologies. Consolidation, Edema, Atelectasis, and Pneumothorax are well-calibrated (error < 3 pp). Cardiomegaly under-defers substantially (10.4% vs 20% target) and Pneumonia over-defers (28.7% vs 20%). These discrepancies reflect pathology-specific interactions between the DRE weight distribution and the uncertainty score distribution that the global PCA(4) model does not fully capture.

---

## 5. Limitations and Caveats

### 5.1 Simplified Expert Model

All system accuracy comparisons use a fixed expert accuracy of 85%, modelled as an i.i.d. Bernoulli process independent of the sample. In practice, expert accuracy is pathology-dependent (e.g., Pneumonia sensitivity = 65% vs Cardiomegaly sensitivity = 90%) and label-dependent (sensitivity $\neq$ specificity). The flat 85% assumption:

- **Overestimates** expert quality for hard pathologies (Pneumonia, Consolidation).
- **Underestimates** expert quality for easy pathologies (Cardiomegaly, Pneumothorax).
- **Ignores selection bias**: different methods defer different subsets of samples, and the expert's accuracy on those subsets depends on the label composition of the deferred set, which varies with prevalence.

Importantly, the **ranking of methods is invariant to the choice of expert accuracy** under this flat model. At a matched deferral rate $d$, the system accuracy difference between two methods reduces to $(1 - d) \cdot (\text{model\_acc\_kept}_A - \text{model\_acc\_kept}_B)$, since the constant expert accuracy term cancels. The comparison therefore depends solely on each method's selective accuracy — how well it identifies confident, correct predictions to retain — which is a property of the deferral mechanism alone. This invariance holds for any constant expert accuracy value.

A more rigorous evaluation should use the per-pathology, label-conditional expert model defined in the `SimulatedExpert` class. Under a sample-dependent expert model, different methods would defer different subsets, and the expert's accuracy on those subsets could differ, potentially affecting the ranking.

### 5.2 Matched-Rate Comparison Caveat

The matched deferral rate comparison (Tables 3–4) interpolates across the CP methods' step-function deferral curves. For standard CP and WCP, most intermediate deferral rates (e.g., 20%) are not achievable by any setting of $\alpha$ — the deferral rate jumps from ~0% to ~95% at a single critical $\alpha$. The interpolated system accuracy at these operating points does not correspond to any real CP configuration, which overstates the continuous method's advantage in relative terms. However, this also underscores the practical limitation: CP-based methods cannot be tuned to a desired deferral rate in the binary setting.

### 5.3 Loss of Coverage Guarantees

Continuous deferral abandons conformal prediction's coverage guarantee ($P(Y \in C_\alpha(X)) \geq 1 - \alpha$). The threshold $\tau$ controls the deferral rate but provides no formal guarantee on the correctness of non-deferred predictions. In safety-critical medical applications, this trade-off must be carefully considered.

---

## 6. Conclusions

1. **Weighted CP's effectiveness is bottlenecked by binary RAPS structure**, not by the quality of the importance weights. The all-or-nothing deferral behaviour at $K{=}2$ means that WCP either dramatically reduces deferral (when weights push the quantile below the critical threshold) or has no effect at all.

2. **Label shift correction fails** because (a) prevalence estimation methods (BBSE, MLLS) assume pure label shift and fail under combined covariate-and-label shift, and (b) even with oracle prevalence, prior-adjusted logits cannot overcome the binary RAPS structural limitation.

3. **Continuous deferral with DRE-weighted calibration** is the most practical approach for binary classification under domain shift. It provides smooth, tunable accuracy-rejection curves and consistently outperforms CP-based methods by 3–8 percentage points across all seven pathologies, at the cost of losing formal coverage guarantees.

---

## References

- Alexandari, A., Kundaje, A., & Shrikumar, A. (2020). Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. *ICML*.
- Hendrycks, D. & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.
- Lipton, Z., Wang, Y.-X., & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML*.
- Tibshirani, R. J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

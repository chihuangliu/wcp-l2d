# Selective Conformal Risk Control for Multi-Label Learning to Defer Under Domain Shift

## Abstract

We propose a two-stage Selective Conformal Risk Control (SCRC) pipeline for multi-label learning to defer under covariate shift (CheXpert → NIH). The method resolves a critical limitation of prior RAPS-based multi-label WCP: uncontrollable deferral rates (100% at $\alpha{\leq}0.15$, 0% at $\alpha{\geq}0.30$). Stage 1 enforces a hard deferral budget $\beta$ by selecting the most uncertain samples via multi-label entropy. Stage 2 calibrates a probability threshold $\lambda^*$ on non-deferred samples using weighted Conformal Risk Control (CRC), controlling the False Negative Rate (FNR) at level $\alpha$. Density ratio estimation (D-WRCP) provides importance weights for covariate shift correction. On the NIH test set, SCRC achieves smooth, continuous deferral control from 5% to 30%, with empirical FNR consistently below the target $\alpha{=}0.10$ (observed: 0.043–0.046), and system accuracy scaling from 0.766 to 0.854 as the deferral budget increases.

---

## 1. Motivation

### 1.1 The Deferral Control Problem

The previous multi-label WCP approach (running $K{=}7$ independent binary CPs with RAPS scoring) produced a fundamental deferral control problem. At $\alpha{=}0.10$, all four aggregation strategies yielded either 100% deferral or 0% deferral on the NIH test set:

| Method | Deferral at $\alpha{=}0.10$ |
|---|---|
| Independent WCP | 100% |
| Bonferroni WCP | 100% |
| Max Score WCP | 100% |
| Mean Score WCP | 0% |

The root cause is the binary RAPS set-size bottleneck: with $K{=}2$ classes per pathology, prediction sets are limited to $\emptyset$, $\{0\}$, $\{1\}$, or $\{0,1\}$. The transition between singleton and full sets is a single sharp threshold, making deferral rates discontinuous in $\alpha$.

### 1.2 SCRC: A Two-Stage Solution

SCRC replaces the RAPS-based prediction sets with a fundamentally different approach:

1. **Stage 1 (Selection):** Compute a multi-label entropy score $H(x) = -\sum_{k=1}^K [p_k \log p_k + (1{-}p_k) \log(1{-}p_k)]$ for each sample. Defer the top-$\beta$ fraction with highest entropy. This *guarantees* deferral rate $= \beta$ by construction.

2. **Stage 2 (CRC):** On the remaining non-deferred samples, find a threshold $\lambda^*$ on per-pathology probabilities $p_k(x)$ such that the weighted empirical FNR $\leq \alpha$. The prediction set is $C(x) = \{k : p_k(x) \geq \lambda^*\}$.

Key insight: by working with probabilities directly (rather than RAPS non-conformity scores), the threshold $\lambda$ produces continuous, monotonic FNR curves — eliminating the step-function behaviour of binary RAPS.

---

## 2. Experimental Setup

### 2.1 Datasets and Splits

| Split | CheXpert | NIH |
|---|---|---|
| Train | 38,720 | — |
| Calibration | 12,907 | — |
| Test | 12,907 | 15,403 |
| DRE pool | — | 15,402 |

**Table 1.** Dataset sizes. Same splits as prior experiments. NIH has 0% NaN; CheXpert has 31–73% NaN per pathology.

### 2.2 Binary Classifiers

Per-pathology logistic regression on standardised DenseNet-121 features (frozen):

| Pathology | Train N | CheXpert AUC | NIH AUC |
|---|---|---|---|
| Atelectasis | 16,172 | 0.788 | 0.678 |
| Cardiomegaly | 16,061 | 0.861 | 0.743 |
| Consolidation | 16,611 | 0.854 | 0.702 |
| Edema | 21,342 | 0.836 | 0.825 |
| Effusion | 26,419 | 0.874 | 0.821 |
| Pneumonia | 10,532 | 0.757 | 0.609 |
| Pneumothorax | 22,217 | 0.725 | 0.531 |

**Table 2.** Per-pathology classifier AUC. All pathologies show AUC degradation from CheXpert to NIH, confirming covariate shift.

### 2.3 Density Ratio Estimation (D-WRCP)

Shared DRE across all pathologies: PCA(4) + Platt-calibrated logistic regression + weight clipping at 20.0.

| Property | Value |
|---|---|
| Domain classifier AUC | 0.962 |
| ESS | 826.5 / 12,907 = 6.4% |
| Weight mean / median / max | 0.591 / 0.042 / 20.0 |

**Table 3.** DRE diagnostics. The high domain AUC (0.962) confirms severe covariate shift. Weight clipping (D-WRCP) bounds the maximum weight at 20.0 to prevent variance explosion.

### 2.4 Probability Outputs

Unlike RAPS-based approaches which use binary logits $[N, 2]$ per pathology, SCRC works with the positive-class probability $p_k(x) \in [0, 1]$ directly:

| Pathology | CheXpert Cal Range | NIH Test Range |
|---|---|---|
| Atelectasis | [0.000, 1.000] | [0.000, 0.993] |
| Cardiomegaly | [0.000, 1.000] | [0.000, 1.000] |
| Consolidation | [0.000, 0.996] | [0.000, 0.999] |
| Edema | [0.001, 0.995] | [0.000, 0.950] |
| Effusion | [0.000, 1.000] | [0.000, 0.995] |
| Pneumonia | [0.000, 0.999] | [0.000, 0.999] |
| Pneumothorax | [0.000, 0.890] | [0.000, 0.757] |

**Table 4.** Probability ranges. Pneumothorax has the most compressed range on NIH (max 0.757), consistent with its largest AUC drop.

---

## 3. Results

### 3.1 Entropy Analysis

Multi-label entropy $H(x) = -\sum_k [p_k \log p_k + (1{-}p_k)\log(1{-}p_k)]$ serves as the uncertainty measure for Stage 1.

| Dataset | Mean $H(x)$ | Std $H(x)$ |
|---|---|---|
| CheXpert (cal) | 2.832 | 1.014 |
| NIH (test) | 1.741 | 1.006 |
| Maximum possible ($K{=}7$) | 4.852 | — |

**Table 5.** Entropy statistics. NIH samples have significantly *lower* entropy than CheXpert (mean 1.741 vs 2.832), reflecting that the NIH probability outputs are more concentrated (closer to 0 or 1). This is because NIH has no NaN labels and lower positive prevalence, leading to more confident negative predictions.

### 3.2 Stage 1: Budget-Constrained Deferral

Sweeping $\beta$ on the NIH test set ($N{=}15{,}403$):

| $\beta$ | N Deferred | N Kept | $\bar{H}$(deferred) | $\bar{H}$(kept) | $\bar{p}$(deferred) | $\bar{p}$(kept) |
|---|---|---|---|---|---|---|
| 0.05 | 770 | 14,633 | 3.797 | 1.632 | 0.416 | 0.111 |
| 0.10 | 1,540 | 13,863 | 3.584 | 1.536 | 0.387 | 0.097 |
| 0.15 | 2,310 | 13,093 | 3.418 | 1.445 | 0.358 | 0.085 |
| 0.20 | 3,080 | 12,323 | 3.268 | 1.359 | 0.330 | 0.075 |
| 0.30 | 4,620 | 10,783 | 3.011 | 1.196 | 0.282 | 0.059 |

**Table 6.** Stage 1 selection. Deferred samples have 2–3$\times$ higher entropy and 4–5$\times$ higher mean probability than kept samples, confirming that entropy effectively separates uncertain from confident predictions. Deferral rate exactly equals $\beta$ by construction.

### 3.3 Stage 2: CRC Calibration

At $\beta{=}0.15$, the calibration set retains 10,971 samples (85%). CRC sweeps $\lambda$ from 0 to 1 to find $\lambda^*$ where weighted FNR $\leq \alpha$:

| $\alpha$ | $\lambda^*$ | Weighted FNR | ESS |
|---|---|---|---|
| 0.05 | 0.0545 | 0.0499 | 792.1 (7.2%) |
| 0.10 | 0.1264 | 0.0996 | 792.1 (7.2%) |
| 0.15 | 0.2581 | 0.1482 | 792.1 (7.2%) |
| 0.20 | 0.4104 | 0.2000 | 792.1 (7.2%) |

**Table 7.** CRC calibration results. $\lambda^*$ increases monotonically with $\alpha$: stricter FNR targets require lower thresholds (predicting more labels as positive). The weighted FNR is tightly controlled at or just below $\alpha$ in all cases. ESS is 7.2%, slightly higher than the full calibration set (6.4%) because Stage 1 preferentially removes high-entropy samples that may have extreme DRE weights.

### 3.4 In-Domain FNR Verification

On CheXpert test (no covariate shift), unweighted SCRC at $\alpha{=}0.10$:

| $\beta$ | FNR | $\leq \alpha$? |
|---|---|---|
| 0.05 | 0.099 | YES |
| 0.10 | 0.099 | YES |
| 0.15 | 0.099 | YES |
| 0.20 | 0.099 | YES |
| 0.25 | 0.098 | YES |
| 0.30 | 0.099 | YES |

**Table 8.** In-domain verification. FNR is controlled at $\leq \alpha$ across all $\beta$ values, confirming correct CRC calibration when there is no distribution shift.

### 3.5 Full SCRC Evaluation on NIH (Out-of-Domain)

Weighted SCRC (with DRE) at $\alpha{=}0.10$:

| $\beta$ | Deferral | FNR (kept) | W-FNR (kept) | System Acc | Model Acc (kept) | $\lambda^*$ |
|---|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.045 | 0.045 | 0.766 | 0.761 | 0.1320 |
| 0.10 | 0.100 | 0.046 | 0.046 | 0.793 | 0.786 | 0.1320 |
| 0.15 | 0.150 | 0.046 | 0.044 | 0.812 | 0.802 | 0.1264 |
| 0.20 | 0.200 | 0.045 | 0.044 | 0.827 | 0.819 | 0.1230 |
| 0.25 | 0.250 | 0.043 | 0.042 | 0.838 | 0.831 | 0.1167 |
| 0.30 | 0.300 | 0.043 | 0.042 | 0.854 | 0.851 | 0.1161 |

**Table 9.** Weighted SCRC on NIH. The empirical FNR (0.043–0.046) is well below the target $\alpha{=}0.10$ across all deferral budgets. System accuracy scales smoothly from 0.766 at $\beta{=}0.05$ to 0.854 at $\beta{=}0.30$, confirming continuous deferral-accuracy control.

Unweighted SCRC (no DRE) at $\alpha{=}0.10$:

| $\beta$ | Deferral | FNR (kept) | System Acc | Model Acc (kept) | $\lambda^*$ |
|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.057 | 0.835 | 0.834 | 0.2065 |
| 0.10 | 0.100 | 0.057 | 0.852 | 0.851 | 0.1978 |
| 0.15 | 0.150 | 0.056 | 0.863 | 0.864 | 0.1876 |
| 0.20 | 0.200 | 0.054 | 0.873 | 0.877 | 0.1787 |
| 0.25 | 0.250 | 0.052 | 0.880 | 0.887 | 0.1685 |
| 0.30 | 0.300 | 0.051 | 0.887 | 0.899 | 0.1612 |

**Table 10.** Unweighted SCRC achieves higher system accuracy at each $\beta$ (e.g. 0.863 vs 0.812 at $\beta{=}0.15$) but with higher FNR (0.056 vs 0.046). Without DRE weights, the CRC calibration is not corrected for covariate shift, resulting in a higher $\lambda^*$ (0.1876 vs 0.1264) that predicts fewer labels positive — hence higher accuracy but more missed true positives.

### 3.6 Per-Pathology FNR/FPR at $\alpha{=}0.10$, $\beta{=}0.15$

| Pathology | FNR | FPR |
|---|---|---|
| Atelectasis | 0.252 | 0.522 |
| Cardiomegaly | 0.699 | 0.105 |
| Consolidation | 0.894 | 0.039 |
| Edema | 0.385 | 0.267 |
| Effusion | 0.355 | 0.273 |
| Pneumonia | 0.853 | 0.090 |
| Pneumothorax | 0.963 | 0.070 |

**Table 11.** Per-pathology FNR/FPR (weighted SCRC). There is extreme heterogeneity: Atelectasis has the lowest FNR (0.252) but highest FPR (0.522), while Pneumothorax has the highest FNR (0.963) but lowest FPR (0.070). This reflects the single-$\lambda$ design: a global threshold cannot simultaneously optimise for pathologies with very different probability distributions. Pathologies with low NIH AUC (Pneumothorax: 0.531, Pneumonia: 0.609) have near-total FNR because their probability outputs rarely exceed $\lambda^*{=}0.1264$.

---

## 4. Comparison: SCRC vs RAPS-Based WCP

### 4.1 Deferral Controllability

| Method | Achievable Deferral Rates (NIH) |
|---|---|
| RAPS WCP (independent, $\tau{=}1$) | 0%, 20.8%, 99.3%, 100% (only 4 discrete values) |
| SCRC (weighted, $\alpha{=}0.10$) | 5%, 10%, 15%, 20%, 25%, 30% (any $\beta$) |

**Table 12.** SCRC provides arbitrary deferral rates by setting $\beta$. RAPS WCP has only a few discrete operating points, with a gap from 0% to 20.8% and from 20.8% to 99.3%.

### 4.2 System Accuracy at Comparable Deferral

| Deferral Rate | SCRC Weighted | SCRC Unweighted | RAPS WCP |
|---|---|---|---|
| ~0% | — | — | 0.924 ($\alpha{\geq}0.30$) |
| ~5% | 0.766 | 0.835 | — |
| ~15% | 0.812 | 0.863 | — |
| ~20% | 0.827 | 0.873 | 0.909 ($\alpha{=}0.20$) |
| ~30% | 0.854 | 0.887 | — |
| ~100% | — | — | 0.862 ($\alpha{=}0.10$) |

**Table 13.** At comparable deferral rates, SCRC unweighted achieves similar accuracy to RAPS WCP (0.873 vs 0.909 at ~20% deferral). The key advantage is that SCRC can operate at *any* deferral rate, whereas RAPS WCP jumps from 0% to 20.8% to 99.3% deferral.

### 4.3 The Weighting Paradox

A striking result: **unweighted SCRC outperforms weighted SCRC on system accuracy** (e.g. 0.863 vs 0.812 at $\beta{=}0.15$), despite weighted SCRC having lower FNR (0.046 vs 0.056).

This is explained by the FNR-FPR trade-off:
- Weighted SCRC uses DRE weights that down-weight most calibration samples (ESS 6.4%), producing a lower $\lambda^*$ (0.1264 vs 0.1876).
- Lower $\lambda^*$ predicts more labels as positive → lower FNR but higher FPR.
- Higher FPR directly reduces system accuracy (more false positive predictions).

The DRE weights successfully correct the FNR guarantee under shift, but at the cost of over-predicting positive labels. This trade-off is inherent to single-threshold multi-label CRC: controlling FNR via a lower $\lambda$ necessarily increases FPR.

---

## 5. Analysis

### 5.1 Why SCRC Solves the Binary CP Bottleneck

The binary RAPS prediction set can only be $\{0\}$, $\{1\}$, $\{0,1\}$, or $\emptyset$. The transition between these states is a single discontinuous threshold in $\alpha$. SCRC bypasses this entirely by:

1. **Decoupling deferral from prediction sets**: Stage 1 uses entropy (a continuous score) rather than prediction set size (a discrete count).
2. **Using probabilities, not non-conformity scores**: The CRC threshold $\lambda$ operates on $p_k(x) \in [0,1]$, producing a continuous FNR curve rather than a step function.

### 5.2 The Per-Pathology FNR Problem

The single global $\lambda^*$ causes severe per-pathology FNR heterogeneity (Table 11). Pathologies with low AUC on NIH (Pneumothorax: 0.531) produce probabilities that are nearly always below $\lambda^*$, giving FNR $\approx 1$. Meanwhile, well-calibrated pathologies (Effusion: 0.821 AUC) achieve moderate FNR (0.355).

This suggests that per-pathology thresholds $\lambda_k^*$ could substantially improve performance, though at the cost of losing the joint FNR guarantee without additional correction (e.g. Bonferroni).

### 5.3 Entropy as an Uncertainty Measure

The entropy-based selection (Stage 1) effectively identifies samples where the model is least confident. Deferred samples have:
- 2–3$\times$ higher entropy (3.4 vs 1.4 at $\beta{=}0.15$)
- 4$\times$ higher mean probability (0.36 vs 0.085)

The higher mean probability of deferred samples indicates they are "harder" cases where the model predicts more pathologies as partially positive (ambiguous), exactly the cases where expert review is most valuable.

---

## 6. Limitations

1. **Single global $\lambda^*$**: A single threshold across $K{=}7$ pathologies cannot adapt to different probability scales per pathology. This leads to near-zero sensitivity for low-AUC pathologies and excessive false positives for high-AUC pathologies.

2. **FNR vs accuracy trade-off**: Weighted SCRC achieves better FNR control but worse system accuracy than unweighted SCRC. The DRE weights' low ESS (6.4%) pushes $\lambda^*$ down aggressively, increasing false positives.

3. **Entropy may not be optimal**: Multi-label entropy treats all pathologies equally. An importance-weighted entropy (weighting by clinical severity or classifier confidence) could better identify clinically meaningful uncertainty.

4. **No per-sample FNR guarantee**: CRC provides a marginal (average) FNR guarantee, not a per-sample one. Individual patients may still have high FNR, which is clinically concerning.

5. **Independent calibration and test selection**: Stage 1 applies selection independently to calibration and test sets. If the entropy distributions differ substantially (they do: CheXpert mean 2.83 vs NIH mean 1.74), the calibration set's non-deferred subset may not represent the test set's non-deferred subset well.

---

## 7. Conclusions

1. **SCRC solves the deferral control problem**: Unlike RAPS-based multi-label WCP (which jumps between 0% and 100% deferral), SCRC provides smooth, continuous deferral at any budget $\beta \in (0, 1)$.

2. **FNR is well-controlled**: At $\alpha{=}0.10$, the empirical FNR on NIH non-deferred samples is 0.043–0.046 (weighted) and 0.051–0.057 (unweighted), both well below the target. In-domain CheXpert verification confirms correct calibration (FNR $= 0.099 \leq 0.10$ at all $\beta$ values).

3. **System accuracy scales with deferral budget**: Weighted SCRC achieves 0.766 (at $\beta{=}0.05$) to 0.854 (at $\beta{=}0.30$), providing a clear accuracy-deferral Pareto front that was impossible with RAPS.

4. **DRE weighting improves FNR control but reduces accuracy**: The weighted variant has lower FNR (0.046 vs 0.056) but lower system accuracy (0.812 vs 0.863 at $\beta{=}0.15$) due to the FPR increase from a lower threshold.

5. **Per-pathology FNR heterogeneity is a key limitation**: A single $\lambda^*$ cannot serve pathologies with AUCs ranging from 0.531 to 0.825. Future work should explore per-pathology or adaptive thresholds within the CRC framework.

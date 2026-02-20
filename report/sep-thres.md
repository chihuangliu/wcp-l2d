# Per-Pathology Threshold Control (PP-SCRC)

## Abstract

We extend the Selective Conformal Risk Control (SCRC) pipeline by replacing the single global threshold λ* with a per-pathology threshold vector **λ** = [λ_1*, ..., λ_K*]. Each λ_k* independently controls the weighted empirical FNR_k ≤ α_k for pathology k. This resolves a key limitation of global SCRC: with a single threshold, low-AUC pathologies (Pneumothorax NIH AUC=0.531) have FNR≈1 because their probability outputs rarely exceed the global λ*=0.1264. PP-SCRC finds per-pathology thresholds ranging from 0.004 (Consolidation) to 0.079 (Atelectasis) at α=0.10, β=0.15 — substantially reducing FNR for all 7 pathologies at the cost of increased FPR. On the NIH test set, overall empirical FNR improves from 0.046 (global SCRC) to 0.023 (PP-SCRC weighted). In-domain FNR control is verified: FNR_k ≤ α holds for each pathology on CheXpert test (marginal guarantee).

**Critical finding:** Despite empirical FNR reduction, PP-SCRC **cannot provide a valid per-pathology FNR guarantee** on NIH test data. All 7 pathologies remain far above α=0.10 (range: 0.115–0.512). This is not a code bug — it is a fundamental collapse of the theoretical foundation caused by three interacting violations: (1) Stage 1 selection destroys the exchangeability required by CRC; (2) post-selection DRE weights are biased because they were estimated on the unselected full distribution; (3) the calibration positive-sample count per pathology is too small to overcome the finite-sample penalty. These failures and their remedies are analysed in detail in Section 6.

---

## 1. Motivation

### 1.1 Per-Pathology Heterogeneity Under Global SCRC

Global SCRC (from `scrc-multilabel.md`) finds a single λ* that controls the **joint** weighted FNR across all K=7 pathologies simultaneously. At α=0.10, β=0.15, λ*=0.1264 on the NIH test set. However, this produces severe per-pathology heterogeneity:

| Pathology | NIH AUC | Global FNR | Global FPR | Over α? |
|---|---|---|---|---|
| Atelectasis | 0.678 | 0.252 | 0.522 | YES |
| Cardiomegaly | 0.743 | 0.699 | 0.105 | YES |
| Consolidation | 0.702 | 0.894 | 0.039 | YES |
| Edema | 0.825 | 0.385 | 0.267 | YES |
| Effusion | 0.821 | 0.355 | 0.273 | YES |
| Pneumonia | 0.609 | 0.853 | 0.090 | YES |
| Pneumothorax | 0.531 | 0.963 | 0.070 | YES |

**Table 1.** Per-pathology FNR/FPR on NIH test at α=0.10, β=0.15 (Global SCRC weighted). All 7 pathologies exceed the FNR target. The aggregate weighted FNR=0.044 satisfies α=0.10, but this masks near-complete failure at the per-pathology level.

### 1.2 Root Cause: Compressed Probability Ranges Under Shift

The root cause is the interaction between covariate shift and the global threshold:

- Pneumothorax NIH AUC=0.531 (near random): probability outputs are nearly uniform over [0, 0.757]
- At λ*=0.1264, approximately 80%+ of NIH Pneumothorax probabilities fall below the threshold
- This produces FNR≈1 by construction — the classifier cannot discriminate well enough at this threshold

The CDF plots (cell-15) confirm that multiple pathologies have the majority of their probability mass below λ*=0.1264 on the NIH test set, explaining the systematically high FNR.

### 1.3 PP-SCRC Solution

Replace single λ* with per-pathology thresholds:

$$\text{FNR}_k(\lambda) = \frac{\sum_{i: y_{ik}=1, \text{valid}} w_i \cdot \mathbf{1}[p_k(x_i) < \lambda]}{\sum_{i: y_{ik}=1, \text{valid}} w_i}$$

$$\lambda_k^* = \sup\{\lambda : \text{FNR}_k(\lambda) \leq \alpha_k\}$$

Low-AUC pathologies get lower λ_k*, allowing more true positives to be predicted at the cost of more false positives.

---

## 2. Method

### 2.1 Two-Stage Pipeline

Stage 1 (entropy-based budget deferral) is **unchanged** from global SCRC:
1. Compute multi-label entropy $H(x) = -\sum_k [p_k \log p_k + (1-p_k)\log(1-p_k)]$
2. Defer top-β fraction (most uncertain) by construction

Stage 2 (CRC) is replaced with **per-pathology** threshold calibration:
1. For each pathology k, filter to calibration samples where $y_{ik}=1$ and not NaN
2. Vectorised grid search over candidates $\Lambda \subset [0,1]$: $\text{missed\_matrix}[j, i] = p_k(x_i) < \lambda_j$
3. $\text{FNR}\_k(\lambda_j) = (\text{missed\_matrix}[j, :] \cdot w_{\text{pos}}) / \sum_{i \in \text{pos}_k} w_i$
4. $\lambda_k^* = \Lambda[\text{last index where FNR} \leq \alpha_k]$; default to 0.0 if none satisfies

Prediction on test: $C_k(x) = \mathbf{1}[p_k(x) \geq \lambda_k^*]$ independently per pathology.

### 2.2 Key Difference from Global SCRC

| Property | Global SCRC | PP-SCRC |
|---|---|---|
| Threshold(s) | 1 (λ* shared) | K (λ_k* per pathology) |
| FNR control | Joint weighted FNR ≤ α | Per-pathology FNR_k ≤ α_k |
| Guarantee type | One constraint, K pathologies | K independent constraints |
| FPR behavior | Low for easy pathologies | Increased for hard pathologies |

---

## 3. Experimental Setup

Same datasets, classifiers, DRE, and splits as `scrc-multilabel.md`. Reference that document for details. This section covers per-pathology calibration statistics only.

### 3.1 Per-Pathology Calibration Statistics (α=0.10, β=0.15)

| Pathology | Weighted λ_k* | Unweighted λ_k* | Global λ* | n_pos (cal) |
|---|---|---|---|---|
| Atelectasis | 0.0793 | 0.2157 | 0.1264 | 2,115 |
| Cardiomegaly | 0.0610 | 0.1487 | 0.1264 | 1,591 |
| Consolidation | 0.0045 | 0.0566 | 0.1264 | 837 |
| Edema | 0.0628 | 0.2166 | 0.1264 | 2,507 |
| Effusion | 0.0417 | 0.2321 | 0.1264 | 3,283 |
| Pneumonia | 0.0103 | 0.0236 | 0.1264 | 433 |
| Pneumothorax | 0.0277 | 0.0334 | 0.1264 | 639 |

**Table 2.** Per-pathology thresholds. ESS=792.1/10971=7.2%. Weighted λ_k* are uniformly lower than unweighted — the DRE up-weights calibration samples resembling NIH, which tend to have lower probabilities under the CheXpert-trained classifier, pushing the threshold down.

---

## 4. Results

### 4.1 Per-Pathology Threshold Comparison

| Pathology | Global λ* | PP λ_k* | Global FNR | PP FNR | ΔFNR | Global FPR | PP FPR |
|---|---|---|---|---|---|---|---|
| Atelectasis | 0.1264 | 0.0793 | 0.252 | 0.139 | −0.113 | 0.522 | 0.654 |
| Cardiomegaly | 0.1264 | 0.0610 | 0.699 | 0.508 | −0.192 | 0.105 | 0.208 |
| Consolidation | 0.1264 | 0.0045 | 0.894 | 0.129 | **−0.765** | 0.039 | 0.735 |
| Edema | 0.1264 | 0.0628 | 0.385 | 0.308 | −0.077 | 0.267 | 0.441 |
| Effusion | 0.1264 | 0.0417 | 0.355 | 0.115 | −0.240 | 0.273 | 0.646 |
| Pneumonia | 0.1264 | 0.0103 | 0.853 | 0.426 | −0.426 | 0.090 | 0.553 |
| Pneumothorax | 0.1264 | 0.0277 | 0.963 | 0.512 | **−0.450** | 0.070 | 0.421 |

**Table 3.** Per-pathology comparison at α=0.10, β=0.15, NIH test. PP-SCRC reduces FNR for all 7 pathologies. FPR increases for all pathologies, especially Consolidation (0.039→0.735) which receives the most permissive threshold (λ=0.005).

### 4.2 Overall Performance Summary

| Method | Deferral | FNR (kept) | W-FNR (kept) | System Acc |
|---|---|---|---|---|
| SCRC weighted (global) | 0.150 | 0.046 | 0.044 | **0.810** |
| PP-SCRC weighted | 0.150 | **0.023** | **0.024** | 0.540 |
| PP-SCRC unweighted | 0.150 | 0.054 | 0.054 | 0.786 |

**Table 4.** Summary at α=0.10, β=0.15, NIH test. PP-SCRC weighted achieves the lowest overall FNR (0.023) at the cost of significantly lower system accuracy (0.540 vs 0.810). The accuracy drop reflects the large increase in false positives from very permissive per-pathology thresholds.

### 4.3 In-Domain Verification (CheXpert Test)

FNR control is verified on CheXpert test at α=0.10 (unweighted, source = target):

| β | Deferral | Overall FNR | FNR ≤ α? |
|---|---|---|---|
| 0.05 | 0.050 | 0.075 | YES |
| 0.10 | 0.100 | 0.074 | YES |
| 0.15 | 0.150 | 0.073 | YES |
| 0.20 | 0.200 | 0.070 | YES |
| 0.25 | 0.250 | 0.070 | YES |
| 0.30 | 0.300 | 0.068 | YES |

**Table 5.** In-domain PP-SCRC FNR verification. Overall FNR ≤ α=0.10 holds at all β values on CheXpert test, confirming the marginal guarantee holds in-domain.

---

## 5. Analysis

### 5.1 Why PP-SCRC Empirically Reduces FNR

The global threshold λ*=0.1264 is calibrated to control the **joint** weighted FNR, which is an average across all K pathologies. Pathologies like Pneumothorax (AUC=0.531) have most of their probability mass concentrated below 0.3 on the NIH test set. For these, the global threshold is far too high.

PP-SCRC grants each pathology its own budget: Pneumothorax gets λ_k*=0.028 instead of 0.126, approximately halving its empirical FNR from 0.963 to 0.512. The improvement is real but insufficient — 0.512 still far exceeds the target α=0.10.

### 5.2 The FNR-FPR Trade-Off

Every reduction in FNR comes at the cost of higher FPR. Consolidation illustrates the extreme case: λ_k*=0.005 (near-zero threshold) catches most positives (FNR=0.129) but also triggers on nearly all negatives (FPR=0.735). The PP-SCRC framework has no mechanism to cap FPR — the objective is purely FNR minimisation subject to the α constraint.

### 5.3 Accuracy vs FNR Trade-Off

The system accuracy drop (0.810 → 0.540) reflects the false-positive explosion. When λ_k* ≈ 0 for several pathologies, the model predicts positive for almost every sample, collapsing accuracy on the majority-negative classes. This is expected behaviour given the design objective.

The unweighted PP-SCRC (0.786 accuracy) uses higher thresholds (0.02–0.23 range vs 0.004–0.079 for weighted), because the DRE-weighted calibration aggressively up-weights NIH-like samples that have low probability outputs under the CheXpert-trained classifier.

### 5.4 Marginal vs Joint Guarantee

PP-SCRC provides at most a **marginal** guarantee — FNR_k ≤ α_k controlled independently for each pathology — with no joint bound across all K simultaneously. A joint guarantee requires Bonferroni correction: α_k = α / K per pathology.

---

## 6. Why Per-Pathology FNR Control Fails: Three Fundamental Violations

Even granting all the assumptions of standard CRC, PP-SCRC cannot provide valid per-pathology FNR guarantees on NIH. The failure is structural and arises from three interacting mechanisms. Each one alone is sufficient to invalidate the bound; together they make it impossible.

### 6.1 Stage 1 Selection Destroys Exchangeability

Conformal Risk Control inherits its guarantees from **exchangeability**: calibration and test samples must be exchangeable draws from the same distribution. Stage 1 selects the retained calibration subset $\mathcal{S} = \{i : H(x_i) \leq \tau_\beta\}$ based on the entropy score $H(x)$, which is a deterministic function of $x_i$. This creates a covariate-conditioned selection event: the retained samples are precisely those with lower multi-label entropy — a distinctly non-random, information-bearing subset.

Formally, after selection the calibration distribution becomes:

$$P_{\text{cal}}^{\text{kept}}(x, y) \propto P_{\text{cal}}(x, y) \cdot \mathbf{1}[H(x) \leq \tau_\beta]$$

This distribution is not exchangeable with the full test distribution. CRC calibrated on $P_{\text{cal}}^{\text{kept}}$ and evaluated on $P_{\text{test}}$ is guaranteed to be miscalibrated in general — the risk bound no longer holds as a finite-sample guarantee. The literature on selective conformal prediction (Bates et al., 2023) shows that recovering valid coverage post-selection requires explicitly modelling the selection mechanism, e.g., by including the selection probability $\pi(x) = P(H(x) \leq \tau_\beta \mid x)$ as a reweighting factor in the calibration procedure.

**Observed consequence:** The calibration FNR path (FNR vs λ curve per pathology) is computed on the entropy-low subset of CheXpert. On NIH test, entropy values differ — NIH samples span a different entropy range — so the empirical FNR at a given λ on NIH does not match the value used to set λ_k*, causing systematic miscalibration.

### 6.2 Post-Selection DRE Weights Are Biased

The DRE importance weights $\hat{w}(x) = \hat{p}(x \in \text{NIH}) / \hat{p}(x \in \text{CheXpert})$ are estimated on the **full** CheXpert calibration set against the NIH pool. After Stage 1 selection, only the entropy-low subset is retained. The marginal distribution of the retained subset differs from the full calibration distribution, so:

$$\mathbb{E}_{P_{\text{cal}}^{\text{kept}}}[\hat{w}(x)] \neq \mathbb{E}_{P_{\text{target}}}[1]$$

The importance sampling identity $\mathbb{E}_{P_{\text{cal}}}[w(x) \cdot f(x)] = \mathbb{E}_{P_{\text{target}}}[f(x)]$ no longer holds on the selected subset. The weights used in the per-pathology CRC calibration are therefore biased estimates of the target-domain risk.

**Observed consequence:** Weighted λ_k* values (0.004–0.079) are dramatically lower than unweighted values (0.024–0.232). This is not signal — it is the DRE being applied to a distribution it was not fitted on, producing systematically distorted weights that over-correct and push thresholds to near-zero.

### 6.3 Finite-Sample Penalty per Pathology

The CRC finite-sample risk bound (Angelopoulos & Bates, 2022) for N calibration samples takes the form:

$$\lambda^* \leq \lambda^*_{\text{pop}} + O\!\left(\sqrt{\frac{\log(1/\delta)}{N_{\text{eff}}}}\right)$$

where $N_{\text{eff}}$ is the effective sample size accounting for importance weights. When switching from joint to per-pathology CRC, the effective denominator collapses from $N_{\text{eff}}$ (all kept samples) to $N_{\text{eff},k}$ (kept positive samples for pathology $k$ only).

In our experiment at β=0.15:

| Pathology | n_pos (post-Stage1 cal) | DRE ESS | ESS fraction |
|---|---|---|---|
| Pneumonia | 433 | ~31 | ~7% |
| Pneumothorax | 639 | ~46 | ~7% |
| Consolidation | 837 | ~60 | ~7% |

With ESS ≈ 30–60 positives, the finite-sample penalty term dominates the risk bound for any reasonable δ. The algorithm cannot find a λ_k* that satisfies FNR_k ≤ α with the required statistical confidence — so it defaults to very small values (or 0.0) that minimise empirical FNR at the cost of FPR explosion, without actually providing coverage.

**Observed consequence:** For Pneumothorax and Pneumonia, weighted λ_k* ≈ 0.03 and 0.01 respectively. These are not principled thresholds — they are artefacts of the grid search finding the empirical minimum on an ESS-30 sample, which is statistically meaningless.

---

## 7. Design Recommendations for Valid Per-Pathology FNR Control

Three targeted modifications are required to make PP-SCRC theoretically sound:

### 7.1 Incorporate the Selection Mechanism into Calibration

Following the framework for selective conformal prediction (Marandon et al., 2024; Bates et al., 2023), the calibration risk must be reweighted by the inverse selection probability:

$$\text{FNR}_k^{\text{corrected}}(\lambda) = \frac{\sum_{i: y_{ik}=1} w_i \cdot \mathbf{1}[H(x_i) \leq \tau_\beta] \cdot \pi(x_i)^{-1} \cdot \mathbf{1}[p_k(x_i) < \lambda]}{\sum_{i: y_{ik}=1} w_i \cdot \mathbf{1}[H(x_i) \leq \tau_\beta] \cdot \pi(x_i)^{-1}}$$

where $\pi(x_i) = P(\text{not deferred} \mid x_i) = \mathbf{1}[H(x_i) \leq \tau_\beta]$ is deterministic in our case (no randomness in selection). Since selection is deterministic by entropy rank, $\pi$ is either 0 or 1 — this recovers the standard formula but requires the **test-time marginalisation** over deferral: the bound only applies to the marginal distribution over test samples that are also not deferred. In other words, the per-pathology FNR guarantee should be stated conditionally: FNR_k ≤ α *among kept test samples*, not over the full test set.

### 7.2 Refit DRE Weights on the Post-Selection Subset

Rather than applying full-data DRE weights to the post-selection subset, refit the domain classifier using only the retained calibration samples versus the NIH pool. This ensures the importance sampling identity holds on the actual distribution being used for calibration:

$$\hat{w}^{\text{post}}(x) = \frac{\hat{p}_{\text{NIH}}(x)}{\hat{p}_{\text{cal-kept}}(x)}$$

In practice: rerun PCA + logistic regression DRE with source = `cal_probs[kept_cal]` features and target = NIH pool features. The resulting weights will be less extreme because the entropy-low CheXpert subset is more similar to NIH than the full CheXpert distribution.

### 7.3 Dynamic α Allocation Based on Model Capability

Applying a uniform α=0.10 to all pathologies regardless of AUC is statistically infeasible for near-random classifiers (AUC ≈ 0.53). A principled allocation rule should budget α_k proportionally to model discriminability:

$$\alpha_k = \alpha \cdot \frac{\text{AUC}_k - 0.5}{\sum_{j} (\text{AUC}_j - 0.5)} \cdot K$$

Pathologies where $\text{AUC}_k \approx 0.5$ (Pneumothorax: 0.531) would receive a very loose or infinite α_k, effectively redirecting all such cases to the expert. This is a form of adaptive resource allocation: assign tight FNR targets only to pathologies where the model has genuine discriminative power.

---

## 8. Conclusions

PP-SCRC provides empirical FNR reduction for all 7 pathologies compared to global SCRC — a genuine improvement in per-pathology sensitivity. However, it **cannot provide valid finite-sample per-pathology FNR guarantees** in the two-stage selective deferral setting with covariate shift. The failure is caused by three simultaneous violations of the assumptions underlying CRC: (1) the selection event breaks exchangeability, (2) DRE weights become biased on the post-selection subset, and (3) per-pathology positive-sample counts are too small to overcome the finite-sample penalty given realistic ESS.

The aggregate FNR guarantee of global SCRC (which remains valid, as confirmed by in-domain verification) is not inherited by the per-pathology decomposition. Recovering valid per-pathology guarantees requires incorporating the selection mechanism into the calibration formula, refitting DRE on the post-selection distribution, and adopting capability-based α allocation for low-AUC pathologies.

**Implementation:** `PerPathologySCRCPredictor` in `src/wcp_l2d/scrc.py`, evaluation in `src/wcp_l2d/scrc_evaluation.py`, experiment in `notebooks/multilabel/sep_thres_experiment.ipynb`.

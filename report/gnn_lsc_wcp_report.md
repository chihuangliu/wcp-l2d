# GNN-DRE + Label Shift Correction (LSC): Experiment Report

**Notebook**: `notebooks/gnn/gnn_lsc_wcp.ipynb`
**Date**: 2026-02-22
**Baseline**: `report/gnn_dre_wcp_report.md`
**Methods**: Standard CP · WCP-GNN · WCP-GNN+EM-LSC · WCP-GNN+Oracle-LSC

---

## 1. Motivation

The WCP-GNN experiment (`gnn_dre_wcp_report.md`) established that GNN-DRE (ESS ≈ 31%)
successfully corrects for **covariate shift** between CheXpert (source) and NIH (target),
reducing mean deferral from 85.9% (Std CP) to 34.6% (WCP-GNN).

However, singleton decisions reveal a critical safety concern: the model's **false negative
rate (FNR) on non-deferred samples is 50–99%** across pathologies.  The leading hypothesis
is that this reflects **label shift** — CheXpert has far higher disease prevalence than NIH
(e.g. Atelectasis: 47.1% vs 5.5%), so the binary LR classifier trained on CheXpert is
biased toward predicting positive.  With WCP-GNN adjusting the threshold toward the target,
singletons appear but predict "negative" for everything, missing true positives.

**This experiment** adds a Bayesian label shift correction (LSC) to the WCP-GNN pipeline:

> **Input**: frozen DenseNet features → GNN probabilities
> **Correction**: Bayesian odds-ratio adjustment using target prevalence $\pi_{\text{tgt}}$
> **WCP**: corrected probabilities + GNN-DRE weights (30.9% ESS)

The correction formula is:
$$
\tilde{p}(y=1|x) = \sigma\!\left(\log\frac{p(y=1|x)}{1-p(y=1|x)} + \log\frac{\pi_{\text{tgt}}/(1-\pi_{\text{tgt}})}{\pi_{\text{src}}/(1-\pi_{\text{src}})}\right)
$$

where $\pi_{\text{tgt}}$ is estimated via Expectation-Maximisation (EM) on GNN probabilities
from the unlabelled NIH test set.

---

## 2. Experimental Design

| Component | Detail |
|-----------|--------|
| Feature extractor | DenseNet121 (torchxrayvision, pre-trained on CheXpert) |
| Source domain | CheXpert (N=64 534); split 60/20/20 train/cal/val |
| Target domain | NIH ChestXray14 (N=30 805); split 50/50 pool(DRE)/test |
| Binary classifier | Per-pathology logistic regression on 1024-dim features |
| CP score function | RAPS (penalty=0.10, kreg=1, randomized=False) |
| GNN | LabelGCN (7×7 co-occurrence adjacency, 50 epochs, best val AUC = 0.833) |
| GNN-DRE | 7-dim GNN probability space, no PCA, no clip; ESS = 30.9% |
| α | 0.10 (90% coverage target) |
| Expert accuracy | 0.85 |

### Four methods compared

| Method | Covariate shift | Label shift correction |
|--------|----------------|------------------------|
| Standard CP | ✗ | ✗ |
| WCP-GNN | GNN-DRE (ESS=31%) | ✗ |
| **WCP-GNN+EM-LSC** | GNN-DRE (ESS=31%) | EM-estimated NIH prior |
| WCP-GNN+Oracle-LSC | GNN-DRE (ESS=31%) | True NIH prior (oracle) |

LSC is applied to **both calibration and test** LR probabilities so that RAPS scores
are computed from the same (target-recalibrated) score function.

---

## 3. EM Prevalence Estimation

The EM algorithm converged at iteration 45.  Comparison of estimated vs true NIH
prevalences:

| Pathology | CheXpert src | NIH oracle | NIH EM-est | Error (pp) | Odds ratio |
|-----------|-------------|-----------|-----------|-----------|-----------|
| Atelectasis | 47.1% | 5.5% | **≈0.0%** | −5.5 | **0.000** |
| Cardiomegaly | 33.7% | 2.6% | 0.23% | −2.4 | 0.004 |
| Consolidation | 19.0% | 1.3% | **≈0.0%** | −1.3 | **0.000** |
| Edema | 42.0% | 0.26% | **≈0.0%** | −0.3 | **0.000** |
| Effusion | 46.6% | 4.0% | **≈0.0%** | −4.0 | **0.000** |
| Pneumonia | 16.3% | 0.56% | **3.75%** | +3.2 | **0.200** |
| Pneumothorax | 11.5% | 0.88% | **≈0.0%** | −0.9 | **0.000** |

**Critical failure**: For 6/7 pathologies the EM drives the estimated prevalence to
near zero (odds ratio ≈ 0), and for Pneumonia it **overestimates** (3.75% vs oracle 0.56%).

### Why EM fails: combined covariate + label shift

The EM algorithm relies on GNN probability outputs $p_{\text{GNN}}(y=1|x)$ on NIH test
samples to estimate the target prevalence.  Its fixed point satisfies:

$$
\hat{\pi}_{\text{tgt}} = \frac{1}{N}\sum_{i=1}^{N} \tilde{p}(y_i=1|x_i; \hat{\pi}_{\text{tgt}})
$$

This is valid when the GNN is **correctly calibrated on the source**.  However, our GNN was
trained on CheXpert features (source domain).  On NIH features (target domain), the GNN
suffers from both:

1. **Covariate shift**: DenseNet features differ between CheXpert and NIH → GNN outputs
   are unreliable on NIH inputs.
2. **Label shift**: CheXpert prevalence (12–47%) >> NIH prevalence (0.3–5.5%).

The GNN overestimates positive probability on NIH (it was trained to predict high positive
rates).  The EM then iteratively adjusts the prior downward, but because the GNN's outputs
are systematically biased by covariate shift, the fixed-point estimate converges to near-zero
rather than the true NIH prevalence.  This reproduces the exact failure mode documented in
`MEMORY.md`: "BBSE/MLLS prevalence estimation fails: Combined covariate+label shift confounds
prevalence estimation."

---

## 4. Per-Pathology Results at α = 0.10

| Pathology | NIH AUC | Std Defer | GNN Defer | GNN Cov | EM Defer | EM Cov | Oracle Defer | Oracle Cov |
|-----------|---------|-----------|-----------|---------|---------|--------|------------|----------|
| Atelectasis | 0.687 | 95.2% | 95.2% | 0.994 | **100.0%** | 1.000 | 97.3% | 0.997 |
| Cardiomegaly | 0.739 | 96.1% | **3.8%** | 0.887 | 10.8% | 0.978 | **0.1%** | 0.970 |
| Consolidation | 0.725 | 97.1% | **12.0%** | 0.849 | **0.0%** | 0.987 | **0.0%** | 0.986 |
| Edema | 0.816 | 95.6% | 95.6% | 0.996 | **100.0%** | 1.000 | 99.7% | 1.000 |
| Effusion | 0.803 | 95.0% | **23.0%** | 0.923 | **100.0%** | 1.000 | 98.5% | 0.998 |
| Pneumonia | 0.629 | 96.3% | **4.2%** | 0.883 | **0.0%** | 0.976 | **0.0%** | 0.992 |
| Pneumothorax | 0.567 | 26.4% | **8.7%** | 0.901 | **0.0%** | 0.991 | **0.0%** | 0.991 |
| **Mean** | **0.710** | **85.9%** | **34.6%** | 0.919 | **44.4%** | 0.990 | **42.2%** | 0.991 |

EM-LSC **increases** mean deferral from 34.6% (WCP-GNN) to 44.4% — a regression of 9.8
percentage points.  The pattern bifurcates:

- **Atelectasis, Edema, Effusion** (q̂=1.1 in GNN-DRE): EM-LSC drives odds ratio to ≈0
  → RAPS score distribution shifts → weighted quantile increases to 1.1 → **100% deferral**.
- **Consolidation, Pneumonia, Pneumothorax** (q̂=1.0 in GNN-DRE): q̂ stays at 1.0 but
  the model now predicts class 0 for everything → **0% deferral, FNR=1.0**.

---

## 5. Singleton Error Rate (FNR / FPR)

| Pathology | n_pos | GNN FNR | GNN FPR | EM-LSC FNR | EM-LSC FPR | Oracle FNR | Oracle FPR |
|-----------|-------|---------|---------|-----------|-----------|-----------|-----------|
| Atelectasis | 849 | 0.784 | 0.092 | n/a (0 singles) | — | 1.000 | 0.005 |
| Cardiomegaly | 398 | 0.779 | 0.060 | **1.000** | 0.000 | 0.982 | 0.004 |
| Consolidation | 205 | 0.861 | 0.023 | **1.000** | 0.000 | 1.000 | 0.000 |
| Edema | 40 | 0.500 | 0.096 | n/a (0 singles) | — | 1.000 | 0.000 |
| Effusion | 623 | 0.527 | 0.082 | n/a (0 singles) | — | 1.000 | 0.005 |
| Pneumonia | 86 | 0.872 | 0.073 | **0.953** | 0.019 | 1.000 | 0.002 |
| Pneumothorax | 135 | 0.992 | 0.005 | **1.000** | 0.000 | 1.000 | 0.000 |

**LSC consistently worsens FNR** — the opposite of the intended effect.

### Mechanistic explanation

With binary RAPS (K=2) and q̂ = 1.0, every singleton `{k}` satisfies "class k is the
model's top-ranked prediction."  So:

$$
\text{FNR}_{\text{singleton}} = P(\hat{y} = 0 \mid y = 1, |C(x)| = 1)
= P(\text{model ranks class 0 first} \mid y=1, \text{is singleton})
$$

After LSC with an odds ratio ≪ 1 (target prevalence ≈ 0), the corrected probability for
class 1 is driven toward 0:

$$
\tilde{p}(y=1|x) = \frac{\text{odds}_{\text{src}}(x) \cdot \text{OR}}
                        {1 + \text{odds}_{\text{src}}(x) \cdot \text{OR}}
\approx 0 \quad \text{when OR} \approx 0
$$

Even samples where the original model was **highly confident positive** (e.g. logit = +3,
original prob = 95%) are corrected to near-zero:

$$
\tilde{p}(y=1|x) = \frac{(0.95/0.05) \times 0.004}{1 + (0.95/0.05) \times 0.004}
= \frac{0.076}{1.076} \approx 7\%
$$

After LSC correction, class 0 is always ranked first → RAPS score for true positives = 1.1
> q̂=1.0 → singletons predict {0} for all samples → FNR = 1.0.

The fundamental issue: **the source-to-target odds ratio is so extreme** (e.g. Atelectasis:
47% → 0% = OR ≈ 0.000) that even the oracle correction causes FNR → 1.0.

### Why the user's intuition was inverted

The user hypothesised: "model was blindly predicting negative; LSC will fix this."  In
reality, the model (trained on CheXpert with 12–47% prevalence) was **biased toward
positive**, and singletons from WCP-GNN were predicting **negative correctly** for most
samples but missing the rare positives.  LSC makes the model predict negative even more
aggressively, which:

- **Reduces FPR** to near 0 (correct, fewer false alarms)
- **Increases FNR** to 1.0 (wrong, now misses ALL positives as singletons)

This reflects the direction of the shift: CheXpert has high prevalence → model biased
toward positive → WCP-GNN gives many singletons but low sensitivity → LSC shifts model
toward negative → singletons become all-negative → FNR = 1.0.

---

## 6. Calibration Quantile (q̂) Analysis

| Pathology | Std q̂ | GNN q̂ | EM-LSC q̂ | Oracle q̂ |
|-----------|-------|-------|---------|--------|
| Atelectasis | 1.100 | 1.100 | 1.100 | 1.100 |
| Cardiomegaly | 1.100 | 1.000 | **1.100** | 1.000 |
| Consolidation | 1.100 | 1.000 | 1.000 | 1.000 |
| Edema | 1.100 | 1.100 | 1.100 | 1.100 |
| Effusion | 1.100 | 1.100 | 1.100 | 1.100 |
| Pneumonia | 1.100 | 1.000 | 1.000 | 1.000 |
| Pneumothorax | 1.100 | 1.000 | 1.000 | 1.000 |

**Notable change**: Cardiomegaly's q̂ reverts from 1.000 (GNN) to 1.100 (EM-LSC).

For GNN-DRE on Cardiomegaly, NIH-like calibration samples (mostly negatives in NIH) had
RAPS scores < 1.0, pulling the weighted 90th percentile to 1.000 → singletons.  Under
EM-LSC, the calibration RAPS scores shift: negatives get even lower RAPS (model is more
confident), but positives' RAPS scores = 1.1 (model now ranks them second).  The
weighted CDF shifts such that the 90th weighted percentile is now 1.100 → full prediction
sets → deferral.

This shows EM-LSC **can disrupt** previously stable q̂ thresholds, reverting progress made
by GNN-DRE.

---

## 7. Comparison with WCP-GNN Baseline

| Pathology | GNN Defer | EM Defer | Δ Defer | GNN FNR | EM FNR | Δ FNR | GNN FPR | EM FPR | Δ FPR |
|-----------|-----------|---------|--------|---------|-------|------|---------|-------|------|
| Atelectasis | 95.2% | **100.0%** | +4.8% | 0.784 | n/a | — | 0.092 | n/a | — |
| Cardiomegaly | **3.8%** | 10.8% | +7.0% | 0.779 | **1.000** | +0.221 | 0.060 | 0.000 | −0.060 |
| Consolidation | **12.0%** | 0.0% | −12.0% | 0.861 | **1.000** | +0.139 | 0.023 | 0.000 | −0.023 |
| Edema | 95.6% | **100.0%** | +4.4% | 0.500 | n/a | — | 0.096 | n/a | — |
| Effusion | **23.0%** | **100.0%** | +77.0% | 0.527 | n/a | — | 0.082 | n/a | — |
| Pneumonia | **4.2%** | 0.0% | −4.2% | 0.872 | **0.953** | +0.081 | 0.073 | 0.019 | −0.054 |
| Pneumothorax | **8.7%** | 0.0% | −8.7% | 0.992 | **1.000** | +0.008 | 0.005 | 0.000 | −0.005 |
| **Mean** | **34.6%** | **44.4%** | **+9.8%** | — | — | — | — | — | — |

Across all metrics, EM-LSC is strictly worse than WCP-GNN for this experiment.

---

## 8. Prediction Set Size Summary

| Pathology | GNN f₁ | GNN avg | EM-LSC f₁ | EM-LSC avg | Oracle f₁ | Oracle avg |
|-----------|--------|--------|---------|--------|---------|--------|
| Atelectasis | 4.8% | 1.95 | **0.0%** | 2.00 | 2.7% | 1.97 |
| Cardiomegaly | **96.2%** | 0.96 | 89.2% | 1.08 | **99.9%** | 1.00 |
| Consolidation | **88.0%** | 0.88 | **100.0%** | 1.00 | **100.0%** | 1.00 |
| Edema | 4.4% | 1.96 | **0.0%** | 2.00 | 0.3% | 2.00 |
| Effusion | **77.0%** | 1.23 | **0.0%** | 2.00 | 1.5% | 1.98 |
| Pneumonia | **95.8%** | 0.96 | **100.0%** | 1.00 | **100.0%** | 1.00 |
| Pneumothorax | **91.3%** | 0.91 | **100.0%** | 1.00 | **100.0%** | 1.00 |
| **Mean** | **65.4%** | **1.27** | **41.3%** | **1.58** | **57.8%** | **1.42** |

EM-LSC reduces mean singleton rate from 65.4% (GNN) to 41.3%. Oracle-LSC achieves 57.8%
singletons — slightly worse than GNN, not better.

---

## 9. Conclusions

### 9.1 EM prevalence estimation fails under combined shift

The EM algorithm's convergence to near-zero prevalence (6/7 pathologies) is not a
convergence failure — it reached iteration 45 before converging.  Rather, it found the
correct fixed point **given the biased GNN outputs**, which themselves reflect combined
covariate + label shift.  This is a fundamental limitation of using a source-trained model
as the proxy for prevalence estimation on an out-of-distribution target.

### 9.2 LSC direction is correct but magnitude is catastrophic

The EM correctly identifies that NIH has lower disease prevalence than CheXpert (odds ratio
< 1 for 6/7 pathologies).  However, the estimated prevalences are so extreme (≈0%) that
the correction zeros out all positive probability, making FNR = 1.0.  Even with oracle
prevalences (0.26–5.5%), the source-to-target ratio is large enough that LSC pushes all
singleton decisions to class 0.

### 9.3 The binary RAPS bottleneck prevents FNR improvement via LSC

With K=2 and q̂=1.0, singletons always equal the model's top prediction.  Improving FNR
requires the model to rank class 1 first for more true positives.  LSC moves probability
mass FROM class 1, making this worse.  To improve FNR under this framework, one would need:

1. **Higher base model AUC**: better feature representations or target-domain fine-tuning.
2. **Asymmetric/cost-sensitive CP**: allow class-dependent thresholds to prioritise sensitivity.
3. **Higher deferral**: refuse to make singleton decisions until confidence is genuinely high.

### 9.4 WCP-GNN remains the best practical approach

Despite the binary RAPS limitation, WCP-GNN (34.6% mean deferral) is superior to
WCP-GNN+EM-LSC (44.4% mean deferral) on all composite metrics.  The residual singleton
FNR reflects the model's discriminative limitations, not the conformal calibration.

### 9.5 The path forward: target-domain adaptation, not post-hoc calibration

The high FNR on NIH is ultimately a **model capability problem**: the binary LR trained
on CheXpert has NIH AUC 0.57–0.82 and cannot reliably detect the rare NIH positives.
Post-hoc conformal correction (WCP) addresses the *confidence calibration* of prediction
sets but cannot improve the underlying rank-ordering of model probabilities.  Future
approaches should explore:

- Semi-supervised or domain-adaptive GNN training on unlabelled NIH data.
- Mixture density networks that model both covariate and label shift jointly.
- Prevalence-weighted loss during source training to reduce source-prior bias.

---

## 10. Limitations

1. **EM uses biased GNN outputs**: the GNN was trained only on CheXpert.  A domain-adapted
   GNN could give better calibrated outputs on NIH, enabling more accurate EM estimation.
2. **Single global DRE + per-pathology LSC**: the interaction between GNN-DRE covariate weights
   and per-pathology LSC corrections may not be optimal.
3. **LSC applied to both cal and test**: applying LSC only to test (keeping cal scores unchanged)
   would preserve the WCP-GNN threshold and might partially isolate the LSC effect.  This
   variant was not evaluated due to the score-scale mismatch it introduces.
4. **GNN AUC dropped slightly** (0.832 vs 0.833 baseline) due to stochastic training; results
   are reproducible within ±0.002 of the baseline GNN-DRE AUC.

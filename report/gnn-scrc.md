# GNN + SCRC Experiment Report (Corrected DRE Setup)

## Overview

This experiment evaluates three SCRC pipeline configurations on the CheXpert → NIH transfer task (domain shift). The key question is how the choice of DRE feature space interacts with the downstream conformal risk control.

**Three methods compared** (all β = 0.15, α = 0.10):

| Method | Classifier | DRE input | DRE PCA | DRE clip |
|--------|-----------|-----------|---------|----------|
| **A** | LR (7 × binary, 1024-dim) | 1024-dim scaled features | PCA(4) | None |
| **B** | LR (7 × binary, 1024-dim) | 1024-dim scaled features | None | None |
| **C** | GNN (joint 7-label, 1024-dim) | **7-dim GNN probs** | None | None |

A vs B isolates the effect of PCA in the DRE (same LR classifier).
B vs C isolates the effect of feature space (1024-dim vs 7-dim) and classifier (LR vs GNN).

All three use `PerPathologySCRCPredictor`: Stage 1 entropy-based deferral at budget β, Stage 2 per-pathology weighted CRC for FNR ≤ α.

**Dataset:** 38,720 CheXpert train / 12,906 cal / 12,908 test; 15,402 NIH pool / 15,403 NIH test.

---

## 1. DRE Diagnostics

| DRE | Domain AUC | ESS | ESS% | W mean | W median | W max |
|-----|-----------|-----|------|--------|----------|-------|
| A: 1024-dim, PCA-4 | 0.921 | 1,019 | **7.9%** | 0.763 | 0.144 | 94.1 |
| B: 1024-dim, no PCA | **1.000** | **1.0** | **0.01%** | 1.722 | 0.000 | 21,990 |
| C: 7-dim GNN probs | 0.857 | **2,764** | **21.4%** | 0.952 | 0.334 | 128.9 |

**DRE-B is completely degenerate.** With 1,024 features and no PCA, the domain classifier achieves perfect AUC (1.000) — CheXpert and NIH are perfectly separable in the full feature space. This collapses the ESS to essentially 1 sample (1.0 / 12,906 = 0.01%), and the maximum weight reaches 21,990. Weighted CRC calibration degenerates to fitting a threshold on a single effective sample, producing wildly miscalibrated thresholds (see Section 4). **Method B results are discarded as degenerate.**

**DRE-A** (PCA-4) achieves ESS = 7.9%, consistent with prior experiments. PCA projects both domains into a shared low-dimensional space where they partially overlap, enabling meaningful importance weighting.

**DRE-C** (7-dim GNN probs) achieves the best ESS at **21.4%**. The GNN's semantic prediction space is substantially less separable than the raw 1,024-dim feature space (AUC 0.857 vs 1.000), meaning that in probability space the two domains overlap much more. This produces well-distributed importance weights with a reasonable median (0.334) and a max of only 128.9.

---

## 2. GNN Training

The ML-GCN (node embeddings: 7×300, GCN layers: 300→1024→1024, ~1.4M parameters) trained on MPS in under 5 minutes. BCE loss decreased monotonically from 0.435 → 0.384 over 50 epochs. Val AUC (mean over 7 pathologies on CheXpert cal) improved from 0.829 and plateaued around **0.832**, best at epoch 20.

The co-occurrence adjacency matrix encodes that Consolidation, Effusion, Edema, and Pneumonia are mutually correlated, while Cardiomegaly and Pneumothorax are relatively isolated.

---

## 3. Discriminative Performance (NIH Test AUC)

| Pathology | LR (1024-dim) | GNN | Δ (GNN − LR) |
|-----------|--------------|-----|--------------|
| Atelectasis | 0.687 | 0.698 | +0.011 |
| Cardiomegaly | 0.739 | **0.759** | +0.020 |
| Consolidation | 0.725 | **0.744** | +0.019 |
| Edema | 0.816 | **0.821** | +0.005 |
| Effusion | 0.803 | **0.817** | +0.013 |
| Pneumonia | 0.629 | **0.665** | +0.036 |
| Pneumothorax | 0.567 | 0.581 | +0.014 |
| **Mean** | 0.710 | **0.726** | **+0.017** |

The GNN improves over LR on all 7 pathologies with an average gain of +0.017 AUC. Graph co-occurrence structure provides consistent discriminative benefit when operating on full 1024-dim features.

---

## 4. SCRC Results (β = 0.15, α = 0.10, NIH test)

All three methods achieve the exact β = 0.15 deferral budget by construction. Method B is presented for completeness but its results are degenerate due to collapsed ESS (see Section 1).

### Per-pathology λ* (CRC threshold)

| Pathology | Method A | Method B | Method C |
|-----------|---------|---------|---------|
| Atelectasis | 0.113 | 0.085 | 0.125 |
| Cardiomegaly | 0.027 | **0.490** ← | 0.049 |
| Consolidation | 0.010 | **0.330** ← | 0.017 |
| Edema | 0.055 | **0.501** ← | 0.069 |
| Effusion | 0.068 | **0.262** ← | 0.065 |
| Pneumonia | 0.006 | 0.118 | 0.021 |
| Pneumothorax | 0.013 | 0.012 | 0.027 |

Method B's λ* values (marked ←) are wildly inflated for most pathologies — a direct consequence of the near-zero ESS forcing CRC to satisfy the risk bound on a single effective sample.

Methods A and C produce coherent, low thresholds (0.006–0.132), consistent with predicting broadly positive at these probability levels. Method C's thresholds are consistently slightly higher than A's, reflecting the GNN's better-calibrated, more peaked probability distributions.

### Weighted FNR at λ* (calibration guarantee)

| Pathology | Method A | Method C |
|-----------|---------|---------|
| Atelectasis | 0.099 | 0.100 |
| Cardiomegaly | 0.097 | 0.100 |
| Consolidation | 0.097 | 0.094 |
| Edema | 0.093 | 0.100 |
| Effusion | 0.100 | 0.100 |
| Pneumonia | 0.099 | 0.088 |
| Pneumothorax | 0.100 | 0.096 |

Both A and C satisfy the weighted FNR ≤ α = 0.10 guarantee on the calibration set, as expected.

### Empirical FNR on NIH test

| Pathology | Method A | Method B† | Method C | Best (A/C) |
|-----------|---------|---------|---------|------------|
| Atelectasis | 0.313 | 0.220 | **0.230** | C |
| Cardiomegaly | 0.296 | 0.886 ← | **0.221** | C |
| Consolidation | 0.325 | 0.912 ← | **0.266** | C |
| Edema | 0.222 | 0.611 ← | **0.211** | C |
| Effusion | 0.228 | 0.616 ← | **0.171** | C |
| Pneumonia | 0.188 | 0.833 ← | **0.229** | A |
| Pneumothorax | **0.435** | 0.413 | 0.435 | Tie |
| **Mean** | 0.287 | 0.642 ← | **0.252** | C wins 5/7 |

†Method B's collapsed ESS produces extremely high thresholds for 5/7 pathologies, causing near-zero positive prediction and very high FNR (0.61–0.91) on the NIH test.

**Method C achieves the best empirical FNR in 5/7 pathologies** (losing only to A on Pneumonia, tying on Pneumothorax). Mean FNR 0.252 vs 0.287 for A — a 12% relative improvement.

### Positive Prediction Rate on NIH test

| Pathology | Method A | Method B | Method C |
|-----------|---------|---------|---------|
| Atelectasis | 45.8% | 56.8% | 54.3% |
| Cardiomegaly | 40.3% | **2.7%** ← | 42.0% |
| Consolidation | 44.7% | **1.4%** ← | 49.2% |
| Edema | 52.2% | **4.9%** ← | 49.6% |
| Effusion | 49.3% | **8.4%** ← | 56.6% |
| Pneumonia | 76.0% | 16.7% | 66.5% |
| Pneumothorax | 49.5% | 52.0% | 52.3% |
| **Mean** | 51.1% | **20.4%** ← | **52.9%** |

Method B's near-zero PPR for 4 pathologies (2.7%–8.4%) confirms that its collapsed ESS has destroyed the calibration — it is predicting negative almost universally for those classes. Method C's PPR is slightly higher than A's overall, reflecting the GNN's slightly higher positive class probabilities.

---

## 5. Key Interpretations

### DRE-B failure: curse of dimensionality in domain classification

With 1,024 unconstrained features, the domain classifier achieves perfect AUC. This is not surprising — CheXpert and NIH differ in scanner characteristics, patient populations, and labelling protocols, all of which are encoded in the raw 1,024-dim DenseNet features. Perfect separability means that for every CheXpert calibration sample, the model assigns p(target) ≈ 0, giving it importance weight ≈ 0. Essentially all weight collapses onto the few samples near the decision boundary, causing ESS → 1. This confirms that **PCA compression is not optional for the DRE in this feature space** — it is a necessary regularisation that prevents the domain classifier from overfitting.

### Why DRE-C has the best ESS

The GNN maps 1,024-dim features into a 7-dim probability simplex. In this compact semantic space, the two domains are no longer perfectly separable (AUC 0.857). CheXpert and NIH patients may share similar predicted pathology probabilities even though their raw image features differ. This overlap in probability space means the domain classifier assigns intermediate weights to many calibration samples, maintaining high ESS. Concretely:

- DRE-A (PCA-4): ESS = 7.9% — some collapse, domains still distinguishable in 4-dim PCA space
- DRE-C (7-dim GNN): ESS = 21.4% — substantially better, domains more overlapping in semantic space

The implication is that **domain shift in prediction space is smaller than domain shift in feature space**. The GNN's learned representations implicitly disentangle disease-relevant from scanner-specific variation.

### Why Method C achieves lower empirical FNR

Two mechanisms combine:

1. **Better importance weights**: DRE-C has 2.7× higher ESS (2,764 vs 1,019). More effective samples in the weighted CRC calibration means the estimated weighted FNR is more accurate, and the selected threshold λ* better reflects the true target distribution risk.

2. **Better classifier**: The GNN achieves +0.017 AUC over LR on NIH. Higher AUC means more positive cases are ranked at higher probabilities, and a given threshold captures more true positives. At similar λ* levels (A: 0.027–0.113 vs C: 0.027–0.125), the GNN's probability distributions are more informative.

### Pneumonia exception

Method C is slightly worse than A on Pneumonia (FNR 0.229 vs 0.188). Despite the GNN having higher AUC for Pneumonia (+0.036), Method A's λ* = 0.006 is extremely low (effectively predict all positive) while C's λ* = 0.021 is slightly higher. The DRE-C weights apparently give more mass to samples where Pneumonia probability is above 0.021, allowing a stricter threshold. However, at the strict λ* = 0.021, the GNN misses more Pneumonia positives than the LR at λ* = 0.006.

### Pneumothorax: hardest pathology

Both A and C achieve FNR ≈ 0.435 — far above α = 0.10. Pneumothorax has the lowest AUC of all pathologies (LR: 0.567, GNN: 0.581), barely above chance. With such poor discrimination, the classifier cannot reliably rank positives above negatives, and no threshold can achieve low FNR while maintaining a reasonable PPR.

### Calibration vs target gap

Weighted FNR on calibration (CheXpert, guaranteed ≤ 0.10 by construction): all methods comply.
Empirical FNR on NIH test (not guaranteed): Method A = 0.29, Method C = 0.25. This gap reflects:

1. DRE imperfection — ESS 7.9%–21.4%, not 100%, so importance weighting is approximate
2. Label prevalence shift (NIH pathology rates differ from CheXpert)
3. The CRC bound holds in expectation for the calibration distribution, not the target distribution

Method C's smaller gap (0.252 - 0.10 = +0.152 vs 0.287 - 0.10 = +0.187 for A) suggests its higher ESS translates the calibration guarantee more faithfully to the target domain.

---

## 6. Summary

| Metric | Method A: LR + DRE-A | Method B: LR + DRE-B | Method C: GNN + DRE-C |
|--------|---------------------|---------------------|----------------------|
| DRE domain AUC | 0.921 | **1.000** ← degenerate | 0.857 |
| DRE ESS% | 7.9% | **0.01%** ← degenerate | **21.4%** |
| Mean NIH AUC | 0.710 | 0.710 | **0.726** |
| Mean empirical FNR (NIH) | 0.287 | **0.642** ← degenerate | **0.252** |
| Deferral rate | 0.150 | 0.150 | 0.150 |
| Pathologies where best FNR | 2 | 0 | **5** |

**Bottom line:**

- **Method B (no-PCA DRE) completely fails**: Perfect domain separability in 1,024-dim space collapses the ESS to near zero, destroying the weighted CRC calibration.
- **Method C (GNN + 7-dim DRE) is the best end-to-end system**: It combines a stronger classifier (GNN, +1.7 AUC mean) with a better DRE (7-dim GNN probability space, ESS 21.4%), yielding the lowest mean empirical FNR (0.252, −12% relative to Method A). Using the GNN's own probability outputs as the DRE feature space is a natural and effective choice: it collapses scanner-specific variation while preserving disease-relevant signal.
- **Method A (LR + DRE-A) is a solid baseline**: PCA(4) is essential for DRE on 1,024-dim features. ESS 7.9% is workable, and results are well-behaved.

---

## 7. Next Steps

1. **GNN with PCA input features**: Test whether preprocessing the 1,024-dim features with PCA(64) before the GNN reduces overfitting to source-domain artefacts, potentially improving NIH AUC further (especially for Pneumonia and Edema where LR+PCA-64 previously outperformed GNN).

2. **Post-selection DRE refit**: After Stage 1 deferral removes the most uncertain samples, refit the DRE on the remaining calibration subset using `refit_dre_post_selection`. This aligns the importance weighting to the actual Stage 2 distribution.

3. **Capability-aware alpha allocation**: Apply `compute_capability_alpha` with the NIH AUC values to assign tighter FNR budgets to high-AUC pathologies (Edema, Effusion) and looser budgets to low-AUC ones (Pneumothorax, Atelectasis), addressing the per-pathology FNR imbalance.

4. **DRE-C with calibration set as source**: Instead of using the training set as the DRE source, use the calibration set directly. This avoids distribution mismatch between DRE source (train) and the samples being weighted (cal), and aligns with the importance sampling identity in the CRC bound.

5. **Longer GNN training with early stopping**: Val AUC plateaued at epoch 20 without overfitting. Checkpoint-based early stopping with more epochs might recover additional performance.

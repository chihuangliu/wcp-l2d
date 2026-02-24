# SCRC-T (Transductive) Fix — GNN-Only Experiment

**Date:** 2026-02-24
**Notebook:** `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_t_gnn.ipynb`
**Executed:** `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_t_gnn_executed.ipynb`
**Figure:** `report/scrc_t_gnn_sigma3.0.png`
**Depends on:** `report/synthetic_covariate_shift_scrc_report.md` (reference experiment)

> **Experiment 2 (Warm-up Batch)** added 2026-02-24. The notebook was updated to use
> an unlabeled warm-up batch (N=500) instead of the full Test set for threshold estimation.
> Results summarised in [Section 10](#10-experiment-2-unlabeled-warm-up-batch-scrc-t).

---

## Abstract

The original synthetic covariate shift SCRC experiment (`synthetic_covariate_shift_scrc.ipynb`)
used `select_for_deferral(entropy, β)` independently on the Cal and Test sets, producing different
absolute entropy thresholds for each. Because the blurred Test set has a different entropy
distribution from the clean Cal set, this breaks the conditional exchangeability assumption
underlying SCRC's coverage guarantee.

**SCRC-T** (transductive) fixes this by deriving a single absolute threshold from the Test
entropy distribution and applying it uniformly to both Cal and Test. This notebook tests the fix
using the GNN-DRE arm only (σ=3.0, β=0.15, α=0.10).

**Key results (three variants tested):**

| Metric | Original SCRC | Full-Test SCRC-T | **Warm-up Batch SCRC-T** |
|--------|--------------|-----------------|--------------------------|
| Threshold source | per-set relative | all 12,907 Test | **N=500 unlabeled** |
| FNR Gap | 0.019 | 0.003 | **0.001** |
| Violation | 0.013 | 0.014 | **0.011** |
| Cal deferral | ~15% | 27.4% | 29.3% |
| Test deferral | ~15% | 15.0% | 16.7% |

Both SCRC-T variants dramatically reduce the FNR gap vs the original (0.003/0.001 vs 0.019).
The warm-up batch (N=500) matches or slightly beats the full-Test variant, demonstrating that
500 unlabeled target samples are sufficient to estimate τ accurately. A surprising finding:
Cal deferral rate (27–29%) greatly exceeds BETA in both SCRC-T variants, because the clean
source GNN produces *higher* per-sample entropy than the blurred target — reversing the
expected direction.

---

## 1. Motivation: Why the Original SCRC Approach Was Flawed

SCRC requires that the calibration and test samples are **exchangeable** conditional on
being kept (non-deferred). Exchangeability holds if the selection rule is symmetric: the
same mechanism determines which samples are deferred from both sets.

The original approach applied `select_for_deferral(entropy, β)` separately to each set:

```
defer_cal = select_for_deferral(entropy_cal, BETA)   # top-β of CAL entropy
defer_tst = select_for_deferral(entropy_tst, BETA)   # top-β of TEST entropy
```

This defers the top-15% of **each set's own entropy distribution**, which corresponds to
different absolute entropy values when Cal and Test have different entropy distributions.
The remaining Cal and Test samples are filtered by different criteria → not exchangeable.

**SCRC-T fix:**

```python
n_defer_tst = int(len(entropy_tst) * BETA)
absolute_threshold = np.partition(entropy_tst, -n_defer_tst)[-n_defer_tst]

defer_tst = entropy_tst > absolute_threshold   # ≤ β by construction
defer_cal = entropy_cal > absolute_threshold   # same threshold, may differ from β
```

Both Cal and Test are now filtered by the same rule: entropy > absolute_threshold. Samples
from both sets that survive filtering are those with "low enough" entropy under the test
distribution's scale. **This restores conditional exchangeability.**

Crucially, this does NOT require equal deferral rates. Under distribution shift, it is expected
and correct that Cal and Test defer different proportions — the guarantee only requires symmetric
selection, not symmetric rates.

---

## 2. Experimental Setup

Identical to `synthetic_covariate_shift_scrc_report.md`, GNN arm only.

```
CheXpert (N=64,534, SEED=42)
├── Source (60% = 38,720)  ─ clean DenseNet121 features
│   ├── Train (75% = 29,040)  → fit LR (init logits) + GNN
│   └── Cal   (25% =  9,680)  → SCRC calibration
└── Target (40% = 25,814)  ─ features re-extracted with Gaussian blur (σ=3.0)
    ├── DRE Pool (50% = 12,907)  → fit GNN-DRE domain classifier
    └── Test     (50% = 12,907)  → SCRC evaluation
```

| Parameter | Value |
|-----------|-------|
| SIGMA | 3.0 |
| BETA (Stage 1 budget) | 0.15 |
| ALPHA (Stage 2 FNR target) | 0.10 |
| DRE arm | GNN-DRE (clip=20) |
| GNN ESS% | 16.9% |
| Domain AUC | 0.8643 |

---

## 3. Stage 1 — SCRC-T Deferral

```
absolute_threshold = 3.8808  (85th percentile of Test entropy)

Test (Target): 1,935 / 12,907 deferred  (15.0%)   ← = β by construction
Cal  (Source): 2,657 /  9,680 deferred  (27.4%)   ← >> β  (unexpected)
```

### 3.1 Unexpected finding: Cal deferral rate >> β

The working assumption was that blurred Test images would produce higher GNN entropy (the GNN
is uncertain about degraded features), making the test-derived threshold generous for Cal.
The result is the opposite: 27.4% of clean Cal samples have entropy exceeding the 85th
percentile of Test entropy.

**Interpretation:** The GNN, trained entirely on source (clean) features, produces a broader,
more diverse probability distribution on the familiar Cal set. It outputs moderate probabilities
for many pathologies on known-distribution images — high per-label uncertainty that compounds
across 7 labels into high multilabel entropy. On blurred Test images, the DenseNet features
activate different regions of the 1024-dim feature space, which the GNN maps to more extreme
(near 0 or near 1) sigmoid outputs. Confident but wrong predictions are low-entropy.

This is consistent with the overconfidence phenomenon in out-of-distribution inference: a model
trained on clean images may be spuriously confident when evaluated on shifted images.

**Does this invalidate SCRC-T?** No. The theoretical validity of SCRC-T only requires that the
same absolute threshold is applied to both sets. The guarantee that the kept samples from Cal are
exchangeable with the kept samples from Test holds as long as selection is by the same rule —
regardless of how many samples are removed from each set. The larger Cal removal (27.4% vs 15%)
reduces the effective calibration set size (from 8,228 to 7,023 samples) but does not break
exchangeability.

### 3.2 Entropy distribution summary

| Set | n | Deferred | Threshold | Type |
|-----|---|---------|----------|------|
| Test | 12,907 | 1,935 (15.0%) | > 3.8808 | Blurred (target) |
| Cal | 9,680 | 2,657 (27.4%) | > 3.8808 | Clean (source) |

---

## 4. Stage 2 — Calibration (GNN arm, non-deferred Cal)

Calibration uses 7,023 Cal samples (those with entropy ≤ 3.8808).

| Pathology | λ* | Cal FNR |
|-----------|-----|---------|
| Atelectasis | 0.341 | 0.099 |
| Cardiomegaly | 0.393 | 0.096 |
| Consolidation | 0.070 | 0.099 |
| Edema | 0.286 | 0.100 |
| Effusion | 0.236 | 0.100 |
| Pneumonia | 0.135 | 0.099 |
| Pneumothorax | 0.051 | 0.097 |
| **Mean** | **0.216** | **0.099** |

Cal FNR ≤ 0.10: **PASS** (all pathologies).

The calibration thresholds are tightly clustered just below α=0.10, consistent with the GNN
providing reliable weighted quantile estimates at 16.9% ESS.

---

## 5. Test Evaluation

10,972 Test samples kept (85% of Test — threshold at exactly the 85th percentile).

### 5.1 Summary

| Metric | Value |
|--------|-------|
| ESS% | 16.9% |
| Cal FNR | 0.099 |
| Test FNR (mean) | 0.103 |
| **FNR Gap** \|mean FNR − α\| | **0.003** |
| **Violation** mean(max(0, FNR−α)) | **0.014** |
| Test FPR (mean) | 0.511 |

### 5.2 Per-pathology breakdown

| Pathology | FNR | FPR | Violation |
|-----------|-----|-----|-----------|
| Atelectasis | 0.080 | 0.569 | 0.000 |
| Cardiomegaly | 0.136 | 0.278 | **0.036** |
| Consolidation | 0.121 | 0.426 | 0.021 |
| Edema | 0.069 | 0.506 | 0.000 |
| Effusion | 0.071 | 0.462 | 0.000 |
| Pneumonia | 0.100 | 0.602 | 0.000 |
| Pneumothorax | 0.141 | 0.733 | **0.041** |
| **Mean** | **0.103** | **0.511** | **0.014** |

Violations concentrated in Cardiomegaly (+0.036) and Pneumothorax (+0.041).
Four pathologies have FNR comfortably below α (Atelectasis, Edema, Effusion, Pneumonia).

---

## 6. Comparison: Original SCRC vs SCRC-T

| Metric | Original SCRC (GNN) | SCRC-T (GNN) | Δ |
|--------|-------------------|-------------|---|
| Cal deferral | 15.0% | 27.4% | +12.4 pp |
| Test deferral | 15.0% | 15.0% | 0 |
| Cal calibration n | 8,228 | 7,023 | −1,205 |
| Cal FNR (mean) | 0.093 | 0.099 | +0.006 |
| Test FNR (mean) | 0.119 | 0.103 | −0.016 |
| **FNR Gap** | **0.019** | **0.003** | **−84%** |
| **Violation** | **0.013** | **0.014** | +0.001 |
| Test FPR | 0.469 | 0.511 | +0.042 |

### 6.1 FNR Gap interpretation

The FNR Gap collapsed from 0.019 to 0.003 — the mean test FNR (0.103) is essentially at α.
This is the core confirmation that SCRC-T improves the calibration guarantee transfer.

The improvement mechanism: by selecting non-deferred Test samples as those below the
85th-percentile of Test entropy, the evaluation subset is the "low-entropy" portion of Test —
the samples on which the GNN is most confident. By calibrating on the similarly-filtered Cal
(also low-entropy by the same absolute criterion), the calibration set is matched to the
evaluation regime in terms of prediction difficulty. The weighted quantile calibrated on
"confident Cal" transfers better to "confident Test" than a quantile calibrated on all Cal
(including samples the GNN is uncertain about) and evaluated on a different confident subset.

### 6.2 Violation interpretation

Violation increased marginally (0.013 → 0.014), driven by Cardiomegaly and Pneumothorax.
FNR Gap and Violation tell complementary stories:

- **FNR Gap** (|mean FNR − α|): Edema (0.069) and Effusion (0.071) are well below α,
  partially cancelling Cardiomegaly (0.136) and Pneumothorax (0.141) in the average.
  The mean (0.103) lands nearly at α, giving small FNR Gap.

- **Violation** (mean max(0, FNR−α)): Edema and Effusion contribute 0 (guaranteed), while
  Cardiomegaly and Pneumothorax contribute their exceedances directly. The sum is not masked
  by the under-coverage pathologies.

The small increase in Violation (0.001) is consistent with the smaller calibration set
(7,023 vs 8,228 samples), which slightly increases variance in the weighted quantile estimate
for Cardiomegaly and Pneumothorax.

### 6.3 FPR increase

Mean FPR increased 0.042 (0.469 → 0.511). This is consistent with the slightly higher Cal FNR
(0.099 vs 0.093), which corresponds to lower λ* thresholds — predicting more positives, increasing
both FNR (for some pathologies) and FPR. In clinical screening, FPR is secondary; a higher FPR
is acceptable if FNR is controlled.

---

## 7. Key Interpretations

### 7.1 SCRC-T restores exchangeability and improves FNR gap

The symmetric absolute threshold is the theoretically correct construction. The 84% improvement
in FNR Gap confirms that the original per-set deferral was introducing a meaningful calibration
transfer error. Under SCRC-T, the calibration better reflects the evaluation regime.

### 7.2 The entropy distribution reversal is a finding, not a failure

The expectation that "clean Cal has lower entropy than blurred Test" was a domain assumption,
not a theoretical requirement of SCRC-T. The actual GNN entropy distributions show the opposite:
the source GNN outputs are more diverse (higher entropy) on familiar source data than on
out-of-distribution blurred data. This is consistent with the known overconfidence of deep
networks under distribution shift.

This finding implies: in real clinical deployment (CheXpert→NIH), the same reversal may
occur if the source GNN is overconfident on the NIH target set. The SCRC-T fix is still
the correct construction regardless of which direction entropy shifts — it is robust to this.

### 7.3 Remaining violations are ESS-limited, not threshold-limited

Cardiomegaly (violation=0.036) and Pneumothorax (violation=0.041) still violate the guarantee.
These are the two pathologies where the GNN achieves lowest AUC on the target test set and where
importance weighting variance is highest. With ESS=16.9% (effective n≈1,187 samples from n=7,023
calibration points), the weighted quantile estimate has non-trivial variance. The Pneumothorax
violations in particular track the same pathology-specific pattern seen across all σ in the
original experiment.

SCRC-T addresses the *threshold* problem but not the *ESS* problem. For pathologies where ESS
is insufficient, violations persist regardless of threshold correction.

### 7.4 Practical recommendation

When deploying SCRC under distribution shift:
1. **Always use an absolute threshold** derived from the target (Test) entropy distribution.
   Do not use per-set relative thresholds.
2. **Expect Cal deferral rate ≠ BETA.** This is correct. A higher Cal deferral rate under
   source-confident GNNs is expected and not an error.
3. **Check ESS separately** from threshold validity. Violations that persist under SCRC-T are
   ESS-driven and require higher-ESS DRE, not threshold adjustment.

---

## 8. Summary

| | Value |
|--|-------|
| Notebook | `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_t_gnn.ipynb` |
| σ | 3.0 |
| β / α | 0.15 / 0.10 |
| GNN-DRE ESS | 16.9% |
| Absolute threshold | 3.8808 |
| Test deferral | 15.0% (= β) |
| Cal deferral | **27.4% (>> β)** — source GNN is more entropic than blurred target |
| Cal FNR | 0.099 |
| Test FNR | 0.103 |
| **FNR Gap** | **0.003** (vs 0.019 original — 84% reduction) |
| **Violation** | **0.014** (vs 0.013 original — essentially unchanged) |
| Violating pathologies | Cardiomegaly (0.036), Pneumothorax (0.041) |

**Conclusion:** SCRC-T restores the correct exchangeability structure and dramatically reduces
the mean FNR gap (0.003 vs 0.019). Violations for two pathologies persist at similar magnitude,
driven by ESS constraints not correctable by threshold symmetry alone. The entropy direction
reversal (Cal > Test) is a genuine empirical finding consistent with source-model overconfidence
on shifted inputs.

---

---

## 10. Experiment 2: Unlabeled Warm-up Batch SCRC-T

**Motivation**: The Full-Test SCRC-T (Section 3) uses the entire Target Test set to derive τ.
In real deployment, the full test set is not available up-front. The warm-up batch variant
simulates a hospital providing a small unlabeled sample of N=500 recent X-rays to calibrate the
deferral threshold — a clinically realistic assumption.

### 10.1 Protocol

```python
# Step 1: Sample N_WARMUP unlabeled target probabilities (labels NOT used)
warmup_idx     = rng.choice(len(p_test_gnn), size=500, replace=False)
entropy_warmup = multilabel_entropy(p_test_gnn[warmup_idx])

# Step 2: Target-anchored threshold
tau_target = np.quantile(entropy_warmup, 1 - BETA)      # 85th pct of 500 samples

# Step 3: Symmetric deferral
defer_cal = entropy_cal > tau_target
defer_tst = entropy_tst > tau_target
```

### 10.2 Stage 1 Results

| | Full-Test SCRC-T | Warm-up Batch SCRC-T |
|-|-----------------|----------------------|
| Threshold τ | 3.8808 (exact 85th pct of 12,907) | **3.8537** (est. from 500) |
| Δτ | — | −0.027 (−0.7%) |
| Cal deferral | 27.4% | **29.3%** |
| Test deferral | 15.0% | **16.7%** |

The warm-up threshold (3.8537) is 0.027 below the full-Test threshold (3.8808) — a small
sampling error from estimating the 85th percentile from 500 vs 12,907 samples. This causes
the warm-up variant to defer slightly more from both sets (lower threshold → more samples
exceed it). Test deferral rises from 15.0% to 16.7%: still close to β=15%, acceptable for
clinical deployment.

Warm-up entropy diagnostics (n=500):
- Mean: 3.2351, Std: 0.6679, Range: [0.548, 4.408]

### 10.3 Calibration

Calibration uses the non-deferred Cal subset (70.7% of 9,680 = 6,843 samples).

| Pathology | λ* | Cal FNR |
|-----------|-----|---------|
| Atelectasis | 0.334 | 0.100 |
| Cardiomegaly | 0.393 | 0.098 |
| Consolidation | 0.070 | 0.099 |
| Edema | 0.286 | 0.100 |
| Effusion | 0.224 | 0.100 |
| Pneumonia | 0.131 | 0.099 |
| Pneumothorax | 0.049 | 0.098 |
| **Mean** | **0.212** | **0.099** |

Cal FNR ≤ 0.10: **PASS** (all pathologies). Thresholds nearly identical to Full-Test SCRC-T
(mean λ* 0.212 vs 0.216), confirming the small Δτ has negligible calibration impact.

### 10.4 Test Evaluation

10,747 Test samples kept (83.3% of Test — slightly more than the full-Test variant's 85%).

| Pathology | FNR | FPR | Violation |
|-----------|-----|-----|-----------|
| Atelectasis | 0.079 | 0.570 | 0.000 |
| Cardiomegaly | 0.137 | 0.273 | **0.037** |
| Consolidation | 0.122 | 0.420 | 0.022 |
| Edema | 0.071 | 0.498 | 0.000 |
| Effusion | 0.067 | 0.473 | 0.000 |
| Pneumonia | 0.096 | 0.606 | 0.000 |
| Pneumothorax | 0.121 | 0.754 | **0.021** |
| **Mean** | **0.099** | **0.514** | **0.011** |

### 10.5 Three-way comparison

| Metric | Original SCRC | Full-Test SCRC-T | Warm-up SCRC-T |
|--------|--------------|-----------------|----------------|
| τ source | per-set | all 12,907 Test | 500 unlabeled |
| τ value | Cal: ~3.67 / Test: ~3.88 | 3.8808 | **3.8537** |
| Cal deferral | 15.0% | 27.4% | 29.3% |
| Test deferral | 15.0% | 15.0% | 16.7% |
| Cal FNR | 0.093 | 0.099 | 0.099 |
| Test FNR | 0.119 | 0.103 | **0.099** |
| **FNR Gap** | 0.019 | 0.003 | **0.001** |
| **Violation** | 0.013 | 0.014 | **0.011** |
| Test FPR | 0.469 | 0.511 | 0.514 |
| Pneumothorax violation | 0.063 | 0.041 | **0.021** |

### 10.6 Key findings

**FNR Gap = 0.001**: The mean Test FNR (0.099) is virtually identical to α=0.10. The
warm-up batch estimator, despite using only 500 samples, produces a threshold that transfers
the calibration guarantee almost perfectly.

**Violation = 0.011 — lowest of all three variants**: Despite having slightly fewer calibration
samples (6,843 vs 7,023 for Full-Test), the warm-up variant achieves lower violation. The
Pneumothorax violation in particular drops from 0.041 (Full-Test) to 0.021 (warm-up). This is
partly explained by the lower τ (3.8537) deferring more from Test (16.7% vs 15%), which removes
slightly more of the hard Pneumothorax cases from evaluation — the kept Test subset is
marginally "easier" for Pneumothorax, reducing FNR.

**N=500 is sufficient**: The threshold estimate from 500 samples (τ=3.8537) is within 0.7%
of the full-Test estimate (τ=3.8808). The resulting metrics are at least as good, suggesting
that for this entropy distribution, 500 samples adequately capture the 85th percentile. This
is consistent with statistical theory: the standard error of a quantile estimator at quantile
q from n samples is ≈ √(q(1−q)/n) / f(Q_q) where f is the density at the quantile. With
n=500, q=0.85, and a smooth entropy distribution, the SE is small relative to the τ range.

**Deployment-ready**: The warm-up batch approach requires only:
1. N=500 unlabeled (no ground-truth labels) target-domain images
2. A pre-trained GNN to extract probabilities
3. The quantile computation (trivial)

This is a practical protocol for real hospital deployment: receive a small batch of recent
cases, compute the deferral threshold, apply it to the source calibration set, deploy.

---

## 11. Files

| File | Description |
|------|-------------|
| `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_t_gnn.ipynb` | Source notebook (6-arm: 3 DRE × 2 threshold) |
| `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_t_gnn_executed.ipynb` | Executed output (6-arm comparison) |
| `report/scrc_t_gnn_sigma3.0.png` | 3-panel bar charts: FNR Gap, Violation, ESS% |
| `report/synthetic_covariate_shift_scrc_report.md` | Reference experiment (original SCRC, 4 arms) |

---

## 12. Experiment 3: 6-Arm Comparison (3 DRE × 2 Threshold Strategies)

**Added:** 2026-02-24. Extends the GNN-only SCRC-T notebook to compare three DRE methods
(GNN-DRE, LR-DRE, MLP-DRE) each under both threshold strategies (Full-Test and Warm-up),
giving 6 arms total. Stage 1 deferral uses GNN entropy for all arms.

### 12.1 DRE Quality

| Method | Domain AUC | ESS% | W_mean | W_max |
|--------|-----------|------|--------|-------|
| GNN-DRE (clip=20) | 0.8643 | **16.89%** | 0.874 | 20.0 |
| LR-DRE  (clip=20) | 0.9981 | **1.44%**  | 0.168 | 20.0 |
| MLP-DRE (clip=20) | 0.9362 | **9.38%**  | 0.753 | 20.0 |

LR-DRE achieves near-perfect domain classification (AUC=0.998) but collapses to 1.44% ESS —
weight concentration is severe at σ=3 (outside the Goldilocks zone). MLP-DRE at 9.38% ESS
provides a useful middle ground between LR and GNN.

### 12.2 Stage 1 Thresholds

| Strategy | τ | Cal defer | Test defer |
|----------|---|-----------|-----------|
| Full-Test (FT) | 3.8808 | 27.45% | 14.99% |
| Warm-up (WU, N=500) | 3.8537 | 29.31% | 16.74% |

Δτ = −0.027 (−0.7%) between WU and FT — small sampling error from estimating the 85th percentile
from 500 vs 12,907 samples.

### 12.3 Calibration — All 6 Arms Pass

All 6 arms achieve Cal FNR ≤ 0.10 (PASS):

| Arm | Cal n | Mean λ* | Mean Cal FNR | Status |
|-----|-------|---------|--------------|--------|
| GNN-FT | 7,023 | 0.216 | 0.099 | PASS |
| LR-FT  | 7,023 | 0.152 | 0.073 | PASS |
| MLP-FT | 7,023 | 0.226 | 0.086 | PASS |
| GNN-WU | 6,843 | 0.212 | 0.099 | PASS |
| LR-WU  | 6,843 | 0.174 | 0.078 | PASS |
| MLP-WU | 6,843 | 0.223 | 0.087 | PASS |

LR-DRE's low Cal FNR (0.073/0.078 vs 0.099 for GNN) reflects aggressive over-estimation of
importance weights — the calibration over-adjusts for the domain shift, setting conservative
λ* values that permit high FNR on Test.

### 12.4 Test Evaluation — 6-Arm Summary

| Arm | ESS% | Cal%def | Tst%def | MnFNR | **FNR Gap** | **Violation** | MnFPR |
|-----|------|---------|---------|-------|------------|--------------|-------|
| GNN-FT | 16.9 | 27.45 | 14.99 | 0.103 | **0.003** | **0.014** | 0.511 |
| LR-FT  |  1.4 | 27.45 | 14.99 | 0.143 | **0.043** | **0.058** | 0.530 |
| MLP-FT |  9.4 | 27.45 | 14.99 | 0.115 | **0.015** | **0.034** | 0.503 |
| GNN-WU | 16.9 | 29.31 | 16.74 | 0.099 | **0.001** | **0.011** | 0.514 |
| LR-WU  |  1.4 | 29.31 | 16.74 | 0.169 | **0.069** | **0.084** | 0.490 |
| MLP-WU |  9.4 | 29.31 | 16.74 | 0.115 | **0.015** | **0.035** | 0.503 |

### 12.5 Per-pathology: Best (GNN-WU) vs Worst (LR-WU)

| Pathology | GNN-WU FNR | LR-WU FNR | α |
|-----------|-----------|----------|---|
| Atelectasis    | 0.079 | 0.048 | 0.10 |
| Cardiomegaly   | 0.137 | 0.098 | 0.10 |
| Consolidation  | 0.122 | 0.154 | 0.10 |
| Edema          | 0.071 | 0.059 | 0.10 |
| Effusion       | 0.067 | 0.094 | 0.10 |
| Pneumonia      | 0.096 | **0.285** | 0.10 |
| Pneumothorax   | 0.121 | **0.448** | 0.10 |
| **Mean**       | **0.099** | **0.169** | 0.10 |

LR-WU violates severely for Pneumonia (+0.185 above α) and Pneumothorax (+0.348 above α).
GNN-WU has FNR ≤ α for Atelectasis, Edema, Effusion, Pneumonia, with violations only for
Cardiomegaly (0.137) and Pneumothorax (0.121).

### 12.6 Key Findings

**Finding 1: SCRC-T does NOT rescue low-ESS DREs.**
The target-anchored threshold restores exchangeability for all arms, but ESS governs whether
the weighted quantile accurately estimates the FNR level. With only 1.4% ESS, LR-DRE's
weighted calibration is too noisy: Cal FNR (0.073) underestimates Test FNR (0.143–0.169).
The SCRC-T threshold fixes the *selection* problem; the *estimation* problem requires better DRE.

**Finding 2: Warm-up vs Full-Test differs by DRE.**
For GNN-DRE (high ESS), WU is marginally better than FT (0.001 vs 0.003 FNR Gap).
For LR-DRE (low ESS), WU is *worse* (0.069 vs 0.043): the lower τ in WU defers slightly
more from Test, exposing a harder subset that the noisy LR-weighted calibration cannot cover.
For MLP-DRE (intermediate ESS), FT ≈ WU (0.015 for both).

**Finding 3: MLP-DRE is a useful middle ground.**
At 9.38% ESS (vs 16.9% GNN, 1.44% LR), MLP-DRE achieves FNR Gap = 0.015 consistently.
This is 6× better than LR-DRE and only 5× worse than GNN-DRE, suggesting that operating
in the 7-dim probability space (without graph structure) already provides substantial
benefit over raw-feature LR-DRE.

**Finding 4: N=500 warm-up is sufficient for high-ESS DREs.**
The τ estimation from 500 samples is within 0.7% of the full-Test estimate, and GNN-WU
achieves the best overall result (FNR Gap = 0.001, Violation = 0.011) of all 6 arms.
For deployment with high-ESS DRE, the warm-up batch protocol is the recommended choice.

**Finding 5: ESS is the dominant factor, not threshold strategy.**
Ranking by FNR Gap: GNN (0.001–0.003) >> MLP (0.015) >> LR (0.043–0.069).
The threshold strategy (FT vs WU) produces at most 0.026 gap difference within each DRE,
while DRE choice produces up to 0.066 gap difference within each threshold. ESS quality
dominates threshold quality.

### 12.7 Practical Recommendations (Updated)

1. **Use GNN-DRE + Warm-up SCRC-T**: Highest ESS (16.9%), deployable with N=500 unlabeled.
2. **Do not use LR-DRE when outside Goldilocks zone** (σ≥2, Domain AUC>0.98): 1.4% ESS
   is insufficient for reliable weighted calibration. FNR Gap = 0.043–0.069 is clinically
   unacceptable.
3. **MLP-DRE is acceptable for resource-constrained settings**: 9.38% ESS, FNR Gap = 0.015.
   No graph structure needed; shares the probability-space DRE benefit with GNN-DRE.
4. **SCRC-T threshold strategy is secondary to DRE quality**: Fix DRE first, then threshold.
5. **All 6 arms pass calibration (Cal FNR ≤ α)**: The guarantee transfers in-sample for all
   DREs, but only high-ESS DREs transfer it to the Test set.

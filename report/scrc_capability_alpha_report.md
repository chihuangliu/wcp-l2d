# SCRC with Dynamic Capability Alpha Allocation — Experiment Report

**Date**: 2026-02-23
**Branch**: `alpha-allocation`
**Notebook**: `notebooks/gnn/scrc_capability_alpha.ipynb`

---

## 1. Motivation

The Selective Conformal Risk Control (SCRC) pipeline applies a two-stage decision process:

1. **Stage 1** — Defer the β-fraction of most-uncertain samples to a human expert (global entropy threshold).
2. **Stage 2** — Per-pathology weighted CRC finds thresholds λ_k* so that the weighted FNR ≤ α_k on retained samples.

**The FPR problem**: A near-random classifier (Pneumothorax AUC = 0.599) with a uniform budget (α_k = 0.10) must set λ_k* ≈ 0 to achieve the required FNR, predicting almost all samples as positive — causing high FPR on retained cases.

**Fix 7.3 — Dynamic Capability Alpha Allocation** redistributes the budget inversely proportional to each pathology's excess AUC above chance:

```
excess_k = max(AUC_k − 0.5, 0.001)
α_k  ∝  1 / excess_k,    with  mean(α_k) = α_target = 0.10
```

Near-random pathologies receive a *loose* budget (e.g. α_k = 0.20 for Pneumothorax), allowing the calibration to set a higher λ_k*, reducing false positives. High-AUC pathologies receive a *tight* budget, maintaining sensitivity.

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Source domain | CheXpert — 38,720 train / 12,906 cal |
| Target domain | NIH — 15,402 DRE pool / 15,403 labelled test |
| Classifier | LabelGCN (best epoch 18/50, val AUC 0.8324) |
| DRE | GNN-DRE: logistic in 7-dim probability space, clip = 20, ESS = 31.6% |
| Global FNR budget (α_target) | 0.10 |
| Global deferral budget (β) | 0.15 → 1,935 cal / 2,310 NIH test deferred |
| Random seed | 42 |

### GNN NIH Test AUC per Pathology

| Pathology | NIH AUC |
|---|---|
| Atelectasis | 0.7096 |
| Cardiomegaly | 0.7664 |
| Consolidation | 0.7471 |
| Edema | 0.8207 |
| Effusion | 0.8314 |
| Pneumonia | 0.6763 |
| **Pneumothorax** | **0.5994** |
| **Mean** | **0.7358** |

### GNN-DRE Diagnostics

| Metric | Value |
|---|---|
| Domain AUC (CheXpert vs NIH) | 0.8461 |
| ESS (effective sample size) | 4080.6 / 12,906 = **31.6%** |
| Weight clip / max | 20.0 |
| Weight mean ± std | 0.94 ± 1.38 |

---

## 3. Alpha Allocation Results

All strategies achieve **budget neutrality** (mean α_k ≈ 0.10):

| Pathology | AUC | Uniform | Linear | Inv. Excess | Softmax (T=0.05) |
|---|---|---|---|---|---|
| Atelectasis | 0.7096 | 0.1000 | 0.1099 | 0.0970 | 0.0539 |
| Cardiomegaly | 0.7664 | 0.1000 | 0.0884 | 0.0763 | 0.0173 |
| Consolidation | 0.7471 | 0.1000 | 0.0957 | 0.0822 | 0.0254 |
| Edema | 0.8207 | 0.1000 | 0.0679 | 0.0634 | 0.0100 |
| Effusion | 0.8314 | 0.1000 | 0.0638 | 0.0613 | **0.0100** |
| Pneumonia | 0.6763 | 0.1000 | 0.1225 | 0.1153 | 0.1049 |
| **Pneumothorax** | **0.5994** | 0.1000 | 0.1517 | **0.2045** | **0.4880** |
| **Mean** | | **0.100** | **0.100** | **0.100** | 0.1014 |

The Inverse Excess strategy gives Pneumothorax (the weakest pathology) a budget of 0.2045 — twice the uniform value — while Softmax allocates an extreme 0.4880 to it.

### Calibrated Thresholds λ_k*

| Strategy | λ* range | Cal FNR range |
|---|---|---|
| Uniform | [0.027, 0.119] | [0.089, 0.100] |
| Linear | [0.029, 0.124] | [0.062, 0.142] |
| Inverse Excess | [0.028, 0.118] | [0.060, 0.200] |
| Softmax | [0.013, 0.135] | [0.005, 0.487] |

Note that Uniform produces the tightest cal FNR range (all pathologies pinned to ≈ 0.10), while Inverse Excess and Softmax allow larger variance — Pneumothorax is calibrated with a *deliberately* loose FNR at calibration time.

---

## 4. Main Results (Task 2 + 3)

### Evaluation Metrics (β = 0.15, α_target = 0.10)

| Method | Mean FPR ↓ | Max FPR ↓ | Mean FNR | FNR Gap ↓ | Clinical Cost ↓ |
|---|---|---|---|---|---|
| **Uniform** | **0.6194** | **0.8039** | 0.1856 | 0.0931 | 1.5475 |
| Linear | 0.6391 | 0.8198 | 0.1730 | **0.0784** | **1.5041** |
| Inverse Excess | 0.6265 | 0.8269 | 0.1852 | 0.0899 | 1.5528 |
| Softmax | 0.7469 | 0.9926 | **0.1631** | 0.0654 | 1.5625 |

**Metric definitions**:
- **Mean FPR**: average false positive rate on retained test samples (7 pathologies)
- **Max FPR**: single-pathology worst case — catastrophic failure indicator
- **FNR Gap**: mean|Test_FNR_k − α_k| — gap between nominal and realised guarantees
- **Clinical Cost**: mean(5 · FNR_k + FPR_k) — FNR penalised 5× for missed diagnoses

---

## 5. Analysis of Findings

### 5.1 The FPR problem is universal, not allocation-specific

All strategies produce high FPR on retained NIH samples (0.62–0.75). This is a **fundamental limitation** of the binary classification setting under compound covariate shift. Even with ESS = 31.6%, the importance-weighted CRC calibration must set very low thresholds (λ_k* ≈ 0.03–0.13) to satisfy FNR ≤ α_k, because the GNN's probability mass for true positives overlaps heavily with that of true negatives in the target domain.

This finding connects to the "Binary CP bottleneck" identified in earlier experiments: with K = 2 classes, there is only one threshold per pathology, and the FNR–FPR trade-off is a single sharp transition.

### 5.2 Softmax (T = 0.05) is catastrophically bad

By concentrating nearly half the budget on Pneumothorax (α = 0.4880), Softmax starves Effusion and Edema of budget (α = 0.01 each). This forces λ* → 0 for those pathologies, making the classifier predict positive for virtually every sample → **Max FPR = 0.9926** (near 1.0 for at least one pathology). This is the clearest demonstration of how extreme budget redistribution can *cause* rather than prevent FPR collapse.

### 5.3 Uniform is surprisingly competitive on aggregate FPR

Contrary to the hypothesis that uniform allocation causes catastrophic FPR spikes on weak pathologies, Uniform achieves the **lowest Mean FPR (0.6194)** and **lowest Max FPR (0.8039)** among all strategies. Why?

- Uniform enforces the same tight budget everywhere. For strong pathologies (Effusion AUC = 0.83), a tight budget still finds an informative λ* ≈ 0.10–0.12 because the probability mass is well-separated.
- For Pneumothorax (AUC = 0.599), the uniform budget (α = 0.10) does force a low λ*, causing high FPR — but the same low λ* is also forced by Inverse Excess for Effusion (which gets α = 0.061, slightly tighter than Uniform's 0.10).

In other words: Inverse Excess reduces FPR on Pneumothorax by loosening its budget, but simultaneously increases FPR on Effusion/Edema by tightening their budgets. The net aggregate effect is near-neutral — or slightly adverse.

### 5.4 Linear achieves the best Clinical Cost (1.5041)

The linear strategy's gradual budget redistribution (proportional to 1 − AUC) provides a better FNR–FPR balance than either extreme:
- Lower mean FNR (0.1730) than Uniform or Inverse Excess
- Moderate FPR redistribution — Pneumothorax gets α = 0.1517 (vs 0.2045 for IE)
- Best clinical cost: **1.5041** (vs 1.5475 for Uniform)

The smaller FNR Gap (0.0784) also indicates that the calibration-to-test transfer is more honest for Linear — the weighted CRC guarantees transfer more accurately to the test set.

### 5.5 Inverse Excess: balanced but not dominant

The proposed Inverse Excess strategy achieves intermediate performance on all metrics. It does not win on any single metric. The key insight is that its budget redistribution is large enough to lose some gains on strong pathologies but not large enough to substantially help weak pathologies under the compound shift present here.

---

## 6. Beta Sweep Results (Task 4)

Using **Inverse Excess** allocation with α_target = 0.10, varying β from 0.05 to 0.30:

| β | N Retained | N Deferred | Mean FNR | Mean FPR | Max FPR | Clinical Cost |
|---|---|---|---|---|---|---|
| 0.05 | 14,633 | 770 | 0.1393 | 0.6562 | 0.8362 | 1.3525 |
| 0.10 | 13,863 | 1,540 | 0.1623 | 0.6416 | 0.8290 | 1.4532 |
| **0.15** | **13,093** | **2,310** | 0.1852 | 0.6265 | 0.8269 | 1.5528 |
| 0.20 | 12,323 | 3,080 | 0.2018 | 0.6268 | 0.8199 | 1.6360 |
| 0.25 | 11,553 | 3,850 | 0.2078 | 0.6218 | 0.8273 | 1.6609 |
| 0.30 | 10,783 | 4,620 | 0.2154 | 0.6227 | 0.8289 | 1.6999 |

### Key observations

1. **FPR plateau**: Mean FPR drops from 0.6562 (β=0.05) to 0.6265 (β=0.15), then plateaus at ≈ 0.622–0.627 for β ≥ 0.15. Deferring more than 15% of the most-uncertain samples yields diminishing returns in FPR reduction because the entropy-based Stage 1 already captures the most ambiguous cases at β = 0.15.

2. **FNR monotonically increases with β**: As β grows, the kept calibration set shrinks and the calibrated thresholds become more conservative (fewer calibration positives → higher required λ_k* for FNR ≤ α_k), raising test FNR to 0.215 at β = 0.30.

3. **Clinical Cost trade-off**: The best clinical cost (1.3525) is at β = 0.05, not β = 0.15. At β = 0.05 only 770 samples are deferred — meaning the model handles 94.6% of cases autonomously with lower overall error rate. But this relies on a favorable retained population.

4. **Pareto frontier is not monotone in both metrics**: Mean FPR improves until β ≈ 0.15–0.20 then degrades slightly. The sweet spot appears to be **β ∈ [0.15, 0.20]** for the best FPR–automation trade-off under Inverse Excess allocation.

---

## 7. Visualisations

### Radar Chart (`report/scrc_radar_chart.png`)

7-axis polar plot of Test FPR per pathology for all 4 strategies.

**Observation**: All strategies show similar FPR profiles. Softmax spikes outward on Effusion/Edema (FPR ≈ 0.99), confirming catastrophic failure from over-tightening those pathologies' budgets. Uniform, Linear, and Inverse Excess are tightly clustered.

### FNR vs FPR Scatter (`report/scrc_fnr_fpr_scatter.png`)

FNR (Y) vs FPR (X) for all 7 pathologies, all 4 strategies.

**Observation**: Pathology points cluster in the upper-right (FPR 0.5–0.85, FNR 0.15–0.25) for Uniform and Inverse Excess. Effusion/Edema points are most leftward (lower FPR, lower FNR) for Uniform; Pneumothorax is furthest right. Inverse Excess shifts Pneumothorax leftward at the cost of pushing Effusion/Edema rightward.

### Beta Sweep Pareto Frontier (`report/scrc_beta_sweep.png`)

Two-panel figure: (left) β vs Mean/Max FPR curves; (right) retained samples vs Mean FPR.

**Observation**: Left panel shows FPR improving steeply from β=0.05 to β=0.15, then plateauing — a clean knee in the curve at β ≈ 0.15. Right panel confirms that the retained-vs-FPR frontier is smooth and interpretable.

---

## 8. Conclusions

### What worked

- The **two-stage SCRC pipeline with GNN-DRE** provides per-pathology FNR control at calibration time across all allocation strategies (cal FNR ≤ α_k satisfied for all 28 pathology×strategy combinations).
- The **GNN probability space** for DRE gives ESS = 31.6% — a 5× improvement over raw 1024-dim feature DRE (6% ESS), with tighter calibration-to-test transfer.
- The **beta sweep** demonstrates a clear Pareto frontier with a knee at β ≈ 0.15, giving clinicians a principled basis for choosing the deferral rate.
- **Linear allocation** achieves the best clinical cost (1.5041) with modest FPR redistribution.

### What didn't work as expected

- The **Inverse Excess** hypothesis (that Uniform causes catastrophic FPR collapse on weak pathologies, fixed by loosening their budgets) is only partially confirmed. The FPR problem is systemic — driven by compound covariate shift — rather than pathology-specific. Loosening Pneumothorax's budget reduces its FPR but tightens Effusion/Edema's budget, shifting the spike rather than eliminating it.
- **Softmax** (T=0.05) causes a genuine catastrophic failure (Max FPR = 0.9926) by over-concentrating the budget, confirming the opposite risk: extreme budget redistribution can be worse than uniform.
- The **Beta sweep FPR plateau** means that arbitrarily increasing deferral rate cannot fully solve the FPR problem — the root cause is the domain shift and classifier calibration, not the deferral threshold.

### Recommendations

| Scenario | Recommended strategy |
|---|---|
| Minimize clinical cost | **Linear** (cost = 1.5041) |
| Minimize worst-case FPR | **Uniform** (max FPR = 0.8039) |
| Balance FNR honesty + FPR | **Linear** (FNR Gap = 0.0784) |
| Avoid catastrophic failure | **Avoid Softmax** with small T |
| Deferral rate choice | **β = 0.15** (knee of Pareto curve) |

---

## 9. Output Files

| File | Description | Size |
|---|---|---|
| `notebooks/gnn/scrc_capability_alpha.ipynb` | Unexecuted notebook | 26 KB |
| `notebooks/gnn/scrc_capability_alpha_executed.ipynb` | Executed with all outputs | 431 KB |
| `report/scrc_radar_chart.png` | Radar chart of per-pathology FPR | — |
| `report/scrc_fnr_fpr_scatter.png` | FNR vs FPR scatter | — |
| `report/scrc_beta_sweep.png` | β sweep Pareto frontier | — |
| `report/scrc_capability_alpha_report.md` | This report | — |

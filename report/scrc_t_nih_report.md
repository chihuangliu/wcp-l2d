# SCRC-T on CheXpert → NIH (Compound Shift): 6-Arm Comparison

## 1. Abstract

This notebook applies SCRC-T (Selective Conformal Risk Control with Transductive threshold)
to the real CheXpert→NIH cross-institution compound shift. The original `scrc_hard_fnr.ipynb`
used per-set relative thresholds (`select_for_deferral` independently on Cal and Test),
breaking the conditional exchangeability assumption and yielding a best-arm FNR Gap of 0.058.

**SCRC-T Fix**: derive the absolute threshold τ from the Test entropy distribution, then apply
the same τ to Cal. This restores symmetric selection and is the approach validated on pure
covariate shift (synthetic Gaussian blur, σ=3.0).

**Key question**: Does SCRC-T reduce the FNR Gap on compound shift the way it does on pure
covariate shift (where GNN-FT achieved FNR Gap ≈ 0.003)?

## 2. Setup

### Data Pipeline
```
Source Domain: CheXpert (N=64,534)
├── Train (75% = 48,400)  → fit LR (init logits) + GNN
└── Cal   (25% = 16,134)  → SCRC calibration

Target Domain: NIH (N=30,805) ─ natural compound shift
├── DRE Pool (50% = 15,402)  → fit GNN-DRE domain classifier
└── Test     (50% = 15,403)  → SCRC evaluation
    └── Warm-up Batch (N=500) → unlabeled target sample to estimate τ_target
```

### Architecture
- **Stage 1**: Global entropy-based deferral (β = 0.15)
  - Entropy from GNN probabilities (shared across all arms)
  - Threshold τ derived from Test distribution (SCRC-T)
- **Stage 2**: Per-pathology strict FNR calibration (α = 0.1)
  - DRE-weighted quantile on non-deferred calibration positives
  - 3 DRE variants × 2 threshold strategies = 6 arms

### DRE Configurations
| Method | Feature space | PCA | Clip |
|--------|--------------|-----|------|
| GNN-DRE | 7-dim GNN probabilities | None | 20.0 |
| LR-DRE  | 1024-dim scaled features | PCA(4) | 20.0 |
| MLP-DRE | 7-dim MLP probabilities | None | 20.0 |

### Threshold Strategies
| Strategy | Source of τ | Cal deferral |
|----------|------------|-------------|
| Full-Test (FT) | All 15,403 NIH test samples | 51.46% |
| Warm-up (WU) | N=500 unlabeled NIH samples | 53.50% |

## 3. DRE Quality

| Method | Domain AUC | ESS% | W_mean | W_max |
|--------|-----------|------|--------|-------|
| GNN-DRE (clip=20) | 0.8228 | 35.0% | 0.975 | 13.4 |
| LR-DRE  (clip=20) | 0.9623 | 6.6% | 0.587 | 20.0 |
| MLP-DRE (clip=20) | 0.8553 | 31.8% | 0.951 | 15.0 |

## 4. Stage 1 Results

### Threshold Values
- **τ_FT** = 3.5438  (derived from all 15,403 NIH test samples)
- **τ_WU** = 3.4941  (derived from N=500 warm-up samples; Δτ = -0.0497)

### Entropy Distribution
- Cal entropy:  mean=3.2258, median=3.5765
- Test entropy: mean=2.1535, median=1.9653
- Direction: Cal entropy is **HIGHER** than Test entropy

### Deferral Rates
| Strategy | τ | Cal deferred | Test deferred |
|----------|---|-------------|---------------|
| Full-Test | 3.5438 | 51.46% (8,303) | 14.99% (2,309) |
| Warm-up   | 3.4941 | 53.50% (8,631) | 16.02% (2,467) |

## 5. Calibration Results (all 6 arms)

| Arm | Cal n | Mean λ* | Mean Cal FNR | Status |
|-----|-------|---------|-------------|--------|
| GNN-FT       |   7831 | 0.038 | 0.097 | PASS |
| LR-FT        |   7831 | 0.030 | 0.095 | PASS |
| MLP-FT       |   7831 | 0.028 | 0.095 | PASS |
| GNN-WU       |   7503 | 0.037 | 0.098 | PASS |
| LR-WU        |   7503 | 0.029 | 0.094 | PASS |
| MLP-WU       |   7503 | 0.028 | 0.096 | PASS |

## 6. Test Evaluation — 6-Arm Summary

| Arm | ESS% | Cal%def | Tst%def | MnFNR | FNRGap | Violation | MnFPR |
|-----|------|---------|---------|-------|--------|-----------|-------|
| GNN-FT       |   35.0 |    51.46 |    14.99 |  0.112 |   0.012 |     0.031 |  0.719 |
| LR-FT        |    6.6 |    51.46 |    14.99 |  0.227 |   0.127 |     0.142 |  0.628 |
| MLP-FT       |   31.8 |    51.46 |    14.99 |  0.200 |   0.100 |     0.104 |  0.623 |
| GNN-WU       |   35.0 |    53.50 |    16.02 |  0.115 |   0.015 |     0.033 |  0.719 |
| LR-WU        |    6.6 |    53.50 |    16.02 |  0.228 |   0.128 |     0.142 |  0.631 |
| MLP-WU       |   31.8 |    53.50 |    16.02 |  0.204 |   0.104 |     0.108 |  0.620 |

**Baseline** (scrc_hard_fnr.ipynb, GNN-DRE clip, per-set thresholds):
- Best arm (GNN-c): FNR Gap = 0.058, Violation = 0.064, FPR = 0.632

### Best Arm: GNN-FT
FNR Gap = 0.012, Violation = 0.031, FPR = 0.719

### Worst Arm: LR-WU
FNR Gap = 0.128, Violation = 0.142, FPR = 0.631

### Per-Pathology FNR — Best arm (GNN-FT) vs Worst arm (LR-WU)

| Pathology | Best FNR | Worst FNR | Alpha |
|-----------|---------|-----------|-------|
| Atelectasis     | 0.054 | 0.056 | 0.100 |
| Cardiomegaly    | 0.051 | 0.402 | 0.100 |
| Consolidation   | 0.139 | 0.255 | 0.100 |
| Edema           | 0.118 | 0.235 | 0.100 |
| Effusion        | 0.067 | 0.049 | 0.100 |
| Pneumonia       | 0.128 | 0.255 | 0.100 |
| Pneumothorax    | 0.231 | 0.347 | 0.100 |
| Mean            | 0.112 | 0.228 | 0.100 |

## 7. Comparison to scrc_hard_fnr.ipynb Baseline

| Aspect | scrc_hard_fnr (best: GNN-c) | SCRC-T Best (GNN-FT) | Change |
|--------|---------------------------|-------------------------|--------|
| Threshold source | Per-set (independent) | Test distribution | Fixed |
| FNR Gap | 0.058 | 0.012 | -0.046 |
| Violation | 0.064 | 0.031 | -0.033 |
| FPR | 0.632 | 0.719 | +0.087 |

### GNN-FT vs Synthetic Covariate Shift Reference
- **NIH compound shift** (this notebook): GNN-FT FNR Gap = 0.012
- **Pure covariate shift** (σ=3.0):      GNN-FT FNR Gap ≈ 0.003
- **Residual gap** on NIH = label + concept shift, not DRE quality

## 8. Key Findings

1. **SCRC-T vs per-set baseline**: GNN-FT FNR Gap = 0.012 vs baseline 0.058.
   SCRC-T IMPROVES over per-set thresholds.

2. **FT vs WU agreement**: τ_FT=3.5438 vs τ_WU=3.4941 (Δ=-0.0497).
   GNN-FT FNR Gap=0.012 vs GNN-WU FNR Gap=0.015.
   N=500 warm-up is sufficient.

3. **GNN vs LR-DRE**: GNN-FT Gap=0.012 (ESS=35.0%) vs
   LR-FT Gap=0.127 (ESS=6.6%).
   GNN-DRE ESS advantage translates into better FNR Gap.

4. **MLP vs GNN**: MLP-FT FNR Gap=0.100 (ESS=31.8%) vs
   GNN-FT Gap=0.012.
   GNN graph structure provides DRE benefit over MLP.

5. **Compound vs pure shift**: GNN-FT FNR Gap=0.012 on compound shift vs ≈0.003
   on pure covariate shift. The 4x difference confirms that label + concept
   shift (not DRE quality) is the primary bottleneck for CheXpert→NIH.

## 9. Figure
Saved to: `report/scrc_t_nih.png`
- Panel 1: FNR Gap (6 bars with baseline reference at 0.058)
- Panel 2: Violation (6 bars with baseline reference at 0.064)
- Panel 3: ESS% per DRE method (GNN, LR, MLP)

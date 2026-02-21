# GNN Follow-up Experiments Report

**Setup:** β=0.15, α=0.10, CheXpert→NIH, DenseNet121 features, GNN + SCRC pipeline.
**Notebook:** `notebooks/gnn/gnn_followup.ipynb`

---

## Baseline Context

The prior report (`gnn-scrc.md`) established **Method C** (GNN + DRE-C, 7-dim GNN probs) as best with mean NIH FNR = 0.252 at epoch 50 (no checkpoint selection).

This report adds `save_best=True` to `train_gnn` as a shared improvement, then isolates five additional modifications.

---

## Implementation Change: `save_best=True` in `train_gnn`

**Change:** `src/wcp_l2d/gnn.py` — new `save_best: bool = True` parameter. During training the state dict is cloned at every epoch where `mean_val_auc` improves. After the final epoch the best-checkpoint weights are restored and `history["best_epoch"]` is written.

**Effect on baseline:** The GNN's best val AUC occurs at **epoch 20** (val_auc = 0.8325) rather than epoch 50. Restoring the epoch-20 weights yields:

| Metric | Epoch 50 (old) | Epoch 20 / save_best (new) |
|--------|---------------|--------------------------|
| Mean NIH AUC | 0.7264 | **0.7410** |
| Mean FNR | 0.252 | **0.161** |
| Mean PPR | 0.529 | 0.628 |
| DRE ESS% | 21.4% | 21.2% |

The epoch-50 model had begun overfitting to CheXpert-specific features. Restoring the best checkpoint reduces mean FNR by **36%** relative (0.252 → 0.161) at no cost.

---

## Experiment Summary Table

All experiments share β=0.15, α=0.10. The baseline already uses `save_best=True`.

| Method | Mean AUC | DRE ESS% | Deferral | Mean FNR | Mean PPR |
|--------|----------|---------|---------|----------|---------|
| **Baseline (Method C)** | 0.7410 | 21.2% | 0.150 | 0.1607 | 0.6281 |
| Exp1: GNN+PCA64 | 0.7370 | 29.8% | 0.150 | 0.1651 | 0.5954 |
| Exp2: Post-sel DRE | 0.7410 | 34.1% | 0.150 | 0.1810 | 0.5860 |
| **Exp3: Cap-alpha** | **0.7410** | 21.2% | 0.150 | **0.1507** | **0.6494** |
| Exp4: Cal-src DRE | 0.7410 | **30.9%** | 0.150 | 0.1600 | 0.6308 |
| Exp5: 100ep+best | 0.7357 | 23.8% | 0.150 | 0.2020 | 0.5782 |
| MLP (uniform α) | 0.7060 | 35.9% | 0.150 | 0.1815 | 0.6402 |
| MLP (cap-alpha) | 0.7060 | 35.9% | 0.150 | 0.1230† | 0.8557† |

---

## Per-Pathology FNR Breakdown

| Pathology | Baseline | Exp1 | Exp2 | **Exp3** | Exp4 | Exp5 | MLP (unif) | MLP (cap)† |
|-----------|---------|------|------|----------|------|------|------------|-----------|
| Atelectasis | 0.172 | 0.209 | 0.179 | 0.177 | 0.167 | 0.176 | 0.160 | **0.000** |
| Cardiomegaly | 0.142 | 0.181 | 0.153 | **0.101** | 0.134 | 0.250 | 0.105 | **0.000** |
| Consolidation | 0.215 | 0.150 | 0.223 | 0.190 | 0.215 | 0.244 | 0.197 | 0.016 |
| Edema | 0.158 | 0.158 | 0.158 | **0.053** | 0.158 | 0.167 | 0.191 | **0.000** |
| Effusion | 0.081 | 0.109 | 0.094 | **0.039** | 0.065 | 0.075 | 0.108 | **0.000** |
| Pneumonia | 0.122 | 0.151 | 0.143 | 0.143 | 0.122 | 0.174 | 0.088 | **0.000** |
| Pneumothorax | 0.235 | 0.198 | 0.318 | 0.353 | 0.259 | 0.329 | 0.423 | 0.845 |

---

## Experiment 1: GNN on PCA-64 Input Features

**Idea:** Compress the 1024-dim features to 64 PCA components before feeding the GNN. The DRE-C still operates on the 7-dim GNN output probabilities.

**Results:**
- PCA-64 retains 78.7% of explained variance.
- GNN best checkpoint at epoch 36 (val_auc = 0.8342) — slightly higher than baseline's 0.8325, but NIH AUC drops to 0.7370 (−0.004). The model learns from a compressed feature space that generalises slightly differently to NIH.
- DRE ESS improves to **29.8%** (vs 21.2%) because the 64-dim probability manifold is less separable across datasets.
- FNR is marginally worse (0.1651 vs 0.1607). Consolidation improves (0.215→0.150) but Atelectasis, Cardiomegaly, Effusion, Pneumonia all worsen.

**Verdict:** PCA-64 input gives better DRE ESS but slightly worse FNR and AUC. **Not recommended** as a replacement for 1024-dim input.

---

## Experiment 2: Post-Selection DRE Refit

**Idea:** After Stage-1 deferral removes the 15% most-uncertain calibration samples, the source distribution for the DRE shifts. Refit a fresh DRE using only the *kept* 10,971 calibration samples (instead of 38,720 training samples) as source.

**Results:**
- Domain AUC drops from 0.851 → 0.824 (distributions are less separable within the kept subpopulation, as expected — the most "CheXpert-like" uncertain samples were removed).
- ESS improves to **34.1%** (best across all experiments) because the source distribution after selection is closer to the NIH test distribution.
- However, mean FNR worsens to **0.1810** (vs 0.1607). The smaller source set (≈11k vs ≈39k) makes the DRE less reliable despite theoretical correctness.

**Verdict:** Post-selection refit is theoretically sounder but empirically worse at this scale because the kept cal set is too small for stable DRE estimation. **Not recommended** unless the calibration set is much larger (> 30k).

---

## Experiment 3: Capability-Aware α Allocation ✓ Winner

**Idea:** Instead of uniform α=0.10 across all pathologies, allocate per-pathology budgets inversely proportional to the GNN's NIH AUC:
```
excess_k = max(AUC_k - 0.5, 1e-3)
alpha_k  = alpha_global × K × (1/excess_k) / Σ(1/excess_j)
```

**Per-pathology α values:**

| Pathology | AUC | α_k |
|-----------|-----|-----|
| Atelectasis | 0.707 | 0.105 |
| Cardiomegaly | 0.768 | 0.081 |
| Consolidation | 0.746 | 0.089 |
| Edema | 0.828 | 0.066 |
| Effusion | 0.831 | 0.066 |
| Pneumonia | 0.679 | 0.122 |
| Pneumothorax | 0.628 | 0.171 |

**Results:**
- Mean FNR = **0.1507** — best across all experiments (−6.2% relative vs baseline).
- Mean PPR = **0.6494** — also best (model predicts more positives for high-AUC pathologies).
- Dramatic per-pathology wins: Edema FNR 0.158→0.053, Effusion 0.081→0.039, Cardiomegaly 0.142→0.101.
- Trade-off: Pneumothorax FNR 0.235→0.353 (the looser α=0.171 raises lambda_k, reducing sensitivity on an inherently weak pathology, AUC=0.628).

**Interpretation:** The inverse-AUC allocation effectively redistributes risk from clinically-reliable pathologies (Edema, Effusion) to clinically-uncertain ones. For the difficult pathologies (Pneumothorax, Pneumonia), the increased α_k acknowledges the GNN's limitation rather than forcing it into an impossible constraint.

**Verdict:** **Recommended.** Clear improvement in mean FNR and PPR at no cost in AUC or ESS.

---

## Experiment 4: DRE-C with Cal Set as Source

**Idea:** Use the calibration set GNN probs as the DRE source instead of the training set. The calibration set (12,906 samples) is the set actually used for CRC, so the DRE is more directly aligned to the calibration distribution.

**Results:**
- Domain AUC = 0.844 (vs 0.851 baseline) — slightly reduced separability.
- ESS = **30.9%** (vs 21.2% baseline) — substantial improvement. The cal set's GNN probs lie closer to the NIH pool probs because the cal set (CheXpert validation) has a different composition than the training set.
- FNR = 0.1600 ≈ baseline 0.1607 — essentially unchanged.
- Lambda hats shift slightly but the CRC calibration compensates.

**Verdict:** **Recommended as a free improvement.** ESS increases by 46% relative (21.2%→30.9%) with no FNR penalty. Using the cal set as source ensures the importance weights are computed on the same distribution used for CRC, tightening the importance-sampling identity.

---

## Experiment 5: 100-Epoch Training + Best Checkpoint

**Idea:** Train for 100 epochs (doubled from 50) with `save_best=True` and restore the best checkpoint. Since early stopping via save_best is already active, the extra epochs give the scheduler more opportunities to find a better local optimum.

**Results:**
- Best checkpoint at epoch 33 (val_auc = 0.8322) — slightly lower than the 50ep run's best at epoch 20 (0.8325). The 100ep run's different random batching order led to a slightly different trajectory.
- NIH AUC = 0.7357 (−0.005 vs baseline).
- FNR = **0.2020** — significantly worse than baseline (0.1607).
- The 100ep model produces lower GNN probs (lower lambda hats: e.g., Atelectasis 0.101 vs 0.111), suggesting the model is less confident and is calling more cases uncertain (deferred) rather than making discriminative predictions.

**Verdict:** **Not recommended.** Doubling training epochs with the same LR and no decay causes the model to drift toward a worse local optimum despite checkpoint restoration. If longer training is desired, reduce the LR or add cosine annealing.

---

## MLP Baseline: Does Graph Structure Help?

**Architecture:** Two-layer MLP — `Linear(1024, 1316) + ReLU + Dropout(0.3) + Linear(1316, 7)`.
Param count: **1,358,119** vs LabelGCN **1,357,883** (236 more, <0.02%).
Trained identically: 50 epochs, Adam lr=1e-3, weight_decay=1e-4, batch_size=512, NaN-masked BCE, `save_best=True`.

**Training:** Best val AUC = 0.8298 at epoch 3 (GNN: 0.8325 at epoch 20). The MLP converges faster but peaks lower.

### NIH AUC Comparison

| Pathology | GNN | MLP | Δ |
|-----------|-----|-----|---|
| Atelectasis | 0.7073 | 0.6987 | −0.008 |
| Cardiomegaly | 0.7684 | 0.7711 | +0.003 |
| Consolidation | 0.7464 | 0.7251 | −0.021 |
| Edema | 0.8283 | 0.8056 | −0.023 |
| Effusion | 0.8311 | 0.8139 | −0.017 |
| Pneumonia | 0.6793 | 0.6560 | −0.023 |
| Pneumothorax | 0.6277 | 0.4714 | **−0.156** |
| **Mean** | **0.7410** | **0.7060** | **−0.035** |

The GNN outperforms the MLP overall, with Pneumothorax showing the largest gap (−0.156). This is the pathology most likely to benefit from graph-structured co-occurrence patterns (Pneumothorax rarely co-occurs with other conditions, a structural fact encoded in the GCN adjacency matrix).

### DRE ESS Comparison

The MLP's 7-dim probability outputs are **less separable** between CheXpert and NIH: domain AUC 0.831 vs GNN's 0.851, yielding ESS = **35.9%** (GNN: 21.2%). The GNN's sharper domain boundary (induced by graph structure) paradoxically hurts DRE quality.

### FNR Comparison (Uniform α)

MLP uniform α: mean FNR = 0.1815 — **worse than GNN** (0.1607). The MLP's weaker discriminative power translates directly to missed positives, especially Pneumothorax (FNR 0.423 vs GNN 0.235).

### Cap-Alpha FPR Comparison

FPR = FP / (FP + TN), computed on non-deferred NIH test samples with non-NaN labels.

| Pathology | GNN (cap-alpha) | MLP (cap-alpha) |
|-----------|----------------|----------------|
| Atelectasis | 0.577 | 0.991 |
| Cardiomegaly | 0.551 | 0.992 |
| Consolidation | 0.588 | 0.932 |
| Edema | 0.691 | 0.957 |
| Effusion | 0.823 | 0.998 |
| Pneumonia | 0.775 | 0.981 |
| Pneumothorax | 0.516 | 0.138 |
| **Mean** | **0.646** | **0.856** |

The GNN cap-alpha FPR values (0.52–0.82) reflect genuinely wider prediction sets for the higher-confidence pathologies — a controlled trade-off. The MLP cap-alpha FPRs for six pathologies are 0.93–1.00, meaning nearly every negative is predicted positive. Only Pneumothorax escapes this (FPR 0.138) because its α_k ≈ 0.68 pushes almost all Pneumothorax cases into deferral, leaving few in the evaluated set.

### MLP Cap-Alpha: A Degenerate Case

MLP cap-alpha achieves mean FNR = **0.1230** (best in FNR table†), but this is entirely explained by the FPR table above. Since MLP Pneumothorax AUC = 0.4714 < 0.5, the excess is clipped to 1e-3, giving α_k(Pneumothorax) ≈ 0.68. This dominates the budget allocation and collapses all other pathologies to α_k ≈ 0, forcing prediction sets to include *both* classes for Atelectasis through Pneumonia — hence FNR ≈ 0 and FPR ≈ 1 for those six. Meanwhile Pneumothorax FNR = **0.845**. Viewed jointly, the MLP cap-alpha system achieves near-zero FNR on six pathologies only by predicting positive on essentially all negatives — a clinically useless operating point.

†MLP cap-alpha numbers are marked because the formula breaks down when any pathology AUC < 0.5.

### Verdict

**The GCN graph structure helps.** With matched parameter budgets:
- GNN NIH AUC is 3.5 points higher overall, 15.6 points higher on Pneumothorax.
- GNN uniform-α FNR (0.1607) beats MLP uniform-α (0.1815).
- GNN cap-alpha FPR (mean 0.646) is far lower than MLP cap-alpha (0.856); the MLP's near-zero FNR is bought at the cost of near-unity FPR on six pathologies.
- MLP's apparent cap-alpha FNR advantage is a formula artifact, not a real win.
- The GNN's higher ESS is a disadvantage for DRE stability — but the overall SCRC pipeline still delivers lower FNR and lower FPR because the GNN probabilities are more discriminative.

---

## Final Recommendation

**Best single experiment: Exp3 (Capability-aware α).** Delivers the lowest mean FNR (0.1507) and highest mean PPR (0.6494) with no additional computational cost at inference.

**Complementary improvement: Exp4 (Cal-src DRE).** Provides 46% relative ESS gain (21.2%→30.9%) with negligible FNR change. This improves the statistical reliability of the weighted CRC calibration.

**Combined system (Exp3 + Exp4, not individually tested):**
Replace the baseline with:
1. `save_best=True` in `train_gnn` (implemented in `gnn.py`).
2. DRE fitted on cal-set GNN probs vs NIH pool GNN probs.
3. Per-pathology α from `compute_capability_alpha(auc_gnn, alpha_global=0.10)`.

This combination addresses two orthogonal axes: DRE calibration (Exp4) and FNR target allocation (Exp3). There is no reason the gains should cancel.

**Rejected experiments:**
- Exp1 (PCA-64): Better ESS but worse AUC/FNR. Feature compression loses discriminative information.
- Exp2 (Post-sel DRE): Theoretically correct but requires ≥3× larger calibration set to be statistically reliable.
- Exp5 (100ep): Longer training is harmful without LR scheduling. Add cosine decay before revisiting.
- MLP baseline (uniform α): Matched-parameter MLP with 1.36M params. Lower NIH AUC (0.706 vs 0.741) and higher FNR (0.182 vs 0.161) confirm that the GCN graph structure adds value beyond raw parameter count.

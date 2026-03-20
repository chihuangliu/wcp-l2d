# Post-hoc Multi-label Learning to Defer under Domain Shift with Graph Neural Network

This project proposes a post-hoc Learning to Defer (L2D) system for multi-label chest X-ray classification that controls deferral rate and false negative rate (FNR) under domain shift. The system uses a frozen pre-trained DenseNet121 backbone and a two-stage Selective Conformal Risk Control (SCRC) framework, with GNN-enhanced Density Ratio Estimation (DRE) to handle covariate shift.

## Key Contributions

1. **GNN-enhanced DRE**: Uses Label-GCN (ML-GCN) output probabilities as the feature space for density ratio estimation, exploiting co-occurrence structure to achieve higher Effective Sample Size (ESS) than LR or MLP baselines.
2. **Warm-up SCRC**: Extends SCRC with a small warm-up set (N=500) from the target domain to estimate safe deferral thresholds without requiring labeled target data.
3. **Deferral bottleneck analysis**: Demonstrates that binary Weighted Conformal Prediction (WCP) produces an all-or-nothing deferral collapse, and that SCRC resolves this.

## Datasets

- **CheXpert** (source): 64,534 frontal chest X-rays, multi-label with NaN for uncertain findings.
- **NIH ChestX-ray14** (target): 30,805 frontal chest X-rays after filtering, binary labels for 14 pathologies. 7 pathologies shared with CheXpert.

Features are extracted from the global average pooling layer of a pre-trained DenseNet121 (`densenet121-res224-chex`) via `torchxrayvision`, producing 1024-dimensional vectors.

## Experiments

### Deferral Bottleneck (Binary WCP)

Demonstrates the all-or-nothing collapse of standard binary WCP.

| Notebook | Description |
|---|---|
| `notebooks/wcp_experiment.ipynb` | Binary WCP on NIH test set; deferral rate vs. confidence level plot |

### Experiment A: Synthetic Pure Covariate Shift

CheXpert 60/40 split; target re-extracted with Gaussian blur (σ ∈ {1.0, 2.0, 3.0}).

| Notebook | Description |
|---|---|
| `notebooks/pure_cov_shift/synthetic_covariate_shift_scrc_gnn.ipynb` | SCRC (warm-up) variant for GNN |
| `notebooks/pure_cov_shift/pure_cov_shift_wcp.ipynb` | Binary WCP comparison under synthetic shift |

Sigma sweep runner: `scripts/run_pure_cov_shift_scrc_t.py <sigma>`

### Experiment B: Real Domain Shift (CheXpert → NIH)

Full CheXpert as source; NIH ChestX-ray14 as target, with compound covariate + MNAR label shift.

| Notebook | Description |
|---|---|
| `notebooks/gnn/scrc_t_nih.ipynb` | Main SCRC 6-arm experiment (LR/MLP/GNN × standard/warm-up) on NIH |
| `notebooks/gnn/scrc_hard_fnr.ipynb` | SCRC with per-set thresholds (reference baseline) |
| `notebooks/gnn/gnn_dre_wcp.ipynb` | GNN-DRE applied to binary WCP on NIH |

## Project Structure

```
src/wcp_l2d/
  data.py              # Dataset loading (CheXpert, NIH)
  features.py          # ExtractedFeatures dataclass
  label_utils.py       # Multi-label → single-label + binary extraction
  dre.py               # AdaptiveDRE (PCA + Platt scaling + weight clipping)
  gnn.py               # LabelGCN (ML-GCN), train_gnn, build_adjacency_matrix
  conformal.py         # ConformalPredictor + WeightedConformalPredictor (RAPS)
  scrc.py              # SCRCPredictor, per-pathology CRC calibration
  scrc_evaluation.py   # SCRC evaluation framework
  evaluation.py        # System accuracy, coverage, deferral plots
  expert.py            # SimulatedExpert
  pathologies.py       # COMMON_PATHOLOGIES (7 shared pathologies)
scripts/
  extract_perturbed_features.py   # Extract blurred CheXpert target features
  run_pure_cov_shift_scrc_t.py    # Sigma sweep runner for Experiment A
notebooks/
  wcp_experiment.ipynb            # Binary WCP deferral bottleneck
  gnn/                            # Experiment B notebooks (CheXpert → NIH)
  pure_cov_shift/                 # Experiment A notebooks (synthetic shift)
  multilabel/                     # Exploratory multilabel CP experiments
report/                           # Generated figures and markdown reports
```

## Environment

- Python 3.12, managed with `uv`
- Key dependencies: PyTorch 2.10, torchxrayvision 1.4, torchcp 1.2.1, scikit-learn 1.8, umap-learn 0.5.11

```bash
uv sync --prerelease=allow
```

Jupyter kernel registered as `wcp-l2d`. To execute a notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=wcp-l2d \
  --ExecutePreprocessor.timeout=900 \
  --output /abs/path/output.ipynb /abs/path/input.ipynb
```

#!/usr/bin/env python3
"""Insert MLP baseline cells into gnn_followup.ipynb and update summary."""

from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "gnn" / "gnn_followup.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

# ---------------------------------------------------------------------------
# New cell: markdown header
# ---------------------------------------------------------------------------

MLP_MD = """\
## MLP Baseline: Matched-Parameter Comparison

Two-layer MLP with H=1316 hidden units (~1.36M params, matching LabelGCN).
Trained identically (50 ep, Adam lr=1e-3, save_best=True). No graph structure, no init_logits residual.
Evaluated with: (a) uniform α=0.10 and (b) capability-aware alpha (same formula as Exp3).\
"""

# ---------------------------------------------------------------------------
# New cell: MLP class + train function + experiment
# ---------------------------------------------------------------------------

MLP_CODE = """\
import copy
import torch.nn as nn

# ---- MLP architecture ----
# param count: 1024*1316 + 1316 + 1316*7 + 7 = 1,358,119  (matches LabelGCN 1,357,883)
class MLP(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=1316, K=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, x):
        return self.net(x)   # raw logits [N, K]


def train_mlp(
    features_train, labels_train,
    features_val,   labels_val,
    feat_dim=1024, hidden_dim=1316, K=7, dropout=0.3,
    epochs=50, batch_size=512,
    lr=1e-3, weight_decay=1e-4,
    device="cpu", seed=42, verbose=False,
    save_best=True,
):
    \"\"\"Train a two-layer MLP with NaN-masked BCE loss. Mirrors train_gnn API.\"\"\"
    torch.manual_seed(seed)
    model = MLP(feat_dim=feat_dim, hidden_dim=hidden_dim, K=K, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Xtr = torch.tensor(features_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(np.where(np.isnan(labels_train), -1.0, labels_train),
                       dtype=torch.float32).to(device)
    Xval = torch.tensor(features_val, dtype=torch.float32).to(device)
    Yval_np = labels_val   # keep as numpy for roc_auc_score

    N = Xtr.shape[0]
    history = {"train_loss": [], "val_auc": [], "best_epoch": []}
    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            logits = model(xb)
            valid_mask = (yb >= 0).float()
            yb_safe = yb.clamp(min=0)
            loss_raw = bce(logits, yb_safe)
            valid_count = valid_mask.sum().clamp(min=1)
            loss = (loss_raw * valid_mask).sum() / valid_count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(epoch_loss / n_batches)

        # Validation AUC (NaN-masked)
        model.eval()
        with torch.no_grad():
            val_logits = model(Xval).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))   # sigmoid
        aucs = []
        for k in range(K):
            valid = ~np.isnan(Yval_np[:, k])
            if valid.sum() < 10 or len(np.unique(Yval_np[valid, k])) < 2:
                continue
            aucs.append(roc_auc_score(Yval_np[valid, k], val_probs[valid, k]))
        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        history["val_auc"].append(mean_auc)

        if save_best and mean_auc > best_val_auc:
            best_val_auc = mean_auc
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = [epoch]

        if verbose:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/n_batches:.4f}  val_auc={mean_auc:.4f}")

    if save_best and best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history


# ---- Verify parameter count ----
_mlp_tmp = MLP()
_n_params = sum(p.numel() for p in _mlp_tmp.parameters())
print(f"MLP param count: {_n_params:,}  (LabelGCN: 1,357,883)")
del _mlp_tmp

# ---- Train MLP ----
print("Training MLP (50 epochs, save_best=True)...")
mlp_base, hist_mlp = train_mlp(
    features_train=Xtr_s,
    labels_train=Y_train,
    features_val=Xcal_s,
    labels_val=Y_cal,
    epochs=50,
    save_best=True,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    device=DEVICE,
    seed=SEED,
    verbose=False,
)
print(f"Best val AUC: {max(hist_mlp['val_auc']):.4f}  at epoch {hist_mlp['best_epoch'][0]}")

# ---- MLP probabilities ----
def mlp_predict_probs(model, features_np):
    model.eval()
    X_t = torch.tensor(features_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t).numpy()
    return 1.0 / (1.0 + np.exp(-logits))

probs_train_mlp = mlp_predict_probs(mlp_base, Xtr_s)
probs_cal_mlp   = mlp_predict_probs(mlp_base, Xcal_s)
probs_nih_mlp   = mlp_predict_probs(mlp_base, Xnih_s)
probs_pool_mlp  = mlp_predict_probs(mlp_base, Xpool_s)
print(f"probs_nih_mlp shape: {probs_nih_mlp.shape}  range: [{probs_nih_mlp.min():.4f}, {probs_nih_mlp.max():.4f}]")

# ---- NIH AUC ----
auc_mlp = nih_auc_per_pathology(probs_nih_mlp)
print(f"\\nMLP NIH AUC per pathology:")
for path, auc in zip(COMMON_PATHOLOGIES, auc_mlp):
    print(f"  {path:<16}  {auc:.4f}")
print(f"Mean NIH AUC: {np.nanmean(auc_mlp):.4f}  (GNN baseline: {np.nanmean(auc_gnn_base):.4f})")

# ---- DRE-C on 7-dim MLP probs (train-as-source) ----
dre_mlp = AdaptiveDRE(n_components=None, weight_clip=None, random_state=SEED)
dre_mlp.fit(probs_train_mlp, probs_pool_mlp)
w_cal_mlp = dre_mlp.compute_weights(probs_cal_mlp)
diag_mlp  = dre_mlp.diagnostics(probs_cal_mlp)
print(f"\\nMLP DRE-C: domain_auc={diag_mlp.domain_auc:.4f}  "
      f"ESS={diag_mlp.ess:.1f}  ESS%={diag_mlp.ess_fraction*100:.1f}%")
print(f"Baseline GNN DRE-C: domain_auc={diag_base.domain_auc:.4f}  ESS%={diag_base.ess_fraction*100:.1f}%")

# ---- (a) Uniform alpha ----
scrc_mlp = PerPathologySCRCPredictor(beta=BETA, alpha=ALPHA, seed=SEED)
crc_mlp  = scrc_mlp.calibrate(probs_cal_mlp, Y_cal, w_cal_mlp, pathology_names=COMMON_PATHOLOGIES)
result_mlp = scrc_mlp.predict(probs_nih_mlp)
m_mlp = compute_empirical_metrics(result_mlp.prediction_sets, result_mlp.defer_mask, Y_nih_test)

# ---- (b) Capability-aware alpha (same formula as Exp3) ----
alpha_k_mlp = compute_capability_alpha(
    nih_aucs=auc_mlp,
    alpha_global=ALPHA,
    pathology_names=COMMON_PATHOLOGIES,
)
scrc_mlp_cap = PerPathologySCRCPredictor(beta=BETA, alpha=alpha_k_mlp, seed=SEED)
crc_mlp_cap  = scrc_mlp_cap.calibrate(probs_cal_mlp, Y_cal, w_cal_mlp, pathology_names=COMMON_PATHOLOGIES)
result_mlp_cap = scrc_mlp_cap.predict(probs_nih_mlp)
m_mlp_cap = compute_empirical_metrics(result_mlp_cap.prediction_sets, result_mlp_cap.defer_mask, Y_nih_test)

print(f"\\n=== MLP Baseline Results ===")
print(f"{'':20s}  {'AUC':>6}  {'ESS%':>6}  {'Defer':>6}  {'FNR':>6}  {'PPR':>6}")
print(f"{'MLP (uniform α)':20s}  {np.nanmean(auc_mlp):.4f}  "
      f"{diag_mlp.ess_fraction*100:5.1f}%  {result_mlp.deferral_rate:.3f}  "
      f"{m_mlp['empirical_FNR'].dropna().mean():.4f}  {m_mlp['PPR'].dropna().mean():.4f}")
print(f"{'MLP (cap-alpha)':20s}  {np.nanmean(auc_mlp):.4f}  "
      f"{diag_mlp.ess_fraction*100:5.1f}%  {result_mlp_cap.deferral_rate:.3f}  "
      f"{m_mlp_cap['empirical_FNR'].dropna().mean():.4f}  {m_mlp_cap['PPR'].dropna().mean():.4f}")
print(f"{'GNN (uniform α)':20s}  {np.nanmean(auc_gnn_base):.4f}  "
      f"{diag_base.ess_fraction*100:5.1f}%  {result_base.deferral_rate:.3f}  "
      f"{m_base['empirical_FNR'].dropna().mean():.4f}  {m_base['PPR'].dropna().mean():.4f}")
print(f"{'GNN (cap-alpha)':20s}  {np.nanmean(auc_gnn_base):.4f}  "
      f"{diag_base.ess_fraction*100:5.1f}%  {result_e3.deferral_rate:.3f}  "
      f"{m_e3['empirical_FNR'].dropna().mean():.4f}  {m_e3['PPR'].dropna().mean():.4f}")

print(f"\\nPer-pathology FNR — MLP(unif) / MLP(cap) / GNN(unif) / GNN(cap):")
for k, path in enumerate(COMMON_PATHOLOGIES):
    print(f"  {path:<16}  "
          f"mlp_u={m_mlp['empirical_FNR'].iloc[k]:.4f}  "
          f"mlp_c={m_mlp_cap['empirical_FNR'].iloc[k]:.4f}  "
          f"gnn_u={m_base['empirical_FNR'].iloc[k]:.4f}  "
          f"gnn_c={m_e3['empirical_FNR'].iloc[k]:.4f}")\
"""

# ---------------------------------------------------------------------------
# Updated summary cell
# ---------------------------------------------------------------------------

SUMMARY_UPDATED = """\
# ---- Build summary DataFrame ----
rows = []

for label, auc_arr, diag, result, m in [
    ("Baseline (Method C)",  auc_gnn_base, diag_base, result_base,  m_base),
    ("Exp1: GNN+PCA64",      auc_gnn_e1,   diag_e1,   result_e1,    m_e1),
    ("Exp2: Post-sel DRE",   auc_gnn_base, diag_e2,   None,         m_e2),
    ("Exp3: Cap-alpha",      auc_gnn_base, diag_base, result_e3,    m_e3),
    ("Exp4: Cal-src DRE",    auc_gnn_base, diag_e4,   result_e4,    m_e4),
    ("Exp5: 100ep+best",     auc_gnn_e5,   diag_e5,   result_e5,    m_e5),
    ("Exp3+4: Combined",     auc_gnn_base, diag_e4,   result_comb,  m_comb),
    ("MLP (uniform α)",      auc_mlp,      diag_mlp,  result_mlp,   m_mlp),
    ("MLP (cap-alpha)",      auc_mlp,      diag_mlp,  result_mlp_cap, m_mlp_cap),
]:
    if result is not None:
        defer_rate = result.deferral_rate
    else:
        defer_rate = defer_nih_e2.mean()

    rows.append({
        "Method":    label,
        "Mean AUC":  round(float(np.nanmean(auc_arr)), 4),
        "DRE ESS%":  round(float(diag.ess_fraction * 100), 1),
        "Deferral":  round(defer_rate, 3),
        "Mean FNR":  round(float(m["empirical_FNR"].dropna().mean()), 4),
        "Mean PPR":  round(float(m["PPR"].dropna().mean()), 4),
    })

summary_df = pd.DataFrame(rows).set_index("Method")
print("=== Full Experiment Comparison (β=0.15, α=0.10) ===")
print(summary_df.to_string())

best_fnr_method = summary_df["Mean FNR"].idxmin()
best_auc_method = summary_df["Mean AUC"].idxmax()
best_ess_method = summary_df["DRE ESS%"].idxmax()
print(f"\\nBest Mean FNR:  {best_fnr_method}  ({summary_df.loc[best_fnr_method, 'Mean FNR']:.4f})")
print(f"Best Mean AUC:  {best_auc_method}  ({summary_df.loc[best_auc_method, 'Mean AUC']:.4f})")
print(f"Best DRE ESS%:  {best_ess_method}  ({summary_df.loc[best_ess_method, 'DRE ESS%']:.1f}%)")

print("\\n=== Per-pathology FNR breakdown ===")
fnr_detail = pd.DataFrame({
    "Baseline":     m_base["empirical_FNR"],
    "Exp3 CapAlp":  m_e3["empirical_FNR"],
    "Exp4 CalSrc":  m_e4["empirical_FNR"],
    "Exp3+4 Comb":  m_comb["empirical_FNR"],
    "MLP Unif":     m_mlp["empirical_FNR"],
    "MLP CapAlp":   m_mlp_cap["empirical_FNR"],
})
print(fnr_detail.round(4).to_string())\
"""

# ---------------------------------------------------------------------------
# Locate insertion point and summary cell
# ---------------------------------------------------------------------------

summary_md_idx = None
summary_code_idx = None
for i, cell in enumerate(nb.cells):
    src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    if cell["cell_type"] == "markdown" and "Summary" in src and "All Experiments" in src:
        summary_md_idx = i
    if cell["cell_type"] == "code" and "Build summary DataFrame" in src:
        summary_code_idx = i

assert summary_md_idx is not None, "Could not find summary markdown cell"
assert summary_code_idx is not None, "Could not find summary code cell"
print(f"Summary markdown at index {summary_md_idx}, code at {summary_code_idx}")

# Insert MLP cells right before the summary markdown cell
new_cells = [
    nbformat.v4.new_markdown_cell(MLP_MD),
    nbformat.v4.new_code_cell(MLP_CODE),
]
for offset, cell in enumerate(new_cells):
    nb.cells.insert(summary_md_idx + offset, cell)

# The summary code cell index shifted by len(new_cells)
summary_code_idx += len(new_cells)

# Replace the summary code cell source
nb.cells[summary_code_idx]["source"] = SUMMARY_UPDATED
nb.cells[summary_code_idx]["outputs"] = []
nb.cells[summary_code_idx]["execution_count"] = None

# Clear outputs of new cells
for cell in new_cells:
    cell["outputs"] = []
    cell["execution_count"] = None

nbformat.write(nb, NB_PATH)
print(f"Updated {NB_PATH}  ({len(nb.cells)} cells total)")

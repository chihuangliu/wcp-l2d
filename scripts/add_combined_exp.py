#!/usr/bin/env python3
"""Insert Exp3+Exp4 combined cells into gnn_followup.ipynb and update summary."""

from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "gnn" / "gnn_followup.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

# ---------------------------------------------------------------------------
# New cells to insert
# ---------------------------------------------------------------------------

COMBINED_MD = """\
## Combined: Exp3 (Cap-alpha) + Exp4 (Cal-src DRE)

Both Exp3 and Exp4 are orthogonal improvements: one changes the α allocation
(how much FNR budget each pathology gets), the other changes the DRE source
(which CheXpert samples are used to estimate importance weights).
Here we apply them simultaneously:\
"""

COMBINED_CODE = """\
print("=== Combined: Exp3 (Cap-alpha) + Exp4 (Cal-src DRE) ===")
print(f"  alpha_k from Exp3: {dict(zip(COMMON_PATHOLOGIES, alpha_k.round(4)))}")
print(f"  DRE source: cal set ({len(probs_cal_gnn)} samples)  ESS%={diag_e4.ess_fraction*100:.1f}%")

scrc_comb = PerPathologySCRCPredictor(beta=BETA, alpha=alpha_k, seed=SEED)
crc_comb  = scrc_comb.calibrate(
    probs_cal_gnn, Y_cal, w_cal_e4, pathology_names=COMMON_PATHOLOGIES
)
result_comb = scrc_comb.predict(probs_nih_gnn)
m_comb = compute_empirical_metrics(result_comb.prediction_sets, result_comb.defer_mask, Y_nih_test)

print(f"\\n=== Combined Exp3+Exp4 ===")
print(f"Mean NIH AUC:  {np.nanmean(auc_gnn_base):.4f}  (same GNN as baseline)")
print(f"DRE ESS%:      {diag_e4.ess_fraction*100:.1f}%  (baseline: {diag_base.ess_fraction*100:.1f}%)")
print(f"Deferral rate: {result_comb.deferral_rate:.3f}")
print(f"Mean FNR:      {m_comb['empirical_FNR'].dropna().mean():.4f}  "
      f"(Exp3: {m_e3['empirical_FNR'].dropna().mean():.4f}  "
      f"Exp4: {m_e4['empirical_FNR'].dropna().mean():.4f}  "
      f"baseline: {m_base['empirical_FNR'].dropna().mean():.4f})")
print(f"Mean PPR:      {m_comb['PPR'].dropna().mean():.4f}")
print(f"Lambda hats:   {dict(zip(COMMON_PATHOLOGIES, crc_comb.lambda_hats.round(3)))}")
print(f"\\nPer-pathology FNR — baseline / Exp3 / Exp4 / Combined:")
for k, path in enumerate(COMMON_PATHOLOGIES):
    print(f"  {path:<16}  "
          f"base={m_base['empirical_FNR'].iloc[k]:.4f}  "
          f"e3={m_e3['empirical_FNR'].iloc[k]:.4f}  "
          f"e4={m_e4['empirical_FNR'].iloc[k]:.4f}  "
          f"comb={m_comb['empirical_FNR'].iloc[k]:.4f}")\
"""

# ---------------------------------------------------------------------------
# Updated summary cell (replaces the last code cell)
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
})
print(fnr_detail.round(4).to_string())\
"""

# ---------------------------------------------------------------------------
# Locate insertion point and summary cell
# ---------------------------------------------------------------------------

# Find the summary markdown cell (second-to-last cell currently)
# and the summary code cell (last cell)
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

# Insert combined cells right before the summary markdown cell
new_cells = [
    nbformat.v4.new_markdown_cell(COMBINED_MD),
    nbformat.v4.new_code_cell(COMBINED_CODE),
]
for offset, cell in enumerate(new_cells):
    nb.cells.insert(summary_md_idx + offset, cell)

# The summary code cell index shifted by len(new_cells)
summary_code_idx += len(new_cells)

# Replace the summary code cell source
nb.cells[summary_code_idx]["source"] = SUMMARY_UPDATED
# Clear old outputs so re-execution is clean
nb.cells[summary_code_idx]["outputs"] = []
nb.cells[summary_code_idx]["execution_count"] = None

# Also clear outputs of the new cells (they have none yet, but just in case)
for cell in new_cells:
    cell["outputs"] = []
    cell["execution_count"] = None

nbformat.write(nb, NB_PATH)
print(f"Updated {NB_PATH}  ({len(nb.cells)} cells total)")

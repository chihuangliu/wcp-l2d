"""Run synthetic_covariate_shift_scrc.ipynb for a given sigma.

Usage:
    uv run python scripts/run_synthetic_cov_shift.py <sigma>

Patches SIGMA in the notebook config cell, executes with nbconvert,
and prints the key result rows extracted from the output.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
NB_SRC    = ROOT / 'notebooks' / 'gnn' / 'synthetic_covariate_shift_scrc.ipynb'
NB_OUTDIR = ROOT / 'notebooks' / 'gnn'

SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
NB_OUT = NB_OUTDIR / f'synthetic_covariate_shift_scrc_sigma{SIGMA:.1f}_executed.ipynb'


def patch_sigma(nb: dict, sigma: float) -> dict:
    """Return a copy of the notebook with SIGMA set in cell-config."""
    import copy
    nb = copy.deepcopy(nb)
    for cell in nb['cells']:
        if cell['id'] == 'cell-config':
            cell['source'] = [
                line if 'SIGMA' not in line or 'PERTURBED' in line
                else f"SIGMA   = {sigma}    # Gaussian blur sigma\n"
                for line in cell['source']
            ]
            break
    return nb


def run(sigma: float):
    print(f'=== sigma={sigma} ===')
    nb = json.load(open(NB_SRC))
    nb = patch_sigma(nb, sigma)

    # Write patched notebook to a temp file
    tmp = ROOT / 'notebooks' / 'gnn' / f'_tmp_sigma{sigma:.1f}.ipynb'
    json.dump(nb, open(tmp, 'w'), indent=1)

    cmd = [
        'uv', 'run', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        f'--ExecutePreprocessor.kernel_name=wcp-l2d',
        f'--ExecutePreprocessor.timeout=1800',
        f'--output', str(NB_OUT),
        str(tmp),
    ]
    print('Running nbconvert ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print('FAILED:')
        print(result.stderr[-3000:])
        return None

    print('Done. Extracting results ...')
    nb_out = json.load(open(NB_OUT))
    results = {}
    for cell in nb_out['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                text += ''.join(o.get('text', []))
        if text:
            results[cell['id']] = text

    # Print key outputs
    for cid in ('cell-verify', 'cell-dre-table', 'cell-eval'):
        if cid in results:
            print(f'\n--- {cid} ---')
            print(results[cid].strip())

    return results


if __name__ == '__main__':
    run(SIGMA)

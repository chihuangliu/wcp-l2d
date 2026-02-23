"""Extract DenseNet121 features for CheXpert target split with Gaussian blur.

Usage:
    uv run python scripts/extract_perturbed_features.py [sigma]

    sigma  Gaussian blur std-dev (default 3.0)

Reads the pre-extracted CheXpert features to determine the 40% target split
(same SEED=42 as the notebook), then re-extracts those images with the given
Gaussian blur sigma and caches the result.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Subset as TorchSubset
from torchvision import transforms
from tqdm import tqdm

from wcp_l2d.features import ExtractedFeatures, load_model
from wcp_l2d.data import load_and_align_dataset, apply_xrv_transforms, make_dataloader
from wcp_l2d.pathologies import COMMON_PATHOLOGIES

# ── config ────────────────────────────────────────────────────────────────────
CHEXPERT_IMGPATH = '/Users/amo/programData/wcp-l2d/data/chexpert'
SIGMA  = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
SEED   = 42
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
FEAT_DIR = ROOT / 'data' / 'features'
CACHE    = FEAT_DIR / f'chexpert_target_perturbed_sigma{SIGMA:.1f}_features.npz'
# ─────────────────────────────────────────────────────────────────────────────


class GaussianBlurNP:
    def __init__(self, sigma): self.sigma = sigma
    def __call__(self, img):   return gaussian_filter(img, sigma=[0, self.sigma, self.sigma])


def main():
    if CACHE.exists():
        print(f'Cache already exists: {CACHE}')
        data = np.load(CACHE, allow_pickle=True)
        print(f'  features: {data["features"].shape}')
        return

    print(f'Device: {DEVICE}  sigma={SIGMA}')

    # ── determine target indices (60/40 split, SEED=42) ───────────────────────
    chex = ExtractedFeatures.load(FEAT_DIR / 'chexpert_densenet121-res224-chex_features.npz')
    rng        = np.random.RandomState(SEED)
    all_pos    = rng.permutation(len(chex.features))
    n_source   = int(0.60 * len(chex.features))
    target_pos = all_pos[n_source:]
    orig_target_idx = chex.indices[target_pos]   # original CheXpert indices
    Y_target        = chex.labels[target_pos]

    print(f'Target subset: {len(orig_target_idx):,} images')

    # ── build dataset with blur ────────────────────────────────────────────────
    print('Loading CheXpert dataset ...')
    chex_ds = load_and_align_dataset('chexpert', CHEXPERT_IMGPATH)
    apply_xrv_transforms(chex_ds)
    chex_ds.transform = transforms.Compose([chex_ds.transform, GaussianBlurNP(SIGMA)])

    tgt_subset = TorchSubset(chex_ds, orig_target_idx.tolist())
    loader     = make_dataloader(tgt_subset, batch_size=256, num_workers=0, shuffle=False)

    # ── extract features ──────────────────────────────────────────────────────
    print('Loading DenseNet121 ...')
    model = load_model(weights='densenet121-res224-chex', device=DEVICE)

    all_features = []
    t0 = time.time()
    print('Extracting ...')
    model.eval()
    with torch.no_grad():
        for imgs, labels, indices in tqdm(loader):
            imgs = imgs.to(DEVICE)
            feats = model.features2(imgs)
            all_features.append(feats.cpu().numpy())

    X_target_raw = np.concatenate(all_features, axis=0)
    elapsed = time.time() - t0
    print(f'Done: {X_target_raw.shape}  in {elapsed/60:.1f} min')

    # ── save ──────────────────────────────────────────────────────────────────
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        features=X_target_raw,
        labels=Y_target,
        indices=orig_target_idx,
        pathologies=np.array(COMMON_PATHOLOGIES),
        dataset_name=np.array('chexpert_perturbed'),
        model_weights=np.array('densenet121-res224-chex'),
    )
    print(f'Saved: {CACHE}')


if __name__ == '__main__':
    main()

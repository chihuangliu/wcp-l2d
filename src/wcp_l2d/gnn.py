"""Lightweight ML-GCN for topology-aware multi-label chest X-ray classification.

Replaces K independent binary logistic regression classifiers with a single
joint Graph Convolutional Network (ML-GCN) that leverages label co-occurrence
structure. Operates entirely on pre-extracted frozen DenseNet121 features —
no image loading or backbone retraining required.

Reference: Chen et al. (2019), "Multi-Label Image Recognition with Graph
Convolutional Networks" (ML-GCN), CVPR 2019.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def build_adjacency_matrix(labels: np.ndarray, tau: float = 0.1) -> torch.Tensor:
    """Build a row-normalised label co-occurrence adjacency matrix.

    Entry A[i, j] estimates P(pathology j present | pathology i present),
    computed from the training label matrix.  Entries below ``tau`` are
    zeroed out to remove noisy correlations.  Self-loops are added and the
    matrix is row-normalised before being returned.

    Args:
        labels: [N, K] float32 label matrix.  NaN entries are treated as
            missing and excluded from co-occurrence counts.
        tau: Sparsification threshold.  Conditional probabilities below
            this value are set to zero.

    Returns:
        A: [K, K] float32 tensor, row-normalised adjacency matrix.
    """
    K = labels.shape[1]
    A = np.zeros((K, K), dtype=np.float32)

    for i in range(K):
        valid_i = ~np.isnan(labels[:, i])
        pos_i = valid_i & (labels[:, i] == 1.0)
        n_pos_i = pos_i.sum()
        if n_pos_i == 0:
            continue
        for j in range(K):
            valid_ij = pos_i & ~np.isnan(labels[:, j])
            n_both = (valid_ij & (labels[:, j] == 1.0)).sum()
            A[i, j] = n_both / n_pos_i

    # Sparsify
    A[A < tau] = 0.0

    # Add self-loops
    A += np.eye(K, dtype=np.float32)

    # Row-normalise
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    A = A / row_sums

    return torch.tensor(A, dtype=torch.float32)


class LabelGCN(nn.Module):
    """Two-layer Graph Convolutional classifier for multi-label prediction.

    The GCN maps K learnable node embeddings through the co-occurrence graph
    to produce K classifier weight vectors in the image feature space.  The
    final logits are the dot-product of image features with these weight
    vectors, optionally blended with an initial (residual) logit estimate.

    Args:
        adjacency: [K, K] row-normalised adjacency matrix (static buffer).
        feat_dim: Dimension of input image features (default: 1024).
        embed_dim: Node embedding dimension (default: 300).
        hidden_dim: GCN hidden layer dimension (default: 1024).
        alpha: Residual blending weight for init_logits (default: 0.7).
            Final logit = alpha * gcn_logit + (1-alpha) * init_logit.
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        feat_dim: int = 1024,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        alpha: float = 0.7,
    ) -> None:
        super().__init__()
        K = adjacency.shape[0]
        self.alpha = alpha

        self.register_buffer("A", adjacency)  # [K, K]
        self.node_emb = nn.Parameter(torch.randn(K, embed_dim) * 0.01)
        self.gcn1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, feat_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(K))

    def _classifier_weights(self) -> torch.Tensor:
        """Compute GCN-derived classifier weight matrix W ∈ R^{K × feat_dim}."""
        h1 = F.relu(self.A @ self.node_emb @ self.gcn1.weight.T)  # [K, hidden]
        W = self.A @ h1 @ self.gcn2.weight.T                       # [K, feat_dim]
        return W

    def forward(
        self,
        img_features: torch.Tensor,
        init_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-label logits.

        Args:
            img_features: [N, feat_dim] image feature vectors.
            init_logits: [N, K] optional residual logits (e.g. from a
                pre-trained logistic regression).  When provided, the output
                is a weighted blend: alpha * gcn + (1-alpha) * init.

        Returns:
            logits: [N, K] raw (pre-sigmoid) logits.
        """
        W = self._classifier_weights()                      # [K, feat_dim]
        gcn_logits = img_features @ W.T + self.bias        # [N, K]
        if init_logits is not None:
            return self.alpha * gcn_logits + (1.0 - self.alpha) * init_logits
        return gcn_logits


def train_gnn(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    adjacency: torch.Tensor,
    init_logits_train: np.ndarray | None = None,
    init_logits_val: np.ndarray | None = None,
    feat_dim: int = 1024,
    embed_dim: int = 300,
    hidden_dim: int = 1024,
    alpha: float = 0.7,
    epochs: int = 50,
    save_best: bool = True,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "mps",
    verbose: bool = True,
) -> tuple[LabelGCN, dict[str, list[Any]]]:
    """Train LabelGCN on pre-extracted features.

    NaN labels are masked out from the loss so that pathologies with missing
    annotations do not pollute gradient updates.

    Args:
        features_train: [N_train, feat_dim] training features (already scaled).
        labels_train: [N_train, K] training labels with NaN for missing.
        features_val: [N_val, feat_dim] validation features (already scaled).
        labels_val: [N_val, K] validation labels with NaN for missing.
        adjacency: [K, K] row-normalised co-occurrence matrix.
        init_logits_train: [N_train, K] optional residual logits for training.
        init_logits_val: [N_val, K] optional residual logits for validation.
        feat_dim: Feature dimensionality.
        embed_dim: Node embedding dimension.
        hidden_dim: GCN hidden layer dimension.
        alpha: Residual blending weight.
        epochs: Number of training epochs.
        save_best: If True (default), restore the model weights from the epoch
            with the highest mean val AUC before returning.  ``history["best_epoch"]``
            records that epoch number.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        weight_decay: L2 regularisation coefficient.
        device: PyTorch device string ("mps", "cuda", or "cpu").
        verbose: Print per-epoch summary.

    Returns:
        model: Trained LabelGCN (on CPU).  If ``save_best=True``, weights are
            restored from the epoch with the highest mean val AUC.
        history: Dict with keys "train_loss" and "val_auc" (lists, one per epoch)
            plus "best_epoch" (single-element list with the 1-indexed best epoch).
    """
    dev = torch.device(device if torch.backends.mps.is_available() or device != "mps" else "cpu")

    model = LabelGCN(adjacency, feat_dim=feat_dim, embed_dim=embed_dim,
                     hidden_dim=hidden_dim, alpha=alpha).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Build training tensors
    X_tr = torch.tensor(features_train, dtype=torch.float32)
    Y_tr = torch.tensor(labels_train, dtype=torch.float32)
    init_tr = (torch.tensor(init_logits_train, dtype=torch.float32)
               if init_logits_train is not None else None)

    if init_tr is not None:
        dataset = TensorDataset(X_tr, Y_tr, init_tr)
    else:
        dataset = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation tensors (full-batch)
    X_val = torch.tensor(features_val, dtype=torch.float32).to(dev)
    Y_val_np = labels_val
    init_val = (torch.tensor(init_logits_val, dtype=torch.float32).to(dev)
                if init_logits_val is not None else None)

    history: dict[str, list[Any]] = {"train_loss": [], "val_auc": []}

    _best_val_auc: float = float("-inf")
    _best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if init_tr is not None:
                xb, yb, lb = batch
                xb, yb, lb = xb.to(dev), yb.to(dev), lb.to(dev)
            else:
                xb, yb = batch
                xb, yb = xb.to(dev), yb.to(dev)
                lb = None

            logits = model(xb, lb)
            mask = ~torch.isnan(yb)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, yb.nan_to_num(0.0), reduction="none"
            )
            loss = (loss_raw * mask.float()).sum() / mask.float().sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation: per-pathology AUROC (NaN-masked)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, init_val).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid

        aucs = []
        K = val_probs.shape[1]
        for k in range(K):
            valid = ~np.isnan(Y_val_np[:, k])
            if valid.sum() < 2 or len(np.unique(Y_val_np[valid, k])) < 2:
                continue
            try:
                auc = roc_auc_score(Y_val_np[valid, k], val_probs[valid, k])
                aucs.append(auc)
            except ValueError:
                pass
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(mean_auc)

        if save_best and mean_auc > _best_val_auc:
            _best_val_auc = mean_auc
            _best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_auc={mean_auc:.4f}")

    if save_best and _best_state is not None:
        model.load_state_dict(_best_state)
        if verbose:
            best_ep = int(np.argmax(history["val_auc"])) + 1
            print(f"Restored best model from epoch {best_ep} (val_auc={_best_val_auc:.4f})")

    history["best_epoch"] = [int(np.argmax(history["val_auc"])) + 1]

    model = model.cpu()
    return model, history

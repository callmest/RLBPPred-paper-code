"""
Cross-comparison experiments: RF features vs SaProt embeddings × RF dataset vs our filtered dataset.

Experiment 1: RF handcraft features + our filtered dataset IDs → RF classifier (balanced ensemble)
Experiment 2: RF dataset IDs + SaProt-650M-AF2 embeddings → our MLP ensemble

Usage:
  # Experiment 1: RF features + our filtered dataset
  python scripts/train_cross_comparison.py \
    --experiment rf_features_our_data \
    --rf-features data/RF/RF_features.xlsx \
    --pos-ids data/dataset_filter/positive_ids.txt \
    --neg-ids data/dataset_filter/negative_ids.txt \
    --output-dir results/cross_comparison/rf_features_our_data

  # Experiment 2: RF dataset + SaProt embeddings + our MLP ensemble
  python scripts/train_cross_comparison.py \
    --experiment saprot_rf_data \
    --pos-ids data/RF/positive_ids.txt \
    --neg-ids data/RF/negative_ids.txt \
    --embedding-dir data/embeddings/saprot-650m-af2 \
    --output-dir results/cross_comparison/saprot_rf_data
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve,
    precision_score, recall_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# MLP model (same as train_sa_mlp.py)
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.0, use_bn=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier:
    def __init__(self, in_dim, hidden_dims=(512, 256), dropout=0.3, use_bn=True,
                 lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y):
        self.model = _MLP(self.in_dim, self.hidden_dims,
                          self.dropout, self.use_bn).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(Xt).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.stack([1 - probs, probs], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ids(id_file: str) -> list:
    with open(id_file) as f:
        return [line.strip() for line in f if line.strip() and line.strip() != "positive"]


def load_rf_features(xlsx_path: str, ids: list) -> tuple:
    """Load RF handcraft features for given IDs. Returns (X, found_ids, missing_ids)."""
    df = pd.read_excel(xlsx_path)
    df_indexed = df.set_index("UniprotID")
    feature_cols = [c for c in df.columns if c != "UniprotID"]

    X, found_ids, missing_ids = [], [], []
    for pid in ids:
        if pid in df_indexed.index:
            X.append(df_indexed.loc[pid, feature_cols].values.astype(float))
            found_ids.append(pid)
        else:
            missing_ids.append(pid)

    return np.array(X) if X else np.array([]), found_ids, missing_ids


def load_embeddings_by_ids(ids: list, embedding_dir: Path) -> tuple:
    """Load SaProt embeddings for given IDs. Returns (X, found_ids, missing_ids)."""
    pt_index = {}
    for pt_path in embedding_dir.rglob("*.pt"):
        pt_index[pt_path.stem] = pt_path

    X, found_ids, missing_ids = [], [], []
    for pid in ids:
        if pid in pt_index:
            data = torch.load(pt_index[pid], map_location="cpu", weights_only=False)
            X.append(data["embeddings"].float().numpy())
            found_ids.append(pid)
        else:
            missing_ids.append(pid)

    return np.array(X) if X else np.array([]), found_ids, missing_ids


# ─────────────────────────────────────────────────────────────────────────────
# Train/val split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(X, y, ids, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    def split_idx(idx):
        n_val = max(1, int(len(idx) * val_ratio))
        val = rng.choice(idx, size=n_val, replace=False)
        train = np.setdiff1d(idx, val)
        return train, val

    pos_train, pos_val = split_idx(pos_idx)
    neg_train, neg_val = split_idx(neg_idx)
    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            [ids[i] for i in val_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Balanced subsampling ensemble — RF version
# ─────────────────────────────────────────────────────────────────────────────

def balanced_ensemble_rf(X_train, y_train, X_val, y_val, n_rounds=300, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    val_probs = np.zeros(len(y_val))
    models = []

    for i in range(n_rounds):
        neg_sample = rng.choice(neg_idx, size=n_pos, replace=False)
        idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(idx)

        clf = RandomForestClassifier(n_estimators=100, random_state=seed + i, n_jobs=-1)
        clf.fit(X_train[idx], y_train[idx])
        val_probs += clf.predict_proba(X_val)[:, 1]
        models.append(clf)

        if (i + 1) % 50 == 0:
            print(f"  round {i+1}/{n_rounds}", flush=True)

    val_probs /= n_rounds
    return val_probs, models, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Balanced subsampling ensemble — MLP version
# ─────────────────────────────────────────────────────────────────────────────

def balanced_ensemble_mlp(X_train, y_train, X_val, y_val, in_dim,
                          n_rounds=300, seed=42, device="cpu"):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    val_probs = np.zeros(len(y_val))
    models = []

    for i in range(n_rounds):
        neg_sample = rng.choice(neg_idx, size=n_pos, replace=False)
        idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(idx)

        clf = MLPClassifier(in_dim, hidden_dims=(512, 256),
                            use_bn=True, dropout=0.3, device=device)
        clf.fit(X_train[idx], y_train[idx])
        val_probs += clf.predict_proba(X_val)[:, 1]
        models.append(clf)

        if (i + 1) % 50 == 0:
            print(f"  round {i+1}/{n_rounds}", flush=True)

    val_probs /= n_rounds
    return val_probs, models, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & plotting
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probs)
    return {
        "auroc":     round(roc_auc_score(y_true, probs), 4),
        "auprc":     round(average_precision_score(y_true, probs), 4),
        "f1":        round(f1_score(y_true, preds, zero_division=0), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, preds), 4),
        "accuracy":  round(accuracy_score(y_true, preds), 4),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
    }


def plot_roc(metrics, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["fpr"], metrics["tpr"], lw=2, color="steelblue",
            label=f'AUC={metrics["auroc"]:.4f}')
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"ROC curve saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-comparison experiments")
    parser.add_argument("--experiment", required=True,
                        choices=["rf_features_our_data", "saprot_rf_data"],
                        help="Which experiment to run")
    parser.add_argument("--pos-ids", required=True)
    parser.add_argument("--neg-ids", required=True)
    parser.add_argument("--rf-features", default=str(Path(__file__).resolve().parents[2] / "data" / "RF" / "RF_features.xlsx"),
                        help="Path to RF_features.xlsx (for experiment 1)")
    parser.add_argument("--embedding-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "embeddings" / "saprot-650m-af2"),
                        help="SaProt embedding directory (for experiment 2)")
    parser.add_argument("--n-rounds", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load IDs ──────────────────────────────────────────────────────────────
    pos_ids = load_ids(args.pos_ids)
    neg_ids = load_ids(args.neg_ids)
    print(f"ID lists: {len(pos_ids)} positive, {len(neg_ids)} negative")

    if args.experiment == "rf_features_our_data":
        # ── Experiment 1: RF features + our filtered dataset ──────────────────
        print("=" * 60)
        print("Experiment 1: RF Handcraft Features + Our Filtered Dataset")
        print("  Classifier: Random Forest (balanced subsampling ensemble)")
        print("=" * 60)

        print(f"\nLoading RF features from {args.rf_features} ...")
        X_pos, found_pos, missing_pos = load_rf_features(args.rf_features, pos_ids)
        X_neg, found_neg, missing_neg = load_rf_features(args.rf_features, neg_ids)
        label = "RF Features + Our Data"
        clf_type = "RF"

    else:
        # ── Experiment 2: SaProt embeddings + RF dataset ──────────────────────
        print("=" * 60)
        print("Experiment 2: SaProt-650M-AF2 Embeddings + RF Dataset")
        print("  Classifier: MLP (balanced subsampling ensemble)")
        print("=" * 60)

        emb_dir = Path(args.embedding_dir)
        print(f"\nLoading SaProt embeddings from {emb_dir} ...")
        X_pos, found_pos, missing_pos = load_embeddings_by_ids(pos_ids, emb_dir)
        X_neg, found_neg, missing_neg = load_embeddings_by_ids(neg_ids, emb_dir)
        label = "SaProt + RF Data"
        clf_type = "MLP"

    print(f"  Positive: {len(found_pos)} loaded, {len(missing_pos)} missing")
    print(f"  Negative: {len(found_neg)} loaded, {len(missing_neg)} missing")

    if missing_pos or missing_neg:
        with open(output_dir / "missing_ids.txt", "w") as f:
            for pid in missing_pos:
                f.write(f"positive\t{pid}\n")
            for pid in missing_neg:
                f.write(f"negative\t{pid}\n")

    if len(found_pos) == 0 or len(found_neg) == 0:
        print("ERROR: No data found for one or both classes. Aborting.")
        return

    # ── Assemble dataset ──────────────────────────────────────────────────────
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * len(found_pos) + [0] * len(found_neg))
    ids = found_pos + found_neg
    in_dim = X.shape[1]

    # Handle NaN in RF features
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"  Warning: {nan_count} NaN values found, filling with column median")
        col_median = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_median[j]

    print(f"\nDataset: {len(y)} samples, dim={in_dim} "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())}, ratio=1:{(y==0).sum()/y.sum():.1f})")

    # ── Train/val split ───────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, val_ids = train_val_split(
        X, y, ids, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Train: {len(y_train)} (pos={int(y_train.sum())}, neg={int((y_train==0).sum())}) | "
          f"Val: {len(y_val)} (pos={int(y_val.sum())}, neg={int((y_val==0).sum())})")

    # ── Train ensemble ────────────────────────────────────────────────────────
    print(f"\nTraining {args.n_rounds}-round balanced subsampling ensemble ({clf_type}) ...")

    if clf_type == "RF":
        val_probs, models, scaler = balanced_ensemble_rf(
            X_train, y_train, X_val, y_val,
            n_rounds=args.n_rounds, seed=args.seed,
        )
    else:
        val_probs, models, scaler = balanced_ensemble_mlp(
            X_train, y_train, X_val, y_val,
            in_dim=in_dim, n_rounds=args.n_rounds,
            seed=args.seed, device=args.device,
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_val, val_probs)
    print(f"\n{'='*60}")
    print(f"Validation Results ({label}):")
    print(f"  AUROC={metrics['auroc']}  AUPRC={metrics['auprc']}  "
          f"F1={metrics['f1']}  Prec={metrics['precision']}  "
          f"Rec={metrics['recall']}  MCC={metrics['mcc']}  ACC={metrics['accuracy']}")
    print(f"{'='*60}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Checkpoint
    ckpt = {
        "models": models, "scaler": scaler,
        "experiment": args.experiment, "clf_type": clf_type,
        "in_dim": in_dim, "n_rounds": args.n_rounds,
        "val_ids": val_ids, "val_probs": val_probs.tolist(),
        "metrics": {k: v for k, v in metrics.items() if k not in ("fpr", "tpr")},
        "dataset": {
            "n_pos_loaded": len(found_pos), "n_neg_loaded": len(found_neg),
            "n_pos_missing": len(missing_pos), "n_neg_missing": len(missing_neg),
        },
    }
    with open(output_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ckpt, f)

    # Results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k not in ("fpr", "tpr")}, f, indent=2)

    # ROC plot
    plot_roc(metrics, f"ROC — {label}", output_dir / "roc_curve.svg")

    # Val predictions CSV
    with open(output_dir / "val_predictions.csv", "w") as f:
        f.write("protein_id,label,probability\n")
        for pid, lab, prob in sorted(
            zip(val_ids, y_val, val_probs), key=lambda x: x[2], reverse=True
        ):
            f.write(f"{pid},{lab},{prob:.6f}\n")
    print(f"Val predictions saved to {output_dir / 'val_predictions.csv'}")


if __name__ == "__main__":
    main()

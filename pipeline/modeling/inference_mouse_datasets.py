#!/usr/bin/env python3
"""
Legacy pickle-ensemble inference on mouse protein datasets.

Use strict_runtime.py predict for the current strict state_dict checkpoint.
Mouse predictions are zero-shot; mouse groups are not training groups.

Usage:
  python scripts/inference_mouse_datasets.py \
    --checkpoint results/sa_mlp_filter/ensemble.pkl \
    --embedding-dir data/embeddings/saprot-650m-af2/mouse_proteome_mean \
    --id-dir data/mouse_datasets \
    --output-dir results/inference_mouse_datasets \
    --device cpu
"""

import argparse
import csv
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# ── MLP model (needed for unpickling checkpoint) ─────────────────────────────

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
    """Sklearn-compatible wrapper around a PyTorch MLP."""

    def __init__(self, in_dim=1280, hidden_dims=(512, 256), dropout=0.3, use_bn=True,
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

# ── Embedding Index ───────────────────────────────────────────────────────────

def build_embedding_index(embedding_dir: Path) -> dict:
    """Build index: protein_id -> embedding array."""
    index = {}
    for pt_file in embedding_dir.glob("*.pt"):
        protein_id = pt_file.stem
        import torch
        data = torch.load(pt_file, map_location="cpu")
        emb = data['embeddings'].numpy() if isinstance(data, dict) else data.numpy()
        index[protein_id] = emb
    return index


# ── Ensemble Prediction ───────────────────────────────────────────────────────

def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Predict probabilities using ensemble of models."""
    probs = np.zeros(len(X))
    for clf in models:
        if hasattr(clf, "predict_proba"):
            probs += clf.predict_proba(X)[:, 1]
    probs /= len(models)
    return probs


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_density(dataset_probs: dict, save_dir: Path):
    """Plot bar chart showing prediction statistics for all datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Light pastel colors
    colors = {
        "significant_positive": "#ffb3ba",       # light red
        "negative": "#bae1ff",                   # light blue
        "potential_positive": "#ffd8a8",         # light orange
    }
    labels = {
        "significant_positive": "Sig. Positive\n(Train Pos)",
        "negative": "Mouse negative\ngroup",
        "potential_positive": "Potential\nPositive",
    }

    all_datasets = ["significant_positive", "negative", "potential_positive"]

    # Compute statistics for all datasets
    stats = []
    for name in all_datasets:
        probs = dataset_probs[name]
        if len(probs) > 0:
            stats.append({
                'name': name,
                'mean_prob': probs.mean(),
                'pct_positive': (probs >= 0.5).mean() * 100,
                'n': len(probs)
            })

    # Sort by mean probability (descending)
    stats_sorted = sorted(stats, key=lambda x: x['mean_prob'], reverse=True)

    dataset_order = [s['name'] for s in stats_sorted]
    mean_probs = [s['mean_prob'] for s in stats_sorted]
    pct_positive = [s['pct_positive'] for s in stats_sorted]
    sample_sizes = [s['n'] for s in stats_sorted]

    x_pos = np.arange(len(dataset_order))
    bar_colors = [colors[name] for name in dataset_order]
    bar_labels = [labels[name] for name in dataset_order]

    # Plot 1: Mean probability
    bars1 = ax1.bar(x_pos, mean_probs, color=bar_colors, alpha=0.9, edgecolor='gray', linewidth=0.8)
    ax1.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_title("Mean RLBP Prediction Probability (Mouse)", fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, val, n) in enumerate(zip(bars1, mean_probs, sample_sizes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}\n(n={n})',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Percentage predicted as RLBP
    bars2 = ax2.bar(x_pos, pct_positive, color=bar_colors, alpha=0.9, edgecolor='gray', linewidth=0.8)
    ax2.set_ylabel("Percentage Predicted as RLBP (%)", fontsize=12)
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_title("Percentage Predicted as RLBP (prob >= 0.5, Mouse)", fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, pct_positive)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)

    fig.tight_layout()

    for fmt in ["svg", "png"]:
        fig.savefig(save_dir / f"prob_barplot.{fmt}", dpi=200, format=fmt)
    plt.close(fig)
    print(f"Bar plot saved to {save_dir}/prob_barplot.svg/.png")

    # Save plot source data
    with open(save_dir / "prob_barplot_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mean_probability", "percentage_positive", "sample_size"])
        for name, mean_p, pct_p, n in zip(dataset_order, mean_probs, pct_positive, sample_sizes):
            writer.writerow([name, f"{mean_p:.4f}", f"{pct_p:.2f}", n])
    print(f"Plot source data saved to {save_dir}/prob_barplot_data.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.warn("Legacy pickle inference. Use strict_runtime.py predict for the current strict checkpoint.", RuntimeWarning)
    parser = argparse.ArgumentParser(description="Inference on mouse datasets")
    parser.add_argument("--checkpoint", default=str(REPOSITORY_ROOT / "results" / "sa_mlp_filter" / "ensemble.pkl"))
    parser.add_argument("--embedding-dir", default=str(REPOSITORY_ROOT / "data" / "embeddings" / "saprot-650m-af2" / "mouse_proteome_mean"))
    parser.add_argument("--id-dir", default=str(REPOSITORY_ROOT / "data" / "mouse_datasets"))
    parser.add_argument("--output-dir", default=str(REPOSITORY_ROOT / "results" / "inference_mouse_datasets"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Inference on Mouse Datasets")
    print("=" * 60)

    # Build embedding index
    print(f"Building embedding index from {args.embedding_dir} ...")
    emb_index = build_embedding_index(Path(args.embedding_dir))
    print(f"  {len(emb_index)} embeddings indexed")
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    models = checkpoint["models"]
    scaler = checkpoint.get("scaler", None)
    in_dim = checkpoint.get("in_dim", 1280)
    print(f"  Ensemble: {len(models)} models, in_dim={in_dim}")
    if scaler:
        print(f"  Scaler: StandardScaler loaded")
    print()

    # Process each dataset
    dataset_names = ["significant_positive", "potential_positive", "negative"]
    dataset_probs = {}

    for dataset_name in dataset_names:
        print("─" * 60)
        id_file = Path(args.id_dir) / f"{dataset_name}.txt"

        if not id_file.exists():
            print(f"Dataset: {dataset_name} - FILE NOT FOUND: {id_file}")
            dataset_probs[dataset_name] = np.array([])
            continue

        # Load IDs
        with open(id_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        print(f"Dataset: {dataset_name} ({len(ids)} IDs)")

        # Match embeddings
        X_list = []
        matched_ids = []
        missing_ids = []

        for protein_id in ids:
            if protein_id in emb_index:
                X_list.append(emb_index[protein_id])
                matched_ids.append(protein_id)
            else:
                missing_ids.append(protein_id)

        if len(X_list) == 0:
            print(f"  No embeddings found!")
            dataset_probs[dataset_name] = np.array([])
            continue

        X = np.vstack(X_list)
        print(f"  Loaded: {len(matched_ids)}, Missing: {len(missing_ids)}")

        # Apply scaler if available
        if scaler is not None:
            X = scaler.transform(X)

        # Predict
        probs = predict_ensemble(models, X)
        dataset_probs[dataset_name] = probs

        # Stats
        print(f"  Prob stats: mean={probs.mean():.4f}, median={np.median(probs):.4f}, std={probs.std():.4f}")
        print(f"  >=0.5: {(probs >= 0.5).sum()} ({(probs >= 0.5).mean()*100:.1f}%)")

        # Save predictions
        output_file = output_dir / f"{dataset_name}_predictions.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["protein_id", "probability"])
            for pid, prob in zip(matched_ids, probs):
                writer.writerow([pid, f"{prob:.6f}"])
        print(f"  Saved to {output_file}")
        print()

    # Save combined predictions
    all_predictions = []
    for dataset_name in dataset_names:
        id_file = Path(args.id_dir) / f"{dataset_name}.txt"
        if not id_file.exists():
            continue
        with open(id_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        probs = dataset_probs[dataset_name]
        matched_ids = [pid for pid in ids if pid in emb_index]

        for pid, prob in zip(matched_ids, probs):
            all_predictions.append((pid, dataset_name, prob))

    combined_file = output_dir / "all_predictions.csv"
    with open(combined_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "dataset", "probability"])
        for pid, ds, prob in all_predictions:
            writer.writerow([pid, ds, f"{prob:.6f}"])
    print(f"Combined predictions saved to {combined_file}")
    print()

    # Plot
    print("Plotting probability density ...")
    plot_density(dataset_probs, output_dir)
    print()

    # Summary table
    print("=" * 60)
    print(f"{'Dataset':<30} {'N':>6} {'Mean':>8} {'Median':>8} {'>=0.5':>6} {'%>=0.5':>8}")
    print("─" * 60)
    for name in dataset_names:
        probs = dataset_probs[name]
        if len(probs) > 0:
            print(f"{name:<30} {len(probs):>6} {probs.mean():>8.4f} {np.median(probs):>8.4f} "
                  f"{(probs >= 0.5).sum():>6} {(probs >= 0.5).mean()*100:>7.1f}%")


if __name__ == "__main__":
    main()

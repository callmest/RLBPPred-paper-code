"""
Legacy pickle-ensemble inference on four protein datasets.

The current strict state_dict_ensemble_v1 checkpoint must be loaded with
strict_runtime.py predict. This script remains for historical pickle models.

Loads pre-extracted embeddings, runs ensemble prediction, saves probabilities,
and plots probability density distributions.

Usage:
  python scripts/inference_four_datasets.py \
    --checkpoint results/sa_mlp_filter/ensemble.pkl \
    --embedding-dir data/embeddings/saprot-650m-af2 \
    --id-dir data/four_datasets \
    --output-dir results/inference_four_datasets \
    --device cpu
"""

import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import MLPClassifier / _MLP so pickle can deserialize the ensemble
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from train_sa_mlp import MLPClassifier, _MLP  # noqa: E402,F401


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ids(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_embedding_index(embedding_dir: Path) -> dict:
    """Build protein_id -> .pt path mapping across all subdirectories."""
    idx = {}
    for pt in embedding_dir.rglob("*.pt"):
        idx[pt.stem] = pt
    return idx


def load_embeddings_by_ids(ids: list, pt_index: dict):
    """Load embeddings for given IDs. Returns (X, found_ids, missing_ids)."""
    X, found, missing = [], [], []
    for pid in ids:
        if pid in pt_index:
            data = torch.load(pt_index[pid], map_location="cpu", weights_only=False)
            X.append(data["embeddings"].float().numpy())
            found.append(pid)
        else:
            missing.append(pid)
    return (np.array(X) if X else np.empty((0, 0))), found, missing


# ── Ensemble prediction ──────────────────────────────────────────────────────

def ensemble_predict(X: np.ndarray, checkpoint: dict) -> np.ndarray:
    models = checkpoint["models"]
    scaler = checkpoint["scaler"]
    if scaler is not None:
        X = scaler.transform(X)
    probs = np.zeros(len(X))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for clf in models:
            probs += clf.predict_proba(X)[:, 1]
    probs /= len(models)
    return probs


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_density(dataset_probs: dict, save_dir: Path):
    """Plot bar chart showing prediction statistics for all datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Light pastel colors
    colors = {
        "sig_pos_ids": "#ffb3ba",       # light red
        "neg_ids": "#bae1ff",           # light blue
        "potential_positive": "#ffd8a8", # light orange
        "untrained_proteome": "#b3e5b3", # light green
    }
    labels = {
        "sig_pos_ids": "High-confidence\npositive benchmark",
        "neg_ids": "Negative\nbenchmark",
        "potential_positive": "Potential\nPositive",
        "untrained_proteome": "Untrained\nProteome",
    }

    all_datasets = ["sig_pos_ids", "neg_ids", "potential_positive", "untrained_proteome"]

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
    ax1.set_title("Mean RLBP Prediction Probability", fontsize=13, fontweight='bold')
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
    ax2.set_title("Percentage Predicted as RLBP (prob >= 0.5)", fontsize=13, fontweight='bold')
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
    parser = argparse.ArgumentParser(description="Inference on four datasets")
    parser.add_argument("--checkpoint", default=str(REPOSITORY_ROOT / "results" / "sa_mlp_filter" / "ensemble.pkl"))
    parser.add_argument("--embedding-dir", default=str(REPOSITORY_ROOT / "data" / "embeddings" / "saprot-650m-af2"))
    parser.add_argument("--id-dir", default=str(REPOSITORY_ROOT / "data" / "four_datasets"))
    parser.add_argument("--output-dir", default=str(REPOSITORY_ROOT / "results" / "inference_four_datasets"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset definitions
    datasets = {
        "sig_pos_ids": "sig_pos_ids.txt",
        "neg_ids": "neg_ids.txt",
        "potential_positive": "potential_positive.txt",
        "untrained_proteome": "untrained_proteome.txt",
    }

    print("=" * 60)
    print("Inference on Four Datasets")
    print("=" * 60)

    # Build embedding index once
    emb_dir = Path(args.embedding_dir)
    print(f"Building embedding index from {emb_dir} ...")
    pt_index = build_embedding_index(emb_dir)
    print(f"  {len(pt_index)} embeddings indexed")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    n_models = len(checkpoint["models"])
    print(f"  Ensemble: {n_models} models, in_dim={checkpoint.get('in_dim')}")

    # Move MLP models to target device
    device = args.device
    for clf in checkpoint["models"]:
        if hasattr(clf, 'model') and clf.model is not None:
            clf.model = clf.model.to(device)
            clf.device = device

    # Run inference per dataset
    dataset_probs = {}
    for name, filename in datasets.items():
        id_path = os.path.join(args.id_dir, filename)
        ids = load_ids(id_path)
        print(f"\n{'─'*60}")
        print(f"Dataset: {name} ({len(ids)} IDs)")

        X, found, missing = load_embeddings_by_ids(ids, pt_index)
        print(f"  Loaded: {len(found)}, Missing: {len(missing)}")

        if len(found) == 0:
            print("  SKIP: no embeddings found")
            dataset_probs[name] = np.array([])
            continue

        probs = ensemble_predict(X, checkpoint)
        dataset_probs[name] = probs

        # Stats
        print(f"  Prob stats: mean={probs.mean():.4f}, median={np.median(probs):.4f}, "
              f"std={probs.std():.4f}")
        print(f"  >=0.5: {(probs >= 0.5).sum()} ({(probs >= 0.5).mean()*100:.1f}%)")

        # Save per-dataset CSV
        csv_path = output_dir / f"{name}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["protein_id", "probability"])
            for pid, prob in sorted(zip(found, probs), key=lambda x: x[1], reverse=True):
                writer.writerow([pid, f"{prob:.6f}"])
        print(f"  Saved to {csv_path}")

    # Save combined source data
    combined_path = output_dir / "all_predictions.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "protein_id", "probability"])
        for name in datasets:
            id_path = os.path.join(args.id_dir, datasets[name])
            ids = load_ids(id_path)
            X, found, _ = load_embeddings_by_ids(ids, pt_index)
            if len(found) > 0:
                probs = dataset_probs[name]
                for pid, prob in zip(found, probs):
                    writer.writerow([name, pid, f"{prob:.6f}"])
    print(f"\nCombined predictions saved to {combined_path}")

    # Plot density
    print("\nPlotting probability density ...")
    plot_density(dataset_probs, output_dir)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Dataset':<25} {'N':>6} {'Mean':>8} {'Median':>8} {'>0.5':>8} {'%>0.5':>8}")
    print(f"{'─'*60}")
    for name in datasets:
        p = dataset_probs[name]
        if len(p) == 0:
            continue
        n_pos = (p > 0.5).sum()
        print(f"{name:<25} {len(p):>6} {p.mean():>8.4f} {np.median(p):>8.4f} "
              f"{n_pos:>8} {n_pos/len(p)*100:>7.1f}%")


if __name__ == "__main__":
    main()

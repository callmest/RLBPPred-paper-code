#!/usr/bin/env python3
"""
Compare negative undersampling strategies for SaProt-650M-AF2 + MLP.

The experiment keeps the model, train/validation split, negative sampling
quantity, and evaluation metrics fixed, and changes only how negative training
samples are selected in each ensemble round:

  1. random undersampling
  2. K-means representative undersampling
  3. K-means hard/boundary undersampling

This is intended to address the reviewer concern that high recall but modest
precision may be related to insufficient negative-space coverage from random
undersampling.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sa_mlp import (  # noqa: E402
    MLPClassifier,
    load_embeddings_by_ids,
    load_ids,
    train_val_split,
)


METRIC_KEYS = ["auroc", "auprc", "f1", "precision", "recall", "mcc", "accuracy"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs),
        "auprc": average_precision_score(y_true, probs),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
        "accuracy": accuracy_score(y_true, preds),
    }


def build_cluster_pools(X_train_scaled, y_train, n_clusters, seed, top_k):
    neg_idx = np.where(y_train == 0)[0]
    pos_idx = np.where(y_train == 1)[0]
    X_neg = X_train_scaled[neg_idx]
    X_pos = X_train_scaled[pos_idx]

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=2048,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X_neg)
    centers = kmeans.cluster_centers_

    nearest_center_idx, center_dist = pairwise_distances_argmin_min(X_neg, centers)
    pos_centroid = X_pos.mean(axis=0, keepdims=True)
    _, pos_dist = pairwise_distances_argmin_min(X_neg, pos_centroid)

    representative_pools = []
    hard_pools = []
    for cluster_id in range(n_clusters):
        local = np.where(labels == cluster_id)[0]
        if len(local) == 0:
            continue
        rep_order = local[np.argsort(center_dist[local])]
        hard_order = local[np.argsort(pos_dist[local])]
        representative_pools.append(neg_idx[rep_order[: max(1, min(top_k, len(rep_order)))]])
        hard_pools.append(neg_idx[hard_order[: max(1, min(top_k, len(hard_order)))]])

    return {
        "representative": representative_pools,
        "hard": hard_pools,
        "n_nonempty_clusters": len(representative_pools),
    }


def sample_negatives(strategy, rng, neg_idx, n_sample, cluster_pools=None):
    if strategy == "random":
        return rng.choice(neg_idx, size=n_sample, replace=False)

    pool_key = "representative" if strategy == "kmeans_representative" else "hard"
    pools = cluster_pools[pool_key]
    selected = []
    for pool in pools:
        selected.append(rng.choice(pool))

    selected = np.array(selected, dtype=int)
    if len(selected) >= n_sample:
        return rng.choice(selected, size=n_sample, replace=False)

    selected_set = set(selected.tolist())
    remaining = np.array([idx for idx in neg_idx if idx not in selected_set], dtype=int)
    extra = rng.choice(remaining, size=n_sample - len(selected), replace=False)
    return np.concatenate([selected, extra])


def train_strategy(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    strategy,
    in_dim,
    n_rounds,
    seed,
    device,
    cluster_pools=None,
):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)
    cumulative_probs = np.zeros(len(y_val), dtype=float)

    start = time.time()
    for i in range(n_rounds):
        round_seed = seed + i + 10000
        set_global_seed(round_seed)

        neg_sample = sample_negatives(strategy, rng, neg_idx, n_pos, cluster_pools)
        idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(idx)

        clf = MLPClassifier(
            in_dim,
            hidden_dims=(512, 256),
            use_bn=True,
            dropout=0.3,
            device=device,
        )
        clf.fit(X_train[idx], y_train[idx])
        cumulative_probs += clf.predict_proba(X_val)[:, 1]

        if (i + 1) % 25 == 0 or (i + 1) == n_rounds:
            avg_probs = cumulative_probs / (i + 1)
            metric = compute_metrics(y_val, avg_probs)
            elapsed = (time.time() - start) / 60
            print(
                f"  {strategy} round={i + 1}/{n_rounds} "
                f"AUROC={metric['auroc']:.4f} AUPRC={metric['auprc']:.4f} "
                f"F1={metric['f1']:.4f} Precision={metric['precision']:.4f} "
                f"Recall={metric['recall']:.4f} MCC={metric['mcc']:.4f} "
                f"elapsed={elapsed:.1f} min",
                flush=True,
            )

    probs = cumulative_probs / n_rounds
    return probs, compute_metrics(y_val, probs)


def write_csv(path, rows):
    fieldnames = ["strategy", "description", "n_rounds"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows, output_prefix):
    labels = [row["strategy"] for row in rows]
    display = {
        "random": "Random",
        "kmeans_representative": "K-means\nrepresentative",
        "kmeans_hard": "K-means\nhard",
    }
    metrics = ["precision", "recall", "f1", "mcc", "auprc", "auroc"]
    colors = ["#4C78A8", "#59A14F", "#E15759"]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        values = [row[metric] for row in rows]
        ax.bar([display[x] for x in labels], values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(values):
            ax.text(i, min(v + 0.025, 1.01), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{output_prefix}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_summary(path, rows, dataset_info, args, cluster_info):
    strategy_name = {
        "random": "随机欠采样",
        "kmeans_representative": "K-means代表性欠采样",
        "kmeans_hard": "K-means边界/困难负样本欠采样",
    }
    lines = [
        "# Negative Undersampling Strategy Comparison",
        "",
        "## Configuration",
        "",
        "- Model: SaProt-650M-AF2 + MLP",
        "- Sampling quantity: 1:1 positives:negatives in each ensemble round",
        f"- Ensemble rounds: {args.rounds}",
        f"- Seed: {args.seed}",
        f"- Device: {args.device}",
        f"- Positive samples loaded: {dataset_info['n_pos']}",
        f"- Negative samples loaded: {dataset_info['n_neg']}",
        f"- Validation positives: {dataset_info['n_val_pos']}",
        f"- Validation negatives: {dataset_info['n_val_neg']}",
        f"- K-means clusters: {cluster_info['n_clusters']}",
        f"- Non-empty clusters: {cluster_info['n_nonempty_clusters']}",
        f"- Candidate pool per cluster: top {args.cluster_top_k}",
        "",
        "## Metrics",
        "",
        "| Strategy | AUROC | AUPRC | F1 | Precision | Recall | MCC | ACC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {strategy_name[row['strategy']]} | {row['auroc']:.4f} | "
            f"{row['auprc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['mcc']:.4f} | {row['accuracy']:.4f} |"
        )

    random_row = next(row for row in rows if row["strategy"] == "random")
    best_precision = max(rows, key=lambda r: r["precision"])
    best_mcc = max(rows, key=lambda r: r["mcc"])
    lines += [
        "",
        "## Interpretation",
        "",
        f"Compared with random undersampling, the highest precision was obtained by "
        f"{strategy_name[best_precision['strategy']]} "
        f"(precision {random_row['precision']:.4f} -> {best_precision['precision']:.4f}).",
        f"The highest MCC was obtained by {strategy_name[best_mcc['strategy']]} "
        f"(MCC {random_row['mcc']:.4f} -> {best_mcc['mcc']:.4f}).",
        "",
        "## Suggested thesis text",
        "",
        (
            "针对固定阈值下模型Precision相对偏低、Recall较高的问题，本文进一步比较了不同负样本"
            "欠采样策略对SaProt-650M-AF2+MLP模型性能的影响。在保持训练/验证划分、分类器结构、"
            "集成轮数和1:1正负样本抽样比例一致的条件下，分别评估随机欠采样、K-means代表性欠采样"
            "以及K-means边界/困难负样本欠采样。结果显示，不同欠采样策略会影响Precision、F1和MCC等"
            "固定阈值指标，提示负样本空间覆盖方式是假阳性控制的重要影响因素。K-means代表性欠采样"
            "侧重覆盖负样本主要分布，K-means困难负样本欠采样则更强调边界样本区分能力，可作为后续"
            "提升模型特异性和降低假阳性的优化方向。"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare negative undersampling strategies.")
    parser.add_argument("--pos-ids", required=True)
    parser.add_argument("--neg-ids", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "results" / "undersampling_strategy_comparison"))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cluster-top-k", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    pos_ids = load_ids(args.pos_ids)
    neg_ids = load_ids(args.neg_ids)
    embedding_dir = Path(args.embedding_dir)

    print("=" * 80)
    print("Undersampling strategy comparison: SaProt-650M-AF2 + MLP")
    print("=" * 80)
    print(f"Loading embeddings from {embedding_dir}")
    X_pos, found_pos, missing_pos = load_embeddings_by_ids(pos_ids, embedding_dir)
    X_neg, found_neg, missing_neg = load_embeddings_by_ids(neg_ids, embedding_dir)
    print(f"Loaded positives: {len(found_pos)} / {len(pos_ids)}")
    print(f"Loaded negatives: {len(found_neg)} / {len(neg_ids)}")

    if missing_pos or missing_neg:
        with open(output_dir / "missing_ids.txt", "w") as f:
            for pid in missing_pos:
                f.write(f"positive\t{pid}\n")
            for pid in missing_neg:
                f.write(f"negative\t{pid}\n")

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * len(found_pos) + [0] * len(found_neg))
    ids = found_pos + found_neg
    in_dim = X.shape[1]
    X_train, y_train, X_val, y_val, val_ids = train_val_split(
        X, y, ids, val_ratio=args.val_ratio, seed=args.seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    n_pos_train = int((y_train == 1).sum())
    n_clusters = n_pos_train
    print(f"Building K-means pools with {n_clusters} clusters")
    cluster_pools = build_cluster_pools(
        X_train_scaled,
        y_train,
        n_clusters=n_clusters,
        seed=args.seed,
        top_k=args.cluster_top_k,
    )

    descriptions = {
        "random": "Random undersampling baseline",
        "kmeans_representative": "Select near-centroid negatives from K-means clusters",
        "kmeans_hard": "Select positives-proximal negatives within K-means clusters",
    }
    rows = []
    raw = {
        "config": vars(args),
        "dataset": {
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "n_val_pos": int(y_val.sum()),
            "n_val_neg": int((y_val == 0).sum()),
            "in_dim": int(in_dim),
        },
        "cluster": {
            "n_clusters": int(n_clusters),
            "n_nonempty_clusters": int(cluster_pools["n_nonempty_clusters"]),
        },
        "metrics": {},
    }

    for strategy in ["random", "kmeans_representative", "kmeans_hard"]:
        print(f"\nRunning strategy: {strategy}")
        probs, metrics = train_strategy(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            strategy=strategy,
            in_dim=in_dim,
            n_rounds=args.rounds,
            seed=args.seed,
            device=args.device,
            cluster_pools=cluster_pools,
        )
        row = {
            "strategy": strategy,
            "description": descriptions[strategy],
            "n_rounds": args.rounds,
            **metrics,
        }
        rows.append(row)
        raw["metrics"][strategy] = metrics
        np.save(output_dir / f"{strategy}_val_probs.npy", probs)

    write_csv(output_dir / "undersampling_strategy_metrics.csv", rows)
    with open(output_dir / "undersampling_strategy_results.json", "w") as f:
        json.dump(raw, f, indent=2)
    plot_metrics(rows, output_dir / "undersampling_strategy_comparison")
    write_summary(
        output_dir / "undersampling_strategy_summary.md",
        rows,
        raw["dataset"],
        args,
        raw["cluster"],
    )

    print("\nDone. Results written to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()

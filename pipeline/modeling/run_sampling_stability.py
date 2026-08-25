#!/usr/bin/env python3
"""
Stability analysis for the balanced subsampling ensemble strategy.

This script fixes the upstream/downstream model to SaProt-650M-AF2 + MLP and
evaluates two factors:
  1. negative sampling ratio per ensemble round (1:1, 1:2, 1:4)
  2. number of ensemble rounds (50, 100, 200, 300)

Outputs are written to a standalone results directory and are intended for
direct inclusion in the thesis as a supplemental robustness analysis.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def train_trace(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    in_dim,
    neg_ratio,
    n_rounds,
    checkpoints,
    seed,
    device,
):
    """
    Train an MLP ensemble and return metrics at selected cumulative rounds.

    For a checkpoint k, probabilities are averaged over the first k trained
    models. This makes round-number stability analysis deterministic and avoids
    rerunning the same 1:1 configuration multiple times.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)
    n_neg_sample = min(len(neg_idx), int(round(n_pos * neg_ratio)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    cumulative_probs = np.zeros(len(y_val), dtype=float)
    checkpoint_set = set(checkpoints)
    checkpoint_metrics = {}

    start = time.time()
    for i in range(n_rounds):
        round_seed = seed + i + int(neg_ratio * 1000)
        set_global_seed(round_seed)

        neg_sample = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(idx)

        clf = MLPClassifier(
            in_dim,
            hidden_dims=(512, 256),
            use_bn=True,
            dropout=0.3,
            device=device,
        )
        clf.fit(X_train_scaled[idx], y_train[idx])
        cumulative_probs += clf.predict_proba(X_val_scaled)[:, 1]

        current_round = i + 1
        if current_round in checkpoint_set:
            avg_probs = cumulative_probs / current_round
            metric = compute_metrics(y_val, avg_probs)
            checkpoint_metrics[current_round] = metric
            elapsed = time.time() - start
            print(
                f"  ratio=1:{neg_ratio:g} round={current_round}/{n_rounds} "
                f"AUROC={metric['auroc']:.4f} AUPRC={metric['auprc']:.4f} "
                f"F1={metric['f1']:.4f} MCC={metric['mcc']:.4f} "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )
        elif current_round % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  ratio=1:{neg_ratio:g} round={current_round}/{n_rounds} "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    return checkpoint_metrics


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_lines(rows, x_key, title, xlabel, output_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    axes = axes.ravel()
    metrics = ["auroc", "auprc", "f1", "mcc"]

    x = [row[x_key] for row in rows]
    for ax, metric in zip(axes, metrics):
        y = [row[metric] for row in rows]
        ax.plot(x, y, marker="o", linewidth=2, color="#2F6F9F")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(f"{output_prefix}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_summary(path, ratio_rows, round_rows, args, dataset_info):
    def fmt(row, key):
        return f"{row[key]:.4f}"

    best_ratio = max(ratio_rows, key=lambda r: r["auprc"])
    final_round = [r for r in round_rows if r["n_rounds"] == max(x["n_rounds"] for x in round_rows)][0]
    prev_round = [r for r in round_rows if r["n_rounds"] == 200]
    delta_200_300 = None
    if prev_round:
        prev = prev_round[0]
        delta_200_300 = {
            key: final_round[key] - prev[key]
            for key in ["auroc", "auprc", "f1", "mcc"]
        }

    lines = [
        "# Sampling Stability Analysis",
        "",
        "## Configuration",
        "",
        f"- Model: SaProt-650M-AF2 + MLP",
        f"- Positive IDs: `{args.pos_ids}`",
        f"- Negative IDs: `{args.neg_ids}`",
        f"- Embedding directory: `{args.embedding_dir}`",
        f"- Validation ratio: {args.val_ratio}",
        f"- Seed: {args.seed}",
        f"- Device: {args.device}",
        f"- Loaded positives: {dataset_info['n_pos']}",
        f"- Loaded negatives: {dataset_info['n_neg']}",
        f"- Validation positives: {dataset_info['n_val_pos']}",
        f"- Validation negatives: {dataset_info['n_val_neg']}",
        "",
        "## Negative Sampling Ratio Experiment",
        "",
        "The ensemble round number was fixed at 300, and the negative-to-positive sampling ratio per round was varied.",
        "",
        "| Positive:negative ratio | AUROC | AUPRC | F1 | Precision | Recall | MCC | ACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ratio_rows:
        lines.append(
            f"| 1:{row['neg_ratio']:g} | {fmt(row,'auroc')} | {fmt(row,'auprc')} | "
            f"{fmt(row,'f1')} | {fmt(row,'precision')} | {fmt(row,'recall')} | "
            f"{fmt(row,'mcc')} | {fmt(row,'accuracy')} |"
        )

    lines += [
        "",
        f"The best AUPRC in this scan was observed at 1:{best_ratio['neg_ratio']:g} "
        f"(AUPRC={best_ratio['auprc']:.4f}), while all tested ratios retained high AUROC values.",
        "",
        "## Ensemble Round Number Experiment",
        "",
        "The negative sampling ratio was fixed at 1:1, and cumulative ensemble predictions were evaluated at different round numbers.",
        "",
        "| Ensemble rounds | AUROC | AUPRC | F1 | Precision | Recall | MCC | ACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in round_rows:
        lines.append(
            f"| {row['n_rounds']} | {fmt(row,'auroc')} | {fmt(row,'auprc')} | "
            f"{fmt(row,'f1')} | {fmt(row,'precision')} | {fmt(row,'recall')} | "
            f"{fmt(row,'mcc')} | {fmt(row,'accuracy')} |"
        )

    if delta_200_300:
        lines += [
            "",
            "From 200 to 300 ensemble rounds, the absolute changes were "
            f"AUROC={abs(delta_200_300['auroc']):.4f}, "
            f"AUPRC={abs(delta_200_300['auprc']):.4f}, "
            f"F1={abs(delta_200_300['f1']):.4f}, and "
            f"MCC={abs(delta_200_300['mcc']):.4f}.",
        ]

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling-ratio and ensemble-round stability analysis."
    )
    parser.add_argument("--pos-ids", required=True)
    parser.add_argument("--neg-ids", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "results" / "sampling_stability"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[1, 2, 4])
    parser.add_argument("--round-checkpoints", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--ratio-rounds", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_ids = load_ids(args.pos_ids)
    neg_ids = load_ids(args.neg_ids)
    embedding_dir = Path(args.embedding_dir)

    print("=" * 72)
    print("Sampling stability analysis: SaProt-650M-AF2 + MLP")
    print("=" * 72)
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
    dataset_info = {
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_val_pos": int(y_val.sum()),
        "n_val_neg": int((y_val == 0).sum()),
        "in_dim": int(in_dim),
    }
    print(
        f"Dataset: pos={dataset_info['n_pos']} neg={dataset_info['n_neg']} "
        f"dim={in_dim}"
    )
    print(
        f"Validation: pos={dataset_info['n_val_pos']} "
        f"neg={dataset_info['n_val_neg']}"
    )

    max_rounds = max(args.round_checkpoints)
    ratio_rows = []
    round_rows = []
    raw_results = {
        "config": vars(args),
        "dataset": dataset_info,
        "ratio_experiment": {},
        "round_experiment": {},
    }

    for ratio in args.ratios:
        checkpoints = args.round_checkpoints if ratio == 1 and max_rounds == args.ratio_rounds else [args.ratio_rounds]
        print(f"\nRunning negative sampling ratio 1:{ratio:g} for {args.ratio_rounds} rounds")
        metrics_by_round = train_trace(
            X_train,
            y_train,
            X_val,
            y_val,
            in_dim=in_dim,
            neg_ratio=ratio,
            n_rounds=args.ratio_rounds,
            checkpoints=checkpoints,
            seed=args.seed,
            device=args.device,
        )
        raw_results["ratio_experiment"][str(ratio)] = {
            str(k): v for k, v in metrics_by_round.items()
        }

        final_metrics = metrics_by_round[args.ratio_rounds]
        row = {
            "neg_ratio": ratio,
            "n_rounds": args.ratio_rounds,
            **final_metrics,
        }
        ratio_rows.append(row)

        if ratio == 1:
            for checkpoint in args.round_checkpoints:
                if checkpoint in metrics_by_round:
                    round_rows.append({
                        "neg_ratio": ratio,
                        "n_rounds": checkpoint,
                        **metrics_by_round[checkpoint],
                    })
                    raw_results["round_experiment"][str(checkpoint)] = metrics_by_round[checkpoint]

    ratio_rows = sorted(ratio_rows, key=lambda r: r["neg_ratio"])
    round_rows = sorted(round_rows, key=lambda r: r["n_rounds"])

    ratio_csv = output_dir / "sampling_ratio_metrics.csv"
    round_csv = output_dir / "ensemble_round_metrics.csv"
    metric_fields = ["neg_ratio", "n_rounds"] + METRIC_KEYS
    write_csv(ratio_csv, ratio_rows, metric_fields)
    write_csv(round_csv, round_rows, metric_fields)

    with open(output_dir / "sampling_stability_results.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    plot_metric_lines(
        ratio_rows,
        "neg_ratio",
        "Effect of negative sampling ratio",
        "Negative samples per positive sample",
        output_dir / "sampling_ratio_stability",
    )
    plot_metric_lines(
        round_rows,
        "n_rounds",
        "Effect of ensemble round number",
        "Number of ensemble rounds",
        output_dir / "ensemble_round_stability",
    )
    write_summary(
        output_dir / "sampling_stability_summary.md",
        ratio_rows,
        round_rows,
        args,
        dataset_info,
    )

    print("\nDone. Results written to:")
    print(f"  {output_dir}")
    print(f"  {ratio_csv}")
    print(f"  {round_csv}")


if __name__ == "__main__":
    main()

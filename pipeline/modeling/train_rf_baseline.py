"""
RF baseline: plain Random Forest on RF handcraft features with RF dataset.
No balanced subsampling ensemble — just standard RF with class_weight='balanced'.

Usage:
  python scripts/train_rf_baseline.py \
    --rf-features data/RF/RF_features.xlsx \
    --pos-ids data/RF/positive_ids.txt \
    --neg-ids data/RF/negative_ids.txt \
    --output-dir results/cross_comparison/rf_baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve,
    precision_score, recall_score, average_precision_score,
)


def load_ids(id_file):
    with open(id_file) as f:
        return [l.strip() for l in f if l.strip() and l.strip() != "positive"]


def load_rf_features(xlsx_path, ids):
    df = pd.read_excel(xlsx_path).set_index("UniprotID")
    feature_cols = [c for c in df.columns]
    X, found, missing = [], [], []
    for pid in ids:
        if pid in df.index:
            X.append(df.loc[pid].values.astype(float))
            found.append(pid)
        else:
            missing.append(pid)
    return np.array(X) if X else np.array([]), found, missing


def train_val_split(X, y, ids, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    def split(idx):
        n = max(1, int(len(idx) * val_ratio))
        v = rng.choice(idx, size=n, replace=False)
        return np.setdiff1d(idx, v), v
    pt, pv = split(pos_idx)
    nt, nv = split(neg_idx)
    ti = np.concatenate([pt, nt])
    vi = np.concatenate([pv, nv])
    return X[ti], y[ti], X[vi], y[vi], [ids[i] for i in vi]


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
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-features", default=str(Path(__file__).resolve().parents[2] / "data" / "RF" / "RF_features.xlsx"))
    parser.add_argument("--pos-ids", required=True)
    parser.add_argument("--neg-ids", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pos_ids = load_ids(args.pos_ids)
    neg_ids = load_ids(args.neg_ids)
    print(f"IDs: {len(pos_ids)} positive, {len(neg_ids)} negative")

    print("Loading RF features ...")
    X_pos, fp, mp = load_rf_features(args.rf_features, pos_ids)
    X_neg, fn, mn = load_rf_features(args.rf_features, neg_ids)
    print(f"  Positive: {len(fp)} loaded, {len(mp)} missing")
    print(f"  Negative: {len(fn)} loaded, {len(mn)} missing")

    X = np.concatenate([X_pos, X_neg])
    y = np.array([1]*len(fp) + [0]*len(fn))
    ids = fp + fn

    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            X[m, j] = med[j]

    print(f"Dataset: {len(y)} samples, dim={X.shape[1]} "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())}, ratio=1:{(y==0).sum()/y.sum():.1f})")

    X_tr, y_tr, X_val, y_val, val_ids = train_val_split(X, y, ids, args.val_ratio, args.seed)
    print(f"Train: {len(y_tr)} | Val: {len(y_val)}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    print("Training plain Random Forest (class_weight='balanced') ...")
    clf = RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    val_probs = clf.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, val_probs)
    print(f"\n{'='*60}")
    print(f"RF Baseline Results:")
    print(f"  AUROC={metrics['auroc']}  AUPRC={metrics['auprc']}  "
          f"F1={metrics['f1']}  Prec={metrics['precision']}  "
          f"Rec={metrics['recall']}  MCC={metrics['mcc']}  ACC={metrics['accuracy']}")
    print(f"{'='*60}")

    with open(out / "results.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k not in ("fpr","tpr")}, f, indent=2)

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics["fpr"], metrics["tpr"], lw=2, color="steelblue",
            label=f'RF Baseline (AUC={metrics["auroc"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC — RF Baseline (Handcraft Features + RF Dataset)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curve.svg", format="svg")
    plt.close(fig)

    with open(out / "val_predictions.csv", "w") as f:
        f.write("protein_id,label,probability\n")
        for pid, lab, prob in sorted(zip(val_ids, y_val, val_probs), key=lambda x: x[2], reverse=True):
            f.write(f"{pid},{lab},{prob:.6f}\n")

    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

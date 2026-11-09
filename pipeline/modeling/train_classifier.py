"""
Balanced Subsampling Ensemble Classifier for RNA-binding protein prediction.

For each upstream embedding model × downstream classifier combination:
  - Randomly split positive/negative into train/val (default 8:2)
  - Run n_rounds of balanced subsampling (sample |pos| negatives per round)
  - Train one classifier per round → ensemble of n_rounds classifiers
  - Evaluate on val set via soft voting (mean probability)
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve,
    precision_score, recall_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# ── MLP (PyTorch) ────────────────────────────────────────────────────────────
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_ROOT = Path(
    os.environ.get("RLBPPRED_EMBEDDINGS", REPOSITORY_ROOT / "data" / "embeddings")
)

UPSTREAM_MODELS = [
    "esmc-300m",
    "esmc_600m",
    "pst_t33",
    "pst_t33_so",
    "saprot-650m-af2",
    "saprot-650m-pdb",
]


def load_embeddings(model_name: str):
    """Return (X, y, ids) numpy arrays for a given upstream model."""
    base = EMBEDDING_ROOT / model_name
    pos_dir = base / "positive_mean"
    neg_dir = base / "negative_mean"

    X, y, ids = [], [], []
    for pt_path in sorted(pos_dir.glob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        X.append(data["embeddings"].float().numpy())
        y.append(1)
        ids.append(pt_path.stem)
    for pt_path in sorted(neg_dir.glob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        X.append(data["embeddings"].float().numpy())
        y.append(0)
        ids.append(pt_path.stem)

    return np.array(X), np.array(y), ids


def train_val_split(X, y, ids, val_ratio=0.2, seed=42):
    """Stratified split: keep class ratio in val set."""
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
# Classifier definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_sklearn_classifiers(device="cpu"):
    xgb_device = "cuda" if device == "cuda" else "cpu"
    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "SVM":                lambda: SVC(kernel="linear", probability=True, C=1.0),
        "RandomForest":       lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "XGBoost":            lambda: xgb.XGBClassifier(n_estimators=100,
                                                         eval_metric="logloss", verbosity=0,
                                                         device=xgb_device),
        "LightGBM":           lambda: lgb.LGBMClassifier(n_estimators=100, verbose=-1),
        "KNN":                lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "GaussianNB":         lambda: GaussianNB(),
    }


# ── MLP wrappers ─────────────────────────────────────────────────────────────

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


class _RNNClassifier(nn.Module):
    """Treat the embedding as a sequence of fixed-size steps for RNN/LSTM."""
    def __init__(self, in_dim, hidden_size, num_layers, rnn_type="RNN", dropout=0.0):
        super().__init__()
        # split embedding into steps: (step_size=64, seq_len=in_dim//64)
        self.step_size = 64
        self.seq_len = in_dim // self.step_size
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.RNN
        self.rnn = rnn_cls(
            input_size=self.step_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, in_dim) → (batch, seq_len, step_size)
        x = x[:, :self.seq_len * self.step_size].view(x.size(0), self.seq_len, self.step_size)
        out, _ = self.rnn(x)
        last = out[:, -1, :]          # last timestep
        return self.fc(last).squeeze(-1)


class _SequenceClassifierWrapper:
    """Sklearn-compatible wrapper for RNN/LSTM."""

    def __init__(self, in_dim, hidden_size, num_layers, rnn_type,
                 dropout=0.0, lr=1e-3, epochs=50, batch_size=64, device="cpu"):
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y):
        self.model = _RNNClassifier(
            self.in_dim, self.hidden_size, self.num_layers,
            self.rnn_type, self.dropout
        ).to(self.device)
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


class MLPClassifier:
    """Sklearn-compatible wrapper around a PyTorch MLP."""

    def __init__(self, in_dim, hidden_dims, dropout=0.0, use_bn=False,
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
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

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
        probs = 1 / (1 + np.exp(-logits))          # sigmoid
        return np.stack([1 - probs, probs], axis=1)


def get_deep_classifiers(in_dim, device="cpu"):
    return {
        # ── MLP variants ──────────────────────────────────────────────────────
        "MLP":   lambda: MLPClassifier(in_dim, [512, 256],      use_bn=True, dropout=0.3, device=device),
        # ── RNN variants ──────────────────────────────────────────────────────
        "RNN":        lambda: _SequenceClassifierWrapper(in_dim, hidden_size=256, num_layers=2, rnn_type="RNN",  dropout=0.2, device=device),
        # ── LSTM variants ─────────────────────────────────────────────────────
        "LSTM":       lambda: _SequenceClassifierWrapper(in_dim, hidden_size=256, num_layers=2, rnn_type="LSTM", dropout=0.2, device=device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def balanced_subsample_ensemble(
    X_train, y_train, X_val, y_val,
    clf_factory, n_rounds=300, seed=42,
    scale_features=True,
):
    """
    Train n_rounds classifiers on balanced subsets.
    Returns val probabilities (mean over ensemble) and list of fitted models.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    val_probs = np.zeros(len(y_val))
    models = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for i in range(n_rounds):
            # balanced sample: all positives + equal number of negatives
            neg_sample = rng.choice(neg_idx, size=n_pos, replace=False)
            idx = np.concatenate([pos_idx, neg_sample])
            rng.shuffle(idx)

            Xb, yb = X_train[idx], y_train[idx]

            clf = clf_factory()
            clf.fit(Xb, yb)
            val_probs += clf.predict_proba(X_val)[:, 1]
            models.append(clf)

            if (i + 1) % 50 == 0:
                print(f"    round {i+1}/{n_rounds}", flush=True)

    val_probs /= n_rounds
    return val_probs, models, scaler


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probs)
    auroc = round(roc_auc_score(y_true, probs), 4)
    return {
        "auroc":     auroc,
        "auprc":     round(average_precision_score(y_true, probs), 4),
        "f1":        round(f1_score(y_true, preds, zero_division=0), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, preds), 4),
        "accuracy":  round(accuracy_score(y_true, preds), 4),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    warnings.warn("General/legacy comparison entry point. Current principal strict401 replay uses strict_runtime.py and frozen manifests.", RuntimeWarning)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    all_results = {}
    roc_data = []   # list of (label, fpr, tpr, auroc)

    upstream_list = args.upstream if args.upstream else UPSTREAM_MODELS

    for upstream in upstream_list:
        print(f"\n{'='*60}")
        print(f"Upstream model: {upstream}")
        print(f"{'='*60}")

        X, y, ids = load_embeddings(upstream)
        in_dim = X.shape[1]
        print(f"  Loaded {len(y)} samples, dim={in_dim} "
              f"(pos={y.sum()}, neg={(y==0).sum()})")

        X_train, y_train, X_val, y_val, val_ids = train_val_split(
            X, y, ids, val_ratio=args.val_ratio, seed=args.seed
        )
        print(f"  Train: {len(y_train)} | Val: {len(y_val)}")

        classifiers = {**get_sklearn_classifiers(device=device),
                       **get_deep_classifiers(in_dim, device=device)}

        clf_list = args.classifiers if args.classifiers else list(classifiers.keys())
        all_results[upstream] = {}

        for clf_name in clf_list:
            if clf_name not in classifiers:
                print(f"  [SKIP] Unknown classifier: {clf_name}")
                continue

            print(f"\n  Classifier: {clf_name}")
            try:
                val_probs, models, scaler = balanced_subsample_ensemble(
                    X_train, y_train, X_val, y_val,
                    clf_factory=classifiers[clf_name],
                    n_rounds=args.n_rounds,
                    seed=args.seed,
                    scale_features=args.scale_features,
                )
                metrics = compute_metrics(y_val, val_probs)
                all_results[upstream][clf_name] = {
                    k: v for k, v in metrics.items() if k not in ("fpr", "tpr")
                }
                print(f"    AUROC={metrics['auroc']}  AUPRC={metrics['auprc']}  "
                      f"F1={metrics['f1']}  Prec={metrics['precision']}  "
                      f"Rec={metrics['recall']}  MCC={metrics['mcc']}  ACC={metrics['accuracy']}")

                roc_data.append((
                    f"{upstream} / {clf_name}",
                    metrics["fpr"], metrics["tpr"], metrics["auroc"]
                ))

                # Save ensemble
                if args.save_models:
                    model_path = output_dir / upstream / clf_name
                    model_path.mkdir(parents=True, exist_ok=True)
                    with open(model_path / "ensemble.pkl", "wb") as f:
                        pickle.dump({"models": models, "scaler": scaler,
                                     "upstream": upstream, "in_dim": in_dim,
                                     "clf_name": clf_name, "n_rounds": args.n_rounds,
                                     "val_ids": val_ids, "val_probs": val_probs,
                                     "metrics": metrics}, f)

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[upstream][clf_name] = {"error": str(e)}

    # Save results summary
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot ROC curves
    if roc_data:
        plot_roc_curves(roc_data, output_dir / "roc_curves.svg")

    # Print leaderboard
    print_leaderboard(all_results)


def plot_roc_curves(roc_data, save_path):
    """Plot all ROC curves on a single figure, sorted by AUROC descending.
    Saves SVG + raw coordinate data (CSV) for publication use.
    """
    save_path = Path(save_path)
    roc_data_sorted = sorted(roc_data, key=lambda x: x[3], reverse=True)
    n = len(roc_data_sorted)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 8))

    for (label, fpr, tpr, auroc), color in zip(roc_data_sorted, colors):
        ax.plot(fpr, tpr, lw=1.2, color=color,
                label=f"{label}  (AUC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Upstream × Downstream Combinations", fontsize=13)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"ROC curves saved to {save_path}")

    # Save raw coordinate data as CSV for publication
    csv_path = save_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("label,auroc,fpr,tpr\n")
        for label, fpr, tpr, auroc in roc_data_sorted:
            for fp, tp in zip(fpr, tpr):
                f.write(f"{label},{auroc:.6f},{fp:.6f},{tp:.6f}\n")
    print(f"ROC coordinate data saved to {csv_path}")


def print_leaderboard(results):
    rows = []
    for upstream, clfs in results.items():
        for clf_name, metrics in clfs.items():
            if "auroc" in metrics:
                rows.append((upstream, clf_name,
                             metrics["auroc"], metrics.get("auprc", float("nan")),
                             metrics["f1"], metrics.get("precision", float("nan")),
                             metrics.get("recall", float("nan")),
                             metrics["mcc"], metrics["accuracy"]))
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n{'='*90}")
    print(f"{'LEADERBOARD (sorted by AUROC)':^90}")
    print(f"{'='*90}")
    print(f"{'Upstream':<20} {'Classifier':<22} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'MCC':>7} {'ACC':>7}")
    print(f"{'-'*90}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:<22} {r[2]:>7.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]:>7.4f} {r[6]:>7.4f} {r[7]:>7.4f} {r[8]:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Balanced Subsampling Ensemble for RBP classification")
    p.add_argument("--upstream", nargs="+", default=None,
                   help="Upstream models to run (default: all 6)")
    p.add_argument("--classifiers", nargs="+", default=None,
                   help="Classifiers to run (default: all)")
    p.add_argument("--n-rounds", type=int, default=300,
                   help="Number of ensemble rounds (default: 300)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Validation set ratio (default: 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="Device for MLP training: cpu or cuda")
    p.add_argument("--no-scale", dest="scale_features", action="store_false",
                   help="Disable StandardScaler (enabled by default)")
    p.add_argument("--save-models", action="store_true", default=True,
                   help="Save ensemble models to disk (default: True)")
    p.add_argument("--no-save-models", dest="save_models", action="store_false",
                   help="Disable saving ensemble models")
    p.add_argument("--output-dir", default=str(REPOSITORY_ROOT / "results" / "classifier_comparison"),
                   help="Output directory for results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

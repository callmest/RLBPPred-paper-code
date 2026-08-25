#!/usr/bin/env python3
"""
UMAP visualization of PST protein embeddings (positive vs negative).

Generates UMAP plots for pst_t33 and pst_t33_so models.
Outputs SVG for publication and saves UMAP coordinates as CSV source data.

Usage:
  python scripts/pst_embedding_umap.py
  python scripts/pst_embedding_umap.py --models pst_t33 pst_t33_so
  python scripts/pst_embedding_umap.py --seed 42 --n-neighbors 30 --min-dist 0.3
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap
from Bio import SeqIO


def load_embeddings(emb_dirs: list, ids: set) -> dict:
    """Load .pt embeddings for given IDs, searching across multiple dirs."""
    result = {}
    for pid in ids:
        for d in emb_dirs:
            pt_file = d / f"{pid}.pt"
            if pt_file.exists():
                data = torch.load(pt_file, weights_only=False)
                emb = data['embeddings']
                if isinstance(emb, torch.Tensor):
                    emb = emb.numpy()
                result[pid] = emb.flatten()
                break
    return result

def get_ids_from_fasta(fasta_file: Path) -> list:
    """Extract protein IDs from a FASTA file (ordered)."""
    return [r.id.split('|')[0] for r in SeqIO.parse(fasta_file, 'fasta')]


def save_source_data(pos_ids, neg_ids, embedding_2d, labels, save_path: Path):
    """Save UMAP coordinates and metadata as CSV for publication."""
    all_ids = pos_ids + neg_ids
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'umap_1', 'umap_2', 'label'])
        for i, pid in enumerate(all_ids):
            writer.writerow([pid, f"{embedding_2d[i, 0]:.6f}",
                             f"{embedding_2d[i, 1]:.6f}",
                             'positive' if labels[i] == 1 else 'negative'])
    print(f"  Source data: {save_path}")


def plot_umap(pos_embs: np.ndarray, neg_embs: np.ndarray,
              pos_ids: list, neg_ids: list,
              title: str, save_stem: Path, source_dir: Path,
              n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42):
    """Run UMAP, save SVG plot and source data CSV."""
    labels = np.array([1] * len(pos_embs) + [0] * len(neg_embs))
    X = np.vstack([pos_embs, neg_embs])

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, n_components=2)
    embedding_2d = reducer.fit_transform(X)

    # Save source data
    save_source_data(pos_ids, neg_ids, embedding_2d, labels,
                     source_dir / f"{save_stem.stem}.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax.scatter(embedding_2d[neg_mask, 0], embedding_2d[neg_mask, 1],
               c='#4393c3', s=8, alpha=0.4, label=f'Negative ({neg_mask.sum()})')
    ax.scatter(embedding_2d[pos_mask, 0], embedding_2d[pos_mask, 1],
               c='#d6604d', s=15, alpha=0.8, label=f'Positive ({pos_mask.sum()})')
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    fig.tight_layout()
    fig.savefig(save_stem.with_suffix('.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  SVG: {save_stem.with_suffix('.svg')}")


def process_model(model_name: str, pos_ids: list, neg_ids: list,
                  emb_dirs: list, output_dir: Path, source_dir: Path, args,
                  fixed_sampled_neg_ids: list = None):
    """Generate full and sampled UMAP plots for one model."""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    pos_set = set(pos_ids)
    neg_set = set(neg_ids)

    print("Loading positive embeddings...")
    pos_embs_dict = load_embeddings(emb_dirs, pos_set)
    print(f"  Loaded {len(pos_embs_dict)} positive embeddings")

    print("Loading negative embeddings...")
    neg_embs_dict = load_embeddings(emb_dirs, neg_set)
    print(f"  Loaded {len(neg_embs_dict)} negative embeddings")

    if len(pos_embs_dict) == 0 or len(neg_embs_dict) == 0:
        print("⚠️  Not enough embeddings, skipping")
        return

    # Keep order consistent
    pos_ids_found = [p for p in pos_ids if p in pos_embs_dict]
    neg_ids_found = [n for n in neg_ids if n in neg_embs_dict]
    pos_embs = np.stack([pos_embs_dict[p] for p in pos_ids_found])
    neg_embs = np.stack([neg_embs_dict[n] for n in neg_ids_found])

    # Plot 1: Full
    print(f"\nPlot 1: Full ({len(pos_embs)} pos vs {len(neg_embs)} neg)")
    plot_umap(pos_embs, neg_embs, pos_ids_found, neg_ids_found,
              f'{model_name} — Full ({len(pos_embs)} pos vs {len(neg_embs)} neg)',
              output_dir / f'{model_name}_umap_full', source_dir,
              n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed)

    # Plot 2: Sampled (balanced)
    if fixed_sampled_neg_ids is not None:
        sampled_neg_ids = [n for n in fixed_sampled_neg_ids if n in neg_embs_dict]
    else:
        n_sample = len(pos_embs)
        random.seed(args.seed)
        sampled_neg_ids = random.sample(neg_ids_found, min(n_sample, len(neg_ids_found)))
    neg_embs_sampled = np.stack([neg_embs_dict[k] for k in sampled_neg_ids])

    print(f"Plot 2: Sampled ({len(pos_embs)} pos vs {len(neg_embs_sampled)} neg)")
    plot_umap(pos_embs, neg_embs_sampled, pos_ids_found, sampled_neg_ids,
              f'{model_name} — Sampled ({len(pos_embs)} pos vs {len(neg_embs_sampled)} neg)',
              output_dir / f'{model_name}_umap_sampled', source_dir,
              n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed)


# ── PST model embedding directory registry ──────────────────────────────
# Maps PST models to their embedding directories
def get_model_emb_dirs(emb_base: Path) -> dict:
    return {
        'pst_t33': [
            emb_base / 'pst_t33' / 'positive_mean',
            emb_base / 'pst_t33' / 'negative_mean',
        ],
        'pst_t33_so': [
            emb_base / 'pst_t33_so' / 'positive_mean',
            emb_base / 'pst_t33_so' / 'negative_mean',
        ],
    }


ALL_MODELS = ['pst_t33', 'pst_t33_so']


def main():
    parser = argparse.ArgumentParser(
        description='UMAP visualization of PST embeddings (SVG + source data)')
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).resolve().parents[2] / 'data'))
    parser.add_argument('--models', nargs='+', default=None,
                        choices=ALL_MODELS,
                        help=f'PST models to plot (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for SVG plots (default: data/umap_plots)')
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampled-ids', type=str, default=None,
                        help='Path to a file with pre-selected negative IDs for sampled plots')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'umap_plots'
    source_dir = output_dir / 'source_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    # Load IDs from FASTA (ordered lists)
    pos_ids = get_ids_from_fasta(data_dir / 'positive.fasta')
    neg_ids = get_ids_from_fasta(data_dir / 'negative.fasta')
    print(f"Positive IDs: {len(pos_ids)}, Negative IDs: {len(neg_ids)}")

    emb_dirs_map = get_model_emb_dirs(data_dir / 'embeddings')

    # Load shared sampled negative IDs if provided
    fixed_sampled_neg_ids = None
    if args.sampled_ids:
        fixed_sampled_neg_ids = Path(args.sampled_ids).read_text().strip().split('\n')
        print(f"Using fixed sampled neg IDs: {len(fixed_sampled_neg_ids)} from {args.sampled_ids}")

    # Auto-detect available models if not specified
    models = args.models or ALL_MODELS
    for model in models:
        dirs = emb_dirs_map.get(model, [])
        if not any(d.exists() for d in dirs):
            print(f"\n⚠️  No embedding dirs found for {model}, skipping")
            continue
        process_model(model, pos_ids, neg_ids, dirs, output_dir, source_dir, args,
                      fixed_sampled_neg_ids=fixed_sampled_neg_ids)

    print(f"\nSVG plots saved to: {output_dir}")
    print(f"Source data saved to: {source_dir}")


if __name__ == '__main__':
    main()

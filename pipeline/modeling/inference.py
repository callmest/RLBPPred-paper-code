#!/usr/bin/env python3
"""
RLBP Inference: predict RNA-ligand binding probability from PDB structures.

Supports two input modes:
  1. End-to-end:    --pdb-dir → extract embeddings → ensemble predict
  2. Pre-extracted: --embedding-dir → ensemble predict

Usage examples:

  # End-to-end with ESMC model
  python scripts/inference.py \
    --pdb-dir data/query_pdb \
    --checkpoint results/classifier_comparison/esmc-300m/LogisticRegression/ensemble.pkl \
    --upstream esmc-300m \
    --device cuda

  # End-to-end with PST model (requires --model-path)
  python scripts/inference.py \
    --pdb-dir data/query_pdb \
    --checkpoint results/classifier_comparison/pst_t33/LogisticRegression/ensemble.pkl \
    --upstream pst_t33 \
    --model-path model/pst_t33.pt \
    --device cuda

  # End-to-end with SaProt model (requires --model-path and --foldseek)
  python scripts/inference.py \
    --pdb-dir data/query_pdb \
    --checkpoint results/classifier_comparison/saprot-650m-af2/LogisticRegression/ensemble.pkl \
    --upstream saprot-650m-af2 \
    --model-path model/saprot_650m_af2/SaProt_650M_AF2.pt \
    --device cuda

  # With pre-extracted embeddings (any upstream)
  python scripts/inference.py \
    --embedding-dir data/embeddings/esmc-300m/query_mean \
    --checkpoint results/classifier_comparison/esmc-300m/LogisticRegression/ensemble.pkl
"""

import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── Make train_classifier importable (for unpickling custom classifiers) ─────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
EMBEDDING_SCRIPT_DIR = SCRIPT_DIR.parent / "embeddings"
sys.path.insert(0, str(EMBEDDING_SCRIPT_DIR))
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from train_classifier import MLPClassifier, _SequenceClassifierWrapper  # noqa: E402
from train_sa_mlp import _MLP as SA_MLP, MLPClassifier as SA_MLPClassifier  # noqa: E402


class _CompatUnpickler(pickle.Unpickler):
    """Handle checkpoints pickled from scripts run as __main__."""

    def find_class(self, module, name):
        if module == "__main__" and name == "_MLP":
            return SA_MLP
        if module == "__main__" and name == "MLPClassifier":
            return SA_MLPClassifier
        return super().find_class(module, name)


# ─────────────────────────────────────────────────────────────────────────────
# PDB → Sequence extraction (for ESMC models)
# ─────────────────────────────────────────────────────────────────────────────

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from PDB file using CA atoms."""
    residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            res_name = line[17:20].strip()
            chain = line[21]
            res_seq = line[22:27].strip()
            key = (chain, res_seq)
            if key in seen:
                continue
            seen.add(key)
            aa = THREE_TO_ONE.get(res_name)
            if aa:
                residues.append(aa)
    return "".join(residues)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction by upstream model
# ─────────────────────────────────────────────────────────────────────────────

def extract_esmc_embeddings(pdb_dir: Path, device: str, model_name: str):
    """Extract ESMC embeddings from PDB files (sequence-based)."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    print(f"Loading ESMC model: {model_name} ...")
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    ids, embeddings = [], []

    for pdb_file in tqdm(pdb_files, desc=f"Extracting {model_name} embeddings"):
        protein_id = pdb_file.stem
        seq = extract_sequence_from_pdb(str(pdb_file))
        if not seq:
            print(f"  Warning: no sequence from {protein_id}, skipping")
            continue

        try:
            with torch.no_grad():
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                emb = logits_output.embeddings
                if hasattr(emb, "tensor"):
                    emb = emb.tensor
                emb = emb.squeeze(0).mean(dim=0).cpu().numpy()  # mean pool

            ids.append(protein_id)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Error processing {protein_id}: {e}")

    return ids, np.array(embeddings)


def extract_pst_embeddings(pdb_dir: Path, device: str,
                           model_path: str, model_name: str):
    """Extract PST embeddings from PDB files (structure-based)."""
    PST_DIR = os.environ.get(
        "PST_DIR", str(REPOSITORY_ROOT / "external" / "PST")
    )
    sys.path.insert(0, PST_DIR)
    sys.path.insert(0, str(Path(PST_DIR) / "scripts"))

    from pst.esm2 import PST

    # Import pdb_to_graph and extract_embedding from existing script
    from extract_pst_embedding import pdb_to_graph, extract_embedding

    include_seq = "so" in model_name.lower()

    print(f"Loading PST model: {model_name} from {model_path} ...")
    model, _ = PST.from_pretrained(model_path, map_location=torch.device(device))
    model.eval()
    model.to(device)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    ids, embeddings = [], []

    for pdb_file in tqdm(pdb_files, desc=f"Extracting {model_name} embeddings"):
        protein_id = pdb_file.stem
        try:
            data, _ = pdb_to_graph(str(pdb_file))
            emb = extract_embedding(data, model, device, include_seq=include_seq)
            ids.append(protein_id)
            embeddings.append(emb.cpu().numpy())
        except Exception as e:
            print(f"  Error processing {protein_id}: {e}")

    return ids, np.array(embeddings)


def extract_saprot_embeddings(pdb_dir: Path, device: str,
                              model_path: str, foldseek_bin: str):
    """Extract SaProt embeddings from PDB files (structure+sequence)."""
    SAPROT_DIR = os.environ.get(
        "SAPROT_DIR", str(REPOSITORY_ROOT / "external" / "SaProt")
    )
    sys.path.insert(0, SAPROT_DIR)

    from extract_saprot_embedding import load_saprot_model, tokenize_saprot_seq
    from generate_saprot_tokens import run_foldseek, apply_plddt_mask

    print(f"Loading SaProt model from {model_path} ...")
    model, alphabet = load_saprot_model(model_path)
    model = model.to(device)
    model.eval()

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    ids, embeddings = [], []

    for i, pdb_file in enumerate(tqdm(pdb_files, desc="Extracting SaProt embeddings")):
        protein_id = pdb_file.stem
        try:
            # Get 3Di tokens via foldseek
            seq_dict = run_foldseek(foldseek_bin, str(pdb_file), process_id=i)
            if not seq_dict:
                print(f"  Warning: foldseek failed for {protein_id}, skipping")
                continue

            chain = sorted(seq_dict.keys())[0]
            aa_seq, struc_seq = seq_dict[chain]

            # Apply pLDDT mask
            struc_seq = apply_plddt_mask(struc_seq, str(pdb_file), chain)

            # Combine: AA + lowercase 3Di
            combined_seq = "".join(a + b.lower() for a, b in zip(aa_seq, struc_seq))

            # Tokenize and extract embedding
            input_ids = tokenize_saprot_seq(combined_seq, alphabet).to(device)
            with torch.no_grad():
                results = model(input_ids, repr_layers=[model.num_layers])
                hidden = results["representations"][model.num_layers]
                emb = hidden[0, 1:-1, :].mean(dim=0)  # remove <cls>/<eos>, mean pool

            ids.append(protein_id)
            embeddings.append(emb.cpu().numpy())
        except Exception as e:
            print(f"  Error processing {protein_id}: {e}")

    return ids, np.array(embeddings)


def load_embeddings_from_dir(embedding_dir: Path):
    """Load pre-extracted mean-pooled embeddings from a directory of .pt files."""
    pt_files = sorted(embedding_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {embedding_dir}")

    ids, embeddings = [], []
    for pt_path in tqdm(pt_files, desc="Loading embeddings"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        ids.append(data["pdb_id"] if "pdb_id" in data else pt_path.stem)
        embeddings.append(data["embeddings"].float().numpy())

    return ids, np.array(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(X: np.ndarray, checkpoint: dict) -> np.ndarray:
    """
    Run ensemble soft-voting prediction.

    Args:
        X: Feature matrix (n_samples, embed_dim)
        checkpoint: Loaded ensemble checkpoint dict

    Returns:
        probabilities: (n_samples,) array of positive-class probabilities
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ESMC_MODELS = {"esmc-300m": "esmc_300m", "esmc_600m": "esmc_600m"}
PST_MODELS = {"pst_t33", "pst_t33_so"}
SAPROT_MODELS = {"saprot-650m-af2", "saprot-650m-pdb"}


def main():
    parser = argparse.ArgumentParser(
        description="RLBP Inference: predict RNA-ligand binding probability from PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input: either PDB directory or pre-extracted embeddings
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--pdb-dir", type=str,
                           help="Directory containing PDB files for end-to-end inference")
    input_grp.add_argument("--embedding-dir", type=str,
                           help="Directory containing pre-extracted .pt embeddings")

    # Checkpoint
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to ensemble.pkl saved by train_classifier.py")

    # Upstream model (required when using --pdb-dir)
    parser.add_argument("--upstream", type=str, default=None,
                        choices=list(ESMC_MODELS.keys()) + list(PST_MODELS) + list(SAPROT_MODELS),
                        help="Upstream embedding model (required with --pdb-dir)")

    # Model-specific options
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to upstream model checkpoint (required for PST and SaProt)")
    parser.add_argument("--foldseek", type=str,
                        default=os.environ.get("FOLDSEEK_BIN", "foldseek"),
                        help="Path to foldseek binary (for SaProt models)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: stdout)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")

    args = parser.parse_args()

    # Validate arguments
    if args.pdb_dir and not args.upstream:
        parser.error("--upstream is required when using --pdb-dir")
    if args.upstream in PST_MODELS and not args.model_path:
        parser.error(f"--model-path is required for PST model: {args.upstream}")
    if args.upstream in SAPROT_MODELS and not args.model_path:
        parser.error(f"--model-path is required for SaProt model: {args.upstream}")

    print("=" * 60)
    print("RLBP Inference")
    print("=" * 60)

    # ── Step 1: Get embeddings ───────────────────────────────────────────────
    if args.embedding_dir:
        print(f"Loading pre-extracted embeddings from {args.embedding_dir}")
        protein_ids, X = load_embeddings_from_dir(Path(args.embedding_dir))
    else:
        pdb_dir = Path(args.pdb_dir)
        assert pdb_dir.is_dir(), f"PDB directory not found: {pdb_dir}"

        if args.upstream in ESMC_MODELS:
            protein_ids, X = extract_esmc_embeddings(
                pdb_dir, args.device, ESMC_MODELS[args.upstream]
            )
        elif args.upstream in PST_MODELS:
            protein_ids, X = extract_pst_embeddings(
                pdb_dir, args.device, args.model_path, args.upstream
            )
        elif args.upstream in SAPROT_MODELS:
            protein_ids, X = extract_saprot_embeddings(
                pdb_dir, args.device, args.model_path, args.foldseek
            )

    print(f"Loaded {len(protein_ids)} proteins, embedding dim={X.shape[1]}")

    # ── Step 2: Load ensemble checkpoint ─────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint = _CompatUnpickler(f).load()

    n_models = len(checkpoint["models"])
    ckpt_upstream = checkpoint.get("upstream", "unknown")
    ckpt_clf = checkpoint.get("clf_name", "unknown")
    print(f"  Ensemble: {n_models} models ({ckpt_clf}), upstream={ckpt_upstream}")

    if "in_dim" in checkpoint and checkpoint["in_dim"] != X.shape[1]:
        print(f"  WARNING: embedding dim mismatch! "
              f"checkpoint expects {checkpoint['in_dim']}, got {X.shape[1]}")

    # ── Step 3: Predict ──────────────────────────────────────────────────────
    print("Running ensemble prediction ...")
    probs = ensemble_predict(X, checkpoint)

    # ── Step 4: Output results ───────────────────────────────────────────────
    results = sorted(zip(protein_ids, probs), key=lambda x: x[1], reverse=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["protein_id", "probability", "prediction"])
            for pid, prob in results:
                writer.writerow([pid, f"{prob:.6f}", int(prob >= args.threshold)])
        print(f"\nResults saved to {output_path}")
    else:
        # Print to stdout
        print(f"\n{'Protein ID':<20} {'Probability':>12} {'Prediction':>12}")
        print("-" * 46)
        for pid, prob in results:
            label = "RLBP" if prob >= args.threshold else "non-RLBP"
            print(f"{pid:<20} {prob:>12.6f} {label:>12}")

    # Summary
    n_pos = sum(1 for _, p in results if p >= args.threshold)
    print(f"\nSummary: {n_pos}/{len(results)} predicted as RLBP "
          f"(threshold={args.threshold})")

    if checkpoint.get("metrics"):
        m = checkpoint["metrics"]
        print(f"Checkpoint val metrics: AUROC={m.get('auroc')}, "
              f"AUPRC={m.get('auprc')}, F1={m.get('f1')}")


if __name__ == "__main__":
    main()

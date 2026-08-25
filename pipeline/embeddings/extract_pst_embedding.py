#!/usr/bin/env python3
"""
Extract PST embeddings from PDB files for RLBP project.

PST (Protein Structure Transformer) requires 3D structure (PDB files) as input.
Generates per-residue embeddings and applies mean pooling for protein-level representations.

Usage:
  python scripts/extract_pst_embedding.py \
    --pdb-dir data/positive_pdb \
    --output-dir data/embeddings/pst-t33/positive_mean \
    --model-path model/pst_t33.pt \
    --model-name pst_t33

  python scripts/extract_pst_embedding.py \
    --pdb-dir data/negative_pdb \
    --output-dir data/embeddings/pst-t33-so/negative_mean \
    --model-path model/pst_t33_so.pt \
    --model-name pst_t33_so \
    --include-seq
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add PST source path
PST_DIR = os.environ.get(
    "PST_DIR", str(Path(__file__).resolve().parents[2] / "external" / "PST")
)
sys.path.insert(0, PST_DIR)
sys.path.insert(0, str(Path(PST_DIR) / "scripts"))

import esm
import torch_geometric.nn as gnn
from biopandas.pdb import PandasPdb
from torch_geometric.data import Data
from pst.esm2 import PST
from pst.utils import AA_THREE_TO_ONE


def pdb_to_graph(pdb_file: str, eps: float = 8.0):
    """Convert PDB file to PyG Data object with 8Å radius graph."""
    esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    pdb_contents = PandasPdb().read_pdb(pdb_file).df["ATOM"]
    ca = pdb_contents[pdb_contents["atom_name"] == "CA"]

    # Extract CA coordinates
    structure = ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
    structure = torch.tensor(structure, dtype=torch.float)

    # Build 8Å radius graph
    edge_index = gnn.radius_graph(structure, r=eps, loop=False, num_workers=0)
    edge_index += 1  # shift for CLS token

    # Extract sequence and tokenize
    sequence = "".join(ca["residue_name"].apply(lambda x: AA_THREE_TO_ONE[x]).tolist())
    x = torch.cat([
        torch.LongTensor([esm_alphabet.cls_idx]),
        torch.LongTensor([esm_alphabet.get_idx(res)
                          for res in esm_alphabet.tokenize(sequence)]),
        torch.LongTensor([esm_alphabet.eos_idx]),
    ])

    # Mask for valid residues (exclude CLS/EOS)
    idx_mask = torch.zeros_like(x, dtype=torch.bool)
    idx_mask[1:-1] = True

    # Add batch and ptr attributes required by PST forward pass
    data = Data(x=x, edge_index=edge_index, idx_mask=idx_mask)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data.ptr = torch.tensor([0, x.size(0)], dtype=torch.long)

    return data, sequence


@torch.no_grad()
def extract_embedding(data, model, device, include_seq=False, aggr=None):
    """
    Extract protein-level embedding from a single PDB structure.

    Args:
        data: PyG Data object
        model: PST model
        device: cuda or cpu
        include_seq: Average structure + sequence representations (for _so models)
        aggr: Layer aggregation (None=last, 'mean', 'concat')

    Returns:
        Protein-level embedding tensor (mean pooled over residues)
    """
    data = data.to(device)

    # Get structure representation
    out = model(data, return_repr=True, aggr=aggr)
    out = out[data.idx_mask]  # Remove CLS/EOS tokens

    # Optionally add sequence representation (for _so models)
    if include_seq:
        data_seq = data.clone()
        data_seq.edge_index = None  # Remove edges for sequence-only
        out_seq = model(data_seq, return_repr=True, aggr=aggr)
        out_seq = out_seq[data_seq.idx_mask]
        out = (out + out_seq) * 0.5

    # Mean pooling over residues for protein-level embedding
    protein_emb = out.mean(dim=0)  # [embed_dim]
    return protein_emb


def process_pdb_dir(pdb_dir: Path, output_dir: Path, model, device,
                    model_name: str, include_seq: bool = False,
                    skip_existing: bool = True):
    """Process all PDB files in a directory and save embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    print(f"\nProcessing {len(pdb_files)} PDB files from {pdb_dir}")

    processed = 0
    skipped = 0
    failed = 0
    failed_ids = []

    for pdb_file in tqdm(pdb_files, desc="Extracting embeddings"):
        protein_id = pdb_file.stem
        output_file = output_dir / f"{protein_id}.pt"

        if skip_existing and output_file.exists():
            skipped += 1
            continue

        try:
            # Convert PDB to graph
            data, sequence = pdb_to_graph(str(pdb_file))

            # Extract embedding
            emb = extract_embedding(data, model, device, include_seq=include_seq)

            # Save
            torch.save({
                'pdb_id': protein_id,
                'sequence': sequence,
                'length': len(sequence),
                'embeddings': emb.cpu(),
                'embedding_dim': emb.shape[-1],
                'pooling': 'mean',
                'model': model_name,
            }, output_file)

            processed += 1

        except Exception as e:
            print(f"\nError processing {protein_id}: {e}")
            failed += 1
            failed_ids.append(protein_id)

    print(f"\nStatistics:")
    print(f"  Total:     {len(pdb_files)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    if failed_ids:
        print(f"  Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
    print(f"  Output:    {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PST embeddings from PDB files"
    )
    parser.add_argument("--pdb-dir", required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .pt files")
    parser.add_argument("--model-path", required=True,
                        help="Path to PST .pt checkpoint")
    parser.add_argument("--model-name", default=None,
                        help="Model name for metadata (auto-detected from path)")
    parser.add_argument("--include-seq", action="store_true",
                        help="Average structure + sequence representations (for _so models)")
    parser.add_argument("--aggr", default=None, choices=[None, "mean", "concat"],
                        help="Layer aggregation: None=last, mean, concat")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    # Auto-detect model name
    if args.model_name is None:
        args.model_name = Path(args.model_path).stem

    print("=" * 60)
    print(f"PST Embedding Extraction")
    print(f"  Model:   {args.model_name} ({args.model_path})")
    print(f"  Device:  {args.device}")
    print(f"  Include seq: {args.include_seq}")
    print("=" * 60)

    # Load model
    print("Loading PST model...")
    try:
        model, model_cfg = PST.from_pretrained(
            args.model_path,
            map_location=torch.device(args.device)
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with CPU map_location...")
        model, model_cfg = PST.from_pretrained(
            args.model_path,
            map_location=torch.device("cpu")
        )

    model.eval()
    model.to(args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Process PDB files
    process_pdb_dir(
        pdb_dir=Path(args.pdb_dir),
        output_dir=Path(args.output_dir),
        model=model,
        device=args.device,
        model_name=args.model_name,
        include_seq=args.include_seq,
        skip_existing=args.skip_existing,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

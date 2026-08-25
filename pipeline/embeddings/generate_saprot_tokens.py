#!/usr/bin/env python3
"""
Generate SaProt-format tokens (AA + 3Di) from PDB files using foldseek.
Outputs FASTA files with combined sequences for SaProt model input.
"""

import os
import sys
import argparse
import shutil
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add SaProt source for pLDDT extraction
SAPROT_DIR = os.environ.get(
    "SAPROT_DIR", str(Path(__file__).resolve().parents[2] / "external" / "SaProt")
)
sys.path.insert(0, SAPROT_DIR)

from Bio.PDB import PDBParser, MMCIFParser


def extract_plddt(pdb_path: str, chain: str) -> np.ndarray:
    """Extract pLDDT scores from PDB file."""
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain_obj = model[chain]

    plddts = []
    for residue in chain_obj:
        residue_plddts = [atom.get_bfactor() for atom in residue]
        plddts.append(np.mean(residue_plddts))

    return np.array(plddts)


def run_foldseek(foldseek_bin: str, pdb_path: str, process_id: int = 0) -> dict:
    """
    Run foldseek on a single PDB file to get 3Di sequence.
    Returns dict: {chain_id: (aa_seq, struc_seq, combined_seq)}
    """
    tmp_save_path = f"/tmp/foldseek_{process_id}_{time.time()}.tsv"
    cmd = f"{foldseek_bin} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 '{pdb_path}' '{tmp_save_path}'"
    os.system(cmd)

    seq_dict = {}
    pdb_name = os.path.basename(pdb_path)

    if not os.path.exists(tmp_save_path):
        return seq_dict

    with open(tmp_save_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            desc, aa_seq, struc_seq = parts[0], parts[1], parts[2]

            # Extract chain ID
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(pdb_name, "").split("_")[-1]

            if chain not in seq_dict:
                seq_dict[chain] = (aa_seq, struc_seq)

    # Cleanup
    for f in [tmp_save_path, tmp_save_path + ".dbtype"]:
        if os.path.exists(f):
            os.remove(f)

    return seq_dict


def apply_plddt_mask(struc_seq: str, pdb_path: str, chain: str,
                     plddt_threshold: float = 70.0) -> str:
    """Mask low-confidence regions in 3Di sequence with '#'."""
    try:
        plddts = extract_plddt(pdb_path, chain)
        if len(plddts) != len(struc_seq):
            print(f"  Warning: pLDDT length mismatch for {pdb_path} chain {chain}: "
                  f"{len(plddts)} vs {len(struc_seq)}, skipping mask")
            return struc_seq

        np_seq = np.array(list(struc_seq))
        np_seq[plddts < plddt_threshold] = "#"
        return "".join(np_seq)
    except Exception as e:
        print(f"  Warning: pLDDT extraction failed for {pdb_path}: {e}")
        return struc_seq


def process_pdb_dir(foldseek_bin: str, pdb_dir: str, output_fasta: str,
                    plddt_mask: bool = True, plddt_threshold: float = 70.0):
    """
    Process all PDB files in a directory, generate SaProt-format FASTA.
    Each entry: >protein_id\ncombined_seq (e.g. MdSaIwLp...)
    """
    pdb_files = sorted(Path(pdb_dir).glob("*.pdb"))
    print(f"Processing {len(pdb_files)} PDB files from {pdb_dir}")

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    success = 0
    failed = []

    with open(output_fasta, "w") as out_f:
        for i, pdb_file in enumerate(tqdm(pdb_files, desc="Generating 3Di")):
            protein_id = pdb_file.stem  # e.g. O00267
            pdb_path = str(pdb_file)

            seq_dict = run_foldseek(foldseek_bin, pdb_path, process_id=i)

            if not seq_dict:
                failed.append(protein_id)
                continue

            # Take the first chain (usually "A")
            chain = sorted(seq_dict.keys())[0]
            aa_seq, struc_seq = seq_dict[chain]

            # Apply pLDDT masking for AlphaFold2 structures
            if plddt_mask:
                struc_seq = apply_plddt_mask(
                    struc_seq, pdb_path, chain, plddt_threshold
                )

            # Combine: AA + lowercase 3Di
            combined_seq = "".join(
                a + b.lower() for a, b in zip(aa_seq, struc_seq)
            )

            out_f.write(f">{protein_id}\n{combined_seq}\n")
            success += 1

    print(f"Done: {success} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate SaProt-format tokens from PDB files"
    )
    parser.add_argument("--pdb-dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--output-fasta", required=True, help="Output FASTA file path")
    parser.add_argument("--foldseek", default=os.environ.get("FOLDSEEK_BIN", "foldseek"),
                        help="Path to foldseek binary")
    parser.add_argument("--plddt-mask", action="store_true", default=True,
                        help="Mask low pLDDT regions with # (default: True)")
    parser.add_argument("--no-plddt-mask", dest="plddt_mask", action="store_false")
    parser.add_argument("--plddt-threshold", type=float, default=70.0,
                        help="pLDDT threshold for masking (default: 70.0)")
    args = parser.parse_args()

    foldseek_bin = shutil.which(args.foldseek) or args.foldseek
    assert os.path.exists(foldseek_bin), f"foldseek not found: {args.foldseek}"
    assert os.path.isdir(args.pdb_dir), f"PDB dir not found: {args.pdb_dir}"

    process_pdb_dir(
        foldseek_bin=foldseek_bin,
        pdb_dir=args.pdb_dir,
        output_fasta=args.output_fasta,
        plddt_mask=args.plddt_mask,
        plddt_threshold=args.plddt_threshold,
    )


if __name__ == "__main__":
    main()

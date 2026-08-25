#!/usr/bin/env python3
"""
Experiment 1.Xb: De novo motif discovery on full-length protein sequences.

Novel Predicted RLBPs (n=342) vs Non-RLBPs (n=9,109) using STREME.
Unlike Exp 1.2b which used IDR subsequences, this uses complete protein sequences.
"""

import os
import subprocess
from pathlib import Path

BASE = Path(os.environ.get("RLBPPRED_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = BASE / "data" / "idr_analysis"
OUT_DIR = BASE / "results" / "idr_analysis" / "motif_discovery_fullseq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STREME_BIN = os.environ.get("STREME_BIN", "streme")

NOVEL_FASTA = DATA_DIR / "candidates_342.fasta"
NEGATIVE_FASTA = DATA_DIR / "negative_9109.fasta"


def run_streme(primary_fasta, control_fasta, output_dir):
    """Run STREME for de novo motif discovery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        STREME_BIN,
        "--p", str(primary_fasta),
        "--n", str(control_fasta),
        "--protein",
        "--minw", "3",
        "--maxw", "15",
        "--thresh", "0.05",
        "--oc", str(output_dir),
    ]
    print(f"Running STREME: Novel full-length vs Negative full-length")
    print(f"  Primary: {primary_fasta} ")
    print(f"  Control: {control_fasta}")
    print(f"  Output:  {output_dir}")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STREME stderr: {result.stderr[:1000]}")
        return None
    else:
        print(f"  STREME completed successfully")
        return output_dir


def parse_streme_results(streme_dir):
    """Parse STREME text output for motif summary."""
    txt_path = streme_dir / "streme.txt"
    if not txt_path.exists():
        print(f"  No streme.txt found in {streme_dir}")
        return []

    motifs = []
    with open(txt_path) as f:
        in_motif = False
        current = {}
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if current:
                    motifs.append(current)
                parts = line.split()
                current = {
                    "id": parts[1] if len(parts) > 1 else "",
                    "consensus": parts[2] if len(parts) > 2 else parts[1],
                }
                in_motif = True
            elif in_motif and line.startswith("letter-probability matrix"):
                # Parse width and nsites
                for token in line.split():
                    if token.startswith("w="):
                        current["width"] = int(token.split("=")[1])
                    elif token.startswith("nsites="):
                        current["nsites"] = int(token.split("=")[1])
                    elif token.startswith("E="):
                        current["evalue"] = float(token.split("=")[1])

        if current:
            motifs.append(current)

    print(f"\n  Found {len(motifs)} significant motifs:")
    for m in motifs:
        print(f"    {m.get('id','?'):12s}  w={m.get('width','?')}  "
              f"nsites={m.get('nsites','?')}  E={m.get('evalue','?')}")

    return motifs


def main():
    print("=" * 60)
    print("Experiment: Full-length motif discovery")
    print("  Novel Predicted RLBPs vs Non-RLBPs")
    print("=" * 60)

    # Verify input files
    for f in [NOVEL_FASTA, NEGATIVE_FASTA]:
        if not f.exists():
            print(f"ERROR: {f} not found")
            return
        n = sum(1 for line in open(f) if line.startswith(">"))
        print(f"  {f.name}: {n} sequences")

    print()
    streme_out = OUT_DIR / "novel_vs_negative"
    result_dir = run_streme(NOVEL_FASTA, NEGATIVE_FASTA, streme_out)

    if result_dir:
        motifs = parse_streme_results(result_dir)
        print(f"\nResults saved to: {result_dir}")
        print(f"  HTML report: {result_dir / 'streme.html'}")
        print(f"  Text output: {result_dir / 'streme.txt'}")


if __name__ == "__main__":
    main()

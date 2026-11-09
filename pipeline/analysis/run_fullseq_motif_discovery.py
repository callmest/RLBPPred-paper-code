#!/usr/bin/env python3
"""
Experiment 1.Xb: De novo motif discovery on full-length protein sequences.

Parameterized, explicitly opt-in upstream STREME execution.
Default invocation only parses the packaged current frozen output; it never
starts a new analysis. Historical 342-candidate inputs are not defaults.
"""

import os
import argparse
import subprocess
from pathlib import Path
try:
    from .current_frozen_analysis import parse_streme_text, parse_streme_xml, motif_summary
except ImportError:
    from current_frozen_analysis import parse_streme_text, parse_streme_xml, motif_summary

BASE = Path(os.environ.get("RLBPPRED_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = BASE / "data" / "idr_analysis"
OUT_DIR = BASE / "results" / "idr_analysis" / "motif_discovery_fullseq"

STREME_BIN = os.environ.get("STREME_BIN", "streme")

NOVEL_FASTA = DATA_DIR / "candidates_401.fasta"
NEGATIVE_FASTA = DATA_DIR / "negative_9109.fasta"


def run_streme(primary_fasta, control_fasta, output_dir):
    """Run STREME for de novo motif discovery."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        STREME_BIN,
        "--p", str(primary_fasta),
        "--n", str(control_fasta),
        "--protein",
        "--minw", "3",
        "--maxw", "15",
        "--thresh", "0.05",
        "--seed", "0", "--hofract", "0.1",
        "--o", str(output_dir),
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
    """Parse XML when present; robust whitespace-aware text fallback."""
    xml_path = Path(streme_dir) / "streme.xml"
    if xml_path.exists():
        return parse_streme_xml(xml_path)
    txt_path = streme_dir / "streme.txt"
    if not txt_path.exists():
        print(f"  No streme.txt found in {streme_dir}")
        return []

    return parse_streme_text(txt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parse-dir", type=Path, default=BASE / "data/frozen_analysis/streme401")
    parser.add_argument("--run-streme", action="store_true", help="Explicitly launch a new upstream analysis; not needed for figure replay.")
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run_streme:
        if not all([args.primary, args.control, args.output_dir]):
            parser.error("--run-streme requires --primary, --control, and a new --output-dir")
        for path, expected in [(args.primary, 401), (args.control, 9109)]:
            count = sum(line.startswith(">") for line in path.open())
            if count != expected:
                parser.error(f"{path.name}: expected {expected} sequences, found {count}")
        result_dir = run_streme(args.primary, args.control, args.output_dir)
        if result_dir is None:
            raise SystemExit(1)
    else:
        result_dir = args.parse_dir
    print(motif_summary(parse_streme_results(result_dir)))


if __name__ == "__main__":
    main()

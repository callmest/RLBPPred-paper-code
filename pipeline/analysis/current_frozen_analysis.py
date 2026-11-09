"""Pure, offline checks and summaries for the current frozen downstream analysis.

No model inference, motif discovery, live annotation downloads, or file writes
occur on import. Motif groups describe sequence composition, not function.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def motif_composition(consensus: str) -> str:
    if not consensus:
        raise ValueError("Empty motif consensus")
    if "C" in consensus:
        return "Cys-containing"
    if (consensus.count("K") + consensus.count("R")) / len(consensus) >= 1 / 3:
        return "Lys/Arg-rich"
    return "Other sequences"


def parse_streme_text(path: Path) -> list[dict]:
    pattern = (r"^MOTIF\s+(\S+)\s+(\S+)\s*\nletter-probability matrix:"
               r"\s+alength=\s*20\s+w=\s*(\d+)\s+nsites=\s*(\d+)\s+E=\s*([\d.eE+-]+)")
    motifs = []
    for rank, match in enumerate(re.finditer(pattern, path.read_text(), re.MULTILINE), 1):
        raw_id, alt, width, nsites, e_value = match.groups()
        consensus = raw_id.split("-", 1)[1] if "-" in raw_id else raw_id
        motifs.append({"rank": rank, "id": raw_id, "alt": alt,
                       "consensus": consensus, "width": int(width),
                       "nsites": int(nsites), "e_value": float(e_value)})
    if not motifs:
        raise ValueError(f"No STREME motifs parsed from {path}")
    return motifs


def parse_streme_xml(path: Path) -> list[dict]:
    """Use test E-values and npassing (unique proteins), never total_sites."""
    root = ET.parse(path).getroot()
    motifs = []
    for rank, element in enumerate(root.findall("./motifs/motif"), 1):
        a = element.attrib
        consensus = a["id"].split("-", 1)[-1]
        e_value = float(a["test_evalue"])
        motifs.append({"rank": rank, "id": a["id"], "alt": a.get("alt", ""),
                       "consensus": consensus, "width": int(a["width"]),
                       "unique_candidate_proteins": int(a["npassing"]),
                       "e_value": e_value, "composition_group": motif_composition(consensus),
                       "significant": e_value < .05, "displayed": e_value <= 1.2})
    if not motifs:
        raise ValueError(f"No STREME motifs parsed from {path}")
    return motifs


def motif_summary(motifs: list[dict]) -> dict:
    displayed = [m for m in motifs if m["e_value"] <= 1.2]
    totals = Counter(motif_composition(m["consensus"]) for m in displayed)
    significant = Counter(motif_composition(m["consensus"]) for m in displayed if m["e_value"] < .05)
    groups = ["Cys-containing", "Lys/Arg-rich", "Other sequences"]
    return {"reported": len(motifs), "significant": sum(m["e_value"] < .05 for m in motifs),
            "displayed": len(displayed),
            "groups": {g: {"significant": significant[g], "displayed": totals[g]} for g in groups}}


def verify_frozen_go(path: Path) -> dict:
    """Audit the complete frozen test family; this does not recompute GO tests."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3082 or len({r["term_id"] for r in rows}) != 3082:
        raise ValueError("Expected 3,082 unique eligible GO terms")
    if not all(int(r["n"]) == 401 and int(r["bg_n"]) == 5237 for r in rows):
        raise ValueError("GO foreground/background must be 401/5,237")
    counts = {aspect: {"tested": sum(r["aspect"] == aspect for r in rows),
                       "significant": sum(r["aspect"] == aspect and float(r["q_bh"]) < .05 for r in rows)}
              for aspect in ["BP", "CC", "MF"]}
    if sum(v["significant"] for v in counts.values()) != 88:
        raise ValueError("Expected 88 significant GO terms")
    return {"tested": len(rows), "significant": 88, "by_aspect": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streme-dir", type=Path, required=True)
    parser.add_argument("--go-csv", type=Path)
    args = parser.parse_args()
    xml = parse_streme_xml(args.streme_dir / "streme.xml")
    text = parse_streme_text(args.streme_dir / "streme.txt")
    if len(xml) != len(text):
        raise ValueError("STREME XML/text motif count mismatch")
    for a, b in zip(xml, text):
        if (a["consensus"], a["width"], a["e_value"], a["unique_candidate_proteins"]) != (b["consensus"], b["width"], b["e_value"], b["nsites"]):
            raise ValueError("STREME XML/text values differ")
    report = {"motifs": motif_summary(xml), "operation": "offline frozen-output audit"}
    if args.go_csv:
        report["go"] = verify_frozen_go(args.go_csv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

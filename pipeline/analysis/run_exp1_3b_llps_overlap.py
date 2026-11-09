#!/usr/bin/env python3
"""Experiment 1.3b: Overlap analysis between predicted RLBPs and known LLPS databases."""


# Historical 342-candidate analysis, retained for provenance only.
# Current frozen publication replay: python scripts/reproduce_figures.py --all
import sys as _historical_sys
if __name__ == "__main__":
    if "--historical-342" not in _historical_sys.argv:
        raise SystemExit("Historical 342-candidate workflow. Use the current frozen replay, or explicitly add --historical-342.")
    _historical_sys.argv.remove("--historical-342")
import json
import os
from pathlib import Path
from scipy import stats
import numpy as np

BASE = Path(os.environ.get("RLBPPRED_ROOT", Path(__file__).resolve().parents[2]))
DATA = BASE / "data"
RESULTS = BASE / "results" / "idr_analysis"
LLPS_DB = DATA / "llps_databases"


def load_protein_ids(fasta_path):
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                uid = line.strip().lstrip(">").split("|")[0].split("-")[0].strip()
                ids.add(uid)
    return ids


def load_phasepdb_human():
    with open(LLPS_DB / "phasepdb_all.json") as f:
        data = json.load(f)
    ids = set()
    for p in data:
        if "Homo sapiens" in p.get("organism", ""):
            uid = p.get("uniprot_id", "").split("-")[0]
            if uid:
                ids.add(uid)
    return ids


def load_drllps_human():
    ids = set()
    with open(LLPS_DB / "drllps_full.txt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5 and "Homo sapiens" in parts[4]:
                uid = parts[1].split("-")[0]
                if uid:
                    ids.add(uid)
    return ids


def load_phasepro_human():
    with open(LLPS_DB / "phasepro_full.json") as f:
        data = json.load(f)
    ids = set()
    for uid, info in data.items():
        if "Homo sapiens" in info.get("organism", ""):
            ids.add(uid.split("-")[0])
    return ids


def load_llpsdb_human():
    ids = set()
    with open(LLPS_DB / "llpsdb_v2_human_ids.txt") as f:
        for line in f:
            uid = line.strip()
            if uid:
                ids.add(uid)
    return ids


def fisher_enrichment(group_ids, llps_ids, background_ids):
    """Fisher exact test for enrichment of LLPS proteins in a group vs background."""
    group = set(group_ids)
    bg = set(background_ids)
    llps = set(llps_ids)

    a = len(group & llps)
    b = len(group - llps)
    c = len(bg & llps)
    d = len(bg - llps)

    table = [[a, b], [c, d]]
    odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")
    pct_group = a / len(group) * 100 if group else 0
    pct_bg = c / len(bg) * 100 if bg else 0

    return {
        "overlap_count": a,
        "group_total": len(group),
        "overlap_pct": round(pct_group, 2),
        "bg_overlap_count": c,
        "bg_total": len(bg),
        "bg_overlap_pct": round(pct_bg, 2),
        "odds_ratio": round(odds_ratio, 2) if odds_ratio != float("inf") else "inf",
        "p_value": p_value,
        "p_value_str": f"{p_value:.2e}",
    }


def main():
    novel = load_protein_ids(DATA / "idr_analysis" / "candidates_342.fasta")
    known = load_protein_ids(DATA / "idr_analysis" / "known_376.fasta")
    negative = load_protein_ids(DATA / "idr_analysis" / "negative_9109.fasta")

    print(f"Protein groups: Novel={len(novel)}, Known={len(known)}, Negative={len(negative)}")

    phasepdb = load_phasepdb_human()
    drllps = load_drllps_human()
    phasepro = load_phasepro_human()
    llpsdb = load_llpsdb_human()
    union_llps = phasepdb | drllps | phasepro | llpsdb

    print(f"\nLLPS databases (human):")
    print(f"  PhaSepDB:  {len(phasepdb)} proteins")
    print(f"  DrLLPS:    {len(drllps)} proteins")
    print(f"  PhaSePro:  {len(phasepro)} proteins")
    print(f"  LLPSDB:    {len(llpsdb)} proteins")
    print(f"  Union:     {len(union_llps)} proteins")
    print(f"  Overlap PhaSepDB∩DrLLPS: {len(phasepdb & drllps)}")

    results = {}

    for db_name, db_ids in [
        ("PhaSepDB", phasepdb),
        ("DrLLPS", drllps),
        ("PhaSePro", phasepro),
        ("LLPSDB", llpsdb),
        ("Union", union_llps),
    ]:
        print(f"\n{'='*60}")
        print(f"Database: {db_name} ({len(db_ids)} human proteins)")
        print(f"{'='*60}")

        db_results = {}
        for group_name, group_ids in [
            ("Novel_RLBP", novel),
            ("Known_RLBP", known),
            ("Non_RLBP", negative),
        ]:
            bg_ids = negative  # Use Non-RLBP as background
            r = fisher_enrichment(group_ids, db_ids, bg_ids)
            db_results[group_name] = r

            overlap_ids = group_ids & db_ids
            print(f"\n  {group_name}: {r['overlap_count']}/{r['group_total']} "
                  f"({r['overlap_pct']}%) | OR={r['odds_ratio']} | p={r['p_value_str']}")
            if overlap_ids and group_name != "Non_RLBP":
                print(f"    Overlapping IDs: {', '.join(sorted(overlap_ids)[:20])}")
                if len(overlap_ids) > 20:
                    print(f"    ... and {len(overlap_ids) - 20} more")

        results[db_name] = db_results

    # Save overlap protein lists
    overlap_details = {}
    for group_name, group_ids in [
        ("Novel_RLBP", novel),
        ("Known_RLBP", known),
    ]:
        overlap_details[group_name] = {}
        for db_name, db_ids in [
            ("PhaSepDB", phasepdb),
            ("DrLLPS", drllps),
            ("PhaSePro", phasepro),
            ("LLPSDB", llpsdb),
            ("Union", union_llps),
        ]:
            overlap_details[group_name][db_name] = sorted(group_ids & db_ids)

    output = {
        "database_sizes": {
            "PhaSepDB_human": len(phasepdb),
            "DrLLPS_human": len(drllps),
            "PhaSePro_human": len(phasepro),
            "LLPSDB_human": len(llpsdb),
            "Union_human": len(union_llps),
        },
        "group_sizes": {
            "Novel_RLBP": len(novel),
            "Known_RLBP": len(known),
            "Non_RLBP": len(negative),
        },
        "enrichment_results": results,
        "overlap_protein_lists": overlap_details,
    }

    out_path = RESULTS / "exp1_3b_llps_overlap.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

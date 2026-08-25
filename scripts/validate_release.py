#!/usr/bin/env python3
"""Validate release structure and manuscript-critical RLBPPred values."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(float(actual) - expected) <= tolerance


def main() -> None:
    args = parse_args(); root = args.root.resolve(); data = root / "data" / "panels"
    results: list[dict[str, object]] = []

    def check(name: str, condition: bool, observed: object = None, expected: object = None) -> None:
        results.append({"check": name, "passed": bool(condition), "observed": observed, "expected": expected})

    required = [
        "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
        "requirements-figures.txt", "environment-training.yml", "Makefile",
        "scripts/export_source_data.py", "scripts/reproduce_figures.py",
        "scripts/validate_release.py", "scripts/download_uniprot_fasta.py",
        "configs/paper.yaml", "docs/reproduce_figures.md",
        "docs/reproduce_model.md", "docs/data_sources.md", "docs/known_limitations.md",
        "data/panels/manifest.json",
    ]
    for relative in required:
        check(f"required:{relative}", (root / relative).exists())

    expected_id_counts = {
        "recurrent_positive_ids.txt": 407,
        "significant_positive_ids.txt": 376,
        "negative_ids.txt": 9109,
        "potential_positive_ids.txt": 1879,
        "untrained_proteome_ids.txt": 5649,
        "high_confidence_candidate_ids.txt": 342,
    }
    observed_id_counts: dict[str, int] = {}
    identifiers_unique = True
    for filename in expected_id_counts:
        values = [line.strip() for line in (root / "data" / "model_inputs" / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        observed_id_counts[filename] = len(values)
        identifiers_unique &= len(values) == len(set(values))
    check("model-input identifier counts", observed_id_counts == expected_id_counts,
          observed_id_counts, expected_id_counts)
    check("model-input identifiers unique", identifiers_unique)

    membership = pd.read_csv(data / "Figure1b.csv")
    set_columns = ["Cristini", "Wang", "Mosler", "Yan", "Shinriki"]
    check("Fig1b union", len(membership) == 1430, len(membership), 1430)
    observed_sizes = membership[set_columns].sum().astype(int).to_dict()
    expected_sizes = {"Cristini": 462, "Wang": 364, "Mosler": 308, "Yan": 434, "Shinriki": 430}
    check("Fig1b set sizes", observed_sizes == expected_sizes, observed_sizes, expected_sizes)
    recurrence = membership[set_columns].sum(axis=1)
    check("Fig1b at least two studies", int((recurrence >= 2).sum()) == 407,
          int((recurrence >= 2).sum()), 407)
    check("Fig1b all five studies", int((recurrence == 5).sum()) == 4,
          int((recurrence == 5).sum()), 4)

    benchmark = pd.read_csv(data / "Figure2b.csv").set_index("Classifier")
    check("Fig2b best AUPRC", close(benchmark.loc["MLP", "SaProt AF2"], 0.9264),
          float(benchmark.loc["MLP", "SaProt AF2"]), 0.9264)

    roc_filter = pd.read_csv(data / "Figure3a.csv")[["label", "auroc"]].drop_duplicates()
    filtered = roc_filter[roc_filter["label"].str.startswith("Filtered Dataset")]["auroc"]
    unfiltered = roc_filter[roc_filter["label"].str.startswith("Unfiltered Dataset")]["auroc"]
    check("Fig3a filtered AUROC", len(filtered) == 1 and close(filtered.iloc[0], 0.9925),
          filtered.tolist(), [0.9925])
    check("Fig3a unfiltered AUROC", len(unfiltered) == 1 and close(unfiltered.iloc[0], 0.9001),
          unfiltered.tolist(), [0.9001])

    metrics = pd.read_csv(data / "SuppFigure2b.csv")
    best = metrics.loc[metrics["AUPRC"].idxmax()]
    observed_best = {"Representation": best["Representation"], "Classifier": best["Classifier"],
                     "AUROC": float(best["AUROC"]), "AUPRC": float(best["AUPRC"])}
    expected_best = {"Representation": "SaProt AF2", "Classifier": "MLP", "AUROC": 0.9925, "AUPRC": 0.9264}
    check("best model", observed_best == expected_best, observed_best, expected_best)

    confusion = pd.read_csv(data / "SuppFigure3a.csv")
    observed_confusion = {(r.true_label, r.predicted_label): int(r.count) for r in confusion.itertuples()}
    expected_confusion = {("Non-RLBP", "Non-RLBP"): 1729, ("Non-RLBP", "RLBP"): 92,
                          ("RLBP", "Non-RLBP"): 4, ("RLBP", "RLBP"): 71}
    check("SupFig3a confusion matrix", observed_confusion == expected_confusion,
          {str(k): v for k, v in observed_confusion.items()}, {str(k): v for k, v in expected_confusion.items()})

    database = pd.read_csv(data / "Figure6a_b.csv")
    scope = database[database["Record type"] == "Database scope"].set_index("Species")["Length or count"]
    observed_scope = {name: int(scope[name]) for name in ["Human", "Mouse", "Human + mouse"]}
    expected_scope = {"Human": 20075, "Mouse": 17002, "Human + mouse": 37077}
    check("Fig6 database scope", observed_scope == expected_scope, observed_scope, expected_scope)
    top1 = database[database["UniProt entry"] == "P11387"]
    check("Fig6 TOP1 query", len(top1) == 1 and close(top1.iloc[0]["Confidence"], 0.999743),
          [] if top1.empty else [{"entry": "P11387", "confidence": float(top1.iloc[0]["Confidence"])}],
          [{"entry": "P11387", "confidence": 0.999743}])

    raw_disorder = pd.read_csv(data / "Figure5c_raw.csv")
    disorder_counts = raw_disorder.groupby("group_label")["disorder_percent"].count().to_dict()
    expected_disorder = {"Novel predicted RLBPs": 342, "Known RLBPs": 376, "Non-RLBPs": 9080}
    check("Fig5c valid MobiDB records", disorder_counts == expected_disorder,
          disorder_counts, expected_disorder)

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    hashes_ok = True
    for item in manifest:
        path = data / f"{item['panel_data']}.csv"
        if not path.exists():
            hashes_ok = False; break
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            hashes_ok = False; break
    check("panel manifest hashes", hashes_ok, len(manifest), 34)

    oversized = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.stat().st_size > 95 * 1024 * 1024]
    check("no GitHub-oversized files", not oversized, oversized, [])
    broken_links = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink() and not path.exists()]
    check("no broken symlinks", not broken_links, broken_links, [])

    passed = sum(bool(item["passed"]) for item in results)
    report = {"passed": passed, "total": len(results), "checks": results}
    report_path = args.report or root / "outputs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in results:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}")
    print(f"{passed}/{len(results)} checks passed; report: {report_path}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

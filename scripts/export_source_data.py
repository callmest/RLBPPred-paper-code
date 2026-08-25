#!/usr/bin/env python3
"""Export final-numbered, panel-level CSV files from the RLBPPred workbooks.

The comprehensive workbook contains all computational panels but retains the
earlier name ``Figure1a`` for the five-study UpSet data.  The corrected v5
workbook contains the cleaned 1,430-record membership matrix under
``Figure1b``.  This script makes the panel CSVs used by the release explicit
and prevents the old Figure 1 numbering from leaking into reproduction runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook


EXCLUDED_SHEETS = {"README", "Panel_manifest", "Figure1a"}


def apply_release_corrections(sheet_name: str, rows: list[list[object]]) -> list[list[object]]:
    """Apply documented label-only corrections required by the final figure."""
    if sheet_name != "Figure3a":
        return rows
    label_index = rows[0].index("label")
    auroc_index = rows[0].index("auroc")
    for row in rows[1:]:
        auroc = float(row[auroc_index])
        if abs(auroc - 0.9925) < 1e-8:
            row[label_index] = "Filtered Dataset (SaProt-650M-AF2 + MLP)"
        elif abs(auroc - 0.9001) < 1e-8:
            row[label_index] = "Unfiltered Dataset (SaProt-650M-AF2 + MLP)"
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Comprehensive source-data workbook (36 sheets).")
    parser.add_argument("--figure1b-input", required=True, type=Path,
                        help="Corrected v5 workbook containing Figure1b.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Destination directory for panel CSV files.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_sheet(workbook, sheet_name: str) -> tuple[str, list[list[object]]]:
    if sheet_name not in workbook.sheetnames:
        raise KeyError(f"{sheet_name!r} is absent from the workbook")
    sheet = workbook[sheet_name]
    description = str(sheet.cell(1, 1).value or "").strip()
    headers = [cell.value for cell in sheet[3]]
    while headers and headers[-1] is None:
        headers.pop()
    if not headers or any(value is None for value in headers):
        raise ValueError(f"Invalid row-3 header in {sheet_name}")

    rows: list[list[object]] = [headers]
    for values in sheet.iter_rows(min_row=4, max_col=len(headers), values_only=True):
        values = list(values)
        if all(value is None for value in values):
            continue
        rows.append(values)
    return description, rows


def write_csv(path: Path, rows: Iterable[Iterable[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)
    return max(0, len(rows) - 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comprehensive = load_workbook(args.input, read_only=False, data_only=True)
    source_sheets = [name for name in comprehensive.sheetnames if name not in EXCLUDED_SHEETS]

    manifest: list[dict[str, object]] = []
    for sheet_name in source_sheets:
        description, rows = rows_from_sheet(comprehensive, sheet_name)
        rows = apply_release_corrections(sheet_name, rows)
        output_path = args.output_dir / f"{sheet_name}.csv"
        row_count = write_csv(output_path, rows)
        manifest.append({
            "panel_data": sheet_name,
            "source_workbook": args.input.name,
            "source_sheet": sheet_name,
            "description": description,
            "rows": row_count,
            "sha256": sha256(output_path),
        })

    comprehensive.close()

    figure1b_workbook = load_workbook(args.figure1b_input, read_only=False, data_only=True)
    description, rows = rows_from_sheet(figure1b_workbook, "Figure1b")
    figure1b_workbook.close()
    figure1b_path = args.output_dir / "Figure1b.csv"
    row_count = write_csv(figure1b_path, rows)
    manifest.append({
        "panel_data": "Figure1b",
        "source_workbook": args.figure1b_input.name,
        "source_sheet": "Figure1b",
        "description": description,
        "rows": row_count,
        "sha256": sha256(figure1b_path),
    })

    manifest.sort(key=lambda item: str(item["panel_data"]))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest)} panel datasets to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export current CSVs from explicit, hash-pinned source-data blocks.

Legacy worksheet names are not current figure numbers. configs/panel_map.json
defines header-inclusive ranges and current outputs. Top-level legacy panel
CSVs are never modified. A nonempty output is accepted only as an identical no-op.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "configs" / "panel_map.json"
FORBIDDEN_SHEETS = {"Figure index", "Archive index", "Figure3e", "SupplementaryFigure1b"}
EXCEL_ERRORS = {"#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!"}


class ExportError(ValueError):
    """Input or output differs from the frozen export contract."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_filename(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", name)) and name not in {".", ".."}


def validate_map(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ExportError("Unsupported panel map schema_version")
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("workbook_sha256", "")):
        raise ExportError("The panel map must pin a workbook SHA-256")
    ids, files = set(), {"manifest.json"}
    for spec in config["datasets"] + config.get("external_files", []):
        if spec["id"] in ids or spec["file"] in files:
            raise ExportError(f"Duplicate dataset ID or filename: {spec['id']}")
        if not safe_filename(spec["file"]):
            raise ExportError(f"Unsafe output filename: {spec['file']}")
        ids.add(spec["id"])
        files.add(spec["file"])
    for spec in config["datasets"]:
        if spec["sheet"] in FORBIDDEN_SHEETS:
            raise ExportError(f"Index/history sheet cannot be current data: {spec['sheet']}")
        left, top, right, bottom = range_boundaries(spec["range"])
        if bottom - top != spec["expected_rows"] or right - left + 1 != len(spec["headers"]):
            raise ExportError(f"Range/header/row-count disagreement in {spec['id']}")
        if len(set(spec["headers"])) != len(spec["headers"]) or any(not h for h in spec["headers"]):
            raise ExportError(f"Missing/duplicate header in {spec['id']}")
        selected = spec.get("select_columns", spec["headers"])
        if len(set(selected)) != len(selected) or not set(selected) <= set(spec["headers"]):
            raise ExportError(f"Invalid selected columns in {spec['id']}")
        if not set(spec.get("unique_key", [])) <= set(selected):
            raise ExportError(f"Unique key is not an exported column in {spec['id']}")


def check_workbook_hash(path: Path, config: dict[str, Any]) -> str:
    actual = sha256(path)
    if actual != config["workbook_sha256"]:
        raise ExportError(f"Workbook SHA-256 mismatch: expected {config['workbook_sha256']}, got {actual}. "
                          "Review the new workbook and update its explicit panel map first.")
    return actual


def extract_block(workbook, spec: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Read one exact block; never infer a table boundary from worksheet dimensions."""
    if spec["sheet"] not in workbook.sheetnames:
        raise ExportError(f"Missing source sheet {spec['sheet']!r}")
    left, top, right, bottom = range_boundaries(spec["range"])
    rows = [list(row) for row in workbook[spec["sheet"]].iter_rows(
        min_row=top, max_row=bottom, min_col=left, max_col=right, values_only=True)]
    if not rows or rows[0] != spec["headers"]:
        raise ExportError(f"Header mismatch in {spec['id']}: {rows[0] if rows else None!r}")
    records = rows[1:]
    if len(records) != spec["expected_rows"]:
        raise ExportError(f"Row count mismatch in {spec['id']}")
    selected = spec.get("select_columns", spec["headers"])
    indices = [spec["headers"].index(h) for h in selected]
    required = [spec["headers"].index(h) for h in spec.get("required_columns", [spec["headers"][0]])]
    for number, row in enumerate(records, start=top + 1):
        if len(row) != len(spec["headers"]) or not any(v is not None for v in row):
            raise ExportError(f"Blank/truncated record in {spec['sheet']} row {number}")
        if any(row[i] is None or row[i] == "" for i in required):
            raise ExportError(f"Missing required value in {spec['sheet']} row {number}")
        if any(isinstance(v, str) and v in EXCEL_ERRORS for v in row):
            raise ExportError(f"Spreadsheet error in {spec['sheet']} row {number}")
        if any(isinstance(v, str) and v.startswith(("Source:", "Note:", "Notes:")) for v in row):
            raise ExportError(f"Provenance note mixed into data in {spec['sheet']} row {number}")
    result = [[row[i] for i in indices] for row in records]
    if spec.get("unique_key"):
        key_indices = [selected.index(h) for h in spec["unique_key"]]
        keys = [tuple(row[i] for i in key_indices) for row in result]
        if len(keys) != len(set(keys)):
            raise ExportError(f"Duplicate record keys in {spec['id']}")
    return selected, result


def csv_bytes(headers: list[str], records: list[list[Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(records)
    return handle.getvalue().encode("utf-8")


def build_package(input_path: Path, map_path: Path = DEFAULT_MAP,
                  figure_source_dir: Path | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate/read all sources into memory before creating any output file."""
    config = json.loads(map_path.read_text(encoding="utf-8"))
    validate_map(config)
    workbook_hash = check_workbook_hash(input_path, config)
    files: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    workbook = load_workbook(input_path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames != config["expected_sheets"]:
            raise ExportError("Workbook sheet inventory/order differs from the reviewed panel map")
        for spec in config["datasets"]:
            headers, records = extract_block(workbook, spec)
            payload = csv_bytes(headers, records)
            files[spec["file"]] = payload
            entries.append({
                "id": spec["id"], "file": spec["file"], "panels": spec["panels"],
                "source_workbook": input_path.name, "source_workbook_sha256": workbook_hash,
                "source_sheet": spec["sheet"], "source_range": spec["range"],
                "header_row": range_boundaries(spec["range"])[1],
                "source_columns": spec["headers"], "columns": headers,
                "rows": len(records), "sha256": digest(payload),
                "plot_filter": spec.get("plot_filter"), "description": spec["description"],
            })
    finally:
        workbook.close()
    if figure_source_dir is not None:
        for spec in config.get("external_files", []):
            source = figure_source_dir / spec["source_file"]
            payload = source.read_bytes()
            if digest(payload) != spec["sha256"]:
                raise ExportError(f"External input SHA-256 mismatch: {source.name}")
            entry = {"id": spec["id"], "file": spec["file"], "panels": spec["panels"],
                     "source_file": spec["source_file"], "source_sha256": digest(payload),
                     "sha256": digest(payload), "description": spec["description"]}
            if source.suffix.lower() == ".csv":
                table = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
                if table[0] != spec["headers"] or len(table) - 1 != spec["expected_rows"]:
                    raise ExportError(f"External CSV header/row mismatch: {source.name}")
                if any(len(r) != len(table[0]) for r in table[1:]):
                    raise ExportError(f"External CSV ragged record: {source.name}")
                entry.update(columns=table[0], rows=len(table) - 1)
            else:
                json.loads(payload)
            files[spec["file"]] = payload
            entries.append(entry)
    # Detect concurrent source edits during extraction.
    check_workbook_hash(input_path, config)
    manifest = {
        "schema_version": 1, "release": config["release"],
        "workbook_sha256": workbook_hash, "panel_map_sha256": sha256(map_path),
        "index_sheet_exported": False, "historical_sheets_exported": [],
        "external_figures_included": figure_source_dir is not None,
        "omitted_optional_inputs": [] if figure_source_dir else [s["id"] for s in config.get("external_files", [])],
        "datasets": entries,
    }
    files["manifest.json"] = json_bytes(manifest)
    return files, manifest


def publish_package(files: dict[str, bytes], output_dir: Path) -> str:
    """Publish to a new/empty directory or return a verified identical no-op."""
    if output_dir.is_symlink():
        raise ExportError("Output directory must not be a symlink")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ExportError("Output exists and is not a directory")
        contents = list(output_dir.iterdir())
        if contents:
            identical = (set(p.name for p in contents) == set(files)
                         and all(p.is_file() and not p.is_symlink()
                                 and p.read_bytes() == files[p.name] for p in contents))
            if identical:
                return "identical-no-op"
            raise ExportError("Refusing nonempty output with different/stale files. "
                              "Choose a new versioned staging directory; no files were changed.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        for name, payload in files.items():
            if not safe_filename(name):
                raise ExportError(f"Unsafe output filename {name}")
            (stage / name).write_bytes(payload)
        if output_dir.exists():
            output_dir.rmdir()  # fails if a concurrent writer populated it
        if output_dir.exists():
            raise ExportError("Output appeared during export; choose another directory")
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Reviewed current source-data workbook.")
    parser.add_argument("--panel-map", type=Path, default=DEFAULT_MAP, help="Explicit, hash-pinned block map.")
    parser.add_argument("--figure-source-dir", type=Path,
                        help="Optional figure_source/figure6 directory with query CSV and metadata JSON.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New/empty destination; differing existing data is never overwritten.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        files, manifest = build_package(args.input, args.panel_map, args.figure_source_dir)
        status = publish_package(files, args.output_dir)
    except (ExportError, OSError) as exc:
        raise SystemExit(f"Export refused: {exc}") from exc
    print(json.dumps({"status": status, "datasets": len(manifest["datasets"]),
                      "files": len(files), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

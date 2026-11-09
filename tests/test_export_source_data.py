"""Unit tests for current-data export; no manuscript workbook is written."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_source_data", ROOT / "scripts/export_source_data.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class FakeSheet:
    def __init__(self, rows):
        self.rows = rows

    def iter_rows(self, min_row, max_row, min_col, max_col, values_only):
        for i in range(min_row - 1, max_row):
            row = self.rows[i] if i < len(self.rows) else []
            yield tuple(row[j] if j < len(row) else None for j in range(min_col - 1, max_col))


class FakeWorkbook(dict):
    @property
    def sheetnames(self):
        return list(self)


def block(**overrides):
    spec = {"id": "Example", "file": "Example.csv", "panels": ["Figure1a"],
            "sheet": "Data", "range": "A3:B5", "headers": ["id", "value"],
            "expected_rows": 2, "unique_key": ["id"], "description": "test"}
    return dict(spec, **overrides)


def config(*specs):
    return {"schema_version": 1, "workbook_sha256": "0" * 64,
            "datasets": list(specs), "external_files": []}


class ExportSourceDataTests(unittest.TestCase):
    def test_current_map_is_valid_and_excludes_index_history(self):
        mapping = json.loads((ROOT / "configs/panel_map.json").read_text())
        exporter.validate_map(mapping)
        self.assertEqual(len(mapping["datasets"]), 29)
        self.assertFalse({s["sheet"] for s in mapping["datasets"]} & exporter.FORBIDDEN_SHEETS)
        self.assertEqual(sum(s["file"] == "Figure1b.csv" for s in mapping["datasets"]), 1)

    def test_index_or_history_as_data_is_rejected(self):
        for name in exporter.FORBIDDEN_SHEETS:
            with self.subTest(name=name), self.assertRaises(exporter.ExportError):
                exporter.validate_map(config(block(sheet=name)))

    def test_multiblock_ranges_exclude_notes_and_other_headers(self):
        wb = FakeWorkbook(Data=FakeSheet([["Title"], [None], ["id", "value"],
            ["a", 0], ["b", 2], ["Notes: not a record"], ["group", "p"], ["g", 0.01]]))
        self.assertEqual(exporter.extract_block(wb, block()), (["id", "value"], [["a", 0], ["b", 2]]))
        next_block = block(id="Tests", range="A7:B8", headers=["group", "p"],
                           expected_rows=1, unique_key=["group"])
        self.assertEqual(exporter.extract_block(wb, next_block)[1], [["g", 0.01]])

    def test_note_inside_declared_range_is_rejected(self):
        wb = FakeWorkbook(Data=FakeSheet([[], [], ["id", "value"], ["a", 1], ["Source: note", None]]))
        with self.assertRaisesRegex(exporter.ExportError, "Provenance note"):
            exporter.extract_block(wb, block())

    def test_header_drift_is_rejected(self):
        wb = FakeWorkbook(Data=FakeSheet([[], [], ["id", "renamed"], ["a", 1], ["b", 2]]))
        with self.assertRaisesRegex(exporter.ExportError, "Header mismatch"):
            exporter.extract_block(wb, block())

    def test_blank_record_is_rejected(self):
        wb = FakeWorkbook(Data=FakeSheet([[], [], ["id", "value"], ["a", 1], []]))
        with self.assertRaisesRegex(exporter.ExportError, "Blank/truncated"):
            exporter.extract_block(wb, block())

    def test_duplicate_output_ids_and_filenames_are_rejected(self):
        for duplicate in [block(file="Another.csv"), block(id="Another")]:
            with self.assertRaisesRegex(exporter.ExportError, "Duplicate dataset"):
                exporter.validate_map(config(block(), duplicate))

    def test_duplicate_keys_are_rejected(self):
        wb = FakeWorkbook(Data=FakeSheet([[], [], ["id", "value"], ["a", 1], ["a", 2]]))
        with self.assertRaisesRegex(exporter.ExportError, "Duplicate record"):
            exporter.extract_block(wb, block())

    def test_select_columns_preserves_values_and_order(self):
        wb = FakeWorkbook(Data=FakeSheet([[], [], ["id", "value"], ["a", 0], ["b", None]]))
        self.assertEqual(exporter.extract_block(wb, block(select_columns=["value", "id"])),
                         (["value", "id"], [[0, "a"], [None, "b"]]))

    def test_missing_and_zero_remain_distinct(self):
        payload = exporter.csv_bytes(["id", "value"], [["a", 0], ["b", None]])
        self.assertEqual(payload, b"id,value\na,0\nb,\n")

    def test_stale_workbook_hash_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.xlsx"
            path.write_bytes(b"fixture, not a workbook")
            mapping = config(block())
            with self.assertRaisesRegex(exporter.ExportError, "SHA-256 mismatch"):
                exporter.check_workbook_hash(path, mapping)
            mapping["workbook_sha256"] = exporter.sha256(path)
            self.assertEqual(exporter.check_workbook_hash(path, mapping), mapping["workbook_sha256"])

    def test_publish_is_identical_no_op_or_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "current"
            files = {"Data.csv": b"id,value\na,1\n", "manifest.json": b"{}\n"}
            self.assertEqual(exporter.publish_package(files, dest), "created")
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.iterdir()}
            self.assertEqual(exporter.publish_package(files, dest), "identical-no-op")
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.iterdir()})
            changed = dict(files, **{"Data.csv": b"changed"})
            with self.assertRaisesRegex(exporter.ExportError, "Refusing nonempty"):
                exporter.publish_package(changed, dest)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.iterdir()})

    def test_stale_unlisted_file_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "current"
            dest.mkdir()
            old = dest / "Figure3e.csv"
            old.write_bytes(b"historical")
            with self.assertRaises(exporter.ExportError):
                exporter.publish_package({"Current.csv": b"current"}, dest)
            self.assertEqual(old.read_bytes(), b"historical")
            self.assertFalse((dest / "Current.csv").exists())

    def test_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.mkdir()
            link = Path(tmp) / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(exporter.ExportError):
                exporter.publish_package({"Data.csv": b"x"}, link)


if __name__ == "__main__":
    unittest.main()

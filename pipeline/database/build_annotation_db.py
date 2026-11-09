#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = [
    'uniprot_id', 'species', 'gene_name', 'gene_names_all', 'gene_description',
    'protein_name', 'functional_annotation', 'confidence', 'reviewed', 'entry_name',
    'length', 'uniprot_page_source', 'pdb_page_source', 'alphafold2_page_source'
]

DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / 'results' / 'RLBPPredDB'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build RLBPPredDB runtime annotation files from source CSV.')
    parser.add_argument('--input', type=Path, default=DEFAULT_DB_DIR / 'all_proteome_annotations_for_db.csv')
    parser.add_argument('--json-out', type=Path, default=DEFAULT_DB_DIR / 'rlbp_annotations.json')
    parser.add_argument('--csv-out', type=Path, default=DEFAULT_DB_DIR / 'rlbp_annotations.csv')
    return parser.parse_args()


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ('true', '1', 'yes', 'reviewed')


def to_float(value: str):
    text = str(value).strip()
    if text in ('', 'NA', 'nan', 'None'):
        return None
    result = float(text)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f'Invalid prediction score: {text}')
    return result


def to_int(value: str):
    text = str(value).strip()
    if text in ('', 'NA', 'nan', 'None'):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f'Input CSV not found: {args.input}')
    if args.json_out.resolve() == args.csv_out.resolve():
        raise ValueError('JSON and CSV outputs must be different files.')
    for path in (args.json_out, args.csv_out):
        if path.exists():
            raise FileExistsError(f'Output exists; choose a new path: {path}')

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.input.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = {k: (row.get(k, '') or '').strip() for k in FIELDS}
            obj['confidence'] = to_float(obj['confidence'])
            obj['length'] = to_int(obj['length'])
            obj['reviewed'] = to_bool(obj['reviewed'])
            rows.append(obj)

    with args.json_out.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False)

    with args.csv_out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out['reviewed'] = 'true' if row['reviewed'] else 'false'
            out['confidence'] = '' if row['confidence'] is None else repr(row['confidence'])
            out['length'] = '' if row['length'] is None else str(row['length'])
            writer.writerow(out)

    print(f'Built {len(rows)} rows -> {args.json_out} / {args.csv_out}')


if __name__ == '__main__':
    main()

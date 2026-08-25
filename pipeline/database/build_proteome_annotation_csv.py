#!/usr/bin/env python3
"""Build import-ready proteome annotation CSVs for RLBPPredDB."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from openpyxl import load_workbook

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HUMAN_XLSX = REPOSITORY_ROOT / 'results/untrained_proteome_analysis/human_all_proteome_042024.xlsx'
MOUSE_XLSX = REPOSITORY_ROOT / 'results/inference_mouse_datasets/mouse_all_proteome.xlsx'
MOUSE_SHEET = 'mouse_all_proteome'

HUMAN_PRED = REPOSITORY_ROOT / 'results/inference_four_datasets/all_predictions.csv'
MOUSE_PRED = REPOSITORY_ROOT / 'results/inference_mouse_datasets/all_predictions.csv'

OUT_DIR = REPOSITORY_ROOT / 'results/RLBPPredDB'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-xlsx', type=Path, default=HUMAN_XLSX)
    parser.add_argument('--mouse-xlsx', type=Path, default=MOUSE_XLSX)
    parser.add_argument('--mouse-sheet', default=MOUSE_SHEET)
    parser.add_argument('--human-predictions', type=Path, default=HUMAN_PRED)
    parser.add_argument('--mouse-predictions', type=Path, default=MOUSE_PRED)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    return parser.parse_args()


def read_prediction_map(path: Path) -> dict[str, float]:
    pred = {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get('protein_id') or '').strip()
            prob = row.get('probability')
            if not pid or prob in (None, ''):
                continue
            pred[pid] = float(prob)
    return pred


def read_proteome_sheet(path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if (sheet_name and sheet_name in wb.sheetnames) else wb[wb.sheetnames[0]]

    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else '' for c in next(rows)]
    idx = {name: i for i, name in enumerate(header)}

    def get(row, key):
        i = idx.get(key)
        if i is None or i >= len(row):
            return ''
        v = row[i]
        return '' if v is None else str(v).strip()

    out = []
    for row in rows:
        entry = get(row, 'Entry')
        if not entry:
            continue
        out.append(
            {
                'Entry': entry,
                'Reviewed': get(row, 'Reviewed'),
                'Entry Name': get(row, 'Entry Name'),
                'Protein names': get(row, 'Protein names'),
                'Gene Names': get(row, 'Gene Names'),
                'Organism': get(row, 'Organism'),
                'Length': get(row, 'Length'),
                'Sequence': get(row, 'Sequence'),
            }
        )
    return out


def first_gene_name(raw_gene_names: str) -> str:
    return raw_gene_names.split()[0].strip() if raw_gene_names else ''


def gene_description_from_protein_name(protein_name: str) -> str:
    if not protein_name:
        return ''
    # Keep a concise description: text before first parenthesis when possible.
    return protein_name.split(' (')[0].strip()


def uniprot_url(uniprot_id: str) -> str:
    return f'https://www.uniprot.org/uniprotkb/{uniprot_id}'


def alphafold_url(uniprot_id: str) -> str:
    return f'https://alphafold.ebi.ac.uk/entry/{uniprot_id}'


def pdb_page_url(uniprot_id: str) -> str:
    # Stable UniProt-linked PDB entry page.
    return f'https://www.ebi.ac.uk/pdbe/entry/uniprot/{uniprot_id}'


def build_rows(records: list[dict[str, str]], pred: dict[str, float], species_label: str) -> tuple[list[dict[str, str]], int]:
    rows = []
    missing_conf = 0

    for rec in records:
        uid = rec['Entry']
        gnames = rec['Gene Names']
        pname = rec['Protein names']
        confidence = pred.get(uid)
        if confidence is None:
            missing_conf += 1

        row = {
            'uniprot_id': uid,
            'species': species_label,
            'gene_name': first_gene_name(gnames),
            'gene_names_all': gnames,
            'gene_description': gene_description_from_protein_name(pname),
            'protein_name': pname,
            'functional_annotation': pname,
            'uniprot_page_source': uniprot_url(uid),
            'pdb_page_source': pdb_page_url(uid),
            'alphafold2_page_source': alphafold_url(uid),
            'confidence': '' if confidence is None else f'{confidence:.6f}',
            'reviewed': rec['Reviewed'],
            'entry_name': rec['Entry Name'],
            'length': rec['Length'],
            'sequence': rec['Sequence'],
        }
        rows.append(row)

    return rows, missing_conf


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human_records = read_proteome_sheet(args.human_xlsx)
    mouse_records = read_proteome_sheet(args.mouse_xlsx, args.mouse_sheet)

    human_pred = read_prediction_map(args.human_predictions)
    mouse_pred = read_prediction_map(args.mouse_predictions)

    human_rows, human_missing = build_rows(human_records, human_pred, 'Human')
    mouse_rows, mouse_missing = build_rows(mouse_records, mouse_pred, 'Mouse')

    out_human = args.output_dir / 'human_proteome_annotations_for_db.csv'
    out_mouse = args.output_dir / 'mouse_proteome_annotations_for_db.csv'
    out_all = args.output_dir / 'all_proteome_annotations_for_db.csv'

    write_csv(out_human, human_rows)
    write_csv(out_mouse, mouse_rows)
    write_csv(out_all, human_rows + mouse_rows)

    print('Saved:', out_human)
    print('Saved:', out_mouse)
    print('Saved:', out_all)
    print(f'Human rows: {len(human_rows)}, missing confidence: {human_missing}')
    print(f'Mouse rows: {len(mouse_rows)}, missing confidence: {mouse_missing}')


if __name__ == '__main__':
    main()

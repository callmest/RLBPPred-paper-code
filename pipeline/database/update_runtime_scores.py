#!/usr/bin/env python3
"""Join frozen strict-model scores to a 14-field annotation snapshot.

No fitting, live annotation retrieval, or public deployment is performed.
Existing outputs and annotation inputs are never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

FIELDS = [
    'uniprot_id', 'species', 'gene_name', 'gene_names_all', 'gene_description',
    'protein_name', 'functional_annotation', 'confidence', 'reviewed', 'entry_name',
    'length', 'uniprot_page_source', 'pdb_page_source', 'alphafold2_page_source',
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(path: Path) -> dict[str, float]:
    result = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'protein_id', 'probability'} <= set(reader.fieldnames or []):
            raise ValueError('Prediction CSV requires protein_id and probability.')
        for row in reader:
            uid = row['protein_id'].strip()
            value = float(row['probability'])
            if not uid or uid in result:
                raise ValueError(f'Empty or duplicate prediction ID: {uid}')
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid probability for {uid}')
            result[uid] = value
    return result


def combine_scores(primary, supplement):
    """Extend a frozen score map without replacing any original prediction."""
    overlap = set(primary) & set(supplement)
    if overlap:
        raise ValueError(f'Database supplement overlaps {len(overlap)} frozen prediction IDs.')
    return {**primary, **supplement}


def merge_scores(records, scores):
    seen = set()
    updated = []
    coverage = collections.defaultdict(collections.Counter)
    for row in records:
        if set(row) != set(FIELDS):
            raise ValueError('Annotation snapshot must contain exactly the 14 runtime fields.')
        species, uid = row['species'], row['uniprot_id']
        if species not in scores or not uid or (species, uid) in seen:
            raise ValueError(f'Unknown species or duplicate/empty annotation ID: {species}/{uid}')
        seen.add((species, uid))
        new = row.copy()
        # Never retain a historical score when the strict model has no score.
        new['confidence'] = scores[species].get(uid)
        updated.append(new)
        coverage[species]['records'] += 1
        coverage[species]['unscored' if new['confidence'] is None else 'scored'] += 1
    for species, species_scores in scores.items():
        extra = set(species_scores) - {uid for sp, uid in seen if sp == species}
        if extra:
            raise ValueError(f'{len(extra)} {species} prediction IDs absent from annotation snapshot.')
    return updated, {key: {name: value[name] for name in ('records', 'scored', 'unscored')}
                     for key, value in coverage.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations-json', required=True, type=Path)
    parser.add_argument('--human-predictions', required=True, type=Path)
    parser.add_argument('--human-supplement', type=Path,
                        help='Optional disjoint strict-model database-completion CSV; never overwrites frozen scores.')
    parser.add_argument('--mouse-predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory must be new or empty.')
    records = json.loads(args.annotations_json.read_text(encoding='utf-8'))
    scores = {'Human': read_scores(args.human_predictions), 'Mouse': read_scores(args.mouse_predictions)}
    if args.human_supplement:
        scores['Human'] = combine_scores(scores['Human'], read_scores(args.human_supplement))
    updated, coverage = merge_scores(records, scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / 'rlbp_annotations.json'
    csv_path = args.output_dir / 'rlbp_annotations.csv'
    json_path.write_text(json.dumps(updated, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in updated:
            out = row.copy()
            out['confidence'] = '' if row['confidence'] is None else repr(row['confidence'])
            out['reviewed'] = 'true' if row['reviewed'] else 'false'
            writer.writerow(out)
    manifest = {
        'profile': 'strict401', 'runtime_field_count': 14, 'coverage': coverage,
        'source_files': [
            {'role': role, 'filename': path.name, 'sha256': digest(path)}
            for role, path in [('annotations', args.annotations_json),
                               ('human_scores', args.human_predictions),
                               ('mouse_scores', args.mouse_predictions)]
                              + ([('human_database_supplement', args.human_supplement)]
                                 if args.human_supplement else [])
        ],
        'outputs': [{'file': path.name, 'sha256': digest(path)} for path in (json_path, csv_path)],
        'missing_scores': 'JSON null; CSV blank; never zero or historical scores',
        'precision': 'full source float precision; six-decimal display belongs to the UI',
        'annotation_fields_and_record_order_preserved': True,
        'original_frozen_scores_preserved': True,
        'discovery_candidates_redefined': False,
    }
    (args.output_dir / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'records': len(updated), 'coverage': coverage}, indent=2))


if __name__ == '__main__':
    main()

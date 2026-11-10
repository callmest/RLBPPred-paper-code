import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'pipeline/database' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class DatabaseTests(unittest.TestCase):
    def test_disjoint_database_completion_preserves_frozen_scores(self):
        lib = module('update_runtime_scores')
        original = {'old': 0.1234567890123456}
        self.assertEqual(lib.combine_scores(original, {'new': .98}),
                         {'old': 0.1234567890123456, 'new': .98})
        self.assertEqual(original, {'old': 0.1234567890123456})
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            lib.combine_scores(original, {'old': original['old']})

    def test_runtime_preserves_precision_and_missing(self):
        lib = module('update_runtime_scores')
        rows = [{key: '' for key in lib.FIELDS} for _ in range(2)]
        for i, row in enumerate(rows):
            row.update(uniprot_id=str(i), species='Human', confidence=0.8)
        updated, coverage = lib.merge_scores(rows, {'Human': {'0': 0.9999898352225621}, 'Mouse': {}})
        self.assertEqual(updated[0]['confidence'], 0.9999898352225621)
        self.assertIsNone(updated[1]['confidence'])
        self.assertEqual(rows[1]['confidence'], 0.8)
        self.assertEqual(coverage['Human'], {'records': 2, 'scored': 1, 'unscored': 1})

    def test_complete_coverage_reports_explicit_zero_missing(self):
        lib = module('update_runtime_scores')
        row = {key: '' for key in lib.FIELDS}
        row.update(uniprot_id='complete', species='Human')
        _, coverage = lib.merge_scores([row], {'Human': {'complete': .5}})
        self.assertEqual(coverage['Human'], {'records': 1, 'scored': 1, 'unscored': 0})

    def test_duplicate_annotation_rejected(self):
        lib = module('update_runtime_scores')
        row = {key: '' for key in lib.FIELDS}
        row.update(uniprot_id='A', species='Human')
        with self.assertRaises(ValueError):
            lib.merge_scores([row, row], {'Human': {}, 'Mouse': {}})

    def test_invalid_and_blank_scores(self):
        lib = module('build_annotation_db')
        self.assertIsNone(lib.to_float(''))
        self.assertEqual(lib.to_float('0'), 0)
        for value in ('garbage', 'inf', '-0.1', '1.1'):
            with self.assertRaises(ValueError):
                lib.to_float(value)


if __name__ == '__main__':
    unittest.main()

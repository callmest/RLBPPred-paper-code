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

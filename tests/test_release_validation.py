"""Offline regression tests for the current-profile validator."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_validation', ROOT / 'scripts/validate_release.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ReleaseValidationTests(unittest.TestCase):
    def test_public_provenance_preserves_non_path_fields(self):
        frozen = ROOT / 'data/frozen_model'
        provenance = json.loads((frozen / 'provenance.json').read_text())
        manifest = json.loads((frozen / 'manifest.json').read_text())
        artifact = manifest['artifacts']['provenance.json']
        self.assertEqual(provenance['source'], 'source-project/RLBP')
        self.assertEqual(provenance['output'], 'experiments/strict_saprot_mlp_reproduction_20261013')
        self.assertEqual(artifact['original_sha256'], '7e3b8d1e38c6384dc1d6a36b6a7048492591711e899efaebe9bb3fe71ea0f51f')
        self.assertEqual(artifact['sanitization']['changed_fields'], ['source', 'output'])
        non_path = {k: v for k, v in provenance.items() if k not in ('source', 'output')}
        canonical = json.dumps(non_path, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(digest, '6e21675314c90b644925b530c8761a008cb21b52e74e155dff71b1afcae3eda8')
        self.assertEqual(digest, artifact['sanitization']['preserved_fields_canonical_sha256'])

    def test_current_profile_passes(self):
        result = validator.run_checks(ROOT)
        self.assertGreaterEqual(result['total'], 200)
        self.assertEqual(result['passed'], result['total'])
        self.assertEqual(result['strict_validation']['screening']['candidate_count'], 401)

    def test_missing_profile_fails_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validator.run_checks(Path(directory))
            self.assertLess(result['passed'], result['total'])
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional independent metric reference unavailable')
    def test_tie_aware_metrics_match_independent_reference(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        labels = [0, 1, 0, 1, 1, 0, 0, 1]
        scores = [.1, .7, .7, .5, .9, .2, .5, .1]
        observed = validator.metrics_from_scores(labels, scores)
        self.assertAlmostEqual(observed['auroc'], roc_auc_score(labels, scores), places=14)
        self.assertAlmostEqual(observed['auprc'], average_precision_score(labels, scores), places=14)


if __name__ == '__main__':
    unittest.main()

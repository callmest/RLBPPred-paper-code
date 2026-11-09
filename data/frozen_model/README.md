# Frozen strict401 model inputs and outputs

This is a local manuscript-v12 working profile, not a claim of a published
GitHub release or DOI. `manifest.json` records artifact SHA256 checksums, the
training protocol, input/scored counts, screening policy, and checkpoint format.
All CSV scores retain their original precision.

`strict_input_manifest.csv` fixes the 376-positive/9,109-negative benchmark,
301/7,288 training split and 75/1,821 validation split. Validation row order is
given by `val_predictions.csv`. `sampling_manifest.json` fixes all 300 rounds'
sampled negative IDs and full training order. No training is needed to inspect
these records or screen the frozen predictions.

`human_all_predictions.csv` has 16,969 scored proteins; the mouse file has
17,002. `candidates_401.csv` is restricted to the 5,640 scored human
untrained-proteome entries with probability >= 0.98. Its `probability` column
matches the frozen 5880 output exactly. `probability_L40S` records the separate
verified machine result; their maximum absolute difference is 5.5631e-9. The
held-out empirical FPR is 1/1,821 = 0.0549149%, satisfying the 0.1% target. The
0.5 classification threshold and whole-group probability summaries are distinct.

`input_validation.json` is an unchanged original run record. The public
`provenance.json` replaces only the original `source` and `output` absolute
paths with logical relative labels; all other JSON fields are unchanged.
Its original and distributed SHA256 checksums and changed fields are recorded
in `manifest.json`, and the original bytes are retained in the private audit
archive, outside Git. The provenance's 0.85 candidate threshold refers to the initial audit
of that same trained run, **not** to this profile's current screening policy.
Use `manifest.json` → `screening` for the current policy.

The approximately 906 MiB `strict_ensemble.pt` and large embeddings are not
included. No working public download URL is asserted. Obtain the trusted
checkpoint separately and verify its manifest SHA256 before deserialization.
The loader supports its `state_dict_ensemble_v1` format and embedded scaler
arrays; historical `ensemble.pkl` files are not interchangeable.

From the repository root:

```bash
python pipeline/modeling/strict_runtime.py validate
python pipeline/modeling/strict_runtime.py screen --output /new/path/candidates.csv
python pipeline/modeling/strict_runtime.py predict \
  --model /trusted/path/strict_ensemble.pt \
  --input /trusted/path/human_input_cache.npz \
  --output /new/path/predictions.csv --device cpu
```

Prediction NPZ input must contain `X` (N × 1,280 float embeddings) and unique
`protein_id` entries. Inference applies the training scaler using the original
float32 in-place operations and averages all 300 member probabilities. Output
paths must not exist and cannot be inside frozen data or model-input folders.

For an explicitly requested full reconstruction (not figure reproduction):

```bash
python pipeline/modeling/strict_runtime.py train \
  --embedding-dir /trusted/path/saprot-650m-af2 \
  --output-dir /new/path/strict_reconstruction --device cuda --execute
```

The command requires exact `positive_mean` and `negative_mean` membership and
the frozen embedding-matrix hash. It uses the explicit split/order manifests,
seeds once as in the original run, and fits the scaler only to training data.
It refuses to train without `--execute`. It does not implement resume; an
interrupted reconstruction must use a new output directory. Hardware/library
differences can affect scores; the output reports differences rather than
asserting byte-identical retraining. No training or new model fitting was
performed while preparing this local release profile.

# Current strict-model workflow

This document describes the strict SaProt-650M-AF2 + MLP profile used for the
current 401-candidate analysis. It does not invoke training when figures are
redrawn. Historical generic ensemble/pickle entrypoints are not substitutes for
the current state-dict/scaler interface.

## 1. Validate and screen frozen predictions (no checkpoint required)

From the repository root:

```bash
python pipeline/modeling/strict_runtime.py validate
mkdir -p outputs
python pipeline/modeling/strict_runtime.py screen --output outputs/candidates401.csv
```

The output filename must not already exist. The validation checks artifact
hashes, all benchmark IDs, exact split/order, all 300 sampling rounds,
human/mouse score coverage and the 401 candidate IDs/scores.

The strict benchmark contains 376 positives and 9,109 negatives. A fixed
class-stratified split holds out 75 positives and 1,821 negatives, leaving
301 positives and 7,288 negatives for training. Each of 300 members uses all
301 training positives plus 301 training negatives sampled without replacement
within that round. Different rounds may reuse negatives. The frozen
`sampling_manifest.json` records the sampled negative IDs and complete shuffled
training order; `strict_input_manifest.csv` fixes the split.

The 0.5 classification threshold is distinct from discovery. At probability
>=0.5, held-out confusion counts are TN=1,732, FP=89, FN=4, TP=71.
AUROC=0.9926487278052353 and AUPRC (average precision)=0.9273565842784495.

Discovery applies probability >=0.98 to the 5,640 scored human
untrained-proteome entries only. It yields 401 candidates. Empirical FPR is
computed on the 1,821 held-out negatives, not the full 9,109 negative group:
1/1,821 = 0.0549149%, below the target of 0.1%. This is an observed held-out
rate, not a guarantee about population FPR or candidate false-discovery rate.

Human score coverage is 376 significant positives, 1,844 potential positives,
5,640 untrained proteins and 9,109 negatives (16,969 total). The original input
lists contain 1,879 potential positives and 5,649 untrained proteins; unscored
inputs must not be assigned zero. Mouse scores cover 76 significant positives,
284 potential positives and 16,642 negatives (17,002 total).

## 2. Pretrained inference: recommended safe bundle

For normal prediction, use the independent
[RLBPPred-inference v0.2.0 package](https://github.com/callmest/RLBPPred-inference/tree/v0.2.0)
and its [fixed GitHub Release assets](https://github.com/callmest/RLBPPred-inference/releases/tag/v0.2.0):

```bash
python -m pip install "rlbppred @ git+https://github.com/callmest/RLBPPred-inference.git@v0.2.0"
rlbppred setup --components model
rlbppred predict --embeddings features.npz --output predictions.csv
```

`features.npz` must contain unique nonempty `protein_id` strings and a finite,
raw, unstandardized float32 `X` matrix of shape N x 1,280 from matching
SaProt-650M-AF2 global-mean features. Do not standardize these inputs again.
The safe `model.safetensors` bundle contains all 300 trained MLP members and
their complete frozen scaler arrays; setup verifies the pinned file sizes and
hashes. The output must be a new file. No training occurs.

Normal users do **not** need the original `strict_ensemble.pt`, a separate
scaler file or this training checkout. For FASTA input, install the `fasta` extra
and follow the inference repository's versioned
[FASTA and offline instructions](https://github.com/callmest/RLBPPred-inference/blob/v0.2.0/README.md).
The safe bundle and that workflow preserve the existing strict model rather
than substituting a different trained model.

### Maintainer compatibility and regression interface only

The following legacy-checkpoint interface is retained for maintainers comparing
historical runs or checking numerical compatibility. It is not the recommended
user installation route, and the original pickle-capable `.pt` is not an
additional public download requirement.

Maintainer-only external assets:

- The approximately 906 MiB `strict_ensemble.pt`, with the exact SHA256 in
  `data/frozen_model/manifest.json`.
- Matching globally mean-pooled SaProt-650M-AF2 embeddings. Input NPZ must
  contain `X` (N x 1,280 numeric embeddings) and unique `protein_id` strings.
- PyTorch installed in a compatible CPU/GPU environment.

```bash
mkdir -p outputs
python pipeline/modeling/strict_runtime.py predict \
  --model /trusted/path/strict_ensemble.pt \
  --input /trusted/path/human_input_cache.npz \
  --output outputs/human_recomputed.csv --device cpu
```

The checkpoint format is `state_dict_ensemble_v1`, with 300 MLP states and
the fitted training scaler arrays. The loader verifies the expected SHA256
before deserializing the trusted checkpoint. This archive includes NumPy
objects and therefore requires pickle-enabled loading; never supply untrusted
weights or a checksum obtained from the same untrusted source.

The model is 1,280 -> 512 -> 256 -> 1, with batch normalization, ReLU and
dropout 0.3 in the hidden blocks. The inference interface applies the frozen
scaler using the original float32 in-place arithmetic and averages the
300 sigmoid probabilities. It does not fit a scaler to inference inputs.
Output scores retain full float precision; model/output source files are not
overwritten. A separately reconstructed trusted checkpoint requires its own
explicit `--model-sha256`.

The original `.pt` checkpoint and historical full-proteome embeddings are not
bundled, and no public endpoint for those original files is asserted. This does
not limit availability of the complete safe pretrained model linked above.
The October repository-only QA did not execute this legacy checkpoint's forward
pass; the inference package separately records actual-weight and two-protein
FASTA regression. Packaging validation did not rerun all 16,969 human and
17,002 mouse predictions. Frozen cross-machine candidate comparisons remain
available in `candidates_401.csv` (`probability` versus `probability_L40S`).

## 3. Optional training reconstruction

Do not run this step merely to reproduce figures. Training is opt-in:

```bash
mkdir -p outputs
python pipeline/modeling/strict_runtime.py train \
  --embedding-dir /trusted/path/saprot-650m-af2 \
  --output-dir outputs/new_strict_reconstruction --device cuda --execute
```

The embedding root must contain exact `positive_mean/<accession>.pt` and
`negative_mean/<accession>.pt` membership. Each trusted file supplies the
`embeddings` tensor. The assembled 9,485 x 1,280 float32 matrix must match the
frozen hash. No embeddings are downloaded or regenerated by this command.

The scaler is fitted only on the 7,589 training proteins. Python, NumPy,
PyTorch and CUDA RNGs are seeded once with 42, matching the recorded run;
a new per-member reseeding scheme is not introduced. Each member is trained
for 50 epochs with Adam (learning rate 0.001), binary cross-entropy with logits
and batch size 64. The command replays frozen split/order/sampling records,
saves new per-round states and a new ensemble, and reports differences from
the frozen validation predictions. It refuses to run without `--execute`.

The observed training environment was Python 3.12.12, NumPy 1.26.4,
scikit-learn 1.8.0 and PyTorch 2.10.0+cu128 on an RTX 5880 (CUDA 12.8).
`environment-training.yml` records these core versions but is not a complete
OS/CUDA dependency lock. Hardware/library differences can change numerical
outputs; byte-identical retraining is not promised. There is no automatic
resume or reuse of an existing output directory.

## 4. Frozen downstream analyses

```bash
python -m pipeline.analysis.current_frozen_analysis \
  --streme-dir data/frozen_analysis/streme401 \
  --go-csv data/panels/current/SupplementaryFigure8_all_GO.csv
```

This parses already completed STREME output and verifies frozen GO summaries;
it launches neither STREME nor GO downloads. The STREME XML uses
`test_evalue` and `npassing` (unique candidate proteins). There are 13 reported
motifs, 7 with E<0.05 and 11 displayed at E<=1.2. Composition groups are
Cys-containing, Lys/Arg-rich and Other sequences, not functional assignments.

For explicitly supplied frozen GO annotations, the separate opt-in
`pipeline/analysis/current_go_enrichment.py` can recompute one-sided Fisher
tests. It requires direct annotation rows, all 401 candidate IDs, 5,640 scored
untrained IDs and the 5,638 returned-ID universe, plus a snapshot identifier.
The comparison background is 5,237 proteins; terms need >=5 members in the
returned pool, and BH adjustment is within BP/CC/MF. It performs no ancestor
propagation or live retrieval. The complete annotation input is not bundled,
so figure replay is the supported offline path; see its `--help` before use.

## Database exports

The runtime schema contains 14 fields, excluding sequence. The original
construction intermediate contains 15 fields including sequence. To join the
frozen strict scores to an existing trusted runtime annotation snapshot:

```bash
python pipeline/database/update_runtime_scores.py \
  --annotations-json /trusted/path/rlbp_annotations.json \
  --human-predictions data/frozen_model/human_all_predictions.csv \
  --mouse-predictions data/frozen_model/mouse_all_predictions.csv \
  --output-dir outputs/new_database_snapshot
```

The output directory must be new or empty. Only `confidence` changes; annotation
fields and record order are preserved. The command produces JSON, CSV and a
build manifest. It rejects duplicate prediction IDs, invalid scores, unmatched
prediction IDs and unexpected runtime fields. Missing strict scores become JSON
null/CSV blank, never zero or a retained historical score. CSV/JSON preserve full
precision; six-decimal formatting belongs only in the browser UI.

Expected current coverage is 37,077 records: human 20,075 (16,969 scored and
3,106 unscored), mouse 17,002 (all scored). The 401 discovery candidates are not
the same as all database proteins above 0.98. The Figure 6 query returns six
reviewed human "DNA topoisomerase" matches sorted by descending score; only
TOP3B belongs to the 401 discovery set.

This operation does not overwrite or deploy the separate RLBPPredDB website.
Live annotation construction remains dependent on exact external snapshots.

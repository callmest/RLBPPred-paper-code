# RLBPPred reproducibility package

Code and frozen results for the current **strict401 / manuscript v12** profile.
This checkout is being prepared for the next GitHub release; the release is not
claimed to be online until its tag and assets are visible.

RLBPPred uses frozen SaProt-650M-AF2 representations and a 300-member balanced
MLP ensemble. The strict benchmark contains 376 positives and 9,109 negatives.
Each member uses 301 training positives and 301 sampled training negatives;
75 positives and 1,821 negatives are held out. Human discovery uses scores
**>=0.98**, with **1/1,821** held-out false positives (empirical FPR **0.0549%**,
below the **0.1%** target), yielding **401 candidates** in the 5,640-protein
scored untrained pool. This empirical target is not a population-level guarantee.
Classification summaries retain the separate threshold **0.5**.

## Two codebases

The [callmest/RLBPPred repository](https://github.com/callmest/RLBPPred)
retains model training, downstream analyses, frozen scientific evidence and
manuscript-figure reproduction. The independent
[callmest/RLBPPred-inference repository](https://github.com/callmest/RLBPPred-inference)
is the publication target for the installable `rlbppred` command: structure
acquisition, SaProt feature extraction and pretrained prediction, as well as
prediction from matching raw, 1,280-dimensional embeddings. It does not depend
on this repository's directory layout or include training and plotting
workflows. The prediction repository's release documentation records which
workflow and asset versions have actually passed acceptance testing.

The existing `pipeline/modeling/strict_runtime.py predict` entrypoint remains
available for compatibility and numerical regression checks. The new inference
interface uses a safe `safetensors` model bundle converted from the
SHA256-verified frozen checkpoint. The release plan supplies weights separately
through the prediction repository's GitHub Releases, not Git or the installation wheel.
The SaProt encoder is obtained directly from its version-pinned official
source. No PyPI publication or Zenodo archive is required or claimed. See
[codebase roles, installation and asset sources](docs/codebase_split.md).

## Quick start: validate and redraw frozen results

Use a checkout containing `data/panels/current/` and `data/frozen_model/`
with the documented hashes. During release preparation, do not assume that the
remote default branch already contains every file in this checkout.

```bash
# Start in the repository root.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/validate_release.py --root . --report outputs/current_validation.json
python scripts/reproduce_figures.py --all \
  --data-dir data/panels/current --output-dir outputs/current_figures
```

These commands validate packaged evidence and generate quantitative panels in
PNG, SVG and PDF. They do not train a model, run STREME, fetch annotations, alter
the manuscript, or publish a website. Final Illustrator layouts, experimental
images, structure surfaces and the database screenshot are not fabricated by
the plotting command. Output directories should be separate from source data.

For the complete offline tests and metadata checks, install the validation
dependencies in a disposable environment:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
make validate
```

These dependency ranges describe supported validation tools, not the original
training environment. The observed validation versions and actual acceptance
results are recorded in [validation status](docs/validation_status.md).

## Three reproducibility levels

| Level | Packaged material | Scope |
|---|---|---|
| Frozen-result replay | Current panel CSVs, complete human/mouse scores, 401 candidates, validation predictions, source hashes | Validate numbers and regenerate quantitative plots without weights or a GPU |
| Frozen-checkpoint inference | State-dict loader, frozen scaler contract, checkpoint hash | Requires the separately supplied trusted checkpoint and matching embeddings; no fitting |
| Training/analysis reconstruction | Exact strict split and 300 sampling rounds, training protocol, parameterized analysis code | Requires external large assets and pinned environments; not claimed as an independently completed end-to-end rerun |

See [model and inference instructions](docs/reproduce_model.md),
[figure mapping and export instructions](docs/reproduce_figures.md), and
[remaining limitations](docs/known_limitations.md).

## Current data and provenance

- `data/panels/current/`: current quantitative data, mapped to six main figures
  and eight supplementary figures. `manifest.json` identifies exact source
  workbook ranges, columns, row counts and hashes.
- `data/frozen_model/`: strict split, all 300 sampling lists/orderings,
  validation metrics, validation/human/mouse predictions and scored candidates.
- `data/model_inputs/`: input accession lists. The 1,879 potential-positive and
  5,649 untrained input IDs are distinct from scored coverage (1,844 and 5,640).
- `data/frozen_analysis/`: available frozen downstream evidence and provenance.
- `configs/paper.yaml`: explicit current model, classification, discovery and
  database conventions; it is not a substitute for the frozen manifests.

Historical top-level `data/panels/*.csv` and the original analysis scripts are
retained for provenance, including the previously corrected Figure2b metric.
They are **not** the default current profile. Do not mix them with
`data/panels/current/`. Old documentation is under `docs/history/20260826/`.

The 59 other representation/classifier comparisons, unfiltered and random-forest
controls, and independent sampling-robustness experiments remain their own
experiments; their results are not replaced by the final ensemble's metrics.

## Database: local deployment only

The separate [RLBPPredDB repository](https://github.com/callmest/RLBPPredDB)
contains the browser application and local deployment instructions. No public
server is required: serve it on `127.0.0.1` using Python's HTTP server.

Current database evidence covers 37,077 annotation records, with 33,971 strict
scores and 3,106 unscored human records. Missing scores remain null/blank, not
zero or historical scores. The browser has 14 fields; the construction
intermediate has 15, including sequence. Scores retain full precision in data
files and are displayed to six decimals. See [database reconstruction](docs/reproduce_model.md#database-exports).
Use a database version matching the documented hashes; its remote publication
status is separate from the reproducibility and prediction releases.

## Documentation

- [Repository map](docs/repository_map.md)
- [Figure reproduction](docs/reproduce_figures.md)
- [Model, inference and database reconstruction](docs/reproduce_model.md)
- [Sources and asset availability](docs/data_sources.md)
- [Known limitations](docs/known_limitations.md)
- [Validation status](docs/validation_status.md)

## Citation and license

Original code is [MIT licensed](LICENSE). See [CITATION.cff](CITATION.cff) for
software metadata and cite the associated manuscript. Release tags and model
assets are hosted on GitHub; no archival DOI is claimed. External datasets, models,
software and experimental images retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

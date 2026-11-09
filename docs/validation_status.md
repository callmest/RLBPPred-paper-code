# Version 1.1.0 validation: 2026-11-09

Maintainer: `callmest`. Training/reproducibility version **1.1.0** is maintained
in `callmest/RLBPPred`; the independent installation and pretrained-prediction
package is version **0.2.0** in `callmest/RLBPPred-inference`. The strict MLP
assets are distributed with the inference v0.2.0 GitHub Release, while the fixed
SaProt checkpoint is downloaded from its official upstream source. Local
software acceptance and remote publication/CI status are separate checks.
See [codebase roles and asset versions](codebase_split.md).

## Checks executed on 9 November

| Check | Result |
|---|---|
| Complete offline pytest suite, including plotting dependencies | 53 passed, 13 subtests passed; no skipped tests |
| Frozen scientific/provenance validation without an external source workbook | 210/210 checks passed |
| CFF schema validation | Passed with cffconvert 2.0.0, schema 1.2.0 |
| Metadata, environment declarations and dependency syntax | Passed |
| Python source syntax | All 60 files parsed without execution |
| Git whitespace check | Passed |
| Public historical provenance redaction | Only `source` and `output` values changed; all other JSON fields exactly match the original |
| Original provenance preservation | Byte-identical private backup and original SHA256 recorded; not added to Git |

The pytest run completed in 51.24 seconds after installing the complete
validation dependencies. The earlier interrupted/partial check is not used as
an acceptance result. Full stdout logs, pytest XML, machine-readable validation
JSON, dependency inventory and command/exit-status records are retained in the
maintainer's separate audit workspace, not mixed into frozen scientific inputs.

The environment was macOS ARM64 with Python 3.12.14, NumPy 2.5.4, pandas 2.3.3,
SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.11.2, openpyxl 3.1.5,
UpSetPlot 0.9.0, PyYAML 6.0.3, pytest 9.2.0 and cffconvert 2.0.0.
`requirements-test.txt` records installable validation dependency ranges;
these are not the observed original training environment or exact training pins.

## Provenance disclosure and boundaries

The public `data/frozen_model/provenance.json` contains logical relative
source/output labels instead of personal absolute paths. Its manifest retains
the original SHA256
`7e3b8d1e38c6384dc1d6a36b6a7048492591711e899efaebe9bb3fe71ea0f51f`,
the distributed SHA256, original/distributed sizes, changed-field list and
canonical SHA256 of every unchanged field. The validator and a regression test
check these records. No model parameters, protocol values, scores, candidate
membership or scientific panels were changed by this redaction.

No training, model inference, STREME discovery, annotation retrieval or complete
45-panel redraw was launched by this validation pass. Figure tests exercise
the offline plotting interfaces; the earlier 45-panel acceptance is recorded
below. No manuscript, official figure, source workbook, supplementary table,
database or old analysis result was overwritten. The prediction package's
real-weight and FASTA acceptance records are separate from this repository's
offline evidence checks. Remote GitHub Actions execution and publication are
not claimed by local tests.

The separate remote run for strict401 commit `75d1160` subsequently passed:
[GitHub Actions validation, 9 November 2026](https://github.com/callmest/RLBPPred/actions/runs/42572774723).
It included metadata checks, frozen-data validation, offline regression tests
and all registered quantitative panel replays. This is a software/replay check,
not a new training run or a repeat of all biological analyses from raw assets.

# Historical repair validation: 2026-10-28

The following is the dated earlier repair record. Its then-outstanding
cffconvert/local dependency checks are superseded by the successful November
validation above; its model-inference limits describe that earlier repair only.

Status: current strict401 working profile repaired and locally validated.
No commit, push, public release or archival DOI was created.

## Completed changes

- Explicit workbook-to-panel map and transactional export; historical/index
  sheets are excluded and exact repeated exports are no-ops.
- Current frozen data are isolated in data/panels/current. The 35 historical
  top-level panel CSV/manifest files were compared against the pre-repair backup
  and are unchanged, including two pre-existing modified panel files.
- Strict input/split, 300 resampling rounds, validation/human/mouse scores,
  401-candidate list and provenance are packaged with hashes.
- Current state-dict/scaler inference interface and opt-in reconstruction
  interface replace ambiguous historical defaults in the documentation.
- FPR uses 1,821 held-out negatives; discovery uses >=0.98 in the scored
  untrained pool. Classification at >=0.5 remains separate.
- Current quantitative plotting registry covers 45 panels, with separate
  manual-asset declarations and protected new output directories.
- STREME text/XML parsing and descriptive composition groups agree with the
  completed 401-candidate run; raw and distributed output hashes are explicit.
- Database score exports preserve full precision and missing values; the
  separate website and its data were not edited.
- README, reproduction instructions, data inventory, environment metadata,
  Makefile and offline CI entrypoints now describe the current profile.

## Checks executed

| Check | Result |
|---|---|
| Offline unit/regression tests | 52 passed |
| Current-profile scientific/provenance checks | 208/208 passed with the original source workbook hash rechecked; 207 without that optional external file |
| Python source syntax | 60 files parsed without execution |
| Git whitespace check and large-file guard | Passed |
| Quantitative panel replay | 45 panels exported in PNG/SVG/PDF; contact sheets visually inspected |
| Text editability | All 45 SVGs retain text and all 45 PDFs contain selectable text |
| SupFig7b legend follow-up | Single panel re-exported and inspected after adding color/area explanations |
| Database trial export | 37,077 records; 33,971 scored; 3,106 missing; all scores and non-score fields/order exactly match the current local snapshot |
| CSV/JSON precision | Full-precision numeric round trip passed |
| Protected documents/site files | 14 pre-recorded SHA256 values unchanged |
| Historical panel data | 35 files unchanged from pre-repair backup |

Independent review checked the strict RNG/split/scaler/sample-order protocol
against the actual run, the trusted checkpoint hash and serialized layer
storage shapes, and preservation of the other 59 model rows, both independent
ROC controls and all sampling-robustness rows. Another review checked current
documentation commands and scientific definitions.

## Boundaries

No model was trained, no STREME run was launched, and no manuscript, official
figure, source workbook, supplementary workbook, website or old analysis result
was overwritten. Generated QA outputs are separate from publication assets.
Historical standalone analysis entrypoints require explicit historical opt-in.

The validation environment used Python 3.13.9 for the full offline test/plot
suite; export/some model checks also ran with Python 3.12.14. These are not a
claim of bitwise training reconstruction. The observed original training core
versions are recorded separately in the model documentation and manifest.

The full 906 MiB checkpoint forward pass was **not executed** during this
repair because this local validation environment did not have PyTorch. The
loader was checked with valid/invalid synthetic state dictionaries, including
all 300 members' keys, shapes, dtypes, finite values and batch-normalization
state. A complete real-weight inference acceptance run remains an external-
asset/environment check, not a completed part of this repair.

CFF metadata was parsed as YAML, but the full cffconvert schema check and the
remote GitHub Actions job were not executed locally. CI configuration includes
those checks for a future authorized push. Large assets and original annotation
snapshots still need stable distribution before claiming end-to-end public
reproducibility; see known_limitations.md.

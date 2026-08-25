# RLBPPred reproducibility package

[![Validate reproducibility package](https://github.com/callmest/RLBPPred-reproducibility/actions/workflows/validate.yml/badge.svg)](https://github.com/callmest/RLBPPred-reproducibility/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-blue.svg)](LICENSE)

This repository contains the code, curated identifiers, panel-level source
data, and documentation associated with **RLBPPred**, a structure-aware protein
language-model framework for proteome-scale prediction of R-loop-binding
proteins.

The release is organized around two reproducibility levels:

- **Fast figure reproduction** rebuilds 26 computational figure panels from
  committed, final-numbered CSV files and validates manuscript-critical values.
- **Full pipeline audit and reconstruction** exposes 44 inventoried first-party
  scripts for embedding extraction, training, inference, downstream analyses,
  and RLBPPredDB construction, while explicitly recording external assets and
  steps that are not yet independently reconstructable.

The repository does not claim that a plotting script can repeat wet-lab
experiments or recreate manually assembled Illustrator panels and browser
screenshots. Those boundaries are listed in
[`docs/known_limitations.md`](docs/known_limitations.md).

## Quick start

Python 3.10 or 3.11 is recommended for the figure-only route.

```bash
git clone https://github.com/callmest/RLBPPred-reproducibility.git
cd RLBPPred-reproducibility

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-figures.txt

python scripts/validate_release.py --root .
python scripts/reproduce_figures.py \
  --all \
  --data-dir data/panels \
  --output-dir outputs/figures
```

The equivalent short commands are:

```bash
make validate-release
make reproduce-figures
```

To generate one panel only:

```bash
python scripts/reproduce_figures.py \
  --figure Fig3e \
  --data-dir data/panels \
  --output-dir outputs/figures
```

Every supported panel is written as both PNG and SVG. Generated outputs are
ignored by Git so a local run does not dirty the repository.

## Reproducibility coverage

| Manuscript content | Release status |
|---|---|
| Fig. 1b; Fig. 2b; Fig. 3a–e; Fig. 5a–d | Reconstructed from committed panel CSV files |
| Supplementary Fig. 1a–c, 2a–d, 3a–d, and 5a–d | Reconstructed from committed panel CSV files |
| Fig. 1a and Fig. 2a | Manual workflow schematics; method and data lineage are documented |
| Fig. 4a,c | Wet-lab panels; source-image and experiment requirements are documented |
| Fig. 4b,d | External APBS/ChimeraX workflow; exact historical maps and camera sessions remain required |
| Fig. 6a,b | Static-site screenshots; database-build scripts and validation records are included |
| Final multipanel layout | Manual Illustrator composition; generated panels are not expected to be byte-identical to the submitted PDFs |

The current manuscript contains Supplementary Figures 1, 2, 3, and 5. There
are no Supplementary Figures 4 or 6 in the final numbering.

See [`docs/reproduce_figures.md`](docs/reproduce_figures.md) for the exact
panel-to-file map and [`docs/reproduce_model.md`](docs/reproduce_model.md) for
the downstream training and structure-to-prediction workflows.

## Released data

`data/panels/` contains 34 panel-level CSV files plus a checksum manifest.
These are the canonical inputs for the fast reproduction route. In particular:

- Figure 1b contains 1,430 normalized records, with 407 proteins present in at
  least two studies and four present in all five studies.
- The five study sizes are 462, 364, 308, 434, and 430.
- Figure 3a uses the final labels: filtered benchmark AUROC 0.9925 and
  unfiltered dataset AUROC 0.9001.

`data/model_inputs/` provides six accession manifests defining the recurrent
positives, final positive and negative benchmark classes, potential positives,
untrained proteome, and 342 high-confidence candidates. Protein sequences can
be retrieved from the [UniProt website REST API](https://www.uniprot.org/help/programmatic_access)
using `scripts/download_uniprot_fasta.py`; archive the database release and
checksums for a frozen run.

The complete external-source inventory and missing snapshot metadata are
listed in [`docs/data_sources.md`](docs/data_sources.md). The code license does
not relicense third-party datasets, models, or executables.

## Validation targets

The automated release validator checks, among other invariants:

- 376 final positives and 9,109 final negatives in the released identifier
  manifests;
- SaProt-AF2 + MLP as the strongest tested combination, with AUROC 0.9925 and
  AUPRC 0.9264;
- the held-out confusion matrix values 1,729/92/4/71 at threshold 0.5;
- RLBPPredDB scope of 20,075 human and 17,002 mouse proteins (37,077 total);
- the TOP1 example, UniProt P11387, with confidence 0.999743;
- panel-file checksums, required documentation, broken links, and files too
  large for ordinary GitHub storage.

The manuscript configuration is summarized in
[`configs/paper.yaml`](configs/paper.yaml). Values that cannot yet be verified
from deposited raw predictions are intentionally excluded from hard acceptance
tests and disclosed in the limitations document.

## Full model and analysis pipeline

The original first-party scripts are grouped under:

```text
pipeline/
├── embeddings/   # ESM-C, PST and SaProt representations; UMAP inputs
├── modeling/     # classifiers, balanced ensemble, robustness and inference
├── analysis/     # GO/domain, LLPS, disorder, chromatin, motif and AA analyses
├── figures/      # legacy publication plotting scripts
└── database/     # RLBPPredDB annotation and runtime-table construction
```

[`pipeline/ORIGIN_MANIFEST.tsv`](pipeline/ORIGIN_MANIFEST.tsv) records the
remote source path, release path, source SHA-256, portability status, and notes
for all 44 scripts. Machine-specific absolute paths were replaced with
arguments, environment variables, or repository-relative defaults without
changing reported scientific thresholds or ensemble settings.

Large pretrained weights, embeddings, protein structures, model checkpoints,
and third-party database snapshots are not stored in Git. Their required
provenance and the remaining environment gaps are documented before any claim
of end-to-end reproduction. The downstream modeling environment is described
in `environment-training.yml`; ESM-C, PST, and SaProt extraction environments
must be isolated and pinned separately because their source trees, checkpoints,
CUDA/PyG combinations, and in some cases Python namespaces differ.

## Repository guide

- [`docs/repository_map.md`](docs/repository_map.md) — where each artifact lives
- [`docs/reproduce_figures.md`](docs/reproduce_figures.md) — per-panel commands,
  inputs, outputs, and checks
- [`docs/reproduce_model.md`](docs/reproduce_model.md) — training, inference,
  resources, seeds, and environment boundaries
- [`docs/data_sources.md`](docs/data_sources.md) — external data/model provenance
- [`docs/known_limitations.md`](docs/known_limitations.md) — unresolved derivations
  and manual/external panels
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — licensing boundaries

## Citation

If you use this release, cite the associated RLBPPred manuscript and the
software metadata in [`CITATION.cff`](CITATION.cff). A DOI-backed archival
record can be added after the repository is connected to an archival service;
no DOI is claimed in this initial GitHub release.

## License

Original code and documentation in this repository are released under the
[MIT License](LICENSE). External software, pretrained models, database records,
published datasets, and generated derivatives remain subject to their own
licenses and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

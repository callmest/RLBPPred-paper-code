Repository rename notice (2026-11-09): this is archived documentation.
The former paper repository is now
[RLBPPred-paper-code](https://github.com/callmest/RLBPPred-paper-code).
The old `callmest/RLBPPred` address now belongs to the main prediction software;
use the paper-code entry point for historical paper links below. The archived
content is otherwise preserved.

# Reproducing RLBPPred training and inference

This document describes the computational path reported in the manuscript and
the boundary between a fast result check and a full retraining. The scripts in
`pipeline/` are an inventory of the original analysis code. Their source and
release paths, checksums, modification status, and preservation notes are
listed in `pipeline/ORIGIN_MANIFEST.tsv`; their scientific roles and required
inputs are documented by the stage descriptions below and the script headers.

The release does not replace an unverified legacy command with an invented
one. Before running any stage, inspect its provenance row in
`ORIGIN_MANIFEST.tsv`, read the script header, and run
`python <script> --help` when the script provides a command-line interface.
The manifest records source path, release path, checksum, modification status,
and a preservation note; exact scientific inputs and outputs remain defined by
the script and the stage descriptions below. Scripts that depend on hard-coded paths or unpublished
intermediate files are identified in `docs/known_limitations.md`.

## Reproducibility levels

For downstream training and inference, create the documented Conda environment:

```bash
conda env create -f environment-training.yml
conda activate rlbppred-training
```

`environment-training.yml` intentionally excludes the ESM-C, PST, and SaProt
embedding stacks because their historical source trees, namespaces, CUDA/PyG
builds, and checkpoint revisions must be isolated and pinned separately. It is
an environment specification for downstream modeling, not proof that the full
embedding pipeline has been reconstructed.

### Level 1: validate the released results

This level is suitable for reviewers who want to check file integrity, panel
data, core counts, metrics, predictions, and database exports without
retraining the 300-member ensemble.

```bash
python scripts/validate_release.py --root .
python scripts/reproduce_figures.py \
  --all \
  --data-dir data/panels \
  --output-dir outputs/figures
```

This route uses released panel data and therefore tests the connection between
reported numerical results and generated plots. It does not independently
recreate the protein embeddings or model checkpoints.

The manuscript-level constants checked by this route are summarized in
`configs/paper.yaml`. This file records the release seed, benchmark and filter
counts, 300-round 1:1 ensemble setting, principal performance values, screening
thresholds, and database totals. It is a compact reference configuration, not
a complete training recipe: external release identifiers, exact split
membership, model architecture/optimizer settings, and large-asset locations
are not all encoded there.

### Released accession manifests

`data/model_inputs/` contains one reviewed UniProt accession per line for the
following manuscript groups:

| Released list | Records |
|---|---:|
| `recurrent_positive_ids.txt` | 407 |
| `significant_positive_ids.txt` | 376 |
| `negative_ids.txt` | 9,109 |
| `potential_positive_ids.txt` | 1,879 |
| `untrained_proteome_ids.txt` | 5,649 |
| `high_confidence_candidate_ids.txt` | 342 |

The 376- and 9,109-record lists can be passed directly to modeling entry
points that accept `--pos-ids` and `--neg-ids`. The other lists support group
scoring and candidate auditing. They do not identify which benchmark proteins
belonged to the historical training versus held-out subsets, and they do not
record the negatives sampled in each ensemble round.

For a new sequence acquisition, the helper below retrieves the current
UniProtKB FASTA records and verifies the returned record count:

```bash
python scripts/download_uniprot_fasta.py \
  --ids data/model_inputs/significant_positive_ids.txt \
  --output data/external/significant_positive.fasta
```

The response is not a frozen historical snapshot. Record the UniProt release,
access date, and output checksum, or use a deposited sequence file with a
verified checksum for archival reproduction. Existing FASTA files can be
converted back to accession lists with `scripts/extract_fasta_ids.py`.

### Level 2: retrain from frozen embeddings and split manifests

This level is intended to start from fixed-length protein embeddings and exact
train/held-out identifiers. Those large embeddings and historical split
manifests are not established as present in the current Git release. Once
deposited, this is the preferred route for testing downstream classification
while avoiding repeated download and encoding of proteome-scale structures.

Required components for this level are:

- a benchmark-membership table with 376 positives and 9,109 negatives;
- explicit train and held-out ID manifests, rather than a newly sampled split;
- mean-pooled embeddings for every representation being tested;
- one configuration file that pins classifier hyperparameters, seeds,
  resampling count, threshold, and output locations;
- a checksum manifest for every embedding shard and split file.

The original modeling scripts are grouped under `pipeline/modeling/`. Use
`pipeline/ORIGIN_MANIFEST.tsv` to confirm provenance and whether a portability
edit was made, then use each script's CLI/header to identify its exact inputs
and outputs. The principal scripts cover general classifier training, the
SaProt-650M-AF2 + MLP ensemble, the handcrafted random-forest baseline,
cross-comparison, human and mouse inference, sampling stability, and alternative
undersampling strategies.

### Level 3: full structure-to-prediction reconstruction

This level starts from sequence/structure identifiers and regenerates
embeddings before retraining. It additionally requires the external data and
model releases listed in `docs/data_sources.md`.

The original encoding scripts are grouped under `pipeline/embeddings/` and
cover ESM-C extraction, PST extraction, SaProt structural-token generation,
SaProt extraction, and mouse structure download. Representative discovery
commands are:

```bash
python pipeline/embeddings/extract_esmc_embedding.py --help
python pipeline/embeddings/extract_pst_embedding.py --help
python pipeline/embeddings/generate_saprot_tokens.py --help
python pipeline/embeddings/extract_saprot_embedding.py --help
python pipeline/embeddings/download_mouse_pdb.py --help
```

If a script does not expose `--help`, inspect its header and the manifest's
preservation note, then document any unresolved path dependencies. It should
not be described as turnkey until those dependencies have been replaced with
explicit arguments.

## Auditable command templates

The commands below use the released CLI names. Text enclosed in angle brackets
is a required local path or release identifier, not a literal argument. Resolve
every placeholder against `docs/data_sources.md` and the frozen release
manifest before running a full reconstruction. These templates expose the
intended stage order; they do not make missing checkpoints or unpublished
intermediate files available.

### Build structure-aware and sequence embeddings

Generate SaProt structural tokens from a structure directory, then mean-pool
the corresponding SaProt representations:

```bash
python pipeline/embeddings/generate_saprot_tokens.py \
  --pdb-dir <pdb-directory> \
  --output-fasta <saprot-structural-token-fasta> \
  --foldseek <foldseek-binary> \
  --plddt-threshold 70

python pipeline/embeddings/extract_saprot_embedding.py \
  --fasta-file <saprot-structural-token-fasta> \
  --output-dir <saprot-mean-embedding-directory> \
  --model-path <saprot-checkpoint> \
  --pooling mean \
  --device cuda
```

The same pair of commands is run separately for the AF2-derived and
PDB-derived SaProt inputs. The precise checkpoint revision, source structure
release, and chain-selection rule remain release metadata that must be filled
in; using the same model family name is not sufficient.

Generate mean-pooled ESM-C embeddings for each frozen FASTA input:

```bash
python pipeline/embeddings/extract_esmc_embedding.py \
  --fasta-file <input-fasta> \
  --output-dir <esmc-300m-mean-embedding-directory> \
  --model esmc_300m \
  --pooling mean \
  --device cuda

python pipeline/embeddings/extract_esmc_embedding.py \
  --fasta-file <input-fasta> \
  --output-dir <esmc-600m-mean-embedding-directory> \
  --model esmc_600m \
  --pooling mean \
  --device cuda
```

Generate PST embeddings from the frozen structure set:

```bash
python pipeline/embeddings/extract_pst_embedding.py \
  --pdb-dir <pdb-directory> \
  --output-dir <pst-mean-embedding-directory> \
  --model-path <pst-checkpoint> \
  --aggr mean \
  --device cuda
```

Run this stage separately for PST-T33 and PST-T33-SO. Whether the historical
run used `--include-seq`, and the exact mapping between that option and the two
manuscript labels, must be confirmed from the archived run configuration before
claiming an exact reconstruction.

Mouse AlphaFold structures can be downloaded by the preserved helper when its
input spreadsheet and remote service remain available:

```bash
python pipeline/embeddings/download_mouse_pdb.py \
  --id-file <mouse-id-workbook> \
  --output-dir <mouse-pdb-directory> \
  --organism mouse
```

### Train the benchmark and principal ensemble

The general benchmark script expects six specifically named subdirectories
under one embedding root. Each must contain `positive_mean/` and
`negative_mean/` directories. With no `--upstream` or `--classifiers` filter,
the command runs all six representations against all ten classifiers:

```bash
RLBPPRED_EMBEDDINGS=<embedding-root> \
python pipeline/modeling/train_classifier.py \
  --n-rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <classifier-comparison-output-directory>
```

The script creates its split internally and does not accept an explicit split
manifest. Therefore, this command is a reconstruction of the curated script's
current behavior, not yet a guaranteed replay of the exact historical held-out
membership. Preserve and compare its validation IDs before interpreting metric
differences.

Train the principal SaProt-650M-AF2 + MLP ensemble from explicit accession
lists and mean-pooled embeddings:

```bash
python pipeline/modeling/train_sa_mlp.py \
  --pos-ids data/model_inputs/significant_positive_ids.txt \
  --neg-ids data/model_inputs/negative_ids.txt \
  --embedding-dir <saprot-af2-mean-embedding-directory> \
  --n-rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <principal-model-output-directory>
```

Expected files include `ensemble.pkl`, `results.json`,
`val_predictions.csv`, and `roc_curve.svg`. The script likewise creates its
validation split internally; an exact release should additionally publish the
realized split and each round's sampled negative IDs.

### Run controls and robustness analyses

The handcrafted-feature baseline and the two cross-dataset controls use the
following entry points:

```bash
python pipeline/modeling/train_rf_baseline.py \
  --rf-features <handcrafted-feature-workbook> \
  --pos-ids <rf-positive-id-list> \
  --neg-ids <rf-negative-id-list> \
  --val-ratio 0.2 \
  --seed 42 \
  --output-dir <rf-baseline-output-directory>

python pipeline/modeling/train_cross_comparison.py \
  --experiment rf_features_our_data \
  --pos-ids <filtered-positive-id-list> \
  --neg-ids <filtered-negative-id-list> \
  --rf-features <handcrafted-feature-workbook> \
  --n-rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <rf-features-our-data-output-directory>

python pipeline/modeling/train_cross_comparison.py \
  --experiment saprot_rf_data \
  --pos-ids <rf-positive-id-list> \
  --neg-ids <rf-negative-id-list> \
  --embedding-dir <saprot-af2-mean-embedding-directory> \
  --n-rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <saprot-rf-data-output-directory>
```

Run the sampling-ratio, convergence, and undersampling comparisons with the
manuscript settings stated explicitly. The undersampling script defaults to
100 rounds, so `--rounds 300` is required for the reported comparison:

```bash
python pipeline/modeling/run_sampling_stability.py \
  --pos-ids data/model_inputs/significant_positive_ids.txt \
  --neg-ids data/model_inputs/negative_ids.txt \
  --embedding-dir <saprot-af2-mean-embedding-directory> \
  --ratios 1 2 4 \
  --round-checkpoints 50 100 200 300 \
  --ratio-rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <sampling-stability-output-directory>

python pipeline/modeling/run_undersampling_strategy_comparison.py \
  --pos-ids data/model_inputs/significant_positive_ids.txt \
  --neg-ids data/model_inputs/negative_ids.txt \
  --embedding-dir <saprot-af2-mean-embedding-directory> \
  --rounds 300 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda \
  --output-dir <undersampling-output-directory>
```

### Score human and mouse datasets

For the four human groups and three mouse groups, provide directories of ID
lists matching the filenames expected by the two preserved scripts:

| Human inference filename | Released source list |
|---|---|
| `sig_pos_ids.txt` | `data/model_inputs/significant_positive_ids.txt` |
| `neg_ids.txt` | `data/model_inputs/negative_ids.txt` |
| `potential_positive.txt` | `data/model_inputs/potential_positive_ids.txt` |
| `untrained_proteome.txt` | `data/model_inputs/untrained_proteome_ids.txt` |

Create a staging directory with those four filenames, preserving the released
file contents. The current Git release does not provide corresponding mouse ID
lists under the filenames expected by `inference_mouse_datasets.py`; obtain and
version those inputs before running the mouse command.

```bash
python pipeline/modeling/inference_four_datasets.py \
  --checkpoint <principal-model-output-directory>/ensemble.pkl \
  --embedding-dir <human-saprot-af2-embedding-root> \
  --id-dir <human-four-dataset-id-directory> \
  --output-dir <human-inference-output-directory> \
  --device cpu

python pipeline/modeling/inference_mouse_datasets.py \
  --checkpoint <principal-model-output-directory>/ensemble.pkl \
  --embedding-dir <mouse-saprot-af2-mean-embedding-directory> \
  --id-dir <mouse-three-dataset-id-directory> \
  --output-dir <mouse-inference-output-directory> \
  --device cpu
```

For a standalone structure or embedding directory, use
`pipeline/modeling/inference.py`. Set `--threshold 0.5` for the reported held-out
classification rule or `--threshold 0.85` for the human discovery rule; always
retain the continuous probability column.

### Build database exports

After freezing the human and mouse prediction tables and annotation workbooks,
build the intermediate proteome tables and then the runtime exports:

```bash
python pipeline/database/build_proteome_annotation_csv.py \
  --human-xlsx <human-annotation-workbook> \
  --mouse-xlsx <mouse-annotation-workbook> \
  --human-predictions <human-prediction-table> \
  --mouse-predictions <mouse-prediction-table> \
  --output-dir <database-intermediate-directory>

python pipeline/database/build_annotation_db.py \
  --input <database-intermediate-directory>/all_proteome_annotations_for_db.csv \
  --json-out <database-output-directory>/rlbp_annotations.json \
  --csv-out <database-output-directory>/rlbp_annotations.csv
```

The first command writes 15 columns including `sequence`; the current runtime
builder writes 14 fields and drops `sequence`. Treat this as a known schema
discrepancy until the implementation or manuscript description is reconciled.

## Reference workflow

### 1. Freeze the input universe

The manuscript benchmark comprises 376 high-confidence RLBPs and 9,109
high-confidence non-RLBPs. The positive labels derive from recurrent evidence
across five published human R-loop proteomics studies. The negative set derives
from the reviewed human proteome after excluding positives and proteins linked
to selected GO terms, enriched domains, R-loopBase regulators, and validation
sets, followed by redundancy, length, and structure-availability filtering.

For a reproducible release, the following files are authoritative:

1. the final positive and negative UniProt accession lists;
2. a per-protein provenance/filter table;
3. the exact train and held-out split manifests;
4. the identifiers reserved as potential positives and untrained-proteome
   candidates;
5. database release/version metadata and checksums.

Do not regenerate the held-out split by enumerating a directory. Several legacy
scripts call `os.listdir()` and convert lists through Python sets, which can
change ordering. Use the released split manifests exactly.

### 2. Generate protein representations

Six frozen upstream representations were compared:

| Representation | Type | Manuscript output dimension | Required external component |
|---|---|---:|---|
| ESMC-300M | Sequence-only | 960 | Exact model identifier and revision: **TO BE CONFIRMED** |
| ESMC-600M | Sequence-only | 1,152 | Exact model identifier and revision: **TO BE CONFIRMED** |
| PST-T33 | Structure-integrated | **TO BE CONFIRMED** | PST code/checkpoint revision and structural graph inputs |
| PST-T33-SO | Structure-integrated | **TO BE CONFIRMED** | PST code/checkpoint revision and structural graph inputs |
| SaProt-650M-AF2 | Structure-aware | 1,280 | SaProt checkpoint, Foldseek build, and AlphaFold structures |
| SaProt-650M-PDB | Structure-aware | 1,280 | SaProt checkpoint, Foldseek build, and experimental/PDB-derived structural inputs |

Residue-level hidden states are reduced by global mean pooling. A release must
state how sequences exceeding the model context length were handled, whether
special tokens were excluded from pooling, how missing residues/chains were
treated, and how proteins without valid structures were removed. These details
cannot be inferred reliably from a final embedding file.

### 3. Benchmark all representation-classifier combinations

Each of the six representations was paired with ten classifiers:

1. logistic regression;
2. support vector machine;
3. random forest;
4. XGBoost;
5. LightGBM;
6. k-nearest neighbors;
7. Gaussian naive Bayes;
8. multilayer perceptron;
9. recurrent neural network;
10. long short-term memory network.

This produces 60 benchmark combinations. Evaluate every combination on the
same released held-out set and report AUROC, AUPRC, F1, precision, recall,
Matthews correlation coefficient, and accuracy. The expected ranking result is
that SaProt-650M-AF2 + MLP is the strongest ranking model, with AUROC 0.9925 and
AUPRC 0.9264 in the manuscript release. SaProt-650M-AF2 + SVM is the preferred
fixed-threshold model in the manuscript, with F1 0.6364, precision 0.4828,
recall 0.9333, MCC 0.6543, and accuracy 0.9578 at the reported threshold.

### 4. Train the principal balanced ensemble

The manuscript's principal model is SaProt-650M-AF2 + MLP. In each of 300
ensemble rounds:

1. retain all training positives;
2. sample an equal number of training negatives without replacement within the
   round;
3. train one downstream MLP on that 1:1 subset;
4. save its checkpoint and the sampled negative IDs;
5. predict each evaluation protein with every ensemble member;
6. average the 300 probabilities to obtain the soft-voting score.

The release should preserve the sampled negative-ID list for every round. A
single global seed is insufficient if library versions, directory order, or
parallel workers change. The model configuration must also pin the MLP
architecture, optimizer, learning rate, weight decay, epoch/early-stopping
rule, batch size, probability calibration behavior, and checkpoint-selection
criterion.

### 5. Run robustness analyses

The original scripts for these analyses are under `pipeline/modeling/` and are
identified in `ORIGIN_MANIFEST.tsv`.

- Sampling ratios: positive:negative ratios 1:1, 1:2, and 1:4 at 300 rounds.
- Ensemble convergence: cumulative predictions after 50, 100, 200, and 300
  rounds with the 1:1 ratio.
- Negative selection: random, K-means representative, and K-means hard-negative
  undersampling.
- Dataset construction control: retrain on the unfiltered 2,220-positive,
  14,749-negative dataset.
- Prior-feature baseline: random forest on 30 handcrafted descriptors using 354
  positives and 16,575 negatives with complete features.

The expected manuscript ranges are AUROC ≥ 0.992 and AUPRC
0.9276–0.9364 across the sampling-ratio comparison, and AUROC 0.992–0.993,
AUPRC 0.921–0.926, and MCC 0.611–0.623 across undersampling strategies.

### 6. Calibrate thresholds and score proteomes

At threshold 0.5, the held-out result reported in the manuscript contains 75
positives and 1,821 negatives, of which 71 positives are recovered and 92
negatives pass the threshold. For proteome-scale discovery, use the held-out
negative pass rate as an empirical false-positive rate rather than retaining
the 0.5 classification threshold.

The principal discovery threshold is 0.85. It corresponds to a 1.15% held-out
negative pass rate and yields 342 candidates from the untrained human proteome.
The separately calculated 1%-FPR threshold is 0.8583. Preserve continuous
scores in all exports; do not retain only binary labels.

The trained human ensemble is then applied without retraining to:

- human significant positives, potential positives, untrained reviewed
  proteins, and negatives;
- mouse significant positives, potential positives, and negatives using mouse
  UniProt sequences and AlphaFold structures encoded with the same
  SaProt-650M-AF2 pipeline.

The mouse predictions are a zero-shot transfer analysis. The manuscript notes
that the mouse negative-set score distribution differs from the human one, so
the human 0.85 discovery threshold must not be represented as a calibrated
mouse threshold.

### 7. Run downstream analyses and database construction

Original downstream scripts are grouped under `pipeline/analysis/`. They cover
GO/domain enrichment, EpiFactors, LLPS database overlap,
MobiDB-lite disorder, STREME motifs, and amino-acid composition. These analyses
consume the frozen 342-candidate list, 376 known RLBPs, the relevant non-RLBP or
untrained-proteome background, and versioned external annotations.

Database construction scripts are under `pipeline/database/`. The final release
should produce separate human and mouse CSV files plus a combined table using a
15-field intermediate schema. The currently inventoried runtime builder retains
only 14 fields and omits `sequence`; this must be reconciled before describing
the deployed runtime schema as 15 fields. Validation targets are 20,075 human records, 17,002 mouse
records, and 37,077 combined records. A query for reviewed human TOP1 at minimum
confidence 0.85 should return UniProt P11387 with score 0.999743.

## Compute and storage requirements

No complete wall-time, peak-memory, GPU-memory, or storage benchmark has been
verified for the full manuscript workflow. Therefore this release must not
invent a tested hardware specification.

The practical expectations are:

- figure-only reproduction from `data/panels/` is CPU-suitable;
- classical classifiers and downstream enrichment can generally run on CPU,
  although 300 repeated fits can be time-consuming;
- proteome-scale ESM-C, PST, and SaProt extraction is the GPU- and
  storage-intensive stage;
- pretrained weights, AlphaFold structures, embeddings, 300 checkpoints, and
  proteome-wide predictions may exceed normal Git repository limits.

Before the archived release is declared fully reproducible, record the tested
operating system, CPU, RAM, GPU model/count, GPU memory, CUDA/driver versions,
wall time, peak memory, and disk use for Levels 1–3. Large licensed assets should
be deposited in an appropriate data/model repository and fetched with verified
checksums rather than committed to Git.

## Randomness and numerical reproducibility

The following stages are stochastic or order-sensitive:

- train/held-out splitting if regenerated instead of using the release
  manifest;
- negative sampling in each ensemble round;
- neural-network initialization, batch ordering, and dropout;
- some CUDA kernels;
- UMAP initialization and neighbor construction;
- K-means initialization;
- filesystem enumeration in legacy scripts.

For a deterministic rerun, pin Python, NumPy, PyTorch, scikit-learn, UMAP,
XGBoost, LightGBM, CUDA, and cuDNN versions; seed Python, NumPy, PyTorch, UMAP,
and K-means; sort every input list; preserve the exact split and per-round
sampling manifests; and record deterministic PyTorch settings. If a kernel
cannot be deterministic, document it and compare predictions and metrics with
an approved tolerance rather than requiring byte-identical checkpoints.

Panel regeneration from the frozen released CSV files should be exact in data
content. Full retraining is not expected to be byte-identical across hardware
unless the above conditions are demonstrated. The reported metric values are
reference checkpoints, not confidence intervals for stochastic reruns.

## Completion criteria

A full reconstruction is complete only when it produces:

- the 376/9,109 benchmark and exact released split membership;
- all 60 benchmark rows with the seven reported metrics;
- the 300-member SaProt-650M-AF2 + MLP ensemble or an explicitly documented
  equivalent reconstruction;
- frozen human and mouse probability tables;
- the 342-candidate table at threshold 0.85;
- the downstream enrichment tables;
- the three RLBPPredDB export files and schema;
- panel data accepted by `scripts/reproduce_figures.py`;
- a passing `python scripts/validate_release.py --root .` report.

Known gaps that currently prevent a claim of complete raw-to-figure
reproducibility are listed in `docs/known_limitations.md`.

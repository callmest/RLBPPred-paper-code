# Known reproducibility limitations

This release is designed to make the available analysis code, panel data, and
provenance auditable without overstating what has been reconstructed. The fast
figure route reproduces computational plots from released panel-level data. It
is not yet equivalent to an independently verified raw-data-to-manuscript run.

## Panels that are not fully automatic

- **Figure 1a** is a manually assembled benchmark-construction schematic.
- **Figure 2a** is a manually assembled model/workflow schematic.
- **Figure 4a and 4c** are wet-lab image panels. Code cannot reproduce the
  underlying immunoprecipitation experiment. A complete data release still
  requires uncropped source images, acquisition metadata, and a crop/contrast
  log.
- **Figure 4b and 4d** were created with externally obtained PDB structures,
  APBS, and manual ChimeraX visualization. Exact APBS input parameters, output
  maps, camera orientations, and ChimeraX session/command files have not yet
  been verified against the final manuscript panels.
- **Figure 6a and 6b** are screenshots of the static RLBPPredDB interface. The
  data build can be checked, but the exact submitted browser rendering and crop
  are manual.
- Final panel lettering, layout, typography, and Illustrator composition are
  manual for all composite figures. The plotting wrapper regenerates panel
  content, not byte-identical final figure pages.

These panels must be labelled manual/external in any reproducibility claim.

## Missing or incomplete derivations

- The legacy **Figure 3b** comparison uses a merged ROC-coordinate CSV for
  which a complete raw-prediction-to-merged-CSV generator was not located.
  Figure reproduction can use the released intermediate, but that is not a
  complete derivation.
- A complete aggregator that rebuilds the final 60-combination classifier CSV
  used by Figure 2b/Supplementary Figure 2b from all raw training outputs was
  not located. The released metric tables are auditable intermediates, not a
  complete raw-log derivation.
- The legacy **Supplementary Figure 3a** confusion-matrix plot contains
  hard-coded counts. The released plot must be checked against held-out labels
  and probabilities before use.
- The legacy **Figure 3a** filtered/unfiltered dataset naming is historically
  inverted in some paths. The final curve labels and manifest, rather than the
  directory name, must be treated as authoritative.
- The corrected **Figure 1b** source was maintained separately from the
  comprehensive source-data workbook. Re-exporting panel CSVs therefore
  requires both `--input` and `--figure1b-input`.
- The released **Figure 3e** panel CSV contains 100 selected rows (20 for each
  of GO:BP, GO:MF, GO:CC, InterPro, and Pfam), of which 80 have FDR < 0.05. It
  does not contain the complete 4,270-term test universe needed to independently
  verify the manuscript statement that 113 terms were significant. The full
  tested-term table must be deposited before that statement can be validated
  from the repository.
- The released **Supplementary Figure 3b** CSV is a 300-point threshold grid
  and does not contain rows at the exact thresholds 0.85, 0.8583, or 0.99. The
  nearest grid values do not reproduce the manuscript's exact candidate counts.
  Frozen per-protein scores or an exact-threshold summary are required to verify
  342 candidates at 0.85, the 1%-FPR threshold, and 21 candidates at 0.99.
- An end-to-end, portable script that starts with the five original proteomics
  downloads and reconstructs every positive/negative filtering decision has
  not yet been verified. The released benchmark/provenance table is therefore
  an essential input, not merely an output.
- `data/model_inputs/` releases the principal human group memberships, but it
  does not contain the exact historical train/held-out split or the negative
  identifiers sampled in each of the 300 ensemble rounds. The training scripts
  will create a new seeded split; that is not proof of identical historical
  membership.
- The released human lists can be staged under the filenames expected by
  `inference_four_datasets.py`. Equivalent versioned mouse ID lists are not
  currently present under the filenames expected by
  `inference_mouse_datasets.py`, so zero-shot mouse inference is not turnkey.
- The current inventory does not establish that every large embedding,
  pretrained weight, 300-member checkpoint set, or full prediction table is
  included in Git. These assets may require an external archival deposit.
- `pipeline/database/build_proteome_annotation_csv.py` writes a 15-column
  intermediate table that includes `sequence`, but
  `pipeline/database/build_annotation_db.py` defines a 14-field runtime schema
  and drops `sequence`. The manuscript currently describes a 15-field resource
  schema. The intended downloadable versus runtime schema must be stated
  explicitly or the code/manuscript made consistent before release.

## Numerical and manuscript issues to resolve

The following values should not all be promoted to automated acceptance tests
until their definitions are reconciled:

1. The current Results state that Figure 1b contains a union of 1,430 proteins,
   with 407 found in at least two studies and four in all five. A subsequent
   paragraph states that mapping the same five studies produced 2,335 unique
   proteins. The five original Table 1 counts sum to 1,998, so the 2,335 value
   requires correction or an explicit explanation of a different universe.
2. GO analyses use a reviewed human-proteome universe of 20,435 proteins,
   whereas RLBPPredDB contains 20,075 human records. This may reflect different
   database releases or filters, but neither is currently pinned sufficiently
   to explain the difference.
3. The manuscript's held-out results contain 75 positives and 1,821 negatives,
   while a legacy dataset-generation script samples 10% of each class. The
   release must provide the exact split manifest and state the actual split
   policy; readers should not regenerate it from the legacy default.
4. The principal manuscript configuration uses 300 balanced ensemble rounds,
   but legacy entry points contain other defaults, including 24 or 100 rounds.
   The final release configuration and per-round sampling manifest must override
   those defaults explicitly.
5. The current manuscript has Supplementary Figures 1, 2, 3, and 5 only. There
   are no Supplementary Figures 4 or 6. Legacy files bearing older numbers must
   not be mapped into nonexistent final panels.

## External-source metadata gaps

Exact releases, access dates, accessions, licence terms, and checksums remain to
be confirmed for several resources, including UniProt, AlphaFoldDB, Gene
Ontology annotations, R-loopBase, InterPro, Pfam, the LLPS databases,
MobiDB-lite, the mouse RLBP annotation source, and the original supplementary
files for the five proteomics studies. Exact checkpoint revisions and hashes
are also needed for ESM-C, PST, and SaProt.

These missing values are shown as **TO BE CONFIRMED** in
`docs/data_sources.md`. They must not be silently replaced with the newest
available release, because that would not document the data actually used.

## Legacy-code portability

The original code was developed for a specific filesystem and compute
environment. The inventory contains scripts that may:

- derive project paths from the current working directory;
- contain absolute paths from the original cluster;
- assume locally installed pretrained models and Foldseek binaries;
- enumerate directories without sorting;
- serialize Python/PyTorch objects that are sensitive to package versions;
- expose configuration through source edits rather than command-line options;
- use defaults that differ from the final manuscript configuration;
- assume GPU availability or a particular CUDA setup without a tested fallback.

Moving a script into `pipeline/` preserves provenance but does not by itself
make it portable. `pipeline/ORIGIN_MANIFEST.tsv` should be consulted before
execution. A script without explicit inputs, outputs, and an environment record
should be described as archived analysis code, not a turnkey workflow.

## Randomness and exact reruns

Model initialization, batch order, per-round negative sampling, UMAP, K-means,
and some GPU kernels are stochastic. Filesystem ordering in legacy scripts can
introduce additional variation. Exact checkpoints may therefore differ across
hardware even when summary performance is similar.

The strongest reproducibility route is to release:

- sorted accession lists;
- the exact train/held-out split;
- all 300 per-round negative-sampling lists;
- random seeds for every library and analysis;
- pinned package, CUDA, and driver versions;
- frozen predictions used by the manuscript;
- metric tolerances justified from repeated runs.

The current reference values are manuscript checkpoints, not established
statistical tolerance intervals.

## Compute benchmarking

Tested wall time, RAM, GPU memory, disk usage, and hardware specifications have
not yet been recorded for a full six-representation, sixty-combination,
300-round reconstruction. Figure-only reproduction is lightweight, but full
proteome-scale structure encoding and checkpoint generation may be expensive.
Do not publish guessed resource estimates. Record measured requirements for the
archived release.

## Licence boundary

An MIT licence for original repository code does not automatically cover
third-party datasets, pretrained model weights, external software, or
experimental source images. Redistribution permission must be checked
resource-by-resource. If redistribution is not allowed, provide a versioned
download procedure and checksum rather than copying the asset into the
repository.

## Claims supported by the current release

The following wording is supportable when the panel data and validator are
present:

> The repository contains the available analysis code, panel-level data, and
> documentation needed to regenerate the computational plots and audit the
> principal reported counts and metrics.

The following wording is not yet supportable:

> Every manuscript result can be reproduced end to end from raw public data by
> running a single command.

A claim of complete raw-to-figure reproducibility should wait until the
numerical discrepancies, external versions, data deposits, environment locks,
manual-panel source assets, split manifests, and missing derivations above are
resolved.

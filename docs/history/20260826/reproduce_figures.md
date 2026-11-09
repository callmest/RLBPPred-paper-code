Repository rename notice (2026-11-09): this is archived documentation.
The former paper repository is now
[RLBPPred-paper-code](https://github.com/callmest/RLBPPred-paper-code).
The old `callmest/RLBPPred` address now belongs to the main prediction software;
use the paper-code entry point for historical paper links below. The archived
content is otherwise preserved.

# Reproducing the manuscript figures

This guide maps the **final manuscript numbering** to the released panel data and
figure-generation entry point. It intentionally distinguishes plots that can be
regenerated from tabular data from diagrams, experimental images, externally
rendered structures, screenshots, and final Illustrator assembly.

The final manuscript contains Figures 1–6 and Supplementary Figures 1, 2, 3,
and 5. There are no Supplementary Figures 4 or 6 in the current manuscript.
Legacy filenames may use older panel numbers; use the identifiers in this guide
and the repository manifests rather than inferring panel identity from a legacy
filename.

## Quick reproduction

The repository includes panel-level CSV files in `data/panels/`. These are the
inputs for the fast, figure-only route and avoid rerunning structure encoding or
model training.

Create an isolated environment and install the figure-only dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-figures.txt
```

The minimum dependency file uses compatible ranges because the historical
patch-level environment was not recovered. Archive a tested lock file with the
tagged release if byte-stable plots are required.

```bash
python scripts/validate_release.py --root .
python scripts/reproduce_figures.py \
  --all \
  --data-dir data/panels \
  --output-dir outputs/figures
```

Equivalent Make targets are `make install-figures`,
`make reproduce-figures`, and `make validate-release`. A single panel can be
selected with, for example, `make reproduce-figures FIGURE=Fig3e`.

To regenerate one panel, replace `--all` with its final panel identifier:

```bash
python scripts/reproduce_figures.py \
  --figure Fig3e \
  --data-dir data/panels \
  --output-dir outputs/figures
```

Run `python scripts/reproduce_figures.py --help` for the identifiers supported
by the release. A panel marked **manual/external** below is deliberately not
claimed as an automatically reproducible output.

The committed panel CSV files were exported from the manuscript source-data
workbooks. Maintainers can rebuild them with:

```bash
python scripts/export_source_data.py \
  --input <path-to-comprehensive-source-data-workbook> \
  --figure1b-input <path-to-corrected-v5-Figure-1b-workbook> \
  --output-dir data/panels
```

The second workbook is required because Figure 1b was corrected separately in
the v5 source-data update. Readers reproducing plots from the committed CSV
files do not need either workbook.

## Interpretation of output fidelity

The wrapper regenerates plotting-layer outputs, not the final multi-panel page.
Minor font metrics, label placement, and anti-aliasing may differ across
operating systems. Scientific content is verified from row counts, category
labels, statistics, thresholds, and model metrics. The final panel arrangement,
panel letters, microscopy or blot cropping, and journal typography were
assembled manually and are outside the automatic comparison.

## Exact panel-data and output map

Every automatic panel writes both `<panel-id>.png` and `<panel-id>.svg` in the
selected output directory. With the commands above, for example, Figure 3e is
written as `outputs/figures/Fig3e.png` and `outputs/figures/Fig3e.svg`.

| Panel identifier | Committed input file(s) |
|---|---|
| `Fig1b` | `data/panels/Figure1b.csv` |
| `Fig2b` | `data/panels/Figure2b.csv` |
| `Fig3a` | `data/panels/Figure3a.csv` |
| `Fig3b` | `data/panels/Figure3b.csv` |
| `Fig3c` | `data/panels/Figure3c.csv` |
| `Fig3d` | `data/panels/Figure3d.csv` |
| `Fig3e` | `data/panels/Figure3e.csv` |
| `Fig5a` | `data/panels/Figure5a.csv` |
| `Fig5b` | `data/panels/Figure5b.csv` |
| `Fig5c` | `data/panels/Figure5c_raw.csv`; validation also uses `Figure5c_stats.csv` and `Figure5c_summary.csv` |
| `Fig5d` | `data/panels/Figure5d.csv` |
| `SupFig1a` | `data/panels/SuppFigure1a.csv` |
| `SupFig1b` | `data/panels/SuppFigure1b.csv` |
| `SupFig1c` | `data/panels/SuppFigure1c.csv` |
| `SupFig2a` | `data/panels/SuppFigure2a.csv` |
| `SupFig2b` | `data/panels/SuppFigure2b.csv` |
| `SupFig2c` | `data/panels/SuppFigure2c_ratio.csv` and `SuppFigure2c_rounds.csv` |
| `SupFig2d` | `data/panels/SuppFigure2d.csv` |
| `SupFig3a` | `data/panels/SuppFigure3a.csv` |
| `SupFig3b` | `data/panels/SuppFigure3b.csv` |
| `SupFig3c` | `data/panels/SuppFigure3c.csv` |
| `SupFig3d` | `data/panels/SuppFigure3d.csv` |
| `SupFig5a` | `data/panels/SuppFigure5a.csv` |
| `SupFig5b` | `data/panels/SuppFigure5b.csv` |
| `SupFig5c` | `data/panels/SuppFigure5c.csv` |
| `SupFig5d` | `data/panels/SuppFigure5d_summary.csv`; validation also uses `SuppFigure5d_raw.csv` and `SuppFigure5d_stats.csv` |

`data/panels/manifest.json` records the source sheet, description, row count,
and SHA-256 checksum for every committed panel CSV. Manual panels have metadata
tables but no automatic image generator: `Figure4a_c.csv` describes the wet-lab
panels, `Figure4b_d.csv` describes the PDB/APBS panels, and `Figure6a_b.csv`
describes the database scope and representative records. These metadata files
are not substitutes for the experimental images, APBS grids, ChimeraX sessions,
or browser screenshots.

## Main figures

### Figure 1 — conservative construction of the RLBP benchmark

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Fig. 1a | Manual schematic; no plotting command | Benchmark-stage counts, provenance/filter table, manuscript Methods, and `docs/reproduce_model.md` | Final workflow diagram assembled manually | Final benchmark: 376 high-confidence positives and 9,109 negatives; recurrence, GO/domain/R-loopBase exclusion, MMseqs2 redundancy filtering, minimum length, and structure-availability filters are shown |
| Fig. 1b | `python scripts/reproduce_figures.py --figure Fig1b --data-dir data/panels --output-dir outputs/figures` | Corrected panel CSV exported with `--figure1b-input` | Figure 1b UpSet plot | Five published studies; union reported in the current Results as 1,430 proteins; 407 proteins detected in at least two studies; four detected in all five studies |

Figure 1a is a scientific workflow diagram, not a statistical plot. The
underlying benchmark lists and stage counts must still be released and checked
by `scripts/validate_release.py`.

### Figure 2 — RLBPPred workflow and model benchmark

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Fig. 2a | Manual schematic; no plotting command | Model and sampling workflow described in `docs/reproduce_model.md` | Final architecture/workflow diagram assembled manually | Six frozen upstream representations, global mean pooling, ten downstream classifiers, 376:9,109 benchmark, balanced resampling, and soft-voting inference are represented |
| Fig. 2b | `python scripts/reproduce_figures.py --figure Fig2b --data-dir data/panels --output-dir outputs/figures` | Sixty representation-classifier results | AUPRC heatmap | 6 representations × 10 classifiers; SaProt-650M-AF2 + MLP is the strongest ranking model; AUROC 0.9925 and AUPRC 0.9264 |

### Figure 3 — filtering, baselines, cross-species prediction, and enrichment

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Fig. 3a | `python scripts/reproduce_figures.py --figure Fig3a --data-dir data/panels --output-dir outputs/figures` | Filtered and unfiltered ROC coordinates | ROC comparison | Filtered AUROC/AUPRC 0.9925/0.9264; unfiltered 0.9001/0.6365 |
| Fig. 3b | `python scripts/reproduce_figures.py --figure Fig3b --data-dir data/panels --output-dir outputs/figures` | RLBPPred and handcrafted-feature baseline ROC coordinates | ROC comparison | RLBPPred AUROC/AUPRC 0.9925/0.9264; random-forest baseline 0.9680/0.4904; at FPR ≤ 0.01, TPR 0.8933 versus 0.4143 |
| Fig. 3c | `python scripts/reproduce_figures.py --figure Fig3c --data-dir data/panels --output-dir outputs/figures` | Mouse group-level predictions | Mouse mean-probability plot | Means: significant positive 0.983, potential positive 0.920, and negative 0.292 |
| Fig. 3d | `python scripts/reproduce_figures.py --figure Fig3d --data-dir data/panels --output-dir outputs/figures` | Human group-level predictions | Human mean-probability plot | Means: significant positive 0.974, potential positive 0.938, untrained proteome 0.309, and negative 0.135 |
| Fig. 3e | `python scripts/reproduce_figures.py --figure Fig3e --data-dir data/panels --output-dir outputs/figures` | Candidate GO, InterPro, and Pfam enrichment results | Enrichment summary | Panel CSV: 100 selected rows (20 per source), 80 with FDR < 0.05; the plot uses ranks 1–5 for each source. The manuscript-level 113/4,270 result requires the unavailable full tested-term table |

The legacy Figure 3a data directories use historically inverted dataset labels.
Use the release manifest and plot labels, not the legacy directory name, to
identify the filtered and unfiltered curves. The merged ROC input for Figure 3b
is a released intermediate; the legacy tree does not contain a complete script
that rebuilds that merged CSV from raw predictions.

### Figure 4 — experimental validation and electrostatic surfaces

| Panel | Reproduction status | Inputs | Output or review procedure | Verification target |
|---|---|---|---|---|
| Fig. 4a | Experimental/manual | Uncropped LIG4 input and S9.6 immunoprecipitation images, with ±RNase H controls | Inspect the Source Data image and documented crop; final panel assembled manually | LIG4 predicted probability 0.866; RNase H reduces the recovered signal |
| Fig. 4b | External/manual | RCSB PDB 3W5O | Remove ligands and waters, calculate APBS potential with the documented external workflow, and render two orientations in ChimeraX | Surface scale −10 to +10 kT/e and the two reported orientations |
| Fig. 4c | Experimental/manual | Uncropped IGHMBP2 input and S9.6 immunoprecipitation images, with ±RNase H controls | Inspect the Source Data image and documented crop; final panel assembled manually | IGHMBP2 predicted probability 0.877; RNase H reduces the recovered signal |
| Fig. 4d | External/manual | RCSB PDB 4B3J | Remove ligands and waters, calculate APBS potential with the documented external workflow, and render two orientations in ChimeraX | Surface scale −10 to +10 kT/e and the two reported orientations |

The repository does not claim to reproduce the wet-lab experiment. A complete
release should contain the uncropped experimental images and a crop/contrast
log. The APBS panels currently depend on an external web service and manual
ChimeraX rendering; no frozen APBS parameter file or ChimeraX session has yet
been verified against the submitted panels.

### Figure 5 — functional and sequence features of predicted RLBPs

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Fig. 5a | `python scripts/reproduce_figures.py --figure Fig5a --data-dir data/panels --output-dir outputs/figures` | EpiFactors category assignments | Category-composition plot | Novel candidates: 11 writers, 11 readers, 12 erasers, 14 remodelers, and eight other chromatin factors |
| Fig. 5b | `python scripts/reproduce_figures.py --figure Fig5b --data-dir data/panels --output-dir outputs/figures` | PhaSepDB, DrLLPS, LLPSDB, PhaSePro, and non-redundant union overlaps | LLPS overlap plot | 152/342 candidates, 339/376 known RLBPs, and 1,029/9,109 non-RLBPs overlap the union; candidate versus non-RLBP OR 6.28, one-sided Fisher p = 6.2 × 10^-51 |
| Fig. 5c | `python scripts/reproduce_figures.py --figure Fig5c --data-dir data/panels --output-dir outputs/figures` | MobiDB-lite disordered-residue fractions | Disorder-fraction comparison | n = 342 candidates, 376 known RLBPs, and 9,080 covered non-RLBPs; candidate versus known is not significant; each RLBP group versus non-RLBP has p < 0.0001 |
| Fig. 5d | `python scripts/reproduce_figures.py --figure Fig5d --data-dir data/panels --output-dir outputs/figures` | STREME motif table | Motif significance plot | Four motifs at E < 0.05; QVKIWFQNRRMKW occurs in 31 candidates with E = 6.6 × 10^-6 |

### Figure 6 — RLBPPredDB

| Panel | Reproduction status | Inputs | Output or review procedure | Verification target |
|---|---|---|---|---|
| Fig. 6a | Manual screenshot | Built static portal and released database CSVs | Serve the site locally or deploy it to GitHub Pages, then capture the home page manually | Navigation exposes Home, Annotation, Data Access, and Help |
| Fig. 6b | Manual screenshot | Built portal and final combined database | Query human TOP1 with minimum confidence 0.85 and capture the result manually | UniProt P11387 with score 0.999743; 37,077 total records = 20,075 human + 17,002 mouse |

The database build scripts are inventoried under `pipeline/database/`. The
current release does not automatically reproduce the exact submitted browser
screenshot or final Illustrator assembly.

## Supplementary figures

### Supplementary Figure 1 — benchmark exclusions and annotation separation

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Supp. Fig. 1a | `python scripts/reproduce_figures.py --figure SupFig1a --data-dir data/panels --output-dir outputs/figures` | GO-, domain-, positive-, and R-loopBase-exclusion membership | Exclusion UpSet plot | Reviewed human universe n = 20,435; 5,693/7,674 proteins in the union are supported by one source; 361 are shared by all three annotation-based sources |
| Supp. Fig. 1b | `python scripts/reproduce_figures.py --figure SupFig1b --data-dir data/panels --output-dir outputs/figures` | GO enrichment for positive and negative sets | Diverging GO enrichment plot | Positive n = 376 and negative n = 9,109, each tested against n = 20,435 using one-sided Fisher tests and within-comparison Benjamini–Hochberg correction |
| Supp. Fig. 1c | `python scripts/reproduce_figures.py --figure SupFig1c --data-dir data/panels --output-dir outputs/figures` | InterPro enrichment table | Positive-set domain enrichment plot | Positive n = 376; bars represent −log10 FDR |

### Supplementary Figure 2 — representations, metrics, and sampling robustness

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Supp. Fig. 2a | `python scripts/reproduce_figures.py --figure SupFig2a --data-dir data/panels --output-dir outputs/figures` | Six mean-pooled embedding matrices and label/sampling manifests | Full and balanced UMAP panels | Each representation compares 376 positives/9,109 negatives with a 376/376 balanced subset; UMAP seed and parameters must come from the release configuration |
| Supp. Fig. 2b | `python scripts/reproduce_figures.py --figure SupFig2b --data-dir data/panels --output-dir outputs/figures` | Sixty-model metric table | AUROC, F1, MCC, precision, and recall heatmaps | Complete 6 × 10 grid for each metric; no missing combinations |
| Supp. Fig. 2c | `python scripts/reproduce_figures.py --figure SupFig2c --data-dir data/panels --output-dir outputs/figures` | Sampling-ratio and cumulative-round results | Stability plots | Ratios 1:1, 1:2, and 1:4 at 300 rounds; cumulative results at 50, 100, 200, and 300 rounds; AUROC ≥ 0.992 and AUPRC 0.9276–0.9364 across ratios |
| Supp. Fig. 2d | `python scripts/reproduce_figures.py --figure SupFig2d --data-dir data/panels --output-dir outputs/figures` | Random, K-means representative, and K-means hard-negative results | Undersampling comparison | AUROC 0.992–0.993, AUPRC 0.921–0.926, and MCC 0.611–0.623 |

### Supplementary Figure 3 — threshold calibration and graded prediction

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Supp. Fig. 3a | `python scripts/reproduce_figures.py --figure SupFig3a --data-dir data/panels --output-dir outputs/figures` | Held-out labels and probabilities at threshold 0.5 | Confusion matrix | 71/75 positives recovered and 92/1,821 negatives called positive; the legacy plotting script contains hard-coded counts and must be checked against released predictions |
| Supp. Fig. 3b | `python scripts/reproduce_figures.py --figure SupFig3b --data-dir data/panels --output-dir outputs/figures` | Threshold scan, held-out-negative pass rates, and untrained-proteome scores | FPR/candidate-count threshold plot | The 300-row grid verifies 9.6827% and 1,665 candidates at threshold 0.5 and shows reference lines at 0.85/0.8583. Exact manuscript values at 0.85, 0.8583, and 0.99 require frozen raw scores or an exact-threshold table not currently released |
| Supp. Fig. 3c | `python scripts/reproduce_figures.py --figure SupFig3c --data-dir data/panels --output-dir outputs/figures` | Human group probabilities | Human call-rate plot | At threshold 0.5: 98.1%, 96.3%, 29.5%, and 9.7% for significant positives, potential positives, untrained proteome, and negatives |
| Supp. Fig. 3d | `python scripts/reproduce_figures.py --figure SupFig3d --data-dir data/panels --output-dir outputs/figures` | Mouse group probabilities | Mouse call-rate plot | At threshold 0.5: 100%, 95.1%, and 26.8% for significant positives, potential positives, and negatives |

### Supplementary Figure 5 — extended candidate characterization

| Panel | Command | Inputs | Generated output | Verification target |
|---|---|---|---|---|
| Supp. Fig. 5a | `python scripts/reproduce_figures.py --figure SupFig5a --data-dir data/panels --output-dir outputs/figures` | Overall EpiFactors membership | Overall enrichment plot | 56/342 candidates, 71/376 known RLBPs, and 65/9,109 non-RLBPs; candidate versus non-RLBP OR 27.2 and p = 3.6 × 10^-49 |
| Supp. Fig. 5b | `python scripts/reproduce_figures.py --figure SupFig5b --data-dir data/panels --output-dir outputs/figures` | Category-specific EpiFactors enrichment | Category enrichment plot | Candidate odds ratios: reader 151.3, remodeler 64.8, eraser 30.1, other 19.8, and writer 8.6; all p < 10^-6 |
| Supp. Fig. 5c | `python scripts/reproduce_figures.py --figure SupFig5c --data-dir data/panels --output-dir outputs/figures` | Motif-to-functional-category assignments | Motif-category count plot | Light bars are all displayed motifs; dark bars are the subset with E < 0.05; categories match the Figure 5d annotation table |
| Supp. Fig. 5d | `python scripts/reproduce_figures.py --figure SupFig5d --data-dir data/panels --output-dir outputs/figures` | Grouped amino-acid counts and denominators | Twenty-residue radar plot | Exactly 20 standard residues for 342 candidates, 376 known RLBPs, and 9,109 non-RLBPs; each group composition sums to 1 within numeric tolerance |

## Tables generated from the same analyses

The release validator should also check the corresponding tabular outputs:

| Table | Content and validation |
|---|---|
| Table 1 | Five published datasets and their original reported counts: Cristini 462, Wang 364, Mosler 308, Yan 434, and Shinriki 430 |
| Supplementary Table 1 | Gene-to-UniProt mapping, filter status, and final benchmark membership; final positive/negative counts 376/9,109 |
| Supplementary Table 2 | Complete metrics for all 60 representation-classifier combinations |
| Supplementary Table 3 | All sampling-ratio, ensemble-round, and undersampling-strategy conditions used in Supplementary Fig. 2c,d |
| Supplementary Table 4 | Exactly 342 human candidates at threshold 0.85, with prediction scores and annotations |
| Supplementary Table 5 | GO/domain, LLPS, motif, and EpiFactors results used by Figures 3 and 5 and Supplementary Figure 5 |
| Supplementary Table 6 | Fifteen-field RLBPPredDB schema and the human, mouse, and combined download files |

## Final checks

After generating the automatic panels, run:

```bash
python scripts/validate_release.py --root .
```

Do not report a fully reproduced manuscript if any manual/external panel lacks
its source asset or provenance note. See `docs/known_limitations.md` for the
current boundaries and `docs/repository_map.md` for the location of every
released component.

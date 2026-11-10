# Current quantitative figure replay

The default profile is `data/panels/current/`, mapped to manuscript v12 after
the strict-model, >=0.98, 401-candidate update. Historical top-level panel files
are retained only as provenance and are not current default inputs.

## Redraw

Run these commands from the updated repository root.

```bash
python -m pip install -r requirements-figures.txt
python scripts/validate_release.py --root .
python scripts/reproduce_figures.py --all \
  --data-dir data/panels/current --output-dir outputs/current_figures
python scripts/reproduce_figures.py --figure Fig5d \
  --output-dir outputs/motif_panel
```

Outputs are PNG, SVG and PDF plus `replay_report.json` with source hashes.
Use a separate new output directory. These are quantitative panel replays, not
byte-identical final Illustrator pages. Manual layouts, experimental images,
structure surfaces and screenshots are not synthesized. Default raster export
is 300 dpi; `--dpi 500` changes raster resolution without altering values.

## Current registry

| Panels | Packaged quantitative source | Meaning |
|---|---|---|
| Fig1a | Figure1a.csv | Exactly 2/3/4/5-study recurrence, union denominator 1,430 |
| Fig1b | Figure1b.csv | Original GO/domain/R-loopBase set intersections; schematic, non-area-proportional Venn |
| Fig1d | Figure1d.csv | Positive/negative GO enrichment, top five significant terms per aspect and direction |
| Fig2c | Figure2c.csv | Ten classifiers x six representations; current strict MLP value, other 59 unchanged |
| Fig3a-b | Figure3a.csv, Figure3b.csv | Strict-model ROC versus separate unfiltered/RF controls |
| Fig3c-d | Figure3c.csv, Figure3d.csv | Seven within-group score distributions: 20 equal-width bins, 140 frequency knots and shape-preserving cubic interpolation (PCHIP); not group-mean bars or candidate counts |
| Fig5a-b | Figure5a.csv, Figure5b.csv | Chromatin-category and LLPS database overlaps |
| Fig5c | Figure5c_raw/summary/stats.csv | Disorder: 401/376/9,059 observed proteins; 50 negative values missing |
| Fig5d | Figure5d.csv | 11 displayed motifs, only 7 significant; all 13 retained in source table |
| SupFig1 | SupplementaryFigure1a.csv | Single-panel benchmark exclusion membership; historical CSV name retained |
| SupFig2a-f, SupFig3a-f | SupplementaryFigure2_3_umap.csv | Frozen UMAP coordinates, no refitting |
| Supplementary Table 2 | SupplementaryTable2.csv | All 60 representation/classifier combinations; performance heatmaps are no longer current SI figures |
| SupFig4a-d | SupplementaryFigure4a/b/cd.csv | Independent sampling-configuration experiments, not final-model replacements |
| SupFig5a-d | SupplementaryFigure5a/b/c/d.csv | Confusion matrix, threshold/FPR scan and human/mouse calls at 0.5 |
| SupFig6a-d | SupplementaryFigure6a/b/c/d.csv | EpiFactors, descriptive motif groups and pooled amino-acid composition |
| SupFig7a-c | SupplementaryFigure7_all_GO.csv | Selected 18 terms from all 3,082 tests; 88 significant |
| Fig6 evidence only | Figure6_query.csv, Figure6_metadata.json | First six records of the default unfiltered local database view, scores and snapshot provenance; zero displayed novel candidates |

The default registry generates 40 quantitative panels. Fig1c, Fig2a-b, Fig4 and Fig6
require separately supplied manual/experimental/browser assets.
The old Fig3e and domain-enrichment supplement are not current panels.

For optional diagnostic heatmaps of the complete performance table, use
`--figure SuppTable2_AUROC` (also F1, MCC, Precision or Recall). These outputs
are excluded from `--all` and are not numbered as manuscript SI figures.
Historical workbook tab names are deliberately retained; the explicit panel
map assigns them to the current displayed numbers rather than treating tab
names as current figure identities.

Figure 3c–d bin percentages are normalized separately within each evidence
group. The final 0.95–1.00 bin includes 1 and is not the >=0.98 candidate count.
PCHIP preserves the 20 frequency knots per group, uses 381 plotted coordinates
per group and runs only between the first/last bin centers (0.025–0.975), with
no extrapolation, threshold filtering, density estimation or statistical
smoothing. Original human positive/negative groups include both training and
held-out proteins. Frozen per-protein probabilities remain in `data/frozen_model/`.

For Fig1b, source sizes are GO 7,050, domain 1,785 and R-loopBase 1,181 unique
proteins (1,182 original nonblank rows include a duplicate). Union=7,674,
single-source=5,693 and three-source=361. Circle areas are not quantitative.

Motif grouping uses consensus composition in this order: any C;
otherwise (K+R)/length>=1/3; otherwise Other sequences. These groups must not
be relabeled as experimentally validated zinc-finger, DNA-binding or
RNA-binding functions. The full-length STREME analysis has a candidate/control
length-distribution limitation; see `data/frozen_analysis/streme401/README.md`.

## Export from the frozen source workbook

```bash
python scripts/export_source_data.py \
  --input /path/to/sourcedata_v3.xlsx \
  --output-dir outputs/new_panel_export \
  --figure-source-dir /path/to/figure_source/figure6
```

The explicit `configs/panel_map.json` fixes workbook SHA256, sheet/range/header,
output name, column schema and row count. It deliberately excludes the index
sheet and historical sheets. Separate blocks in Figure5c are exported as
separate data tables; the three-motif-group summary is not confused with the
13-row motif table. Figure6 evidence is optional at export time but required
for the complete current-profile validator.

Export is transactional after input validation, refuses conflicting files and
treats an exact repeated export as an identical no-op. A changed workbook is
not silently accepted: audit its changes and deliberately revise the panel map
before exporting a new profile. Do not bypass checksum protection by merely
renaming an edited workbook.

## Verification

```bash
python scripts/validate_release.py --root . \
  --source-workbook /path/to/sourcedata_v3.xlsx \
  --report outputs/current_validation.json
python -m unittest discover -s tests -v
```

Validation checks hashes, schemas, ranges, exact split/sampling records,
threshold counts, held-out denominators, group summaries, controls, motif
composition, GO correction family and database-query evidence.
This is not a claim that unbundled embeddings, experimental assays or all
upstream external annotations have been independently regenerated.

The archived `scripts/reproduce_figures_legacy.py` and
`docs/history/20260826/` describe the historical profile only.

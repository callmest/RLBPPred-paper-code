# Repository map

| Location | Current role |
|---|---|
| `configs/paper.yaml` | Strict401 scientific settings; no published-release claim |
| `configs/panel_map.json` | Current panel IDs, explicit workbook ranges and expected schemas |
| `data/panels/current/` | Default current frozen plot inputs and provenance manifest |
| `data/panels/*.csv`, top-level manifest | Historical panel snapshot; not the current default |
| `data/model_inputs/` | Input membership lists and current401 candidate IDs |
| `data/model_inputs/historical/` | Original342 candidate IDs, retained for provenance |
| `data/frozen_model/` | Split, sampling order, scores, metrics, candidates and model identity |
| `data/frozen_analysis/` | Available raw downstream evidence and provenance |
| `pipeline/modeling/strict_runtime.py` | Current validation, screening, checkpoint inference and explicitly gated training |
| `scripts/export_source_data.py` | Range-aware, single-workbook export to a new output directory |
| `scripts/reproduce_figures.py` | Current quantitative figure replay; six main/eight supplementary figure mapping |
| `scripts/reproduce_figures_legacy.py` | Historical plotting interface, not for the current manuscript |
| `scripts/validate_release.py` | Current semantic checks, hashes and consistency tests |
| `pipeline/database/` | Full-precision runtime exports and accession-matched strict scores |
| `pipeline/analysis/`, `pipeline/figures/` | Current helpers and explicitly historical original analyses |
| `pipeline/embeddings/` | Encoder preparation scripts requiring separately pinned external assets |
| `pipeline/ORIGIN_MANIFEST.tsv` | Provenance of original archived scripts, not hashes of every later revision |
| `docs/history/20260826/` | Previous documentation, preserved and non-authoritative for strict401 |
| `tests/` | Regression tests that do not fit models or run STREME |
| `outputs/` | Ignored generated output; never an authoritative input |

Current manuscript numbering is Figures1–6 and Supplementary Figures1–8.
See `reproduce_figures.md` for panel-level mapping. Workbook sheet names are
historical identifiers and are not assumed to equal current figure numbers.

Do not delete benchmark GO because of a similar historical filename: the old
`SuppFigure1b.csv` is GO, now Figure1d. The old `SuppFigure1c.csv` is historical
benchmark-domain enrichment. Domain-related exclusion remains part of the
current benchmark definition, regardless of whether domain enrichment is shown.

The database application remains a separate RLBPPredDB repository, deployed on
the user's local computer. The reproduction repository records its query data,
construction steps, schema, coverage and source hashes without claiming a public
hosted service or embedding large weights into Git.

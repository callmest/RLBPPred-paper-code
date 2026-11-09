# Data provenance and asset availability

This inventory distinguishes bundled evidence from external inputs. A local
working profile is not a published archive, and a SHA256 is not a download URL.
No release DOI, source accession or database version has been invented.

## Bundled current evidence

| Asset | Location | Provenance and limits |
|---|---|---|
| Current panel data | data/panels/current/ | Exact workbook blocks and SHA256 in manifest.json; full current mapping in configs/panel_map.json |
| Strict benchmark IDs/split | data/model_inputs/ and data/frozen_model/strict_input_manifest.csv | 376/9,109 labels; train 301/7,288; held out 75/1,821 |
| All 300 resampling rounds | data/frozen_model/sampling_manifest.json | Sampled negative IDs and shuffled training order |
| Frozen predictions | data/frozen_model/ | Human 16,969; mouse 17,002; held-out 1,896; complete scores with hashes |
| 401 candidate IDs/scores | data/frozen_model/candidates_401.csv | Scored untrained human pool, >=0.98; separate cross-machine scores retained |
| STREME output | data/frozen_analysis/streme401/ | Completed 401-vs-9,109 run, path-sanitized text/XML; original and distributed hashes recorded |
| GO test family | data/panels/current/SupplementaryFigure8_all_GO.csv | All 3,082 eligible tests, 88 significant; not only selected display terms |
| Database-query evidence | data/panels/current/Figure6_query.csv and Figure6_metadata.json | Six records, full scores, 14-field runtime coverage and source hashes |

The authoritative workbook is sourcedata_v3.xlsx, SHA256
`189e992a3b493176760f0273806adf0fcde9ff7d9defbfe0440ce00a67c20195`.
It is not copied into this repository by the repair. Current CSV exports are
read-only derivatives of its selected blocks. The index and old domain/figure
sheets are excluded. Supplementary tables and the Word file remain separate.

The published-study membership derivative has union 1,430 and study sizes
462/364/308/434/430 (Cristini/Wang/Mosler/Yan/Shinriki). These are cleaned mapped
membership counts, not substitutes for raw proteomics repository accessions.
The source-set Venn uses 7,050 GO-related, 1,785 domain-related and 1,181 unique
R-loopBase proteins. The original R-loopBase input list has 1,182 nonblank rows,
including one duplicate; this is not the row count of the exported binary
membership matrix.

## External assets still required for reconstruction

| Resource | Use | Availability/required record |
|---|---|---|
| Strict 300-member checkpoint | New inference without fitting | Approximately 906 MiB, state_dict_ensemble_v1; SHA256 below; not bundled or publicly archived here |
| SaProt mean embeddings | Strict training and new inference | 1,280 features; exact strict matrix hash in input_validation.json; large files not bundled |
| Pretrained ESM-C/PST/SaProt weights and tokenizer/Foldseek inputs | Upstream embedding extraction | Need exact provider revision, weight hashes, code commits, licences and environment locks |
| Five original proteomics studies | Original positive-label assembly | Need exact source tables/accessions, identifier mapping and redistribution terms; membership CSV alone is insufficient |
| UniProt/GO/InterPro/R-loopBase snapshots | Proteome universe, exclusions, annotations | Frozen derivative hashes do not establish upstream release/access dates; preserve the actual snapshot, query and mapping record |
| EpiFactors and four LLPS resources | Chromatin/phase-separation annotations | Current overlap results are bundled; original versioned input tables and mapping rules are needed to reconstruct them |
| MobiDB annotations | Disorder analysis | Current per-protein data and missing flags are bundled; exact upstream snapshot/method remains external |
| Full-length FASTA controls/candidates + MEME Suite | STREME reconstruction | Completed run used STREME 5.5.9; no rerun is required for output parsing/plotting |
| Frozen direct GO annotation rows and returned-ID list | Recompute enrichment | 401 foreground and 5,237 comparison proteins among 5,638 returned IDs; exact input snapshot is not bundled |
| Human/mouse 14-field annotation snapshot | Database-score join | Existing snapshot is required; updater changes scores only and does not fetch annotations |
| Experimental images, structures and visualization sessions | Fig4 | Separate originals and preprocessing records; cannot be regenerated from panel CSVs |

Checkpoint SHA256:
`72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41`.

The public strict-run provenance replaces only the original `source` and
`output` absolute paths with logical relative labels. All other JSON fields,
including the initial 0.85 screening audit setting, are unchanged. The manifest
records the original and distributed hashes and changed fields; original bytes
are retained outside Git in the private audit archive. The `manifest.json`
current screening policy is >=0.98; preserving historical provenance does not
authorize reusing its older candidate definition.

STREME used protein alphabet, widths 3-15, threshold 0.05, seed 0 and holdout
fraction 0.1. Of 13 reported motifs, 7 are significant at E<0.05 and 11 meet the
display cutoff E<=1.2. Use XML test_evalue and npassing, not total site count.
Composition labels are descriptive only. GO uses direct annotations (no
ancestor propagation), pool count >=5 and within-aspect BH correction.

## Database distinction

The database contains 37,077 annotation records: 20,075 human and 17,002 mouse.
There are 33,971 current strict scores; 3,106 human records lack scores.
This annotation universe is not identical to the 20,435-protein historical
exclusion universe. Do not treat these denominators as interchangeable.
Runtime JSON/CSV has 14 fields; the construction intermediate additionally
contains sequence. Only UI display rounds scores to six decimals.

## Before public archival

Supply stable download locations and licence/terms for the checkpoint,
embeddings and original analysis inputs; record actual source versions,
accessions, retrieval queries/dates and checksums. Confirm source redistribution
permissions. Create a tested immutable release and archival DOI only after
those assets and environments are ready. The local repair does not claim
those publication steps are complete.

Original code is MIT licensed; this does not relicense third-party datasets,
pretrained weights or experimental images. Retain provider attribution and
terms. Historical inventories are in docs/history/20260826/ and should not be
used as current model/figure specifications.

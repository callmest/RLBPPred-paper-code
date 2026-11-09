Repository rename notice (2026-11-09): this is archived documentation.
The former paper repository is now
[RLBPPred-paper-code](https://github.com/callmest/RLBPPred-paper-code).
The old `callmest/RLBPPred` address now belongs to the main prediction software;
use the paper-code entry point for historical paper links below. The archived
content is otherwise preserved.

# External data, models, and software

This document records the provenance information required to reproduce the
manuscript without implying that third-party resources were created or
redistributed by the RLBPPred authors. An exact release, accession, access date,
licence, or model revision is marked **TO BE CONFIRMED** when it is not stated
in the current manuscript or cannot be verified from the available files.

Before publication, mirror this inventory in a machine-readable manifest with
at least these fields:

```text
resource_name
resource_type
provider
release_or_version
accession_or_persistent_identifier
source_url
date_accessed
licence_or_terms
manuscript_use
local_derivative
sha256
citation
```

Do not replace a missing value with the current date or the newest database
release. The correct value is the snapshot actually used to generate the
reported results.

## Published R-loop proteomics inputs

The five human datasets used to assemble the positive labels are externally
published data. The counts below are the original study-level counts shown in
Table 1, not proof that the deposited local files correspond to a particular
repository accession.

| Dataset | Manuscript citation | Study metadata in Table 1 | Reported proteins | Fields still required |
|---|---|---|---:|---|
| Cristini et al. | Reference 17 | 2018; HeLa; RNA/DNA hybrid immunoprecipitation; MS/MS | 462 | Exact supplementary file or repository accession, download URL/date, licence/terms, local filename and checksum |
| Wang et al. | Reference 16 | 2018; B cell; dual RNA/DNA hybrid pull-down; LC-MS/MS | 364 | Exact supplementary file or repository accession, download URL/date, licence/terms, local filename and checksum |
| Mosler et al. | Reference 15 | 2021; HEK293T; R-loop proximity labelling; LC-MS/MS | 308 | Exact supplementary file or repository accession, download URL/date, licence/terms, local filename and checksum |
| Yan et al. | Reference 14 | 2022; HEK293; R-loop proximity labelling; LC-MS/MS | 434 | Exact supplementary file or repository accession, download URL/date, licence/terms, local filename and checksum |
| Shinriki et al. | Reference 18 | 2022; HEK293FT; RNase H1 mutant immunoprecipitation; MS | 430 | Exact supplementary file or repository accession, download URL/date, licence/terms, local filename and checksum |

The repository must preserve the original identifier column, the gene-symbol
normalization table, the UniProt mapping decision, study membership, and every
exclusion reason. A citation alone is not enough to reconstruct the input.

## Protein sequences, structures, ontologies, and annotations

| Resource | Manuscript use | Confirmed information | Fields still required |
|---|---|---|---|
| UniProtKB reviewed proteomes | Human/mouse sequences, reviewed status, identifiers, descriptions, functional annotations, and database records | Manuscript reference 40; human GO universe reported as n = 20,435; final RLBPPredDB contains 20,075 human and 17,002 mouse records | Exact UniProt release, proteome/taxonomy identifiers, query, download date, source filenames, licence/terms, checksums, and explanation of the 20,435 versus 20,075 human counts |
| AlphaFold Protein Structure Database | Human and mouse predicted structures; SaProt-650M-AF2 inputs; external links | Manuscript reference 27 | Exact database/model release, accession-to-file mapping, download method/date, handling of multiple files/fragments, terms, and checksums |
| RCSB Protein Data Bank | Experimental structures for APBS panels | PDB 3W5O for LIG4 and 4B3J for IGHMBP2 | Exact downloaded assembly/chain, file format, download date, checksum, and any preprocessing record |
| Gene Ontology Resource | Twenty-one-term exclusion panel and BP/MF/CC enrichment | Twenty-one terms were manually curated; 7,050 reviewed human proteins with at least one term were excluded | Ontology and annotation release dates, all 21 GO identifiers, evidence-code policy, propagation rules, query/download method, licence, source file and checksum |
| R-loopBase | Negative-set regulator exclusion | Manuscript reference 31 | Release/version, regulator table name, species filter, identifier mapping, download URL/date, licence/terms and checksum |
| InterPro | Domain-based exclusion and enrichment | Manuscript reference 33 | Release/version, member-database policy, query/download method/date, licence and checksum |
| Pfam | Domain enrichment | Used together with InterPro in candidate analysis | Release/version, mapping method, query/download date, licence and checksum |
| EpiFactors | Writer/reader/eraser/remodeler/other annotations | Version 2.1 is stated in the manuscript | Exact file/release date, download URL/date, identifier mapping, category-resolution rules, licence and checksum |
| PhaSepDB | LLPS overlap | Candidate overlap reported separately and in a non-redundant union | Release/version, download URL/date, species and identifier filters, licence and checksum |
| DrLLPS | LLPS overlap | Candidate overlap reported separately and in a non-redundant union | Release/version, download URL/date, species and identifier filters, licence and checksum |
| LLPSDB | LLPS overlap | Candidate overlap reported separately and in a non-redundant union | Release/version, download URL/date, species and identifier filters, licence and checksum |
| PhaSePro | LLPS overlap | Candidate overlap reported separately and in a non-redundant union | Release/version, download URL/date, species and identifier filters, licence and checksum |
| MobiDB-lite | Fraction of residues classified as disordered | Coverage reported for 342 candidates, 376 known RLBPs, and 9,080 non-RLBPs | Software/database version, execution mode or downloaded file, parameters, date, licence and checksum |
| Wu et al. mouse RLBP annotations | Mouse significant and potential positive groups | The manuscript attributes the compilation to Wu et al. | Exact citation/reference mapping, supplementary file/accession, group-definition rules, download date, licence and checksum |
| Kumar et al. data/features | Handcrafted-feature random-forest baseline | Manuscript reference 19; 30 descriptors; 354 positives and 16,575 negatives with complete features | Exact input file or reconstruction method for every descriptor, access route/date, feature units, missing-value policy, licence and checksum |

## Pretrained models and structure-processing tools

Model names alone are insufficient because hosted checkpoints and code can
change. Record the exact repository, tag/commit, checkpoint identifier, file
hash, and upstream licence before archival.

| Component | Manuscript role | Confirmed information | Fields still required |
|---|---|---|---|
| ESMC-300M | Sequence-only representation | 960-dimensional pooled representation | Provider repository/model identifier, immutable revision, weight checksum, tokenizer version and licence |
| ESMC-600M | Sequence-only representation | 1,152-dimensional pooled representation | Provider repository/model identifier, immutable revision, weight checksum, tokenizer version and licence |
| PST-T33 | Structure-integrated representation | Frozen representation in the six-model benchmark | Code commit, checkpoint name/hash, graph construction version/parameters and licence |
| PST-T33-SO | Structure-integrated representation | Frozen representation in the six-model benchmark | Code commit, checkpoint name/hash, graph construction version/parameters and licence |
| SaProt-650M-AF2 | Principal structure-aware representation | 1,280-dimensional; strongest ranking model when paired with MLP | Provider repository/model identifier, immutable revision, checkpoint hash, tokenizer version and licence |
| SaProt-650M-PDB | Structure-aware comparison representation | 1,280-dimensional | Provider repository/model identifier, immutable revision, checkpoint hash, definition/provenance of the PDB-derived structural input and licence |
| Foldseek | Generates 3Di structural tokens for SaProt | Used by SaProt tokenization | Exact version/build or commit, command and parameters, binary checksum and licence |
| MMseqs2 | Within- and between-class sequence redundancy filtering | 35% sequence identity and 90% coverage reported in Results | Exact version/build, complete command, coverage mode and remaining parameters, binary checksum and licence |
| STREME/MEME Suite | Full-length protein motif discovery | Protein alphabet; width 3–15; E < 0.05 | Exact MEME Suite version, complete command, background/control construction, random seed if applicable and licence |
| APBS | Electrostatic surface calculation | PDB 3W5O and 4B3J; displayed at −10 to +10 kT/e | APBS server/software version, date used, input preparation, protonation/charge/radius settings, grid parameters, dielectric/ionic conditions, output files and terms |
| ChimeraX | Visualization of APBS surfaces | Version 1.12 stated in the manuscript | Full command/session, orientation matrices, colour scale settings and licence/terms |

Python, R, PyTorch, scikit-learn, UMAP, XGBoost, LightGBM, plotting libraries,
CUDA and cuDNN versions should be pinned in the environment lock and recorded
in the archived release. The current manuscript does not provide enough detail
to reconstruct those versions.

## Data generated by this study

These outputs should be deposited separately from third-party raw resources and
described with their own licence and preferred citation.

| Generated asset | Minimum content | Recommended release location/status |
|---|---|---|
| Benchmark labels | UniProt accession, class, study recurrence, each exclusion flag, sequence/structure QC, redundancy cluster, final membership | Public data archive with DOI; panel-level derivative may also be in Git |
| Split and sampling manifests | Exact train/held-out IDs plus negative IDs for every ensemble round | Version-controlled if small; include checksums |
| Embeddings | Model/revision, protein ID, pooling method, array shape/dtype | External archive when too large for Git; do not apply MIT automatically |
| Model checkpoints | Main 300-member ensemble and any baseline checkpoints needed for inference | Model/data archive with hashes and explicit licence |
| Human and mouse scores | Continuous soft-voting probabilities, group membership, thresholded status, model release | Public archive and database downloads |
| Candidate table | Exactly 342 human candidates at threshold 0.85 with annotations | Supplementary Table 4 and machine-readable repository file |
| Figure panel data | One documented CSV/TSV per automatic panel | `data/panels/`, generated from the source-data workbooks |
| Source-data workbooks | Comprehensive manuscript source data and corrected Figure 1b source | Journal Source Data and/or public archive; record version/hash |
| Experimental source images | Uncropped LIG4 and IGHMBP2 input/IP images with ±RNase H, acquisition metadata, and crop/contrast log | Journal Source Data or public archive; preserve original bit depth |
| RLBPPredDB exports | Human CSV, mouse CSV, combined CSV, and an explicit data dictionary; current intermediate builder writes 15 fields including `sequence`, whereas the runtime builder retains 14 and omits `sequence` | Public archive plus site download page; reconcile downloadable and runtime schemas before submission |
| Supplementary tables | Tables 1–6 in machine-readable form | Supplementary Information and/or public archive |

## Licence and redistribution boundaries

The repository's MIT licence may cover original RLBPPred source code only. It
must not be presented as relicensing:

- published supplementary datasets;
- UniProt, AlphaFoldDB, PDB, GO, R-loopBase, InterPro/Pfam, LLPS, MobiDB, or
  EpiFactors content;
- third-party pretrained model weights or source code;
- APBS, ChimeraX, Foldseek, MMseqs2, MEME Suite, or other software;
- experimental data for which contributor or institutional permission has not
  been confirmed.

For each redistributed third-party file, verify that redistribution is allowed
and preserve its attribution/terms. Otherwise provide a download script or
instructions that retrieve the file from the original provider and verify its
checksum.

## Data Availability checklist

Before replacing the manuscript's current placeholder statement, confirm:

1. the DOI or persistent link for benchmark labels, source data, predictions,
   database exports, and experimental source images;
2. the GitHub release/tag, commit SHA, code licence, and archival DOI;
3. exact releases/accessions/access dates for every external resource above;
4. whether large embeddings and model checkpoints will be public and where;
5. whether any third-party terms prohibit redistribution;
6. that every figure/table points to a named panel-data file;
7. that the repository, archived record, Source Data workbook, and manuscript
   use the same final figure numbering and numerical values.

Until those fields are resolved, retain the **TO BE CONFIRMED** markers in this
document rather than publishing an unsupported accession or version.

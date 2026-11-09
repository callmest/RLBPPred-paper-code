# Reproducibility boundaries for the strict401 profile

The current default replays frozen evidence for manuscript v12. It is not a
claim that every original acquisition, wet-lab experiment, embedding extraction,
training run and downstream analysis has been repeated from public raw inputs.
Historical notes under `history/20260826/` describe the former profile, not
unresolved discrepancies in the current manuscript.

## Assets not bundled

The 300-member checkpoint, all embedding tensors, pretrained encoders, protein
structures and several third-party annotation snapshots are not included in Git.
The strict checkpoint's format and SHA-256 are specified in
`data/frozen_model/manifest.json`; no public download URL is invented. Complete
human/mouse and validation predictions are bundled, allowing result-level
verification without these large assets. A trusted checkpoint and matching
features are required to actually run inference. A separate archive, its license
and a stable download location still need to be arranged before claiming public
checkpoint availability.

Frozen identifiers do not freeze a future live UniProt query. External annotation
and sequence sources must have their actual snapshot/version and checksums; a
download of the newest release is not proof of reproduction. See `data_sources.md`.

## Evidence and statistical definitions

- Candidate discovery is restricted to 5,640 scored untrained human proteins.
  The 401 candidates do not represent all high-scoring database records.
- Empirical FPR is measured on 1,821 held-out negatives, not all 9,109 benchmark
  negatives. One passing negative at >=0.98 gives 0.0549%; the target is <=0.1%.
  This is not a confidence bound or a guaranteed deployment FPR. Mouse FPR has
  not been calibrated by this human experiment.
- Figure3c–d describe all scored members of each group, including training
  proteins where applicable; they are not independent held-out evaluations.
- Classification/confusion and group call rates use >=0.5, separately from
  candidate selection. Independent controls retain their original protocols.
- STREME reports 13 motifs; 7 satisfy E<0.05, while 11 meet E<=1.2 for display.
  Cys-containing, Lys/Arg-rich and Other are sequence-composition groups, not
  validated domains, binding functions or evidence of direct R-loop binding.
- Candidate GO uses the frozen, directly annotated untrained pool: 401 versus
  5,237 other annotated proteins, 3,082 eligible tests, aspect-wise BH correction.
  The 88 significant results and 18 displayed terms are distinct subsets.
- Disorder coverage is 401/376/9,059 valid records. Fifty of 9,109 negatives
  lack the frozen annotation; missing values are not zero disorder.

## Manual and external figure content

Figure1c and Figure2a–b are manual workflows. Figure4a,b,d,e are experimental
image/quantification panels; Figure4c,f are structural surface views. IGHMBP2
uses PDB4B3F. PDB3W5O represents part of LIG4, not its full-length structure.
Uncropped experimental originals, acquisition details, processing logs and
complete APBS/ChimeraX sessions are separate evidence requirements. Tabular
replay must not fabricate or relabel experimental images.

Figure6 is a browser screenshot from a six-record human database query, not six
new candidates. Current query metadata and full-precision scores are bundled;
the separate locally deployed browser app produces the screenshot.

Quantitative replay preserves data and definitions, but does not promise
byte-identical Illustrator typography, panel arrangement or manually placed
annotations. The final manuscript figures remain authoritative for composition.

## Training and upstream reconstruction

The strict split and 300-round sampled negative IDs and training order are now
frozen. The current loader matches `state_dict_ensemble_v1`; legacy pickle
entry points are explicitly historical. Hardware/library differences can still
affect new training. No retraining was performed for this packaging update, and
byte-identical checkpoints across devices are not promised.

The 60-combination table contains audited frozen results, but a completely
verified generator from every historical raw training log remains unavailable.
Likewise, the five original studies' complete raw-download-to-filtered-benchmark
pipeline and all external analysis inputs have not been independently replayed.
Those restrictions are retained rather than hidden by successful CSV plotting.

The repository validation checks current data hashes and scientific invariants;
it does not certify unbundled assets, experimental reproducibility or a remote
GitHub branch. Publication status and archival DOI remain separate tasks.

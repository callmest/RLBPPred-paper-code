# Strict-model database completion snapshot

`human_additional_predictions.csv` contains the 3,106 previously absent human
database scores, produced with the same fixed strict SaProt-650M-AF2 +
300-member MLP ensemble, without retraining. The distributed file is an exact
copy of that completed prediction output; `manifest.json` pins its hash/size.

These IDs are disjoint from `data/frozen_model/human_all_predictions.csv`.
Combining the two score maps gives 20,075 human database scores; the unchanged
17,002 mouse scores give 37,077 scored records. The added scores include 35
values >=0.98, making 1,344 human database records above that threshold across
all evidence groups. This does **not** extend the fixed 5,640-protein discovery
pool or redefine its 401 candidates. The per-row threshold flags are not
candidate-membership labels and are not a newly calibrated FPR guarantee.

No original frozen score, model weight, split, threshold or candidate list is
replaced by this snapshot. The manuscript's Figure 3 distributions continue to
use the original frozen 16,969 human and 17,002 mouse probabilities.

For a current complete database export, use the optional `--human-supplement`
argument of `pipeline/database/update_runtime_scores.py`; see
[database exports](../../docs/reproduce_model.md#database-exports). The merge
rejects any overlapping supplemental ID, even if its score matches, and checks
all IDs against the selected annotation snapshot. Without that option the
command still deliberately reconstructs the earlier incomplete frozen-score
coverage; it does not silently fetch or invent missing scores.

Figure 6 evidence labels such as `evidence_group=not_scored` refer to membership
in the original frozen analysis files. They do not mean the completed runtime
record currently has a missing `confidence` value.

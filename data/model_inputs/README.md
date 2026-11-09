# Released identifier sets

These files contain one reviewed UniProt accession per line and define the
principal protein groups used in the manuscript:

| File | Records | Role |
|---|---:|---|
| `recurrent_positive_ids.txt` | 407 | Proteins recurring in at least two of the five input studies before final quality and redundancy filters |
| `significant_positive_ids.txt` | 376 | Final high-confidence positive benchmark |
| `negative_ids.txt` | 9,109 | Final conservative negative benchmark |
| `potential_positive_ids.txt` | 1,879 | Single-study or otherwise potential-positive group |
| `untrained_proteome_ids.txt` | 5,649 | Reviewed human proteins reserved for candidate discovery |
| `high_confidence_candidate_ids.txt` | 401 | Candidates selected from the scored human untrained-proteome pool at score >= 0.98 |

The potential-positive and untrained-proteome files are complete input
collections, not their scored subsets. Frozen strict-model outputs cover
1,844/1,879 potential positives and 5,640/5,649 untrained proteins (35 and nine
inputs without scores). Do not silently replace input collections with
prediction coverage. All 376 positives and 9,109 negatives were scored.

The current frozen split and its exact ordering are in
`../frozen_model/strict_input_manifest.csv`; the 300 sampling/order records are
in `../frozen_model/sampling_manifest.json`. These ID files' input order must
not be used to regenerate the principal split. Use
`python pipeline/modeling/strict_runtime.py validate` from the repository root.
The scored candidate table is `../frozen_model/candidates_401.csv`; its
`probability` matches the frozen 5880 human prediction table, while
`probability_L40S` preserves the verified cross-machine result.

The previous 342-candidate list is preserved only for historical provenance in
`historical/high_confidence_candidate_ids_342_threshold085.txt`. It is not the
current discovery list and must not be mixed with the strict401 analysis.

The lists are data inputs, not software, and remain subject to the provenance
and attribution notes in [`../../docs/data_sources.md`](../../docs/data_sources.md).
Sequences can be retrieved from UniProtKB with, for example:

```bash
python scripts/download_uniprot_fasta.py \
  --ids data/model_inputs/significant_positive_ids.txt \
  --output data/external/significant_positive.fasta
```

The downloaded sequence snapshot can change if UniProt records are updated.
For archival reproduction, record the UniProt release, access date, and file
checksum alongside the downloaded FASTA.

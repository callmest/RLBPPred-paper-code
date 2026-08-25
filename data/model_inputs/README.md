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
| `high_confidence_candidate_ids.txt` | 342 | Candidates selected at the adopted 0.85 threshold |

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

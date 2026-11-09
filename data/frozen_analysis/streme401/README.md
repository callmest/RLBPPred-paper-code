# Frozen 401-candidate STREME output

This package replays an already completed STREME 5.5.9 analysis; it does not run
motif discovery or model training. The original host-specific run-directory
prefix was removed from the text and XML paths. All motif/statistical values,
probability matrices and scientific attributes remain unchanged.

The run used 401 candidate full-length sequences and 9,109 negative full-length
sequences. Command settings: `--protein --minw 3 --maxw 15 --thresh 0.05 --seed 0
--hofract 0.1`. Internal discovery/holdout counts were 361/40 candidates and
8,199/910 controls. Raw output contains 13 motifs; 7 have test E < 0.05. The
publication displays 11 motifs with E <= 1.2. The stopping parameter is not an
instruction to drop every nonsignificant reported motif from the raw output.

Read the XML `test_evalue` for reported E and `npassing` for the number of unique
candidate proteins, not `total_sites`. The strongest motif is IHTGEKPYKCEEC
(E = 2.2e-7, 55 proteins). Sequence-composition groups are assigned in order:
any C; otherwise (K + R)/consensus length >= 1/3; otherwise Other sequences.
Significant/displayed counts are 3/3, 4/6 and 0/2. These are descriptive groups,
not experimentally validated functional annotations. Consensus labels are
mnemonics for PWMs, not literal sequence-search patterns.

STREME warned of different candidate/control sequence lengths. These E-values
refer to the completed full-length, unmatched-control analysis and do not prove
length-independent or experimentally validated motif function.

Offline audit:

```bash
python -m pipeline.analysis.current_frozen_analysis \
  --streme-dir data/frozen_analysis/streme401 \
  --go-csv data/panels/current/SupplementaryFigure8_all_GO.csv
```

Original unmodified output SHA256:

- streme.txt: `e4efef0b1ce8adbb30fadfb1529596abd09c2db6aeadf5498a81375a232a2d3c`
- streme.xml: `7252bf16f4f06f21a608aca1764ac354f35e7a5372300528c31ef6a49037f0b4`

These checksums identify the original outputs, not the path-sanitized files in
this directory. `manifest.json` records both original and distributed hashes.
Full upstream reproduction additionally requires the frozen
FASTA inputs and STREME installation; this package alone supports output parsing
and figure replay, not a claim that all upstream inputs have been distributed.

# Panel-level source data

The CSV files in this directory are the canonical inputs for the fast figure
reproduction route. Their names follow the **final manuscript numbering**.
`manifest.json` records each file's source sheet, description, row count, and
SHA-256 checksum.

Two release corrections are deliberate:

1. `Figure1b.csv` is the cleaned five-study membership matrix used for the
   final UpSet panel: 1,430 unique identifiers; study sizes 462, 364, 308, 434,
   and 430; 407 proteins recurring in at least two studies; and four recurring
   in all five. It supersedes an earlier alias-unresolved 1,468-row sheet.
2. `Figure3a.csv` labels the AUROC 0.9925 curve as the **filtered benchmark**
   and the AUROC 0.9001 curve as the **unfiltered dataset**, matching the final
   PDF. Some historical folders used inverted labels.

The files reconstruct plotted values; they do not by themselves replace the
external raw datasets or the full model-training assets. See
[`../../docs/known_limitations.md`](../../docs/known_limitations.md).

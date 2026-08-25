# Third-party notices

The MIT license in [`LICENSE`](LICENSE) applies only to original code and
documentation authored for this repository. It does **not** relicense external
software, pretrained models, model checkpoints, databases, datasets, or other
third-party material. Those resources remain subject to their own licenses,
terms of use, citation requirements, and access conditions.

No third-party source tree, pretrained checkpoint, or raw third-party dataset
should be committed here unless its license explicitly permits redistribution
and the required notice is recorded below.

## Python dependencies

The Python packages named in `requirements-figures.txt` and
`environment-training.yml` are installed from their upstream distributions and
are not vendored by this repository. Each package is governed by its upstream
license. The environment files record compatible version ranges because the
historical execution environment was not preserved as a lock file.

## Models and specialist software

| Resource | Upstream source | Use in the workflow | Redistribution status |
|---|---|---|---|
| ESM-C / ESM software | [Biohub/esm](https://github.com/Biohub/esm) | Protein language-model embeddings | Software and model terms must be checked separately; no checkpoints are redistributed here. |
| ESM-2 / `fair-esm` | [facebookresearch/esm](https://github.com/facebookresearch/esm) | Sequence-model compatibility used by structural models | The upstream source repository is MIT licensed; model weights remain subject to their accompanying terms. |
| Protein Structure Transformer (PST) | [BorgwardtLab/PST](https://github.com/BorgwardtLab/PST) | Structure-aware embeddings | Upstream source is BSD-3-Clause licensed; no source snapshot or checkpoint is redistributed here. Record the exact commit and checkpoint provenance before release. |
| SaProt | [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | Structure-aware protein embeddings | Upstream source is MIT licensed; no source snapshot or checkpoint is redistributed here. Record the exact commit and checkpoint provenance before release. |
| Foldseek | [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) | Generation of 3Di structural tokens | Used as an external executable and not redistributed. Upstream source is GPL-3.0 licensed. |
| MEME Suite / STREME | [MEME Suite](https://meme-suite.org/) | Motif discovery | Used as an external executable and not redistributed. Users must follow the upstream license and citation instructions. |
| LLPhyScore | Original project/publication distribution | LLPS-related feature calculation | The historical local copy is intentionally excluded pending provenance and redistribution review. |
| catGRANULE 2.0 | Original project/publication distribution | Granule/phase-separation-related prediction | The historical local copy is intentionally excluded pending provenance and redistribution review. |

## Databases and datasets

The workflow may retrieve or consume records from UniProt, Gene Ontology,
InterPro, Pfam, MobiDB, AlphaFold DB, and the Protein Data Bank, together with
published R-loop-binding protein datasets. Repository users are responsible for
following the current terms, licenses, attribution requirements, and citation
guidance of each source. Derived tables should retain stable identifiers and
source citations so that provenance remains auditable.

## Archival-completeness checklist

The public repository discloses the following items that should be completed
before claiming a frozen, end-to-end archival reconstruction:

1. Record exact PST and SaProt source commits and the provenance/checksums of
   every checkpoint required for reproduction.
2. Confirm that no copied LLPhyScore or catGRANULE source is present unless its
   redistribution terms have been documented.
3. Add a row for every additional external executable, model, or dataset.
4. Keep large checkpoints and restricted data outside Git; provide accession
   numbers, DOI-backed archives, or documented download procedures instead.

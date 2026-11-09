# Panel data: current and historical profiles

Use **current/** for manuscript v12 and all default reproduction commands.

- `current/manifest.json` records exact source workbook ranges, schema, row
  counts and hashes. The explicit mapping is `../../configs/panel_map.json`.
- Current screening is >=0.98, empirical held-out FPR 1/1,821 (0.0549%) and
  401 candidates from 5,640 scored untrained human proteins.
- Current classification summaries use >=0.5, a separate decision threshold.
- Current plots cover six main/eight supplementary figure groups; manual or
  experimental panels are documented but not fabricated by the replay command.
- The source workbook index and historical domain-enrichment sheet are not
  current panel datasets.

The top-level CSVs and top-level manifest are **historical**. They have been
preserved, including the pre-existing corrected Figure2b metric, and are no
longer advertised as the canonical current figure inputs. Do not overwrite
them with a mixed export or combine them with current files. Historical
numbering (including Figure3e and SuppFigure1c) is not current numbering.

See [figure mapping and export instructions](../../docs/reproduce_figures.md)
and [remaining limitations](../../docs/known_limitations.md).

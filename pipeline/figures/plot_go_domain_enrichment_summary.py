#!/usr/bin/env python3
"""Plot GO/InterPro/Pfam enrichment summary bar chart (top 5 per source).

Reads go_domain_enrichment_gt_0.85_top_terms.csv and replaces InterPro/Pfam
accession IDs with human-readable names.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results" / "untrained_proteome_analysis"

# Human-readable names for InterPro accessions
INTERPRO_NAMES = {
    "IPR020479": "Homeodomain, metazoa",
    "IPR001827": "Homeobox protein, antennapedia type, conserved site",
    "IPR001356": "Homeodomain",
    "IPR009057": "Homeodomain-like superfamily",
    "IPR017995": "Homeobox protein, antennapedia type",
    "IPR017970": "Homeobox, conserved site",
    "IPR004161": "Translation elongation factor EFTu-like, domain 2",
    "IPR009000": "Translation protein, beta-barrel domain superfamily",
    "IPR014001": "Helicase superfamily 1/2, ATP-binding domain",
    "IPR001680": "WD40 repeat",
    "IPR006594": "LIS1 homology motif",
    "IPR018359": "Bromodomain, conserved site",
    "IPR000795": "Translational (tr)-type GTP-binding domain",
    "IPR007125": "Core Histone H2A/H2B/H3 domain",
    "IPR010926": "Class I myosin tail homology domain",
    "IPR036072": "Class I myosin, motor domain",
    "IPR031157": "Tr-type G domain, conserved site",
    "IPR038718": "SNF2-like, N-terminal domain superfamily",
    "IPR000330": "SNF2, N-terminal domain",
    "IPR049730": "SNF2/RAD5-like, C-terminal helicase domain",
}

# Human-readable names for Pfam accessions
PFAM_NAMES = {
    "PF00046": "Homeodomain",
    "PF03144": "Elongation factor Tu domain 2",
    "PF00009": "Elongation factor Tu GTP binding domain",
    "PF00125": "Core histone H2A/H2B/H3/H4 domain",
    "PF06017": "Unconventional myosin tail, actin- and lipid-binding",
    "PF00176": "SNF2-related domain",
    "PF08474": "Myelin transcription factor 1",
    "PF21322": "Lysine-specific demethylase 6/UTY, C-terminal helical domain",
    "PF21326": "KDM6, GATA-like",
    "PF00012": "Hsp70 protein",
    "PF00439": "Bromodomain",
    "PF00271": "Helicase conserved C-terminal domain",
    "PF16211": "C-terminus of histone H2A",
    "PF00004": "ATPase family associated with various cellular activities (AAA)",
    "PF01530": "Zinc finger, C2HC type",
    "PF00240": "Ubiquitin family",
    "PF17862": "AAA+ lid domain",
    "PF21227": "Myb-like DNA-binding domain",
    "PF04503": "Single-stranded DNA binding protein, SSDP",
    "PF26094": "Teashirt homolog 3-like, homeodomain",
}

# Colors matching figure_1d.svg
SOURCE_COLORS = {
    "GO:BP": "#66c2a5",
    "GO:MF": "#fc8d62",
    "GO:CC": "#8da0cb",
    "INTERPRO": "#e78ac3",
    "PFAM": "#a6d854",
}

TOP_N = 5

COMPACT_LABELS = {
    "Homeobox protein, antennapedia type, conserved site": "Antennapedia-type homeobox site",
    "Homeodomain-like superfamily": "Homeodomain-like superfamily",
    "Homeobox protein, antennapedia type": "Antennapedia-type homeobox",
    "histone H3K27me2/H3K27me3 demethylase activity": "H3K27me2/3 demethylase activity",
    "negative regulation of transcription by RNA polymerase II": "Negative regulation of Pol II transcription",
    "regulation of nucleotide-excision repair": "Regulation of nucleotide-excision repair",
    "Elongation factor Tu domain 2": "EF-Tu domain 2",
    "Core histone H2A/H2B/H3/H4 domain": "Core histone H2A/H2B/H3/H4",
    "Unconventional myosin tail, actin- and lipid-binding": "Unconventional myosin tail",
}


def resolve_name(source: str, term_id: str, term_name: str) -> str:
    if source == "INTERPRO":
        return INTERPRO_NAMES.get(term_id, term_name)
    if source == "PFAM":
        return PFAM_NAMES.get(term_id, term_name)
    # GO: strip leading "; " added by the analysis script
    return term_name.lstrip("; ")


def compact_name(source: str, term_id: str, term_name: str) -> str:
    name = resolve_name(source, term_id, term_name)
    return COMPACT_LABELS.get(name, name)


def load_top_terms(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    import math
    top_terms = load_top_terms(BASE / "go_domain_enrichment_gt_0.85_top_terms.csv")

    sources = ["GO:BP", "GO:MF", "GO:CC", "INTERPRO", "PFAM"]
    entries = []
    for src in sources:
        src_rows = [r for r in top_terms if r["source"] == src][:TOP_N]
        for r in src_rows:
            fdr = float(r["fdr_bh"])
            entries.append({
                "source": src,
                "term_id": r["term_id"],
                "label": compact_name(src, r["term_id"], r["term_name"]),
                "neg_log10_fdr": -math.log10(max(fdr, 1e-300)),
                "fdr_bh": fdr,
            })

    # Keep the strongest terms first from left to right in the landscape panel.
    entries.sort(key=lambda e: e["neg_log10_fdr"], reverse=True)

    n = len(entries)
    labels = [e["label"] for e in entries]
    values = [e["neg_log10_fdr"] for e in entries]
    colors = [SOURCE_COLORS[e["source"]] for e in entries]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(12.2, 4.25))

    x = np.arange(n)
    ax.bar(x, values, color=colors, edgecolor="#000000", linewidth=0.5, width=0.72)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=52, ha="right", rotation_mode="anchor", fontsize=6.8)
    ax.set_ylabel("Enrichment Score (-log\u2081\u2080 FDR)")
    ax.set_xlabel("")
    ax.set_title("Top Enriched GO/Domain Terms (Untrained Proteome, prob>0.85)")

    ax.grid(False)
    ax.set_axisbelow(True)

    # Spines: only left and bottom, width 0.8
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Legend matching figure_1d right-side category legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=SOURCE_COLORS[s], edgecolor="#000000",
                            linewidth=0.5, label=s) for s in sources]
    ax.legend(handles=legend_handles, title="Category", loc="upper right",
              frameon=True, framealpha=1.0, edgecolor="#cccccc",
              fontsize=8, title_fontsize=8, ncol=5,
              bbox_to_anchor=(1.0, 1.0), borderpad=0.4,
              handlelength=1.3, columnspacing=1.0)

    ax.margins(x=0.008)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.36)

    out_prefix = BASE / "go_domain_enrichment_gt_0.85_summary"
    for ext in ("svg", "png", "pdf"):
        fig.savefig(str(out_prefix) + f".{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_prefix}.svg/.png/.pdf")


if __name__ == "__main__":
    main()

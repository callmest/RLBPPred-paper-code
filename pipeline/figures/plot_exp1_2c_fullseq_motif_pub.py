#!/usr/bin/env python3
"""Experiment 1.2c: Publication figures for full-length protein STREME motif discovery."""


# Historical 342-candidate analysis, retained for provenance only.
# Current frozen publication replay: python scripts/reproduce_figures.py --all
import sys as _historical_sys
if __name__ == "__main__":
    if "--historical-342" not in _historical_sys.argv:
        raise SystemExit("Historical 342-candidate workflow. Use the current frozen replay, or explicitly add --historical-342.")
    _historical_sys.argv.remove("--historical-342")
import re
from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
STREME_DIR = BASE / "results" / "idr_analysis" / "motif_discovery_fullseq" / "novel_vs_negative"
OUT_DIR = BASE / "results" / "idr_analysis"

FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CAT_COLORS = {
    "Homeodomain (HTH)":          "#D62728",
    "Gln/Pro-rich IDR":           "#B8860B",
    "Basic / nucleic acid binding":"#4E79A7",
    "Walker A / P-loop":          "#5B8C85",
    "Ser/Glu repeat":             "#8DA0CB",
    "Acidic stretch":             "#B07AA1",
    "Charge-rich":                "#4E79A7",
    "Other":                      "#9AA5B1",
}

MOTIF_ANNOTATIONS = {
    "QVKIWFQNRRMKW":  {"category": "Homeodomain (HTH)", "short": "Homeodomain recognition helix (helix III)"},
    "TLELEKEFHFNRYL": {"category": "Homeodomain (HTH)", "short": "HTH DNA-binding domain"},
    "QQPPQPQTQ":      {"category": "Gln/Pro-rich IDR",  "short": "Q/P-rich transactivation domain"},
    "KRKR":           {"category": "Basic / nucleic acid binding", "short": "Polybasic NLS-like motif"},
    "AKEK":           {"category": "Basic / nucleic acid binding", "short": "Lysine-spaced charge motif"},
    "GTGKTL":         {"category": "Walker A / P-loop",  "short": "Walker A box (GxxGKT); NTPase"},
    "SESE":           {"category": "Ser/Glu repeat",     "short": "Ser-Glu alternating charge"},
    "KKEL":           {"category": "Basic / nucleic acid binding", "short": "Lysine-rich coiled-coil"},
    "DEEEEEEEEEDDED": {"category": "Acidic stretch",     "short": "Poly-Glu/Asp acidic stretch"},
    "KKKAKM":         {"category": "Basic / nucleic acid binding", "short": "Poly-Lys histone-tail like"},
}

E_VALUE_THRESHOLD = 0.05
DISPLAY_E_CUTOFF = 1.2


def parse_streme(streme_txt):
    text = streme_txt.read_text()
    motifs = []
    for match in re.finditer(
        r"MOTIF (\S+) (\S+)\nletter-probability matrix: alength= 20 w= (\d+) nsites= (\d+) E= ([\d.e+-]+)",
        text,
    ):
        raw_id = match.group(1)
        consensus = raw_id.split("-", 1)[1] if "-" in raw_id else raw_id
        motifs.append({
            "id": match.group(2),
            "consensus": consensus,
            "width": int(match.group(3)),
            "sites": int(match.group(4)),
            "e_value": float(match.group(5)),
        })
    return motifs


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, prefix):
    for ext in ["svg", "pdf", "png"]:
        out = OUT_DIR / f"{prefix}.{ext}"
        fig.savefig(str(out), dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    streme_txt = STREME_DIR / "streme.txt"
    all_motifs = parse_streme(streme_txt)
    print(f"Parsed {len(all_motifs)} motifs from full-length STREME")

    display_motifs = [m for m in all_motifs if m["e_value"] <= DISPLAY_E_CUTOFF]
    n_sig = sum(1 for m in display_motifs if m["e_value"] < E_VALUE_THRESHOLD)
    n_total = len(all_motifs)
    print(f"  Significant (E<0.05): {n_sig}")
    print(f"  Displayed (E<={DISPLAY_E_CUTOFF}): {len(display_motifs)}")

    n_motifs = len(display_motifs)

    # --- Left panel: motif E-value bar chart with annotations ---
    e_vals = [-np.log10(max(m["e_value"], 1e-10)) for m in display_motifs]
    colors = []
    for m in display_motifs:
        ann = MOTIF_ANNOTATIONS.get(m["consensus"], {})
        cat = ann.get("category", "Other")
        colors.append(CAT_COLORS.get(cat, "#BBBBBB"))

    fig_bar, ax_bar = plt.subplots(figsize=(8.2, 4.2))
    fig_bar.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.14)

    y_pos = np.arange(n_motifs)
    ax_bar.barh(y_pos, e_vals, color=colors, edgecolor="black",
                linewidth=0.45, alpha=0.82, height=0.65)

    for i, m in enumerate(display_motifs):
        if e_vals[i] > 1.0:
            ax_bar.text(0.12, i, f"n={m['sites']}", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
        else:
            ax_bar.text(e_vals[i] + 0.06, i, f"n={m['sites']}", va="center",
                        fontsize=6.5, color="#333333")

        ann = MOTIF_ANNOTATIONS.get(m["consensus"], {})
        short_desc = ann.get("short", "")
        if short_desc:
            ax_bar.text(max(e_vals) + 0.25, i, short_desc,
                        va="center", fontsize=6.5, color="#444444", style="italic")

    ylabels = [m["consensus"] for m in display_motifs]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ylabels, fontsize=7.5, fontfamily="monospace")
    ax_bar.set_xlabel(r"$-\log_{10}$(E-value)")
    ax_bar.invert_yaxis()
    style_axis(ax_bar)

    sig_line = -np.log10(E_VALUE_THRESHOLD)
    ax_bar.axvline(x=sig_line, color="#A31E24", linestyle="--", linewidth=0.8, alpha=0.75)
    ax_bar.text(sig_line + 0.05, n_motifs - 0.15, "E = 0.05",
                fontsize=7, color="#A31E24", alpha=0.9)

    x_max_for_annot = max(e_vals) + 0.25 + max(
        len(MOTIF_ANNOTATIONS.get(m["consensus"], {}).get("short", ""))
        for m in display_motifs
    ) * 0.055
    ax_bar.set_xlim(-0.1, x_max_for_annot)
    save_figure(fig_bar, "exp1_2c_fullseq_motif_significance_pub")

    # --- Right panel: functional category summary (all displayed motifs) ---
    categories = {}
    for m in display_motifs:
        ann = MOTIF_ANNOTATIONS.get(m["consensus"], {})
        cat = ann.get("category", "Other")
        if cat not in categories:
            categories[cat] = {"total": 0, "sig": 0}
        categories[cat]["total"] += 1
        if m["e_value"] < E_VALUE_THRESHOLD:
            categories[cat]["sig"] += 1

    sorted_cats = sorted(categories.items(), key=lambda x: -x[1]["total"])
    cat_names = [c[0] for c in sorted_cats]
    cat_totals = [c[1]["total"] for c in sorted_cats]
    cat_sigs = [c[1]["sig"] for c in sorted_cats]
    cat_cols = [CAT_COLORS.get(c, "#BBBBBB") for c in cat_names]

    fig_cat, ax_cat = plt.subplots(figsize=(3.8, 3.8))
    fig_cat.subplots_adjust(left=0.42, right=0.96, top=0.97, bottom=0.16)

    y_cat = np.arange(len(cat_names))
    ax_cat.barh(y_cat, cat_totals, color=cat_cols,
                edgecolor="black", linewidth=0.45, alpha=0.28, height=0.55, label="Total motifs")
    ax_cat.barh(y_cat, cat_sigs, color=cat_cols,
                edgecolor="black", linewidth=0.45, alpha=0.86, height=0.55, label="Significant motifs")

    for i in range(len(cat_names)):
        if cat_sigs[i] > 0:
            ax_cat.text(cat_totals[i] + 0.15, i, f"{cat_sigs[i]}/{cat_totals[i]}",
                        va="center", fontsize=7, color="#333333")
        else:
            ax_cat.text(cat_totals[i] + 0.15, i, f"0/{cat_totals[i]}",
                        va="center", fontsize=7, color="#999999")

    ax_cat.set_yticks(y_cat)
    ax_cat.set_yticklabels(cat_names, fontsize=7.5)
    ax_cat.set_xlabel("Number of motifs")
    ax_cat.invert_yaxis()
    style_axis(ax_cat)
    ax_cat.set_xlim(0, max(cat_totals) + 1.5)
    ax_cat.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_cat.legend(fontsize=6.5, loc="lower right", frameon=False)
    save_figure(fig_cat, "exp1_2c_fullseq_motif_category_pub")

    # --- Write figure legend ---
    legend_md = f"""# Figure: Full-Length Protein De Novo Motif Discovery (Experiment 1.2c)

## Method

De novo motif discovery was performed using STREME v5.5.9 (Bailey, Bioinformatics 2021;
DOI: 10.1093/bioinformatics/btab203) on **full-length protein sequences** of Novel
predicted RLBPs (342 proteins) versus Non-RLBP control proteins (9,109 proteins).
Parameters: protein alphabet, motif width 3-15 residues, E-value threshold 0.05.
Total runtime: {10377.93/3600:.1f} hours on a single CPU.

## Panel Description

**Motif significance figure**: Horizontal bar chart showing discovered motifs (E <= {DISPLAY_E_CUTOFF})
ranked by statistical significance (-log10 E-value). Bar color indicates functional
category assignment. The red dashed line marks E = 0.05 significance threshold.
Numbers inside bars indicate the count of proteins containing each motif (n = sites).
Italic annotations describe functional interpretation of each motif.

**Functional category figure**: Summary of motif functional categories. Dark bars show significant
motifs (E < 0.05); light bars show total motifs in each category. Fractions indicate
significant/total counts.

## Results

{n_total} total motifs discovered; {n_sig} significant at E < 0.05:

| Rank | Consensus | Width | Sites | E-value | Category | Annotation |
|------|-----------|-------|-------|---------|----------|------------|
"""
    for i, m in enumerate(display_motifs, 1):
        ann = MOTIF_ANNOTATIONS.get(m["consensus"], {})
        cat = ann.get("category", "Other")
        short = ann.get("short", "")
        sig = " *" if m["e_value"] < E_VALUE_THRESHOLD else ""
        legend_md += f"| {i} | {m['consensus']} | {m['width']} | {m['sites']} | {m['e_value']:.2e}{sig} | {cat} | {short} |\n"

    legend_md += """
\\* Significant at E < 0.05.

## Key Findings

1. **Homeodomain recognition helix** (QVKIWFQNRRMKW, E=6.6e-6): The most significant
   motif corresponds to the highly conserved homeodomain recognition helix (helix III of the
   homeodomain HTH fold). This motif cross-validates with Embedding Cluster C11 (93.9% RLBP,
   enriched in HOX/homeobox transcription factors), providing independent confirmation from
   two orthogonal methods.

2. **HTH DNA-binding motif** (TLELEKEFHFNRYL, E=2.9e-3): Second motif also maps to HTH-fold
   DNA-binding domain, reinforcing the homeodomain TF enrichment signal.

3. **Gln/Pro-rich IDR motif** (QQPPQPQTQ, E=1.4e-2): Glutamine/proline-rich repeat found in
   transcription factor activation domains, consistent with the transactivation function of
   homeobox proteins.

4. **Basic nucleic acid binding** (KRKR, E=4.6e-2): Short polybasic motif characteristic of
   general nucleic acid binding through electrostatic backbone contacts.

## Biological Interpretation

The dominance of homeodomain-related motifs in full-length sequence analysis (vs. IDR-focused
motifs in Exp 1.2a/1.2b) reflects that the structured DNA-binding domain is the primary
distinguishing feature of the novel predicted RLBP subgroup. This is consistent with the
hypothesis that homeobox transcription factors represent a novel class of R-loop binding proteins
that may interact with R-loops through their DNA-binding homeodomain, potentially at gene
regulatory elements.
"""

    legend_path = OUT_DIR / "exp1_2c_fullseq_motif_pub_legend.md"
    legend_path.write_text(legend_md)
    print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()

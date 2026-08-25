#!/usr/bin/env python3
"""Replot Figure 1b (UpSet plot of human negative sample construction)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import from_contents, plot

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_XLSX = REPOSITORY_ROOT / "data" / "figure1b_human_negative_samples_construction.xlsx"
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "negative_selection"
SHEET_NAME = "all catagories"
BASE_NAME = "figure_1b"


def plot_figure_1b(data: pd.DataFrame, save_dir: Path) -> None:
    human_proteome = list(set(list(data["Human Proteome"].dropna())))
    positive = list(set(list(data["Postive Genes"].dropna())))
    go = list(set(list(data["GO-related Genes"].dropna())))
    domain = list(set(list(data["Domain-Related"].dropna())))
    rloopbase = list(set(list(data["RloopBase Regulators"].dropna())))
    dict_content = {
        "Human Proteome": human_proteome,
        "All Positive": positive,
        "GO-related": go,
        "Domain-related": domain,
        "RloopBase": rloopbase,
    }
    upset_content = from_contents(dict_content)
    fig = plt.figure(figsize=(4, 3.5), dpi=600)
    plot(upset_content, fig=fig, show_counts="%d")
    for fmt in ("pdf", "png", "svg"):
        fig.savefig(save_dir / f"{BASE_NAME}.{fmt}", format=fmt, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_1b_data = pd.read_excel(SOURCE_XLSX, sheet_name=SHEET_NAME)
    plot_figure_1b(figure_1b_data, OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR / BASE_NAME}.pdf/.png/.svg")


if __name__ == "__main__":
    main()

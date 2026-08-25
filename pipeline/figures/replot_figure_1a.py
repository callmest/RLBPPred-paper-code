#!/usr/bin/env python3
"""Replot Figure 1a (UpSet plot of significant positive RLBPs across five studies)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import from_contents, plot

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_XLSX = REPOSITORY_ROOT / "data" / "figure1a_human_significant_positive_rlbps.xlsx"
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "positive_selection"
SHEET_NAME = "process"
BASE_NAME = "figure_1a"


def plot_figure_1a(data: pd.DataFrame, save_dir: Path) -> None:
    c_data = list(set(list(data["Cristini"].dropna())))
    m_data = list(set(list(data["Mosler"].dropna())))
    s_data = list(set(list(data["Shinriki"].dropna())))
    y_data = list(set(list(data["Yan"].dropna())))
    w_data = list(set(list(data["Wang"].dropna())))
    dict_content = {
        "Cristini": c_data,
        "Mosler": m_data,
        "Shinriki": s_data,
        "Yan": y_data,
        "Wang": w_data,
    }
    upset_content = from_contents(dict_content)
    fig = plt.figure(figsize=(4, 3.5), dpi=600)
    plot(upset_content, fig=fig, show_counts="%d")
    for fmt in ("pdf", "png", "svg"):
        fig.savefig(save_dir / f"{BASE_NAME}.{fmt}", format=fmt, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_1a_data = pd.read_excel(SOURCE_XLSX, sheet_name=SHEET_NAME)
    plot_figure_1a(figure_1a_data, OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR / BASE_NAME}.pdf/.png/.svg")


if __name__ == "__main__":
    main()

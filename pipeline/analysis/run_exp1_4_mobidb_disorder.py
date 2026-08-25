#!/usr/bin/env python3
"""Compare MobiDB-lite disorder fractions across three RLBP groups."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats


API_URL = "https://mobidb.org/api/download_page"
GROUPS = ["Novel_RLBP", "Known_RLBP", "Non_RLBP"]
GROUP_LABELS = {
    "Novel_RLBP": "Novel predicted RLBPs",
    "Known_RLBP": "Known RLBPs",
    "Non_RLBP": "Non-RLBPs",
}
GROUP_COLORS = {
    "Novel_RLBP": "#B84A3A",
    "Known_RLBP": "#4F6F9F",
    "Non_RLBP": "#9AA5B1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def load_fasta_ids(path: Path) -> list[str]:
    ids = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                acc = line[1:].strip().split("|")[0].split("-")[0].strip()
                if acc:
                    ids.append(acc)
    return sorted(set(ids))


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_batch(session: requests.Session, accessions: list[str], retries: int = 5) -> list[dict]:
    params = {
        "acc": ",".join(accessions),
        "format": "json",
        "projection": (
            "acc,length,prediction-disorder-mobidb_lite,"
            "prediction-disorder-th_50"
        ),
    }
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, timeout=120)
            response.raise_for_status()
            return response.json().get("data", [])
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_or_load(accessions: list[str], cache_path: Path, batch_size: int, pause: float,
                  refresh: bool) -> dict[str, dict]:
    if cache_path.exists() and not refresh:
        with gzip.open(cache_path, "rt") as handle:
            return {entry["acc"]: entry for entry in (json.loads(line) for line in handle)}

    records: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "RLBP-disorder-analysis/1.0"})
    all_batches = list(chunks(accessions, batch_size))
    for index, batch in enumerate(all_batches, start=1):
        for entry in fetch_batch(session, batch):
            records[entry["acc"]] = entry
        print(f"MobiDB batch {index}/{len(all_batches)}: {len(records)}/{len(accessions)} entries")
        time.sleep(pause)

    with gzip.open(cache_path, "wt") as handle:
        for acc in sorted(records):
            handle.write(json.dumps(records[acc], separators=(",", ":")) + "\n")
    return records


def build_table(group_ids: dict[str, list[str]], records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for acc in group_ids[group]:
            entry = records.get(acc)
            if entry is None:
                rows.append({
                    "uniprot_id": acc,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "length": np.nan,
                    "disordered_residues": np.nan,
                    "disorder_fraction": np.nan,
                    "disorder_percent": np.nan,
                    "mobidb_lite_regions": "",
                    "status": "missing_mobidb_entry",
                })
                continue

            length = entry.get("length")
            lite = entry.get("prediction-disorder-mobidb_lite")
            majority = entry.get("prediction-disorder-th_50")
            if lite is not None:
                count = int(lite.get("content_count", 0))
                regions = lite.get("regions", [])
                status = "mobidb_lite_regions"
            elif majority is not None:
                count = 0
                regions = []
                status = "no_mobidb_lite_region"
            else:
                count = np.nan
                regions = []
                status = "missing_disorder_prediction"

            fraction = count / length if length and not pd.isna(count) else np.nan
            rows.append({
                "uniprot_id": acc,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "length": length,
                "disordered_residues": count,
                "disorder_fraction": fraction,
                "disorder_percent": fraction * 100 if not pd.isna(fraction) else np.nan,
                "mobidb_lite_regions": json.dumps(regions, separators=(",", ":")),
                "status": status,
            })
    return pd.DataFrame(rows)


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def compute_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    valid = df.dropna(subset=["disorder_percent"])
    summary_rows = []
    arrays = {}
    for group in GROUPS:
        values = valid.loc[valid["group"] == group, "disorder_percent"].to_numpy()
        arrays[group] = values
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summary_rows.append({
            "group": group,
            "group_label": GROUP_LABELS[group],
            "n_total": int((df["group"] == group).sum()),
            "n_valid": len(values),
            "n_missing": int((df["group"] == group).sum() - len(values)),
            "median_percent": median,
            "q1_percent": q1,
            "q3_percent": q3,
            "mean_percent": values.mean(),
            "sd_percent": values.std(ddof=1),
        })

    kw = stats.kruskal(*(arrays[group] for group in GROUPS))
    pairs = [("Novel_RLBP", "Known_RLBP"),
             ("Novel_RLBP", "Non_RLBP"),
             ("Known_RLBP", "Non_RLBP")]
    pair_rows = []
    raw_p = []
    for first, second in pairs:
        test = stats.mannwhitneyu(arrays[first], arrays[second], alternative="two-sided")
        rank_biserial = 2 * test.statistic / (len(arrays[first]) * len(arrays[second])) - 1
        raw_p.append(test.pvalue)
        pair_rows.append({
            "group_1": first,
            "group_2": second,
            "n_1": len(arrays[first]),
            "n_2": len(arrays[second]),
            "u_statistic": test.statistic,
            "p_value": test.pvalue,
            "rank_biserial_correlation": rank_biserial,
        })
    adjusted = bh_adjust(raw_p)
    for row, p_adj in zip(pair_rows, adjusted):
        row["p_adjusted_bh"] = p_adj
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows), kw.pvalue


def p_label(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_figure(df: pd.DataFrame, pairwise: pd.DataFrame, out_prefix: Path) -> None:
    valid = df.dropna(subset=["disorder_percent"])
    values = [valid.loc[valid["group"] == group, "disorder_percent"].to_numpy()
              for group in GROUPS]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(4.65, 3.75))
    positions = np.arange(1, 4)

    violins = ax.violinplot(values, positions=positions, widths=0.78,
                            showmeans=False, showmedians=False, showextrema=False,
                            bw_method=0.22)
    for body, group in zip(violins["bodies"], GROUPS):
        body.set_facecolor(GROUP_COLORS[group])
        body.set_edgecolor("#222222")
        body.set_linewidth(0.6)
        body.set_alpha(0.72)

    box = ax.boxplot(values, positions=positions, widths=0.22, patch_artist=True,
                     showfliers=False,
                     medianprops={"color": "#FFFFFF", "linewidth": 1.4},
                     whiskerprops={"color": "#222222", "linewidth": 0.8},
                     capprops={"color": "#222222", "linewidth": 0.8})
    for patch, group in zip(box["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(20260714)
    for position, group, group_values in zip(positions, GROUPS, values):
        display = group_values if len(group_values) <= 350 else rng.choice(group_values, 350, replace=False)
        jitter = rng.normal(0, 0.07, len(display))
        ax.scatter(position + jitter, display, s=5, color="#222222", alpha=0.18,
                   linewidths=0, rasterized=True, zorder=2)

    pair_positions = {("Novel_RLBP", "Known_RLBP"): (1, 2),
                      ("Novel_RLBP", "Non_RLBP"): (1, 3),
                      ("Known_RLBP", "Non_RLBP"): (2, 3)}
    bracket_levels = [104, 116, 110]
    for (_, row), y in zip(pairwise.iterrows(), bracket_levels):
        x1, x2 = pair_positions[(row["group_1"], row["group_2"])]
        ax.plot([x1, x1, x2, x2], [y - 2, y, y, y - 2], color="#222222", linewidth=0.7)
        ax.text((x1 + x2) / 2, y + 0.8, p_label(row["p_adjusted_bh"]),
                ha="center", va="bottom", fontsize=8)

    labels = []
    for group, vals in zip(GROUPS, values):
        labels.append(f"{GROUP_LABELS[group]}\n(n={len(vals):,})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MobiDB-lite disordered residues (%)")
    ax.set_ylim(-3, 123)
    ax.set_xlim(0.45, 3.55)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.98)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)


def write_summary(path: Path, summary: pd.DataFrame, pairwise: pd.DataFrame,
                  kw_p: float, status_counts: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=["float"]).columns:
            display[column] = display[column].map(lambda value: f"{value:.4g}")
        headers = [str(column) for column in display.columns]
        rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
        widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
        render = lambda row: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        return "\n".join([render(headers), separator, *(render(row) for row in rows)])

    lines = [
        "# MobiDB-lite disorder fraction analysis",
        "",
        "MobiDB-lite predicted IDR coverage was calculated as the number of residues "
        "within MobiDB-lite disorder regions divided by the full protein length. "
        "Proteins returned by MobiDB with a majority-consensus prediction but no "
        "MobiDB-lite region were assigned 0%; proteins without disorder prediction "
        "coverage were treated as missing.",
        "",
        f"Kruskal-Wallis test: p={kw_p:.4e}.",
        "",
        "## Group summaries",
        "",
        markdown_table(summary),
        "",
        "## Pairwise Mann-Whitney U tests",
        "",
        markdown_table(pairwise),
        "",
        "## Annotation status",
        "",
        markdown_table(status_counts),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    data_dir = base / "data" / "idr_analysis"
    results = base / "results" / "idr_analysis"
    results.mkdir(parents=True, exist_ok=True)

    group_ids = {
        "Novel_RLBP": load_fasta_ids(data_dir / "candidates_342.fasta"),
        "Known_RLBP": load_fasta_ids(data_dir / "known_376.fasta"),
        "Non_RLBP": load_fasta_ids(data_dir / "negative_9109.fasta"),
    }
    all_ids = sorted(set().union(*(set(ids) for ids in group_ids.values())))
    print("Groups:", {group: len(ids) for group, ids in group_ids.items()})
    print("Unique accessions:", len(all_ids))

    cache = results / "exp1_4_mobidb_raw.jsonl.gz"
    records = fetch_or_load(all_ids, cache, args.batch_size, args.pause, args.refresh)
    table = build_table(group_ids, records)
    table_path = results / "exp1_4_mobidb_disorder_fraction.csv"
    table.to_csv(table_path, index=False, float_format="%.8f")

    summary, pairwise, kw_p = compute_statistics(table)
    summary.to_csv(results / "exp1_4_mobidb_group_summary.csv", index=False, float_format="%.8g")
    pairwise.insert(0, "kruskal_wallis_p", kw_p)
    pairwise.to_csv(results / "exp1_4_mobidb_disorder_statistics.csv", index=False,
                    float_format="%.8g")

    status_counts = (table.groupby(["group", "status"], observed=True).size()
                     .rename("n").reset_index())
    status_counts.to_csv(results / "exp1_4_mobidb_annotation_status.csv", index=False)

    out_prefix = results / "exp1_4_mobidb_disorder_fraction_pub"
    plot_figure(table, pairwise, out_prefix)
    write_summary(results / "exp1_4_mobidb_disorder_summary.md",
                  summary, pairwise, kw_p, status_counts)
    print("Saved analysis outputs under", results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare amino-acid composition among three RLBP-related protein groups."""

from collections import Counter
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "aa_composition"
OUT.mkdir(parents=True, exist_ok=True)
DATA = PROJECT / "data" / "idr_analysis"

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
GROUP_MARKERS = {"Novel_RLBP": "o", "Known_RLBP": "s", "Non_RLBP": "^"}
FASTA = {
    "Novel_RLBP": DATA / "candidates_342.fasta",
    "Known_RLBP": DATA / "known_376.fasta",
    "Non_RLBP": DATA / "negative_9109.fasta",
}

# Ordering and outer classes follow the supplied reference radar.
CLASSES = [
    ("Acidic", ["E", "D"]),
    ("Basic", ["H", "K", "R"]),
    ("Hydrophobic", ["W", "F", "M", "L", "I", "V", "A", "P"]),
    ("Hydrophilic", ["G", "C", "Y", "T", "S", "Q", "N"]),
]
AA_ORDER = ["N", "E", "D", "H", "K", "R", "W", "F", "M", "L",
            "I", "V", "A", "P", "G", "C", "Y", "T", "S", "Q"]
AA_SET = set(AA_ORDER)


def read_fasta(path: Path):
    identifier = None
    chunks = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks).upper()
                identifier = line[1:].split()[0].split("|")[0]
                chunks = []
            else:
                chunks.append(line)
    if identifier is not None:
        yield identifier, "".join(chunks).upper()


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def calculate_composition():
    rows = []
    pooled_rows = []

    for group in GROUPS:
        pooled = Counter()
        protein_count = 0
        excluded_residues = 0
        for protein_id, sequence in read_fasta(FASTA[group]):
            counts = Counter(sequence)
            standard_length = sum(counts[aa] for aa in AA_ORDER)
            if standard_length == 0:
                continue
            excluded_residues += len(sequence) - standard_length
            protein_count += 1
            pooled.update({aa: counts[aa] for aa in AA_ORDER})
            row = {
                "group": group,
                "group_label": GROUP_LABELS[group],
                "uniprot_id": protein_id,
                "sequence_length": len(sequence),
                "standard_residue_count": standard_length,
            }
            row.update({aa: counts[aa] / standard_length * 100 for aa in AA_ORDER})
            rows.append(row)

        pooled_total = sum(pooled.values())
        for aa in AA_ORDER:
            pooled_rows.append({
                "group": group,
                "group_label": GROUP_LABELS[group],
                "n_proteins": protein_count,
                "standard_residue_count": pooled_total,
                "excluded_nonstandard_residues": excluded_residues,
                "amino_acid": aa,
                "class": next(name for name, members in CLASSES if aa in members),
                "pooled_percentage": pooled[aa] / pooled_total * 100,
            })

    per_protein = pd.DataFrame(rows)
    pooled_df = pd.DataFrame(pooled_rows)
    return per_protein, pooled_df


def run_statistics(per_protein: pd.DataFrame):
    pairs = [
        ("Novel_RLBP", "Non_RLBP"),
        ("Known_RLBP", "Non_RLBP"),
        ("Novel_RLBP", "Known_RLBP"),
    ]
    results = []
    for aa in AA_ORDER:
        for group_a, group_b in pairs:
            a = per_protein.loc[per_protein.group == group_a, aa].to_numpy()
            b = per_protein.loc[per_protein.group == group_b, aa].to_numpy()
            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
            # Rank-biserial correlation: positive means group A tends higher.
            rbc = 2 * stat / (len(a) * len(b)) - 1
            results.append({
                "amino_acid": aa,
                "class": next(name for name, members in CLASSES if aa in members),
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(a),
                "n_b": len(b),
                "median_a_pct": np.median(a),
                "median_b_pct": np.median(b),
                "median_difference_pct_point": np.median(a) - np.median(b),
                "mann_whitney_u": stat,
                "rank_biserial_correlation": rbc,
                "p_value": p_value,
            })
    stats = pd.DataFrame(results)
    stats["p_adjust_bh"] = benjamini_hochberg(stats["p_value"].to_numpy())
    stats["significant_fdr_0.05"] = stats["p_adjust_bh"] < 0.05
    return stats


def add_class_arcs(ax, ymax):
    n = len(AA_ORDER)
    # Sector boundaries are expressed in half-step coordinates. Hydrophilic
    # wraps across the top of the radar and therefore extends beyond index 20.
    sectors = [
        ("Acidic", 0.5, 2.5),
        ("Basic", 2.5, 5.5),
        ("Hydrophobic", 5.5, 13.5),
        ("Hydrophilic", 13.5, 20.5),
    ]
    for class_name, start, end in sectors:
        pad = 0.12
        theta = np.linspace(2 * np.pi * (start + pad) / n,
                            2 * np.pi * (end - pad) / n, 100)
        ax.plot(theta, np.full_like(theta, ymax * 1.18), color="#202020",
                linewidth=0.75, clip_on=False)
        angle = 2 * np.pi * ((start + end) / 2) / n
        rotation = np.degrees(angle)
        if 90 < rotation < 270:
            rotation += 180
        ax.text(angle, ymax * 1.30, class_name, ha="center", va="center",
                fontsize=7.2, rotation=rotation, rotation_mode="anchor",
                clip_on=False)


def plot_radar(pooled_df: pd.DataFrame):
    n = len(AA_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    maximum = pooled_df["pooled_percentage"].max()
    ymax = max(10, math.ceil(maximum / 2) * 2)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(6.3, 5.4))
    ax = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.88, bottom=0.08)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for group in GROUPS:
        subset = pooled_df[pooled_df.group == group].set_index("amino_acid")
        values = np.array([subset.loc[aa, "pooled_percentage"] for aa in AA_ORDER])
        closed_values = np.r_[values, values[0]]
        ax.plot(closed_angles, closed_values, color=GROUP_COLORS[group],
                linewidth=1.35,
                label=f"{GROUP_LABELS[group]} (n={int(subset['n_proteins'].iloc[0]):,})")

    ax.set_xticks(angles)
    ax.set_xticklabels(AA_ORDER, fontsize=8.5)
    ax.tick_params(axis="x", pad=5)
    radial_ticks = np.arange(2, ymax + 0.1, 2)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{x:g}" for x in radial_ticks], fontsize=6.5,
                       color="#555555")
    ax.set_ylim(0, ymax)
    ax.set_rlabel_position(92)
    ax.grid(color="#AEB2B8", linewidth=0.55, alpha=0.78)
    ax.spines["polar"].set_color("#222222")
    ax.spines["polar"].set_linewidth(0.8)
    add_class_arcs(ax, ymax)
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.50), frameon=False,
              fontsize=7.4, handlelength=2.7)
    prefix = OUT / "aa_composition_three_groups_radar_pub"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{prefix}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def write_summary(pooled_df: pd.DataFrame, stats: pd.DataFrame):
    pivot = pooled_df.pivot(index="amino_acid", columns="group",
                            values="pooled_percentage").loc[AA_ORDER]
    top = {g: pivot[g].sort_values(ascending=False).head(5) for g in GROUPS}
    significant = stats[stats["significant_fdr_0.05"]]

    lines = [
        "# 三类RLBP相关蛋白的氨基酸组成比较",
        "",
        "## 分析目的",
        "",
        "比较新预测RLBP、已知RLBP与非RLBP三组蛋白的20种标准氨基酸组成，",
        "评估模型筛选所得候选蛋白是否呈现与已知RLBP相似或区别于非RLBP背景的序列组成特征。",
        "",
        "## 方法",
        "",
        "分析对象为342个新预测RLBP、376个已知RLBP和9,109个非RLBP。",
        "对每条全长蛋白序列统计20种标准氨基酸的残基数。雷达图中的占比采用组内所有标准氨基酸残基汇总后计算，",
        "即某氨基酸残基总数除以该组20种标准氨基酸残基总数。非标准字符不计入分母。",
        "20种氨基酸按照参考雷达图的顺序排列，并在外围标示Hydrophilic、Acidic、Basic和Hydrophobic四类。",
        "此外，在逐蛋白水平计算各氨基酸百分比，并采用双侧Mann–Whitney U检验进行三组两两比较；",
        "全部60次检验统一采用Benjamini–Hochberg方法校正，FDR<0.05视为差异显著。",
        "",
        "## 结果",
        "",
    ]
    for group in GROUPS:
        description = "、".join(f"{aa}（{value:.2f}%）" for aa, value in top[group].items())
        lines.append(f"{GROUP_LABELS[group]}中占比最高的5种氨基酸为：{description}。")
    lines.extend([
        "",
        f"逐蛋白两两比较共进行{len(stats)}项检验，其中{len(significant)}项在BH校正后达到FDR<0.05。",
        "具体差异方向、效应量和校正后p值见`aa_composition_pairwise_statistics.csv`。",
        "雷达图用于展示三组整体氨基酸组成轮廓；由于组内蛋白数量差异较大，统计推断基于逐蛋白占比而非仅依据汇总曲线。",
        "",
        "## 输出文件",
        "",
        "- `aa_composition_three_groups_radar_pub.svg/.pdf/.png`：论文版雷达图。",
        "- `aa_composition_group_summary.csv`：各组汇总氨基酸占比。",
        "- `aa_composition_per_protein.csv`：逐蛋白氨基酸占比。",
        "- `aa_composition_pairwise_statistics.csv`：逐氨基酸两两统计比较。",
        "- `plot_aa_composition_radar.py`：完整可复现脚本。",
        "",
        "## 图注建议",
        "",
        "三类RLBP相关蛋白的氨基酸组成比较。雷达图展示新预测RLBP、已知RLBP和非RLBP中20种标准氨基酸的总体残基占比。",
        "氨基酸按照参考雷达图分为Hydrophilic、Acidic、Basic和Hydrophobic四类。各组占比由组内对应氨基酸残基总数除以标准氨基酸残基总数计算。",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_protein, pooled_df = calculate_composition()
    stats = run_statistics(per_protein)
    pooled_df.to_csv(OUT / "aa_composition_group_summary.csv", index=False, float_format="%.6f")
    per_protein.to_csv(OUT / "aa_composition_per_protein.csv", index=False, float_format="%.6f")
    stats.to_csv(OUT / "aa_composition_pairwise_statistics.csv", index=False, float_format="%.8g")
    plot_radar(pooled_df)
    write_summary(pooled_df, stats)
    print(f"Saved amino-acid composition analysis to {OUT}")
    print(pooled_df.groupby("group")["n_proteins"].first().to_string())


if __name__ == "__main__":
    main()

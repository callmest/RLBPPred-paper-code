#!/usr/bin/env python3
"""GO and domain enrichment for untrained proteome candidates using UniProt."""

from __future__ import annotations

import csv
import math
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE = REPOSITORY_ROOT / "results" / "untrained_proteome_analysis"
FG_FILE = BASE / "untrained_proteome_prob_gt_0.85.csv"
BG_FILE = REPOSITORY_ROOT / "results" / "inference_four_datasets" / "untrained_proteome_predictions.csv"

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
FIELDS = (
    "accession,gene_primary,go_id,go_p,go_f,go_c,xref_interpro,xref_pfam"
)

GO_PATTERN = re.compile(r"(.+?)\s*\[(GO:\d+)\]")


def read_ids(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [r["protein_id"].strip() for r in csv.DictReader(f)]


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_go_name_field(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    for m in GO_PATTERN.finditer(text):
        name = m.group(1).strip()
        go_id = m.group(2).strip()
        out[go_id] = name
    return out


def fetch_uniprot_annotations(ids: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    ann: dict[str, dict] = {}
    go_names: dict[str, str] = {}

    parts = chunked(ids, 40)
    for i, part in enumerate(parts, start=1):
        text = None
        batch_sizes = [40, 20, 10, 5, 1]
        for bs in batch_sizes:
            try:
                text_blocks = []
                for sub in chunked(part, bs):
                    query = " OR ".join(f"accession:{x}" for x in sub)
                    params = {
                        "query": query,
                        "format": "tsv",
                        "fields": FIELDS,
                    }
                    url = f"{UNIPROT_STREAM}?{urllib.parse.urlencode(params)}"
                    with urllib.request.urlopen(url, timeout=90) as r:
                        text_blocks.append(r.read().decode("utf-8", errors="replace"))
                # keep first header, append rows from others
                merged = []
                for j, block in enumerate(text_blocks):
                    lines = block.splitlines()
                    if not lines:
                        continue
                    if j == 0:
                        merged.extend(lines)
                    else:
                        merged.extend(lines[1:])
                text = "\n".join(merged) + ("\n" if merged else "")
                break
            except Exception:
                text = None
                continue
        if text is None:
            raise RuntimeError(f"Failed to fetch UniProt annotations for batch {i}")

        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
        for row in rows:
            pid = (row.get("Entry") or "").strip()
            if not pid:
                continue

            go_ids = [
                x.strip()
                for x in (row.get("Gene Ontology IDs") or "").split(";")
                if x.strip().startswith("GO:")
            ]
            ipr_ids = [
                x.strip()
                for x in (row.get("InterPro") or "").split(";")
                if x.strip()
            ]
            pfam_ids = [
                x.strip() for x in (row.get("Pfam") or "").split(";") if x.strip()
            ]

            go_bp = parse_go_name_field(row.get("Gene Ontology (biological process)") or "")
            go_mf = parse_go_name_field(row.get("Gene Ontology (molecular function)") or "")
            go_cc = parse_go_name_field(row.get("Gene Ontology (cellular component)") or "")
            go_names.update(go_bp)
            go_names.update(go_mf)
            go_names.update(go_cc)

            ann[pid] = {
                "GO": set(go_ids),
                "GO:BP": set(go_bp.keys()),
                "GO:MF": set(go_mf.keys()),
                "GO:CC": set(go_cc.keys()),
                "INTERPRO": set(ipr_ids),
                "PFAM": set(pfam_ids),
            }

        if i % 5 == 0:
            print(f"Fetched chunks: {i}", flush=True)
        time.sleep(0.12)

    return ann, go_names


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k: int, n: int, m: int, N: int) -> float:
    max_x = min(n, m)
    if k > max_x:
        return 0.0
    denom = log_comb(N, n)
    vals = []
    for x in range(k, max_x + 1):
        lp = log_comb(m, x) + log_comb(N - m, n - x) - denom
        vals.append(lp)
    ref = max(vals)
    return min(1.0, sum(math.exp(v - ref) for v in vals) * math.exp(ref))


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        raw = pvals[idx] * m / i
        prev = min(prev, raw)
        q[idx] = min(1.0, prev)
    return q


def enrich(
    fg_ids: list[str],
    bg_ids: list[str],
    ann: dict[str, dict],
    sources: list[str],
    go_names: dict[str, str],
) -> list[dict]:
    bg_set = set(bg_ids)
    fg_set = set(fg_ids)
    N = len(bg_set)
    n = len(fg_set)

    results = []
    for src in sources:
        term_bg: dict[str, set[str]] = defaultdict(set)
        term_fg: dict[str, set[str]] = defaultdict(set)

        for pid in bg_set:
            for t in ann.get(pid, {}).get(src, set()):
                term_bg[t].add(pid)
        for pid in fg_set:
            for t in ann.get(pid, {}).get(src, set()):
                term_fg[t].add(pid)

        rows = []
        for term, fg_members in term_fg.items():
            k = len(fg_members)
            if k == 0:
                continue
            m = len(term_bg.get(term, set()))
            if m == 0:
                continue
            p = hypergeom_sf(k, n, m, N)
            bg_rate = m / N if N else 0.0
            fg_rate = k / n if n else 0.0
            fe = (fg_rate / bg_rate) if bg_rate > 0 else float("inf")
            rows.append(
                {
                    "source": src,
                    "term_id": term,
                    "term_name": go_names.get(term, term),
                    "fg_hits": k,
                    "fg_size": n,
                    "bg_hits": m,
                    "bg_size": N,
                    "fg_rate": fg_rate,
                    "bg_rate": bg_rate,
                    "fold_enrichment": fe,
                    "p_value": p,
                    "fg_proteins": ";".join(sorted(fg_members)),
                }
            )

        if rows:
            qvals = bh_fdr([r["p_value"] for r in rows])
            for r, q in zip(rows, qvals):
                r["fdr_bh"] = q
        results.extend(rows)

    return results


def write_outputs(results: list[dict]) -> None:
    all_csv = BASE / "go_domain_enrichment_gt_0.85.csv"
    top_csv = BASE / "go_domain_enrichment_gt_0.85_top_terms.csv"
    sig_csv = BASE / "go_domain_enrichment_gt_0.85_significant_fdr_0.05.csv"

    results_sorted = sorted(results, key=lambda r: (r["source"], r["fdr_bh"], r["p_value"]))

    cols = [
        "source",
        "term_id",
        "term_name",
        "fg_hits",
        "fg_size",
        "bg_hits",
        "bg_size",
        "fg_rate",
        "bg_rate",
        "fold_enrichment",
        "p_value",
        "fdr_bh",
        "fg_proteins",
    ]

    with all_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results_sorted)

    with sig_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results_sorted:
            if r["fdr_bh"] <= 0.05:
                w.writerow(r)

    with top_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "rank", "term_id", "term_name", "fdr_bh", "p_value", "fold_enrichment", "fg_hits", "bg_hits"])
        for src in ["GO:BP", "GO:MF", "GO:CC", "INTERPRO", "PFAM"]:
            top = [r for r in results_sorted if r["source"] == src][:20]
            for i, r in enumerate(top, start=1):
                w.writerow(
                    [
                        src,
                        i,
                        r["term_id"],
                        r["term_name"],
                        f"{r['fdr_bh']:.6g}",
                        f"{r['p_value']:.6g}",
                        f"{r['fold_enrichment']:.4f}",
                        r["fg_hits"],
                        r["bg_hits"],
                    ]
                )

    print(f"Saved: {all_csv}")
    print(f"Saved: {sig_csv}")
    print(f"Saved: {top_csv}")


def main() -> None:
    fg_ids = read_ids(FG_FILE)
    bg_ids = read_ids(BG_FILE)
    print(f"Foreground IDs (>0.85): {len(fg_ids)}")
    print(f"Background IDs (untrained): {len(bg_ids)}")

    ann, go_names = fetch_uniprot_annotations(bg_ids)
    print(f"Annotated proteins fetched: {len(ann)}")

    # Keep only foreground IDs available in annotation table
    fg_ids = [x for x in fg_ids if x in ann]
    bg_ids = [x for x in bg_ids if x in ann]
    print(f"Foreground usable IDs: {len(fg_ids)}")
    print(f"Background usable IDs: {len(bg_ids)}")

    results = enrich(
        fg_ids=fg_ids,
        bg_ids=bg_ids,
        ann=ann,
        sources=["GO:BP", "GO:MF", "GO:CC", "INTERPRO", "PFAM"],
        go_names=go_names,
    )
    print(f"Total enriched rows: {len(results)}")
    write_outputs(results)

    for src in ["GO:BP", "GO:MF", "GO:CC", "INTERPRO", "PFAM"]:
        top = sorted(
            [r for r in results if r["source"] == src],
            key=lambda x: (x["fdr_bh"], x["p_value"]),
        )[:5]
        print(f"\nTop {src}:")
        for r in top:
            print(
                f"{r['term_id']} | {r['term_name']} | FDR={r['fdr_bh']:.3e} "
                f"| FE={r['fold_enrichment']:.2f} | {r['fg_hits']}/{r['fg_size']}"
            )


if __name__ == "__main__":
    main()

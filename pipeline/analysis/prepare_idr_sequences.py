#!/usr/bin/env python3
"""Download protein sequences from UniProt for the three comparison groups."""

import csv
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


BASE = Path(os.environ.get("RLBPPRED_ROOT", Path(__file__).resolve().parents[2]))
DATA = BASE / "data"
RESULTS = BASE / "results"
OUT_DIR = DATA / "idr_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_candidate_ids(path):
    ids = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["protein_id"])
    return ids


def load_fasta_as_dict(fasta_path):
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}


def download_uniprot_fasta(uniprot_ids, output_path, batch_size=50):
    """Download sequences from UniProt REST API in batches."""
    if output_path.exists():
        existing = load_fasta_as_dict(output_path)
        missing = [uid for uid in uniprot_ids if uid not in existing]
        if not missing:
            print(f"  {output_path.name}: all {len(uniprot_ids)} sequences already downloaded")
            return
        print(f"  {output_path.name}: {len(existing)} exist, downloading {len(missing)} more")
        uniprot_ids = missing
    else:
        existing = {}

    records = []
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i : i + batch_size]
        query = "+OR+".join(f"accession:{uid}" for uid in batch)
        full_url = f"https://rest.uniprot.org/uniprotkb/stream?query={query}&format=fasta"
        for attempt in range(3):
            try:
                req = urllib.request.Request(full_url)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = resp.read().decode("utf-8")
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  Retry {attempt+1}/3 for batch {i//batch_size+1}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  FAILED batch {i//batch_size+1}, skipping {len(batch)} IDs")
            continue

        from io import StringIO
        for rec in SeqIO.parse(StringIO(text), "fasta"):
            acc = rec.id.split("|")[1] if "|" in rec.id else rec.id
            records.append(SeqRecord(Seq(str(rec.seq)), id=acc, description=""))

        downloaded_so_far = len(existing) + len(records)
        total = len(existing) + len(uniprot_ids)
        print(f"  Downloaded {downloaded_so_far}/{total}")
        time.sleep(1)

    all_records = []
    for uid, seq in existing.items():
        all_records.append(SeqRecord(Seq(seq), id=uid, description=""))
    all_records.extend(records)

    with open(output_path, "w") as f:
        SeqIO.write(all_records, f, "fasta")
    print(f"  Saved {len(all_records)} sequences to {output_path}")


def main():
    print("=== Preparing sequences for IDR analysis ===\n")

    # 1. 376 known RLBPs - already have FASTA
    pos_path = DATA / "positive.fasta"
    pos_seqs = load_fasta_as_dict(pos_path)
    print(f"Known RLBPs: {len(pos_seqs)} sequences from {pos_path}")

    # 2. 9,109 negatives - already have FASTA
    neg_path = DATA / "negative.fasta"
    neg_seqs = load_fasta_as_dict(neg_path)
    print(f"Negatives: {len(neg_seqs)} sequences from {neg_path}")

    # 3. 342 candidates - need to download from UniProt
    cand_csv = RESULTS / "untrained_proteome_analysis" / "untrained_proteome_prob_gt_0.85_with_gene_names.csv"
    cand_ids = load_candidate_ids(cand_csv)
    print(f"Candidates: {len(cand_ids)} IDs from {cand_csv}")

    cand_fasta = OUT_DIR / "candidates_342.fasta"
    print(f"\nDownloading candidate sequences from UniProt...")
    download_uniprot_fasta(cand_ids, cand_fasta)

    # Verify
    if cand_fasta.exists():
        cand_seqs = load_fasta_as_dict(cand_fasta)
        print(f"\nFinal counts:")
        print(f"  Known RLBPs:  {len(pos_seqs)}")
        print(f"  Negatives:    {len(neg_seqs)}")
        print(f"  Candidates:   {len(cand_seqs)}")
        missing = set(cand_ids) - set(cand_seqs.keys())
        if missing:
            print(f"  WARNING: {len(missing)} candidates not downloaded: {list(missing)[:5]}...")
    else:
        print("ERROR: Failed to create candidate FASTA")
        sys.exit(1)

    # Create symlinks for convenience
    for name, src in [("known_376.fasta", pos_path), ("negative_9109.fasta", neg_path)]:
        dst = OUT_DIR / name
        if not dst.exists():
            dst.symlink_to(src)
            print(f"  Linked {dst} -> {src}")

    print("\nDone. All sequences ready in:", OUT_DIR)


if __name__ == "__main__":
    main()

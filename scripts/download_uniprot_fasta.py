#!/usr/bin/env python3
"""Download UniProtKB FASTA records for a released accession list."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://rest.uniprot.org/uniprotkb/stream"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True, help="One accession per line.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between requests.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 200:
        raise SystemExit("--batch-size must be between 1 and 200")

    accessions = [line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
    if len(accessions) != len(set(accessions)):
        raise SystemExit("The accession list contains duplicates")

    chunks: list[str] = []
    for start in range(0, len(accessions), args.batch_size):
        batch = accessions[start:start + args.batch_size]
        query = "(" + " OR ".join(f"accession:{accession}" for accession in batch) + ")"
        url = ENDPOINT + "?" + urlencode({"format": "fasta", "query": query})
        request = Request(url, headers={"User-Agent": "RLBPPred-reproducibility/1.0"})
        with urlopen(request, timeout=120) as response:
            chunks.append(response.read().decode("utf-8"))
        if start + args.batch_size < len(accessions):
            time.sleep(args.pause)

    text = "".join(chunks)
    returned = sum(line.startswith(">") for line in text.splitlines())
    if returned != len(accessions):
        raise RuntimeError(f"Requested {len(accessions)} accessions but received {returned} FASTA records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Downloaded {returned} UniProtKB records to {args.output}")


if __name__ == "__main__":
    main()

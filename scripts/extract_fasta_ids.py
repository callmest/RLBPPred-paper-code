#!/usr/bin/env python3
"""Extract unique accession identifiers from FASTA headers."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    identifiers: list[str] = []
    seen: set[str] = set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            token = line[1:].strip().split()[0]
            if "|" in token:
                fields = token.split("|")
                token = fields[1] if len(fields) > 1 else fields[0]
            if token and token not in seen:
                identifiers.append(token); seen.add(token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    print(f"Wrote {len(identifiers)} identifiers to {args.output}")


if __name__ == "__main__":
    main()

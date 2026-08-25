"""
Download mouse protein PDB structures from AlphaFold DB.

Usage:
  python scripts/download_mouse_pdb.py \
    --id-file data/mouse/mouse_ids.xlsx \
    --output-dir data/mouse_all_pdb \
    --organism mouse
"""

import argparse
import os
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


def download_alphafold_pdb(uniprot_id: str, output_dir: Path, organism: str = "mouse") -> bool:
    """
    Download PDB file from AlphaFold DB for a given UniProt ID.

    Uses API to get the correct download URL with proper version.

    Returns True if successful, False otherwise.
    """
    output_path = output_dir / f"{uniprot_id}.pdb"

    # Skip if already exists
    if output_path.exists():
        return True

    try:
        # Query API to get PDB download URL
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        api_response = requests.get(api_url, timeout=30)

        if api_response.status_code != 200:
            return False

        data = api_response.json()
        if not data or len(data) == 0:
            return False

        # Get the PDB download URL from API response
        pdb_url = data[0].get('pdbUrl')
        if not pdb_url:
            return False

        # Download PDB file
        pdb_response = requests.get(pdb_url, timeout=30)
        if pdb_response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(pdb_response.content)
            return True
        else:
            return False

    except Exception as e:
        # Silent failure for individual proteins
        return False


def main():
    parser = argparse.ArgumentParser(description="Download mouse PDB files from AlphaFold DB")
    parser.add_argument("--id-file", required=True, help="Path to mouse_ids.xlsx")
    parser.add_argument("--output-dir", required=True, help="Output directory for PDB files")
    parser.add_argument("--organism", default="mouse", help="Organism name")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load UniProt IDs
    print(f"Loading UniProt IDs from {args.id_file}")
    df = pd.read_excel(args.id_file)
    uniprot_ids = df['UniprotID'].unique().tolist()
    print(f"Total unique IDs: {len(uniprot_ids)}")

    # Check existing files
    existing_files = set(f.stem for f in output_dir.glob("*.pdb"))
    print(f"Already downloaded: {len(existing_files)}")

    # Filter IDs to download
    ids_to_download = [uid for uid in uniprot_ids if uid not in existing_files]
    print(f"Remaining to download: {len(ids_to_download)}")

    if len(ids_to_download) == 0:
        print("All PDB files already downloaded!")
        return

    # Download
    success_count = 0
    failed_ids = []

    print(f"\nDownloading PDB files from AlphaFold DB...")
    for uniprot_id in tqdm(ids_to_download, desc="Downloading"):
        success = download_alphafold_pdb(uniprot_id, output_dir, args.organism)
        if success:
            success_count += 1
        else:
            failed_ids.append(uniprot_id)

        # Rate limiting
        time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Total IDs: {len(uniprot_ids)}")
    print(f"Already existed: {len(existing_files)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(failed_ids)}")
    print(f"Total PDB files: {len(list(output_dir.glob('*.pdb')))}")

    if failed_ids:
        failed_file = output_dir / "failed_downloads.txt"
        with open(failed_file, 'w') as f:
            for uid in failed_ids:
                f.write(f"{uid}\n")
        print(f"\nFailed IDs saved to {failed_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract ESM embeddings from FASTA files using ESMC model

This script processes protein sequences from FASTA files and generates
embeddings using the ESM3 ESMC model, saving each embedding as a
separate .pt file named by ID.

python scripts/extract_esmc_embedding.py --fasta-file data/positive.fasta --output-dir data/embeddings/positive
python scripts/extract_esmc_embedding.py --fasta-file data/positive.fasta --output-dir data/embeddings/positive_mean --pooling mean
python scripts/extract_esmc_embedding.py --fasta-file data/positive.fasta --output-dir data/embeddings/esmc_600m/positive_mean --pooling mean --model esmc_600m
"""
import os
import argparse
import torch
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm

# Respect the caller's Hugging Face endpoint; retain the source environment's
# mirror only as a fallback.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


class ESMEmbeddingExtractor:
    """Extract protein embeddings using ESMC model"""

    def __init__(self, model_name: str = "esmc_300m", device: str = "cuda"):
        """
        Initialize the embedding extractor

        Args:
            model_name: ESM model name
            device: Device to run model on ('cuda' or 'cpu')
        """
        print(f"Loading ESMC model: {model_name} on {device}...")
        self.model = ESMC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name
        print("✓ Model loaded successfully")

    def extract_embedding(self, sequence: str, pooling: str = "raw") -> torch.Tensor:
        """
        Extract embedding for a single sequence

        Args:
            sequence: Protein sequence string
            pooling: 'raw' for per-token embeddings [1, seq_len, embed_dim],
                     'mean' for mean-pooled embedding [embed_dim]

        Returns:
            Embeddings tensor
        """
        with torch.no_grad():
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.model.encode(protein)

            logits_output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            # Extract tensor from ForwardTrackData if needed
            embeddings = logits_output.embeddings

            # Convert ForwardTrackData to tensor if necessary
            if hasattr(embeddings, 'tensor'):
                embeddings = embeddings.tensor

            if pooling == "mean":
                # Mean pool over sequence length: [1, seq_len, embed_dim] -> [embed_dim]
                embeddings = embeddings.squeeze(0).mean(dim=0)

            return embeddings

    def process_fasta(
        self,
        fasta_file: Path,
        output_dir: Path,
        skip_existing: bool = True,
        pooling: str = "raw"
    ) -> None:
        """
        Process a FASTA file and extract embeddings for all sequences

        Args:
            fasta_file: Path to input FASTA file
            output_dir: Directory to save embeddings
            skip_existing: Skip sequences with existing embedding files
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse FASTA file and count sequences
        print(f"\n📁 Processing: {fasta_file}")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        total_sequences = len(sequences)
        print(f"Found {total_sequences} sequences")

        # Statistics
        processed = 0
        skipped = 0
        failed = 0

        # Process each sequence
        for record in tqdm(sequences, desc="Extracting embeddings"):
            # Parse header: >pdb_id|chain_id|length
            header_parts = record.id.split('|')
            pdb_id = header_parts[0]

            # Output file path
            output_file = output_dir / f"{pdb_id}.pt"

            # Skip if already exists
            if skip_existing and output_file.exists():
                skipped += 1
                continue

            try:
                # Extract embedding
                sequence = str(record.seq)
                embeddings = self.extract_embedding(sequence, pooling=pooling)

                torch.save({
                    'pdb_id': pdb_id,
                    'sequence': sequence,
                    'length': len(sequence),
                    'embeddings': embeddings.cpu(),
                    'embedding_dim': embeddings.shape[-1],
                    'pooling': pooling,
                    'model': self.model_name
                }, output_file)

                processed += 1

            except Exception as e:
                print(f"\n❌ Error processing {pdb_id}: {e}")
                failed += 1
                continue

        # Print statistics
        print(f"\n📊 Statistics:")
        print(f"   Total sequences:  {total_sequences}")
        print(f"   Processed:        {processed}")
        print(f"   Skipped:          {skipped}")
        print(f"   Failed:           {failed}")
        print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ESM embeddings from FASTA files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process train dataset (default)
  python extract_esm_embedding.py

  # Process both train and valid datasets
  python extract_esm_embedding.py --datasets train valid

  # Use CPU instead of GPU
  python extract_esm_embedding.py --device cpu

  # Process specific FASTA file
  python extract_esm_embedding.py --fasta-file custom.fasta --output-dir ./embeddings
        """
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default=str(REPOSITORY_ROOT / 'data'),
        help='Base directory containing FASTA files'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['train', 'valid'],
        choices=['train', 'valid'],
        help='Datasets to process (default: train valid)'
    )

    parser.add_argument(
        '--fasta-file',
        type=str,
        default=None,
        help='Specific FASTA file to process (overrides --datasets)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides default naming)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='esmc_300m',
        choices=['esmc_300m', 'esmc_600m'],
        help='ESMC model to use (default: esmc_300m)'
    )

    parser.add_argument(
        '--pooling',
        type=str,
        default='raw',
        choices=['raw', 'mean'],
        help='Embedding output mode: raw per-token [1, seq_len, dim] or mean-pooled [dim] (default: raw)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run model on (default: cuda)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip sequences with existing embeddings (default: True)'
    )

    parser.add_argument(
        '--no-skip-existing',
        dest='skip_existing',
        action='store_false',
        help='Reprocess all sequences even if embeddings exist'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🧬 ESM Embedding Extraction (ESMC)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Pooling: {args.pooling}")

    # Initialize extractor
    extractor = ESMEmbeddingExtractor(model_name=args.model, device=args.device)

    # Process specific file or datasets
    if args.fasta_file:
        # Process single FASTA file
        fasta_path = Path(args.fasta_file)
        output_path = Path(args.output_dir) if args.output_dir else fasta_path.parent / "embeddings"

        extractor.process_fasta(
            fasta_file=fasta_path,
            output_dir=output_path,
            skip_existing=args.skip_existing,
            pooling=args.pooling
        )
    else:
        # Process datasets
        base_dir = Path(args.base_dir)

        for dataset in args.datasets:
            fasta_file = base_dir / f"{dataset}.fasta"

            if not fasta_file.exists():
                print(f"⚠️  FASTA file not found: {fasta_file}")
                continue

            # Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir) / dataset
            else:
                output_dir = base_dir / f"{args.model}_{dataset}_embedding"

            extractor.process_fasta(
                fasta_file=fasta_file,
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                pooling=args.pooling
            )

    print("\n" + "=" * 70)
    print("✅ Embedding extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

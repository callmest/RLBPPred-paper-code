#!/usr/bin/env python3
"""
Extract SaProt embeddings from SaProt-format FASTA files (AA+3Di tokens).
Uses ESM repository loading method (.pt checkpoint).

Usage:
  python scripts/extract_saprot_embedding.py \
    --fasta-file data/saprot_input_tokens/positive.fasta \
    --output-dir data/embeddings/saprot-650m-af2/positive_mean \
    --model-path model/saprot_650m_af2/SaProt_650M_AF2.pt \
    --pooling mean

  python scripts/extract_saprot_embedding.py \
    --fasta-file data/saprot_input_tokens/negative.fasta \
    --output-dir data/embeddings/saprot-650m-pdb/negative_mean \
    --model-path model/saprot_650m_pdb/SaProt_650M_PDB.pt \
    --pooling mean
"""
import os
import sys
import argparse
import itertools
import torch
from pathlib import Path
from tqdm import tqdm

SAPROT_DIR = os.environ.get(
    "SAPROT_DIR", str(Path(__file__).resolve().parents[2] / "external" / "SaProt")
)
sys.path.insert(0, SAPROT_DIR)

import esm
from esm.model.esm2 import ESM2

FOLDSEEK_SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWY#"
FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"


def build_saprot_alphabet():
    """Build SaProt alphabet with AA+3Di token pairs."""
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_tok, struc_tok in itertools.product(FOLDSEEK_SEQ_VOCAB, FOLDSEEK_STRUC_VOCAB):
        tokens.append(seq_tok + struc_tok)

    alphabet = esm.data.Alphabet(
        standard_toks=tokens, prepend_toks=[], append_toks=[],
        prepend_bos=True, append_eos=True, use_msa=False,
    )
    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}
    return alphabet


def load_saprot_model(model_path: str):
    """Load SaProt model using ESM repository method."""
    alphabet = build_saprot_alphabet()
    data = torch.load(model_path, map_location="cpu")
    weights = data["model"]
    config = data["config"]

    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )

    # Load weights
    model_dict = model.state_dict()
    for k, v in weights.items():
        if k in model_dict:
            model_dict[k] = v
    model.load_state_dict(model_dict)
    return model, alphabet


def tokenize_saprot_seq(combined_seq: str, alphabet):
    """
    Tokenize a SaProt combined sequence (e.g. 'MdSaIwLp' or 'M#S#T#Q#').
    Each token is 2 chars: uppercase AA + lowercase 3Di.
    Returns tensor with <cls> prepended and <eos> appended.
    """
    # Split into 2-char tokens
    seq_tokens = [combined_seq[i:i+2] for i in range(0, len(combined_seq), 2)]

    # Convert to uppercase token format for alphabet lookup
    token_ids = [alphabet.tok_to_idx.get("<cls>")]
    for tok in seq_tokens:
        # Token format in alphabet: e.g. "Md", "Sa", "M#", "S#"
        idx = alphabet.tok_to_idx.get(tok)
        if idx is None:
            idx = alphabet.tok_to_idx.get("<unk>")
        token_ids.append(idx)
    token_ids.append(alphabet.tok_to_idx.get("<eos>"))

    return torch.tensor([token_ids], dtype=torch.long)


def parse_saprot_fasta(fasta_file: str):
    """Parse SaProt-format FASTA (combined AA+3Di sequences)."""
    records = []
    current_id = None
    current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    return records


def extract_embeddings(model, alphabet, fasta_file, output_dir, device,
                       model_name, pooling="mean", skip_existing=True):
    """Extract embeddings for all sequences in a SaProt-format FASTA file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_saprot_fasta(fasta_file)
    print(f"\nProcessing: {fasta_file}")
    print(f"Found {len(records)} sequences")

    processed = 0
    skipped = 0
    failed = 0

    for protein_id, combined_seq in tqdm(records, desc="Extracting embeddings"):
        output_file = output_dir / f"{protein_id}.pt"

        if skip_existing and output_file.exists():
            skipped += 1
            continue

        try:
            input_ids = tokenize_saprot_seq(combined_seq, alphabet).to(device)

            with torch.no_grad():
                results = model(input_ids, repr_layers=[model.num_layers])
                # Shape: [1, seq_len_with_special, embed_dim]
                hidden = results["representations"][model.num_layers]

                # Remove <cls> (idx 0) and <eos> (last token)
                seq_hidden = hidden[0, 1:-1, :]  # [seq_len, embed_dim]

                if pooling == "mean":
                    emb = seq_hidden.mean(dim=0)  # [embed_dim]
                else:
                    emb = seq_hidden  # [seq_len, embed_dim]

            # Extract original AA sequence (every other char starting at 0)
            aa_seq = combined_seq[0::2]

            torch.save({
                'pdb_id': protein_id,
                'sequence': aa_seq,
                'length': len(aa_seq),
                'embeddings': emb.cpu(),
                'embedding_dim': emb.shape[-1],
                'pooling': pooling,
                'model': model_name,
            }, output_file)

            processed += 1

        except Exception as e:
            print(f"\nError processing {protein_id}: {e}")
            failed += 1

    print(f"\nStatistics:")
    print(f"  Total:     {len(records)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SaProt embeddings (ESM repo loading)"
    )
    parser.add_argument("--fasta-file", required=True,
                        help="SaProt-format FASTA file (AA+3Di combined)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .pt files")
    parser.add_argument("--model-path", required=True,
                        help="Path to SaProt .pt checkpoint")
    parser.add_argument("--model-name", default=None,
                        help="Model name for metadata (auto-detected from path)")
    parser.add_argument("--pooling", default="mean", choices=["raw", "mean"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    # Auto-detect model name from path
    if args.model_name is None:
        # e.g. model/saprot_650m_af2/SaProt_650M_AF2.pt -> saprot-650m-af2
        parent_dir = Path(args.model_path).parent.name
        args.model_name = parent_dir.replace("_", "-")

    print("=" * 60)
    print(f"SaProt Embedding Extraction")
    print(f"  Model:   {args.model_name} ({args.model_path})")
    print(f"  Device:  {args.device}")
    print(f"  Pooling: {args.pooling}")
    print("=" * 60)

    print("Loading model...")
    model, alphabet = load_saprot_model(args.model_path)
    model = model.to(args.device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    extract_embeddings(
        model=model,
        alphabet=alphabet,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        device=args.device,
        model_name=args.model_name,
        pooling=args.pooling,
        skip_existing=args.skip_existing,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

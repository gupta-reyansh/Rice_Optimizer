"""
Convert a CDS FASTA file into the training CSV format expected by CodonTransformer.
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from CodonTransformer.CodonData import read_fasta_file
from CodonTransformer.CodonUtils import ORGANISM2ID


def load_training_dataframe(
    input_fasta: str,
    organism: str,
    keep_all_records: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Read a CDS FASTA file and return the CodonTransformer training DataFrame.

    Args:
        input_fasta (str): Path to the input CDS FASTA file.
        organism (str): Supported organism name from ORGANISM2ID.
        keep_all_records (bool): Whether to keep records flagged as incorrect_seq.

    Returns:
        Tuple[pd.DataFrame, int]: Training DataFrame and total FASTA records parsed.
    """
    if organism not in ORGANISM2ID:
        raise ValueError(
            f"Unsupported organism: {organism}. "
            "Please use an organism name that already exists in ORGANISM2ID."
        )

    input_path = Path(input_fasta)
    fasta_df = read_fasta_file(
        input_file=str(input_path),
        save_to_file=None,
        organism=organism,
    )

    total_records = len(fasta_df)
    if not keep_all_records:
        fasta_df = fasta_df.loc[fasta_df["correct_seq"]].copy()

    if fasta_df.empty:
        raise ValueError(
            "No records were eligible for export. "
            "Check whether the FASTA contains valid CDS sequences for this organism."
        )

    training_df = fasta_df.loc[:, ["dna", "protein"]].copy()
    training_df["organism"] = organism

    return training_df, total_records


def build_training_csv(
    input_fasta: str,
    organism: str,
    output_csv: str,
    keep_all_records: bool = False,
) -> tuple[int, int]:
    """
    Read a CDS FASTA file and save a CodonTransformer training CSV.

    Args:
        input_fasta (str): Path to the input CDS FASTA file.
        organism (str): Supported organism name from ORGANISM2ID.
        output_csv (str): Path to the output CSV file.
        keep_all_records (bool): Whether to keep records flagged as incorrect_seq.

    Returns:
        tuple[int, int]: Number of records written and total FASTA records parsed.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_df, total_records = load_training_dataframe(
        input_fasta=input_fasta,
        organism=organism,
        keep_all_records=keep_all_records,
    )
    training_df.to_csv(output_path, index=False)

    return len(training_df), total_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CDS FASTA file into CodonTransformer training CSV format."
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        required=True,
        help="Path to the input CDS FASTA file.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help="Supported organism name used for translation and CSV generation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file with dna, protein, organism columns.",
    )
    parser.add_argument(
        "--keep_all_records",
        action="store_true",
        help="Keep records even if read_fasta_file marks them as incorrect_seq.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_records, total_records = build_training_csv(
        input_fasta=args.input_fasta,
        organism=args.organism,
        output_csv=args.output_csv,
        keep_all_records=args.keep_all_records,
    )
    skipped_records = total_records - written_records
    print(f"Saved {written_records} records to {args.output_csv}")
    if skipped_records > 0:
        print(f"Skipped {skipped_records} records flagged as incorrect_seq")


if __name__ == "__main__":
    main()

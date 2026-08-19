"""
Train a single-species CodonTransformer model from scratch directly from a CDS FASTA file.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from CodonTransformer.CodonData import prepare_training_data, read_fasta_file


REPO_ROOT = Path(__file__).resolve().parent


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "species"


def normalize_organism(value: str) -> str:
    organism = value.strip()
    if not organism:
        raise ValueError("organism must not be empty.")
    return organism


def resolve_work_dir(work_dir: str, organism: str) -> Path:
    if work_dir:
        return Path(work_dir).expanduser().resolve()
    return (REPO_ROOT / "workflows" / slugify(organism)).resolve()


def prepare_pretrain_inputs(
    input_fasta: str,
    organism: str,
    work_dir: Path,
    keep_all_records: bool,
    codon_table: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    raw_dir = work_dir / "data" / "raw"
    processed_dir = work_dir / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parsed_records_path = raw_dir / "parsed_cds_records.csv"
    training_csv_path = raw_dir / "training_sequences.csv"
    training_json_path = processed_dir / "training_data.json"

    parsed_df = read_fasta_file(
        input_file=str(Path(input_fasta).expanduser().resolve()),
        save_to_file=None,
        organism=organism,
        codon_table=codon_table,
    )
    if parsed_df.empty:
        raise ValueError("No FASTA records were parsed from the input file.")

    parsed_df.to_csv(parsed_records_path, index=False)

    training_df = parsed_df.copy()
    if not keep_all_records:
        training_df = training_df.loc[training_df["correct_seq"]].copy()

    training_df = training_df.loc[training_df["dna"].str.len() % 3 == 0].copy()
    if training_df.empty:
        raise ValueError(
            "No eligible CDS records remained after filtering. "
            "Consider checking sequence validity or using --keep_all_records."
        )

    training_df = training_df.loc[:, ["dna", "protein"]].copy()
    training_df["organism"] = organism
    training_df.to_csv(training_csv_path, index=False)

    prepare_training_data(
        dataset=training_df,
        output_file=str(training_json_path),
        organism_to_id={organism: 0},
    )

    return parsed_df, training_df, training_csv_path, training_json_path


def run_pretrain(
    training_json_path: Path, checkpoint_dir: Path, args: argparse.Namespace
) -> None:
    from pretrain import main as pretrain_main

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pretrain_args = argparse.Namespace(
        tokenizer_path=args.tokenizer_path,
        train_data_path=str(training_json_path),
        checkpoint_dir=str(checkpoint_dir),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_gpus=args.num_gpus,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        save_interval=args.save_interval,
        seed=args.seed,
        type_vocab_size=1,
        debug=args.debug,
    )
    pretrain_main(pretrain_args)


def write_run_metadata(
    work_dir: Path,
    organism: str,
    input_fasta: str,
    parsed_df: pd.DataFrame,
    training_df: pd.DataFrame,
    training_csv_path: Path,
    training_json_path: Path,
    args: argparse.Namespace,
) -> Path:
    metadata_path = work_dir / "run_metadata.json"
    payload = {
        "input_fasta": str(Path(input_fasta).expanduser().resolve()),
        "organism": organism,
        "organism_id": 0,
        "codon_table": args.codon_table,
        "work_dir": str(work_dir),
        "parsed_records": len(parsed_df),
        "selected_records": len(training_df),
        "keep_all_records": args.keep_all_records,
        "training_csv": str(training_csv_path),
        "training_json": str(training_json_path),
        "checkpoint_dir": str((work_dir / "checkpoints").resolve()),
        "tokenizer_path": args.tokenizer_path or "adibvafa/CodonTransformer",
        "type_vocab_size": 1,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-species CodonTransformer retraining pipeline from a CDS FASTA file."
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        required=True,
        help="Path to the CDS FASTA file.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help="Target organism name stored in the generated dataset metadata.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="",
        help="Working directory for generated files. Defaults to <repo_root>/workflows/<organism_slug>.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Optional tokenizer.json path. If omitted, the default Hugging Face tokenizer is used.",
    )
    parser.add_argument(
        "--codon_table",
        type=int,
        default=None,
        help="Optional explicit NCBI codon table id for FASTA translation.",
    )
    parser.add_argument(
        "--keep_all_records",
        action="store_true",
        help="Keep records flagged as incorrect_seq when building the training set.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="Maximum number of epochs to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=6,
        help="Number of batches to accumulate gradients.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save checkpoint every N epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable pretrain.py debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.organism = normalize_organism(args.organism)

    work_dir = resolve_work_dir(args.work_dir, args.organism)
    checkpoint_dir = work_dir / "checkpoints"
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    parsed_df, training_df, training_csv_path, training_json_path = prepare_pretrain_inputs(
        input_fasta=args.input_fasta,
        organism=args.organism,
        work_dir=work_dir,
        keep_all_records=args.keep_all_records,
        codon_table=args.codon_table,
    )

    metadata_path = write_run_metadata(
        work_dir=work_dir,
        organism=args.organism,
        input_fasta=args.input_fasta,
        parsed_df=parsed_df,
        training_df=training_df,
        training_csv_path=training_csv_path,
        training_json_path=training_json_path,
        args=args,
    )

    print(f"Parsed FASTA records: {len(parsed_df)}")
    print(f"Selected records for retraining: {len(training_df)}")
    print(f"Training CSV: {training_csv_path}")
    print(f"Training JSON: {training_json_path}")
    print(f"Run metadata: {metadata_path}")

    run_pretrain(
        training_json_path=training_json_path,
        checkpoint_dir=checkpoint_dir,
        args=args,
    )

    print(f"Retraining completed. Checkpoints are available in {checkpoint_dir}")


if __name__ == "__main__":
    main()

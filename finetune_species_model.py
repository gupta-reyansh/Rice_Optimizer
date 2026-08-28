"""
Fine-tune the CodonTransformer model for a single species directly from a CDS FASTA file.
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


def run_finetune(
    training_json_path: Path, checkpoint_dir: Path, args: argparse.Namespace
) -> None:
    from finetune import main as finetune_main

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    finetune_args = argparse.Namespace(
        dataset_dir=str(training_json_path),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_filename=args.checkpoint_filename,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_gpus=args.num_gpus,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        save_every_n_steps=args.save_every_n_steps,
        seed=args.seed,
        debug=args.debug,
    )
    finetune_main(finetune_args)


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
        description="Run the single-species CodonTransformer fine-tuning pipeline from a CDS FASTA file."
    )
    parser.add_argument("--input_fasta", required=True, help="Path to the CDS FASTA file.")
    parser.add_argument("--organism", required=True, help="Target organism name.")
    parser.add_argument(
        "--work_dir",
        default="",
        help="Working directory. Defaults to <repo_root>/workflows/<organism_slug>.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="",
        help="Recorded tokenizer path for metadata; finetune.py loads its configured tokenizer.",
    )
    parser.add_argument("--codon_table", type=int, default=None)
    parser.add_argument(
        "--keep_all_records",
        action="store_true",
        help="Keep records flagged as incorrect_seq.",
    )
    parser.add_argument(
        "--checkpoint_filename", default="finetune.ckpt", help="Saved checkpoint filename."
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--save_every_n_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")
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
    print(f"Selected records for fine-tuning: {len(training_df)}")
    print(f"Training CSV: {training_csv_path}")
    print(f"Training JSON: {training_json_path}")
    print(f"Run metadata: {metadata_path}")

    run_finetune(
        training_json_path=training_json_path,
        checkpoint_dir=checkpoint_dir,
        args=args,
    )

    print(f"Fine-tuning completed. Checkpoints are available in {checkpoint_dir}")


if __name__ == "__main__":
    main()

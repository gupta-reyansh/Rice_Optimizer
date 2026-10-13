"""
Compare codon-optimization metrics across five differently-formatted result sets:

  1. nif1 + nif2 combined FASTA-style files
     (../Local Notebooks/nifh1.fna, nifh2.fna)
  2. geneious_sequences.csv        (sequence column: "sequence")
  3. idt_processed.csv             (sequence column: "Optimized Sequence")
  4. ModelTest/rice_nucleic_predictions.csv (sequence column: "predicted_dna")
  5. ModelTest/model_predictions_constrained.csv (same, with constrained sequences)
  6. ModelTest/rice_mito_predictions.csv (rice_mito.ckpt predictions, same layout;
     regenerate with predict_checkpoint.py)

Metrics (ported from Fasta_to_Metrics.ipynb):
  - get_gc_content
  - get_gc3_content
  - get_CAI_value (scored against weights built from
    Mitochondria_training.fasta, the gene-level mitochondrial CDS set)
  - avg codon-level similarity to the native original (get_sequence_similarity)
  - RNA secondary structure preservation: avg change in MFE (kcal/mol) of the
    first 50 nt (5' region) between each optimized sequence and its native
    original, folded with ViennaRNA at its default 37 C (optimized - original,
    so 0 = native structure fully preserved)

Pairing of "original" -> "optimized" sequence for the similarity metric:
  - idt_processed.csv            : row order matches ../Local Notebooks/nifH_data.csv
  - geneious_sequences.csv       : "description" id prefix matches nifH_data.csv["id"]
  - unconstrained model(No gc control) : own "sequence" column is the original
  - model_predictions_constrained.csv : same as above
  - nif1+nif2 (combined)         : native sequences themselves, no pairing (N/A)
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import RNA
from Bio import SeqIO

from CodonTransformer.CodonEvaluation import get_sequence_similarity

BASE_DIR = Path(__file__).resolve().parent
LOCAL_NOTEBOOKS_DIR = BASE_DIR.parent / "Local Notebooks"

REFERENCE_FASTA = BASE_DIR / "Data" / "Mitochondria_training.fasta"
NIF_FASTA_FILES = [
    LOCAL_NOTEBOOKS_DIR / "nifh1.fna",
    LOCAL_NOTEBOOKS_DIR / "nifh2.fna",
]
GENEIOUS_CSV = BASE_DIR / "Data" / "geneious_sequences.csv"
IDT_CSV = BASE_DIR / "Data" / "idt_processed.csv"
MODEL_CSV = BASE_DIR / "ModelTest" / "rice_nucleic_predictions.csv"
MODEL_CONSTRAINED_CSV = BASE_DIR / "ModelTest" / "model_predictions_constrained.csv"
RICE_MITO_CSV = BASE_DIR / "ModelTest" / "rice_mito_predictions.csv"
NIFH_DATA_CSV = LOCAL_NOTEBOOKS_DIR / "nifH_data.csv"

VALID_BASES = set("ATCG")

TARGET_ROW = {
    "dataset": "target",
    "avg_gc_percent": 43.51,
    "avg_gc3_percent": 40.08,
    "avg_cai": 1,
    "avg_codon_similarity_to_original_percent": 1,
    "avg_delta_5prime_mfe_kcal_per_mol": 0,  # native structure fully preserved
}

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def get_gc_content(dna: str) -> float:
    """Calculate GC percentage in a DNA sequence."""
    if not dna:
        return 0.0
    gc_count = dna.count("G") + dna.count("C")
    return (gc_count / len(dna)) * 100


def get_gc3_content(dna: str) -> float:
    """Return GC percentage among the third base of each codon."""
    return get_gc_content(dna[2::3])


def get_mfe(dna: str) -> float:
    """Minimum free energy (kcal/mol) of the folded mRNA, via ViennaRNA."""
    _, mfe = RNA.fold(dna.replace("T", "U"))
    return mfe


def get_delta_5prime_mfe(original: str, optimized: str, window: int = 50) -> float:
    """Change in MFE over the first `window` nt: optimized minus original."""
    return get_mfe(optimized[:window]) - get_mfe(original[:window])


def get_cai_weights(sequences: List[str]) -> Dict[str, float]:
    """Calculate standard CAI relative adaptiveness weights (w_i)."""
    codon_counts: Counter = Counter()
    for seq in sequences:
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if codon in GENETIC_CODE and GENETIC_CODE[codon] != "*":
                codon_counts[codon] += 1

    max_counts: Dict[str, int] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa != "*":
            max_counts[aa] = max(max_counts.get(aa, 0), codon_counts[codon])

    weights: Dict[str, float] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa == "*":
            continue
        max_c = max_counts.get(aa, 0)
        count = codon_counts.get(codon, 0)
        # Zero-count codons get a small non-zero floor to avoid log(0) in CAI.
        weights[codon] = (count / max_c) if max_c > 0 and count > 0 else 0.01
    return weights


def get_CAI_value(sequence: str, weights: Dict[str, float]) -> float:
    """Calculate Codon Adaptation Index (CAI) using geometric mean."""
    log_sum = 0.0
    valid_codons = 0
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        if codon in weights:
            log_sum += math.log(weights[codon])
            valid_codons += 1
    if valid_codons == 0:
        return 0.0
    return math.exp(log_sum / valid_codons)


def clean_coding_sequence(raw: str) -> Optional[str]:
    """Uppercase, convert RNA to DNA, and keep only complete valid CDSs."""
    if not isinstance(raw, str):
        return None
    seq = raw.strip().upper().replace("U", "T")
    if seq and len(seq) % 3 == 0 and set(seq).issubset(VALID_BASES):
        return seq
    return None


def load_fasta_sequences(paths: List[Path]) -> List[str]:
    """Read and validate coding sequences from one or more FASTA/.fna files."""
    sequences = []
    for path in paths:
        for record in SeqIO.parse(path, "fasta"):
            cleaned = clean_coding_sequence(str(record.seq))
            if cleaned:
                sequences.append(cleaned)
    return sequences


def load_csv_column(path: Path, column: str) -> List[str]:
    """Read and validate coding sequences from a single CSV column."""
    df = pd.read_csv(path)
    sequences = []
    for raw in df[column].astype(str):
        cleaned = clean_coding_sequence(raw)
        if cleaned:
            sequences.append(cleaned)
    return sequences


def load_pairs_from_model_csv(path: Path) -> List[Tuple[str, str]]:
    """model_predictions*.csv: 'sequence' is native, 'predicted_dna' is optimized."""
    df = pd.read_csv(path)
    pairs = []
    for orig, pred in zip(df["sequence"], df["predicted_dna"]):
        o, p = clean_coding_sequence(orig), clean_coding_sequence(pred)
        if o and p:
            pairs.append((o, p))
    return pairs


def translate(dna: str) -> str:
    return "".join(GENETIC_CODE[dna[i:i + 3]] for i in range(0, len(dna), 3))


def load_pairs_idt(path: Path, test_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """idt_processed.csv has no id column, so match each optimized sequence to the
    original that encodes the same protein (row order only breaks ties between
    originals that share a protein)."""
    df = pd.read_csv(path)
    originals_by_protein: Dict[str, List[str]] = {}
    for orig in test_df["sequence"]:
        o = clean_coding_sequence(orig)
        if o:
            originals_by_protein.setdefault(translate(o), []).append(o)

    pairs = []
    for pred in df["Optimized Sequence"]:
        p = clean_coding_sequence(pred)
        if p and originals_by_protein.get(translate(p)):
            pairs.append((originals_by_protein[translate(p)].pop(0), p))
    return pairs


def load_pairs_geneious(path: Path, test_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """geneious_sequences.csv: match via the id embedded in 'description'."""
    df = pd.read_csv(path)
    id_to_original = dict(zip(test_df["id"], test_df["sequence"]))
    pairs = []
    for desc, pred in zip(df["description"], df["sequence"]):
        gid = desc.split(" # ")[0]
        orig = id_to_original.get(gid)
        o, p = clean_coding_sequence(orig) if orig else None, clean_coding_sequence(pred)
        if o and p:
            pairs.append((o, p))
    return pairs


def avg_codon_similarity(pairs: Optional[List[Tuple[str, str]]]) -> Optional[float]:
    if not pairs:
        return None
    scores = [
        get_sequence_similarity(orig, pred, truncate=True, window_length=3)
        for orig, pred in pairs
    ]
    return sum(scores) / len(scores)


def avg_delta_5prime_mfe(pairs: Optional[List[Tuple[str, str]]]) -> Optional[float]:
    if not pairs:
        return None
    deltas = [get_delta_5prime_mfe(orig, pred) for orig, pred in pairs]
    return sum(deltas) / len(deltas)


def score_dataset(name: str, sequences: List[str], weights: Dict[str, float]) -> Dict:
    if not sequences:
        raise ValueError(f"No valid sequences loaded for dataset '{name}'.")

    gc_values = [get_gc_content(s) for s in sequences]
    gc3_values = [get_gc3_content(s) for s in sequences]
    cai_values = [get_CAI_value(s, weights) for s in sequences]

    return {
        "dataset": name,
        "avg_gc_percent": sum(gc_values) / len(gc_values),
        "avg_gc3_percent": sum(gc3_values) / len(gc3_values),
        "avg_cai": sum(cai_values) / len(cai_values),
    }


def main() -> None:
    print(f"Building CAI reference weights from {REFERENCE_FASTA.name}...")
    reference_sequences = load_fasta_sequences([REFERENCE_FASTA])
    if not reference_sequences:
        raise RuntimeError(f"No valid reference sequences found in {REFERENCE_FASTA}")
    weights = get_cai_weights(reference_sequences)

    test_df = pd.read_csv(NIFH_DATA_CSV, index_col=0)
    datasets = {
        "nif1+nif2 (combined)": load_fasta_sequences(NIF_FASTA_FILES),
        "geneious_sequences.csv": load_csv_column(GENEIOUS_CSV, "sequence"),
        "idt_processed.csv": load_csv_column(IDT_CSV, "Optimized Sequence"),
        "unconstrained model(No gc control)": load_csv_column(MODEL_CSV, "predicted_dna"),
        "model_predictions_constrained.csv": load_csv_column(MODEL_CONSTRAINED_CSV, "predicted_dna"),
        "final model": load_csv_column(RICE_MITO_CSV, "predicted_dna"),
    }

    pairs_by_dataset = {
        "nif1+nif2 (combined)": None,  # native sequences, no optimized counterpart
        "geneious_sequences.csv": load_pairs_geneious(GENEIOUS_CSV, test_df),
        "idt_processed.csv": load_pairs_idt(IDT_CSV, test_df),
        "unconstrained model(No gc control)": load_pairs_from_model_csv(MODEL_CSV),
        "model_predictions_constrained.csv": load_pairs_from_model_csv(MODEL_CONSTRAINED_CSV),
        "final model": load_pairs_from_model_csv(RICE_MITO_CSV),
    }

    results = [score_dataset(name, seqs, weights) for name, seqs in datasets.items()]
    for r in results:
        r["avg_codon_similarity_to_original_percent"] = avg_codon_similarity(
            pairs_by_dataset[r["dataset"]]
        )
        r["avg_delta_5prime_mfe_kcal_per_mol"] = avg_delta_5prime_mfe(
            pairs_by_dataset[r["dataset"]]
        )

    summary_df = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "avg_gc_percent": r["avg_gc_percent"],
                "avg_gc3_percent": r["avg_gc3_percent"],
                "avg_cai": r["avg_cai"],
                "avg_codon_similarity_to_original_percent": r[
                    "avg_codon_similarity_to_original_percent"
                ],
                "avg_delta_5prime_mfe_kcal_per_mol": r[
                    "avg_delta_5prime_mfe_kcal_per_mol"
                ],
            }
            for r in results
        ]
        + [TARGET_ROW]
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    output_path = BASE_DIR / "dataset_comparison_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()

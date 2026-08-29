"""
Calculate average CAI, GC content, GC3 content, and normalized MFE for a FASTA.
The FASTA path is set directly in the script.
"""

from pathlib import Path
from typing import Dict, List
from collections import Counter

from Bio import SeqIO


def get_GC_content(dna: str) -> float:
    """Calculate GC percentage in a DNA sequence."""
    if not dna:
        return 0.0
    gc_count = dna.count('G') + dna.count('C')
    return (gc_count / len(dna)) * 100


def get_CSI_weights(sequences: List[str]) -> Dict[str, float]:
    """Calculate codon usage weights from a list of DNA sequences."""
    codon_counts = Counter()
    aa_counts = Counter()
    
    # Standard genetic code
    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    
    # Count codon usage
    for seq in sequences:
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in genetic_code:
                codon_counts[codon] += 1
                aa_counts[genetic_code[codon]] += 1
    
    # Calculate relative codon usage weights
    weights = {}
    for codon, count in codon_counts.items():
        aa = genetic_code.get(codon)
        if aa and aa in aa_counts:
            weights[codon] = count / aa_counts[aa]
        else:
            weights[codon] = 0.0
    
    return weights


def get_CSI_value(sequence: str, weights: Dict[str, float]) -> float:
    """Calculate Codon Usage Index (CAI/CSI) for a sequence."""
    if not sequence or len(sequence) < 3:
        return 0.0
    
    codon_values = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in weights:
            weight = weights[codon]
            if weight > 0:
                codon_values.append(weight)
    
    if not codon_values:
        return 0.0
    
    return sum(codon_values) / len(codon_values)


def read_sequences(input_fasta: str) -> List[str]:
    """Read valid, complete DNA coding sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence = str(record.seq).upper().replace("U", "T")
        if sequence and len(sequence) % 3 == 0 and set(sequence) <= set("ATCG"):
            sequences.append(sequence)
    return sequences


def get_gc3_content(dna: str) -> float:
    """Return GC percentage among the third base of each codon."""
    third_bases = dna[2::3]
    return get_GC_content(third_bases)


def get_normalized_mfe(dna: str) -> float:
    """Return the RNA folding free energy normalized by sequence length."""
    try:
        import RNA
    except ImportError as error:
        raise RuntimeError(
            "MFE calculation requires ViennaRNA. Install it with "
            "`pip install ViennaRNA`."
        ) from error

    _, mfe = RNA.fold(dna.replace("T", "U"))
    return mfe / len(dna)


def calculate_metrics(input_fasta: str) -> Dict[str, float]:
    """Calculate averages for all valid sequences in a FASTA file."""
    sequences = read_sequences(input_fasta)
    if not sequences:
        raise ValueError("No valid DNA coding sequences were found in the FASTA file.")

    cai_weights = get_CSI_weights(sequences)
    cai_values = [get_CSI_value(sequence, cai_weights) for sequence in sequences]
    gc_values = [get_GC_content(sequence) for sequence in sequences]
    gc3_values = [get_gc3_content(sequence) for sequence in sequences]
    mfe_values = [get_normalized_mfe(sequence) for sequence in sequences]

    return {
        "sequence_count": len(sequences),
        "average_cai": sum(cai_values) / len(cai_values),
        "average_gc_percent": sum(gc_values) / len(gc_values),
        "average_gc3_percent": sum(gc3_values) / len(gc3_values),
        "average_mfe_per_nt": sum(mfe_values) / len(mfe_values),
    }


FASTA_PATH = "Rice_Optimizer/NCBI_combined_training.fasta"


def main() -> None:
    metrics = calculate_metrics(str(FASTA_PATH))
    print(f"Sequences analyzed: {metrics['sequence_count']}")
    print(f"Average CAI: {metrics['average_cai']:.6f}")
    print(f"Average GC%: {metrics['average_gc_percent']:.6f}")
    print(f"Average GC3%: {metrics['average_gc3_percent']:.6f}")
    print(f"Average MFE per nucleotide: {metrics['average_mfe_per_nt']:.6f}")


if __name__ == "__main__":
    main()
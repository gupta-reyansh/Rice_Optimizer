# Rice Optimizer

Rice Optimizer is a computational project that uses CodonTransformer models fine-tuned on rice (*Oryza sativa*) DNA to design bacterial *nifH* sequences for potential expression in rice. Given a NifH protein sequence, the model predicts a DNA sequence that encodes the same protein while following the codon patterns of rice.

The main model targets the **rice mitochondrial genome**. The nif gene family is very sensitive to oxygen, and mitochondria provide a low-oxygen environment where nitrogenase proteins can be expected to survive. Published studies that expressed nif genes in rice used the same approach, which is what led to this project. An earlier model trained on rice **nuclear** genes was built first, to test whether a transformer could beat existing codon-optimization tools at all.

This project was developed for a science fair and may support future research. It does not claim that an optimized sequence will be expressed successfully or that it will make rice nitrogen-fixing.

## Models

| Model | Trained on | Hugging Face |
| --- | --- | --- |
| **Rice Mitochondria Optimizer v2** (main, the "final model") | Rice mitochondrial genes, each a complete coding sequence with start and stop codons | [gupta-reyansh123/Rice_Mitochondria_Optimizer_v2](https://huggingface.co/gupta-reyansh123/Rice_Mitochondria_Optimizer_v2) |
| Rice Mitochondria Optimizer (earlier version) | Rice mitochondrial genomes cut into ~6 kb sections | [gupta-reyansh123/Rice_Mitochondria_Optimizer](https://huggingface.co/gupta-reyansh123/Rice_Mitochondria_Optimizer) |
| Rice Nucleic Optimizer (earlier proof of concept) | Rice nuclear cDNA | [gupta-reyansh123/Rice_Nucleic_Optimizer](https://huggingface.co/gupta-reyansh123/Rice_Nucleic_Optimizer) |

Load any of them with:

```python
from transformers import AutoTokenizer, BigBirdForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
model = BigBirdForMaskedLM.from_pretrained("gupta-reyansh123/Rice_Mitochondria_Optimizer_v2")
```

## How It Works

Both models start from the pretrained CodonTransformer and are fine-tuned on rice coding sequences. The model reads the whole sequence rather than picking codons one at a time, so its predictions reflect patterns learned from complete genes.

<img width="453" height="686" alt="Gemini_Generated_Image_5kvhi95kvhi95kvh" src="https://github.com/user-attachments/assets/b63e7979-22b9-4aa3-aff8-3ca1c4c05d44" />

The mitochondria models add **GC control** during fine-tuning. An extra loss term pushes the GC content of the predicted sequences toward a target (0.432), and a small penalty discourages negative cis-regulatory motifs. These pieces were combined with the base CodonTransformer training code in [finetune.py](finetune.py).

For the final model, the GC constraint was deliberately **relaxed in favor of 5′ MFE**. GC content and MFE both relate to RNA stability, but MFE measures stability more directly, so the final model trades some GC accuracy for a better-preserved 5′ structure.

## Data

### Mitochondria

The earlier mitochondria models were trained on genome sections. The final model was trained on whole genes instead: splitting the genomes into many small segments gave more data points, but it was the wrong trade, because quality matters more than quantity here. The gene-level set is described second below.

[Data/Mitochondria.fasta](Data/Mitochondria.fasta) (earlier models) comes from the [NCBI organelle genomes database for rice](https://www.ncbi.nlm.nih.gov/datasets/organelle/?taxon=4530) and contains four mitochondrial genome records: GenBank ON854123.1, RefSeq NC_066488.1, BA000029.3 and NC_011033.1. Each full genome was split into even sections of about 6 kb, giving **264 sequences** (1,613,448 bp in total, 43.9% GC).

[Data/Mitochondria_training.fasta](Data/Mitochondria_training.fasta) (final model) is a gene-level set built from the NCBI gene annotation for the two RefSeq genomes (NC_011033.1 and NC_066488.1; the GenBank copies are identical). Each protein-coding gene was cut out of the genome by its annotated coordinates and kept only if it is a complete open reading frame (ATG start, one terminal stop, no internal stops). This gives **69 genes**, 50 of them unique sequences. Fourteen gene families with introns, trans-splicing or RNA-edited starts were left out because their coding sequence cannot be rebuilt from the gene range alone. The source NCBI files (`genome.fna`, `organelle_gene_report.jsonl`, `organelle_report.jsonl`, `protein.faa`, `dataset_catalog.json`) are in [Data/](Data/). This set is also the reference for the CAI weights in the evaluation below.

### Nucleic (earlier model)

[Data/Rice_Nucleic_Data.fasta](Data/Rice_Nucleic_Data.fasta) comes from the [Ensembl Plants *Oryza sativa* cDNA dataset](https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/plants/fasta/oryza_sativa/cdna/). Sequences were prepared by:

- Removing sequences shorter than 300 bp.
- Keeping sequences with valid start and stop codons.
- Keeping sequences whose length is divisible by 3.
- Keeping sequences with a tRNA adaptation index (tAI) above 0.4.

This left **4,452 sequences**.

## Training

The mitochondria models were fine-tuned from the pretrained CodonTransformer on a single GPU on Google Colab, using [finetune.py](finetune.py). These are the settings used with full GC control; the final model relaxed the GC constraint in favor of 5′ MFE, so its exact GC settings differ:

```bash
python finetune.py \
  --dataset_dir <training_data.json> \
  --checkpoint_dir <checkpoint_dir> \
  --checkpoint_filename <name>.ckpt \
  --batch_size 8 \
  --max_epochs 10 \
  --num_workers 1 \
  --accumulate_grad_batches 1 \
  --num_gpus 1 \
  --learning_rate 1e-4 \
  --warmup_fraction 0.1 \
  --save_every_n_steps 69 \
  --seed 42 \
  --gc_penalty_weight 20 \
  --gc_target 0.4320 \
  --gc_tolerance 0.005 \
  --curriculum_epochs 2 \
  --motif_penalty_weight 0.14
```

To train a single-species model from a FASTA file, see [Docs/Single_Species_Retraining _Guide.md](Docs/Single_Species_Retraining%20_Guide.md) and [train_species_model.py](train_species_model.py).

## Making Predictions

Use [ModelDemo.ipynb](ModelDemo.ipynb). It lets you pick the mitochondria model (default) or the nucleic model, then supports single-sequence and batch prediction.

The batch workflow uses [ModelTest/nifH_data.csv](ModelTest/nifH_data.csv) as input and writes predictions to [ModelTest/rice_mito_predictions.csv](ModelTest/rice_mito_predictions.csv) (mitochondria model) or [ModelTest/rice_nucleic_predictions.csv](ModelTest/rice_nucleic_predictions.csv) (nucleic model). The output includes a `predicted_dna` column.

## Evaluation

The mitochondria models' predictions for 100 *nifH* sequences were compared with the native *nifH* sequences and with two industry codon-optimization tools, Geneious and IDT. GenScript was tried first because it has a rice option, but it has no custom codon usage table, so Geneious was used instead. These two tools are a sample of widely used optimizers, not an exhaustive comparison.

"Unconstrained model" is fine-tuned without GC control, and "Constrained model" is fine-tuned with GC control. "Final model" is [Rice_Mitochondria_Optimizer_v2](https://huggingface.co/gupta-reyansh123/Rice_Mitochondria_Optimizer_v2), trained on the gene-level data with GC control relaxed in favor of 5′ MFE.

| Dataset | GC % | GC3 % | CAI | Codon similarity to original % | Δ5′ MFE (kcal/mol) |
| --- | --- | --- | --- | --- | --- |
| Native *nifH* (nif1 + nif2) | 53.28 | 62.57 | 0.681 | n/a | n/a |
| Geneious | 44.71 | 39.58 | 0.777 | 34.51 | 0.815 |
| IDT | 44.46 | 39.21 | 0.781 | 34.21 | 0.652 |
| Unconstrained model (no GC control) | 41.59 | 29.34 | 0.857 | 32.10 | 1.039 |
| Constrained model | 44.35 | 37.85 | 0.805 | 38.78 | 2.252 |
| **Final model** | 41.85 | 31.39 | 0.819 | 38.57 | 0.020 |
| Target | 43.51 | 40.08 | 1.0 | 1.0 | 0 |

How the metrics are computed:

- **CAI** uses relative-adaptiveness weights built from [Data/Mitochondria_training.fasta](Data/Mitochondria_training.fasta); a sequence's score is the geometric mean of its codon weights.
- **Δ5′ MFE** is the average change in minimum free energy of the first 50 nucleotides, optimized minus original (ViennaRNA, 37 °C). 0 means the native 5′ structure is preserved, and smaller absolute values are better.
- **Codon similarity** is the codon-level similarity of each optimized sequence to its native original.

The final model preserves the native 5′ RNA structure far better than every other set (0.02 versus 0.65 to 2.25 kcal/mol) and matches the constrained model on codon similarity (38.57 versus 38.78). Its CAI (0.819) is above Geneious, IDT and the constrained model. Its GC and GC3 are lower than the 43.51% and 40.08% targets; this is the intended trade-off from relaxing GC control in favor of MFE. These results come from a single gene family (*nifH*) and 100 sequences.

- [dataset_comparison_summary.csv](dataset_comparison_summary.csv) holds these numbers. Regenerate it with `python compare_datasets.py`; the final model's predictions come from `python predict_checkpoint.py`.
- [ModelEvaluation/Rice_Mitochondria_Model/](ModelEvaluation/Rice_Mitochondria_Model/) has the charts for the mitochondria model.
- [ModelEvaluation/Rice_Nucleic_Model/](ModelEvaluation/Rice_Nucleic_Model/) has the charts for the nucleic model.
- [ModelTest/](ModelTest/) has the test sequences and predictions.

## Repository Contents

| Path | Purpose |
| --- | --- |
| [CodonTransformer/](CodonTransformer/) | Data, prediction, evaluation, and utility modules (based on CodonTransformer) |
| [ModelDemo.ipynb](ModelDemo.ipynb) | Prediction notebook with a model picker |
| [finetune.py](finetune.py) | Fine-tuning with GC and motif control |
| [finetune_species_model.py](finetune_species_model.py) | Fine-tune a single species directly from a FASTA file |
| [train_species_model.py](train_species_model.py) | Train a single-species model from scratch |
| [pretrain.py](pretrain.py) | Pretraining workflow |
| [compare_datasets.py](compare_datasets.py) | Compute GC, GC3, CAI, codon similarity and Δ5′ MFE for each dataset |
| [predict_checkpoint.py](predict_checkpoint.py) | Run a Lightning checkpoint on the test sequences and write predictions |
| [Data/](Data/) | Training FASTA files, NCBI source files, and the Geneious and IDT outputs |
| [ModelTest/](ModelTest/) | Test data and model predictions |
| [ModelEvaluation/](ModelEvaluation/) | Comparison charts |
| [Docs/](Docs/) | Third-party documentation (see below) |

Install the dependencies with `pip install -r requirements.txt`.

## Limitations

1. Codon optimization does not guarantee successful protein expression. A high CAI or rice-like codon profile does not prove that optimized *nifH* will be expressed or function correctly in rice.
2. *nifH* is only one part of nitrogen fixation. Functional nitrogen fixation requires additional nitrogenase components and supporting biological machinery.
3. The models are computationally trained. Their predictions are based on sequence patterns learned from existing data and do not represent the complete biological environment of a rice cell.

## License

This project is distributed under the Apache License 2.0. See [LICENSE](LICENSE).

## Third-Party Documents

Two files in [Docs/](Docs/) are not original to this project. Each has a source note at the top.

| File | Source | License |
| --- | --- | --- |
| [Docs/CodonTransformerReadme.md](Docs/CodonTransformerReadme.md) | README of [Adibvafa/CodonTransformer](https://github.com/Adibvafa/CodonTransformer) | Apache 2.0, Copyright 2024 Adibvafa Fallahpour |
| [Docs/Single_Species_Retraining _Guide.md](Docs/Single_Species_Retraining%20_Guide.md) | `NEW_SPECIES_MODEL_GUIDE.md` from [missall999/CodonTransformer](https://github.com/missall999/CodonTransformer), a fork of the project above | Apache 2.0, Copyright 2024 Adibvafa Fallahpour |

The code in [CodonTransformer/](CodonTransformer/), [pretrain.py](pretrain.py) and [finetune.py](finetune.py) is also based on CodonTransformer.

## Citations

Fallahpour, Adibvafa, et al. "CodonTransformer: A Multispecies Codon Optimizer Using Context-Aware Neural Networks." Nature Communications, vol. 16, no. 1, Apr. 2025, p. 3205, https://doi.org/10.1038/s41467-025-58588-7.

Missall. "GitHub - Missall999/CodonTransformer: CodonTransformer (1M+ Downloads); the Tool for Codon Optimization, Optimizing DNA for Protein Expression." GitHub, https://github.com/missall999/CodonTransformer.

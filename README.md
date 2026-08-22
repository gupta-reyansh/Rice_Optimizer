# Rice Optimizer

Rice Optimizer is a computational project that uses a rice-adapted CodonTransformer model to design bacterial *nifH* DNA sequences for potential expression in rice (*Oryza sativa*). The model takes a NifH protein sequence and predicts a DNA sequence that preserves the encoded protein while better reflecting codon usage patterns learned from rice coding sequences.

This project was developed as a science fair project and may also support future research work. It does not claim that an optimized sequence will be expressed successfully or that it will make rice nitrogen-fixing.

## How It Works

The project extends a pretrained CodonTransformer model to support rice as a target organism. The model was fine-tuned using rice coding sequences and then used to predict DNA sequences for bacterial *nifH* protein sequences.

![Uploading Gemini_Generated_Image_5kvhi95kvhi95kvh.jpg…]()
<img width="1024" height="559" alt="9e13abf6-ba64-4879-88a7-1bb6623e61ea" src="https://github.com/user-attachments/assets/2c4bb9da-6008-48ec-a711-2aa2fdefebda" />

The model uses sequence context rather than selecting codons independently at each position. This allows the prediction to reflect patterns learned from complete coding sequences.

## Data

The training data came from the [Ensembl Plants *Oryza sativa* cDNA dataset](https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/plants/fasta/oryza_sativa/cdna/).

The sequences were prepared by:

- Removing small sequences(<300 bp).
- Keeping sequences with valid start and stop codons.
- Keeping sequences with bp length divisible by 3
- Keeping sequences with a transfer RNA adaptation index (tAI) above 0.4.

## Installation

Create and activate a Python environment, then install the required packages:

```powershell
pip install -r requirements.txt
```

## Making Predictions

Use [CodonTransformerDemo.ipynb](CodonTransformerDemo.ipynb) to make predictions. The notebook includes single-sequence and batch prediction workflows.

The batch workflow uses [ModelTest/test_data.csv](ModelTest/test_data.csv) as input and writes predictions to [ModelTest/model_predictions.csv](ModelTest/model_predictions.csv). The prediction output includes a `predicted_dna` column.

## Training a Species Model

The [train_species_model.py](train_species_model.py) script shows the training method used to prepare and train a single-species model from a CDS FASTA file. Researchers can adapt it for another organism or training dataset.

```powershell
python train_species_model.py --input_fasta <CDS_FASTA> --organism "<organism>"
```

The script filters and prepares the FASTA records, creates training data, and launches the training workflow. Additional options are available through:

```powershell
python train_species_model.py --help
```

## Evaluation

- [ModelTest](ModelTest) contains the test sequences and Rice Optimizer predictions.
- [ModelComparison](ModelComparison) contains the figures comparing this model with other online codon-optimization solutions.

## Repository Contents

| Path | Purpose |
| --- | --- |
| [CodonTransformer/](CodonTransformer/) | Model data, prediction, evaluation, and utility modules |
| [CodonTransformerDemo.ipynb](CodonTransformerDemo.ipynb) | Main prediction notebook |
| [train_species_model.py](train_species_model.py) | Species-model training workflow |
| [fasta_to_training_csv.py](fasta_to_training_csv.py) | FASTA-to-training-data preparation script |
| [pretrain.py](pretrain.py) | Pretraining workflow |
| [finetune.py](finetune.py) | Fine-tuning workflow |
| [ModelTest/](ModelTest/) | Test data and model predictions |
| [ModelComparison/](ModelComparison/) | Comparison results |

## Limitations

1. Codon optimization does not guarantee successful protein expression. A high CAI or rice-like codon profile does not prove that optimized *nifH* will be expressed or function correctly in rice.
2. *nifH* is only one part of nitrogen fixation. Functional nitrogen fixation requires additional nitrogenase components and supporting biological machinery.
3. The model is computationally trained. Its predictions are based on sequence patterns learned from existing data and do not represent the complete biological environment of a rice cell.
4. Rice is not a native target organism in the original CodonTransformer. Extending the model to support *Oryza sativa* and fine-tuning it with rice coding sequences may introduce implementation errors or be limited by the available rice-specific training data.
5. Available *nifH* sequences are biologically diverse. Differences in GC content, codon usage, and evolutionary history may affect model performance across sequence groups.

## License

This project is distributed under the terms of the license in [LICENSE](LICENSE).

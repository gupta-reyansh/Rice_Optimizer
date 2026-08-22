# Rice Optimizer

Rice Optimizer is a computational project that uses a rice-adapted CodonTransformer model to design bacterial *nifH* DNA sequences for potential expression in rice (*Oryza sativa*). The model takes a NifH protein sequence and predicts a DNA sequence that preserves the encoded protein while better reflecting codon usage patterns learned from rice coding sequences.

This project was developed as a science fair project and may also support future research work. It does not claim that an optimized sequence will be expressed successfully or that it will make rice nitrogen-fixing.

## How It Works

The project extends a pretrained CodonTransformer model to support rice as a target organism. The model was fine-tuned using rice coding sequences and then used to predict DNA sequences for bacterial *nifH* protein sequences.

<img width="453" height="686" alt="Gemini_Generated_Image_5kvhi95kvhi95kvh" src="https://github.com/user-attachments/assets/b63e7979-22b9-4aa3-aff8-3ca1c4c05d44" />

The model uses sequence context rather than selecting codons independently at each position. This allows the prediction to reflect patterns learned from complete coding sequences.

## Data

The training data came from the [Ensembl Plants *Oryza sativa* cDNA dataset](https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/plants/fasta/oryza_sativa/cdna/).

The sequences were prepared by:

- Removing small sequences(<300 bp).
- Keeping sequences with valid start and stop codons.
- Keeping sequences with bp length divisible by 3
- Keeping sequences with a transfer RNA adaptation index (tAI) above 0.4.

## Making Predictions

Use [CodonTransformerDemo.ipynb](CodonTransformerDemo.ipynb) to make predictions. The notebook includes single-sequence and batch prediction workflows.

The batch workflow uses [ModelTest/test_data.csv](ModelTest/test_data.csv) as input and writes predictions to [ModelTest/model_predictions.csv](ModelTest/model_predictions.csv). The prediction output includes a `predicted_dna` column.

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

## Citations

Fallahpour, Adibvafa, et al. "CodonTransformer: A Multispecies Codon Optimizer Using Context-Aware Neural Networks." Nature Communications, vol. 16, no. 1, Apr. 2025, p. 3205, https://doi.org/10.1038/s41467-025-58588-7.

Missall. “GitHub - Missall999/CodonTransformer: CodonTransformer (1M+ Downloads); the Tool for Codon Optimization, Optimizing DNA for Protein Expression.” GitHub, github.com/missall999/CodonTransformer.

"""
Run a Lightning .ckpt (e.g. rice_mito.ckpt) on ModelTest/nifH_data.csv and write a
predictions CSV with a `predicted_dna` column, using the same settings as the batch
cell in ModelDemo.ipynb.

Usage:
    python predict_checkpoint.py [--checkpoint rice_mito.ckpt] \
        [--input ModelTest/nifH_data.csv] [--output ModelTest/rice_mito_predictions.csv]
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonPrediction import predict_dna_sequence

BASE_DIR = Path(__file__).resolve().parent
TOKENIZER = "adibvafa/CodonTransformer"
# Architecture of the fine-tuned rice mitochondria model (2048 positions, 1 token type);
# the base CodonTransformer config (4096 positions, 164 token types) does not match.
CONFIG = "gupta-reyansh123/Rice_Mitochondria_Optimizer"


def load_model(checkpoint: Path, device: torch.device) -> BigBirdForMaskedLM:
    """Load BigBird weights from a Lightning checkpoint (keys prefixed 'model.')."""
    model = BigBirdForMaskedLM(AutoConfig.from_pretrained(CONFIG))
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "rice_mito.ckpt"))
    parser.add_argument("--input", default=str(BASE_DIR / "ModelTest" / "nifH_data.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "ModelTest" / "rice_mito_predictions.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = load_model(Path(args.checkpoint), device)

    dataset = pd.read_csv(args.input, index_col=0)
    dataset["predicted_dna"] = None
    for index, data in tqdm(dataset.iterrows(), total=len(dataset), unit=" Sequences"):
        output = predict_dna_sequence(
            protein=data["protein"],
            organism=0,
            device=device,
            tokenizer=tokenizer,
            model=model,
        )
        dataset.loc[index, "predicted_dna"] = output.predicted_dna

    dataset.to_csv(args.output)
    print(f"Wrote {len(dataset)} predictions to {args.output}")


if __name__ == "__main__":
    main()

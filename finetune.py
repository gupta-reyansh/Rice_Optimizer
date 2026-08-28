"""
File: finetune.py
-------------------
Finetune the CodonTransformer model.

The pretrained model is loaded directly from Hugging Face.
The dataset is a JSON file. You can use prepare_training_data from CodonData to
prepare the dataset. The repository README has a guide on how to prepare the
dataset and use this script.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonUtils import (
    MAX_LEN,
    TOKEN2MASK,
    IterableJSONData,
)


class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _ensure_mask_per_example(selected, inputs):
        eligible = inputs >= 5
        missing = ~selected.any(dim=1) & eligible.any(dim=1)
        if not missing.any():
            return selected

        rows = missing.nonzero(as_tuple=False).flatten()
        cols = eligible[missing].to(dtype=torch.int64).argmax(dim=1)
        selected[rows, cols] = True
        return selected

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = tokenized["input_ids"].clone()

        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[torch.where(inputs < 5)] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()
        selected = self._ensure_mask_per_example(selected, inputs)

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        if replaced.any():
            inputs[replaced] = torch.tensor(
                list(map(TOKEN2MASK.__getitem__, inputs[replaced].tolist())),
                dtype=inputs.dtype,
            )

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        return tokenized


class plTrainHarness(pl.LightningModule):
    def __init__(self, model, learning_rate, warmup_fraction):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError, OverflowError):
            return optimizer

        if total_steps <= 0:
            return optimizer

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.warmup_fraction,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss


class DumpStateDict(pl.callbacks.ModelCheckpoint):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__(
            dirpath=checkpoint_dir, every_n_train_steps=every_n_train_steps
        )
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        wrapped_model = getattr(trainer.model, "module", trainer.model)
        model = getattr(wrapped_model, "model", wrapped_model)
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )


class LossHistoryPlotter(pl.Callback):
    def __init__(self, output_dir, csv_filename="training_loss_history.csv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_filename
        self.png_path = self.output_dir / "training_loss_history.png"
        self.history = []

    def add_loss(self, step, loss):
        if step is None or loss is None:
            return
        self.history.append({"step": int(step), "loss": float(loss)})
        self._save_csv()
        self._save_plot()

    def _save_csv(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "loss"])
            for row in self.history:
                writer.writerow([row["step"], row["loss"]])

    def _save_plot(self):
        if not self.history:
            return

        steps = [entry["step"] for entry in self.history]
        losses = [entry["loss"] for entry in self.history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, color="tab:blue", linewidth=2)
        ax.set_title("Training Loss Over Time")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=150)
        plt.close(fig)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("loss")
        if loss is None:
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]

        if loss is None:
            return

        if hasattr(loss, "detach"):
            loss = loss.detach().cpu()
        self.add_loss(trainer.global_step, float(loss))

    def on_train_end(self, trainer, pl_module):
        self._save_csv()
        self._save_plot()


def main(args):
    """Finetune the CodonTransformer model."""
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("gupta-reyansh123/Rice_Optimizer")
    harnessed_model = plTrainHarness(model, args.learning_rate, args.warmup_fraction)

    # Load the training data
    train_data = IterableJSONData(args.dataset_dir, dist_env="slurm")
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=0 if args.debug else args.num_workers,
        persistent_workers=False if args.debug else True,
    )

    # Setup trainer and callbacks
    save_checkpoint = DumpStateDict(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        every_n_train_steps=args.save_every_n_steps,
    )
    loss_plotter = LossHistoryPlotter(output_dir=args.checkpoint_dir)
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=1 if args.debug else args.num_gpus,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=[save_checkpoint, loss_plotter],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Finetune the model
    trainer.fit(harnessed_model, data_loader)
    print(f"Loss history CSV: {os.path.join(args.checkpoint_dir, 'training_loss_history.csv')}")
    print(f"Loss plot: {os.path.join(args.checkpoint_dir, 'training_loss_history.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the CodonTransformer model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Filename for the saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=512,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)

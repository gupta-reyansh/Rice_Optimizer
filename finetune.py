"""
File: finetune.py
------------------
Fine-tune the CodonTransformer model on rice (Oryza sativa) JSON datasets
prepared via CodonData.prepare_training_data. The pretrained base model is
loaded from Hugging Face. See README for usage details.
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM, logging as hf_logging

import torch.nn.functional as F
from CodonTransformer.CodonUtils import (
    C_indices,
    G_indices,
    MAX_LEN,
    TOKEN2MASK,
    IterableJSONData,
)

# Reduce excessive INFO logs from transformers
hf_logging.set_verbosity_warning()

class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list((map(TOKEN2MASK.__getitem__, inputs[replaced].numpy())))
        ).long()

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
        # Unmasked codon ids, used by the GC loss to rebuild the fully-masked (inference) input.
        tokenized["original_ids"] = targets

        return tokenized


class plTrainHarness(pl.LightningModule):
    """
    PyTorch Lightning training harness for the rice CodonTransformer with Augmented-Lagrangian Method (ALM) GC control.

    GC is measured under inference conditions: a second forward pass masks every codon down to its
    amino acid, and the expected GC is taken over the synonymous codons only (renormalised), per
    sequence. This is the same quantity that decoding produces, so the penalty cannot be reduced by
    moving probability between amino acids or by copying visible codons.

    This class implements the training loop for fine-tuning CodonTransformer on rice sequences
    with precise GC content control using an Augmented-Lagrangian Method. The ALM approach allows
    the model to learn codon preferences while maintaining GC content within a target range (e.g., 43.61%
    for rice).

    Key features:
    - Masked language modeling (MLM) loss for codon prediction
    - ALM-based GC content constraint enforcement
    - Curriculum learning: warm-up epochs before enforcing GC constraints
    - Adaptive penalty coefficient (rho) adjustment based on constraint violation progress

    The ALM method minimizes: L = L_MLM + λ·(GC - μ) + (ρ/2)(GC - μ)²
    where λ is the Lagrangian multiplier and ρ is the penalty coefficient.

    Args:
        model: BigBirdForMaskedLM model to train
        learning_rate: Learning rate for optimizer
        warmup_fraction: Fraction of training steps for warmup
        gc_penalty_weight: Weight for simple GC penalty (legacy, unused if ALM enabled)
        tokenizer: Tokenizer for the model
        gc_target: Target GC content (default: 0.4361 for rice)
        use_lagrangian: Enable ALM method (default: False)
        lagrangian_rho: Initial penalty coefficient for ALM (default: 10.0)
        curriculum_epochs: Number of warm-up epochs before enforcing GC constraint (default: 3)
        alm_tolerance: Primal tolerance for ALM convergence (default: 1e-5)
        alm_dual_tolerance: Dual tolerance for constraint violation (default: 1e-5)
        alm_penalty_update_factor: Factor for updating penalty coefficient (default: 10.0)
        alm_initial_penalty_factor: Initial penalty coefficient (default: 20.0)
        alm_tolerance_update_factor: Factor for updating tolerance (default: 0.1)
        alm_rel_penalty_increase_threshold: Relative improvement threshold for penalty updates (default: 0.1)
        alm_max_penalty: Maximum penalty value to prevent ill-conditioning (default: 1e6)
        alm_min_penalty: Minimum penalty value (default: 1e-6)
        gc3_target: Optional target for GC at the third codon position (default: None = not enforced)
        gc_tolerance: Half-width of the no-penalty band around the GC targets (default: 0.02)
        gc_temperature: Softmax temperature used only when measuring GC. Below 1 sharpens the
            distribution toward its mode, approximating argmax decoding (default: 1.0)
    """
    def __init__(self, model, learning_rate, warmup_fraction, gc_penalty_weight, tokenizer,
                 gc_target=0.4361, use_lagrangian=False, lagrangian_rho=10.0, curriculum_epochs=3,
                 alm_tolerance=1e-5, alm_dual_tolerance=1e-5, alm_penalty_update_factor=10.0,
                 alm_initial_penalty_factor=20.0, alm_tolerance_update_factor=0.1,
                 alm_rel_penalty_increase_threshold=0.1, alm_max_penalty=1e6, alm_min_penalty=1e-6,
                 gc3_target=None, gc_tolerance=0.02, gc_temperature=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.gc_penalty_weight = gc_penalty_weight
        self.tokenizer = tokenizer

        # Augmented-Lagrangian GC Control parameters
        self.gc_target = gc_target
        self.gc3_target = gc3_target
        self.gc_tolerance = gc_tolerance
        self.gc_temperature = gc_temperature
        self.use_lagrangian = use_lagrangian
        self.lagrangian_rho = lagrangian_rho
        self.curriculum_epochs = curriculum_epochs

        # Enhanced ALM parameters (inspired by alpaqa research)
        self.alm_tolerance = alm_tolerance
        self.alm_dual_tolerance = alm_dual_tolerance
        self.alm_penalty_update_factor = alm_penalty_update_factor
        self.alm_initial_penalty_factor = alm_initial_penalty_factor
        self.alm_tolerance_update_factor = alm_tolerance_update_factor
        self.alm_rel_penalty_increase_threshold = alm_rel_penalty_increase_threshold
        self.alm_max_penalty = alm_max_penalty
        self.alm_min_penalty = alm_min_penalty

        # Initialize Lagrangian multiplier as buffer (persists across checkpoints)
        self.register_buffer("lambda_gc", torch.tensor(0.0))

        # Adaptive penalty coefficient (rho) - starts as parameter, becomes adaptive
        self.register_buffer("rho_adaptive", torch.tensor(self.lagrangian_rho))

        # Step counter for periodic lambda updates
        self.register_buffer("step_counter", torch.tensor(0))

        # ALM convergence tracking
        self.register_buffer("previous_constraint_violation", torch.tensor(float('inf')))
        self.register_buffer("constraint_violation_history", torch.zeros(10))  # Track last 10 values
        self.register_buffer("alm_iteration_counter", torch.tensor(0))

        # Configure BigBird to use sparse attention (set once to avoid per-step prints)
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'set_attention_type'):
            try:
                self.model.bert.set_attention_type("block_sparse")
            except Exception:
                # Fallback silently if method missing
                pass

        # Create GC lookup table for codons
        self._create_gc_lookup_table()

    def _create_gc_lookup_table(self):
        """Build the token-level lookups the GC loss needs (all registered so they follow the model to GPU).

        gc_lookup_tensor:   token index -> GC fraction of the codon
        gc3_lookup_tensor:  token index -> 1.0 if the third base is G/C
        mask_lookup_tensor: token index -> amino-acid "unk" token (the fully-masked inference input)
        synonym_mask:       [amino-acid unk token, codon token] -> True if the codon encodes that amino acid
        """
        from CodonTransformer.CodonUtils import TOKEN2INDEX

        vocab_size = len(TOKEN2INDEX)
        gc_lookup = torch.zeros(vocab_size)
        gc3_lookup = torch.zeros(vocab_size)
        mask_lookup = torch.arange(vocab_size)
        synonym_mask = torch.zeros(vocab_size, vocab_size, dtype=torch.bool)

        for token, idx in TOKEN2INDEX.items():
            if idx >= 26:  # codon tokens only; 0-25 are special and "<aa>_unk" tokens
                # Last three characters are the codon (e.g. "k_aaa" -> "AAA", "__taa" -> "TAA")
                codon = token[-3:].upper()
                gc_lookup[idx] = (codon.count('G') + codon.count('C')) / 3.0
                gc3_lookup[idx] = float(codon[2] in "GC")
                mask_lookup[idx] = TOKEN2MASK[idx]
                synonym_mask[TOKEN2MASK[idx], idx] = True

        self.register_buffer("gc_lookup_tensor", gc_lookup)
        self.register_buffer("gc3_lookup_tensor", gc3_lookup, persistent=False)
        self.register_buffer("mask_lookup_tensor", mask_lookup, persistent=False)
        self.register_buffer("synonym_mask", synonym_mask, persistent=False)

    def _inference_condition_gc(self, model_inputs, original_ids):
        """Per-sequence expected GC and GC3 when every codon is masked to its amino acid.

        Mirrors decoding: only the synonymous codons of each position compete, so the
        result is what the model would produce for a protein it is asked to optimise.
        """
        attention_mask = model_inputs["attention_mask"]
        valid = (original_ids >= 26) & (attention_mask == 1)  # codon positions only
        masked_ids = self.mask_lookup_tensor[original_ids]
        logits = self.model(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            token_type_ids=model_inputs["token_type_ids"],
        ).logits.float()[valid]
        synonyms = self.synonym_mask[masked_ids[valid]]
        probs = torch.softmax(logits.masked_fill(~synonyms, float("-inf")) / self.gc_temperature, dim=-1)

        seq_index = valid.nonzero()[:, 0]
        counts = valid.sum(dim=1).clamp(min=1)
        zeros = torch.zeros(valid.shape[0], device=probs.device)
        seq_gc = zeros.index_add(0, seq_index, probs @ self.gc_lookup_tensor) / counts
        seq_gc3 = zeros.index_add(0, seq_index, probs @ self.gc3_lookup_tensor) / counts
        return seq_gc, seq_gc3

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # CosineAnnealingWarmRestarts scheduler
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.trainer.estimated_stepping_batches // 4),  # First restart after 1/4 of training
                T_mult=2,  # Double the restart period each time
                eta_min=self.learning_rate * 0.01,  # Minimum learning rate (1% of max)
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        """
        Training step with ALM-based GC content control.

        This method implements the core training loop:
        1. Forward pass through the model to get MLM loss
        2. Second forward pass with every codon masked to its amino acid; expected GC/GC3 is taken
           over the synonymous codons only, per sequence (what decoding will produce)
        3. Apply the GC penalty (hinge, or ALM if enabled) after the curriculum warm-up
        4. Update Lagrangian multiplier and penalty coefficient adaptively

        The ALM loss combines:
        - MLM loss: standard masked language modeling loss
        - GC constraint: λ·(GC - μ) + (ρ/2)(GC - μ)²

        Args:
            batch: Batch of tokenized sequences with labels
            batch_idx: Batch index

        Returns:
            Total loss (MLM + GC constraint)
        """
        # Forward pass (MLM). original_ids is only needed by the GC loss.
        original_ids = batch["original_ids"]
        model_inputs = {k: v for k, v in batch.items() if k != "original_ids"}
        outputs = self.model(**model_inputs)
        mlm_loss = outputs.loss

        # Increment step counter
        self.step_counter += 1

        # GC control, measured under inference conditions (see _inference_condition_gc)
        gc_loss = 0
        if self.use_lagrangian or self.gc_penalty_weight > 0:
            # Enforced only after the curriculum warm-up; the extra forward pass is skipped before that
            if self.current_epoch >= self.curriculum_epochs:
                seq_gc, seq_gc3 = self._inference_condition_gc(model_inputs, original_ids)
                mean_gc = seq_gc.mean()

                # Log the GC that decoding will actually see (name kept for the monitoring callbacks)
                self.log("mean_gc_window", mean_gc, on_step=True, prog_bar=True)
                self.log("mean_gc3", seq_gc3.mean().detach(), on_step=True, prog_bar=True)

                if self.use_lagrangian:
                    # Enhanced Self-Tuning Augmented-Lagrangian approach
                    gc_deviation = mean_gc - self.gc_target
                    current_violation = torch.abs(gc_deviation)

                    # Update constraint violation history for adaptive penalty adjustment
                    new_history = torch.zeros_like(self.constraint_violation_history)
                    new_history[:-1] = self.constraint_violation_history[1:]
                    new_history[-1] = current_violation
                    self.constraint_violation_history = new_history

                    # Self-tuning penalty coefficient (rho) update - inspired by alpaqa
                    if self.step_counter % 20 == 0 and self.step_counter > 0:
                        # Check if constraint violation is improving
                        violation_improvement = self.previous_constraint_violation - current_violation
                        relative_improvement = violation_improvement / max(self.previous_constraint_violation, 1e-8)

                        # Adaptive rho update based on constraint violation progress
                        if current_violation > self.alm_dual_tolerance:
                            # If violation is still too high, check if we're making progress
                            if relative_improvement < self.alm_rel_penalty_increase_threshold:
                                # Not improving fast enough, increase penalty
                                new_rho = self.rho_adaptive * self.alm_penalty_update_factor
                                self.rho_adaptive = torch.clamp(new_rho, self.alm_min_penalty, self.alm_max_penalty)

                                # Update Lagrangian multiplier
                                self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()
                            else:
                                # Making good progress, just update multiplier
                                self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()
                        else:
                            # Violation is acceptable, just update multiplier
                            self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()

                        # Update previous violation for next iteration
                        self.previous_constraint_violation = current_violation
                        self.alm_iteration_counter += 1

                    # Augmented-Lagrangian loss: λ·(mean_gc - μ) + (ρ/2)(mean_gc - μ)²
                    lagrangian_term = self.lambda_gc * gc_deviation
                    penalty_term = (self.rho_adaptive / 2) * (gc_deviation ** 2)
                    gc_loss = lagrangian_term + penalty_term

                    # Enhanced logging for ALM system monitoring
                    self.log("lambda_gc", self.lambda_gc, on_step=True, prog_bar=True)
                    self.log("rho_adaptive", self.rho_adaptive, on_step=True, prog_bar=True)
                    self.log("gc_deviation", gc_deviation, on_step=True, prog_bar=True)
                    self.log("constraint_violation", current_violation, on_step=True, prog_bar=False)
                    self.log("alm_iteration", self.alm_iteration_counter, on_step=True, prog_bar=False)

                else:
                    # Hinge penalty (no multiplier, so no windup): zero inside the tolerance band,
                    # linear outside it, applied to each sequence separately
                    gc_loss = F.relu(torch.abs(seq_gc - self.gc_target) - self.gc_tolerance).mean()

                if self.gc3_target is not None:
                    gc_loss = gc_loss + F.relu(torch.abs(seq_gc3 - self.gc3_target) - self.gc_tolerance).mean()

                self.log("gc_loss", gc_loss, on_step=True, prog_bar=True)

        # Combine losses
        if self.use_lagrangian:
            total_loss = mlm_loss + gc_loss
        else:
            total_loss = mlm_loss + self.gc_penalty_weight * gc_loss

        self.log_dict(
            dictionary={
                "loss": total_loss,
                "mlm_loss": mlm_loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return total_loss


class DumpStateDict(pl.Callback):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__()
        self.dirpath = checkpoint_dir
        self.every_n_train_steps = every_n_train_steps
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = pl_module.model
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )


class ALMMonitoringCallback(pl.Callback):
    """Monitor ALM behavior and log convergence metrics."""

    def __init__(self, log_every_n_steps=20, convergence_window=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.convergence_window = convergence_window

        # Track ALM convergence metrics
        self.lambda_history = []
        self.rho_history = []
        self.violation_history = []
        self.step_history = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor ALM system on each training step."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_step = trainer.global_step

            # Log ALM metrics every N steps
            if current_step % self.log_every_n_steps == 0:
                # Extract ALM state from logged metrics
                lambda_gc = trainer.logged_metrics.get('lambda_gc', 0)
                rho_adaptive = trainer.logged_metrics.get('rho_adaptive', 0)
                constraint_violation = trainer.logged_metrics.get('constraint_violation', 0)
                gc_deviation = trainer.logged_metrics.get('gc_deviation', 0)

                # Store history for convergence analysis
                self.lambda_history.append(float(lambda_gc))
                self.rho_history.append(float(rho_adaptive))
                self.violation_history.append(float(constraint_violation))
                self.step_history.append(current_step)

                # Keep only recent history for efficiency
                if len(self.lambda_history) > self.convergence_window:
                    self.lambda_history = self.lambda_history[-self.convergence_window:]
                    self.rho_history = self.rho_history[-self.convergence_window:]
                    self.violation_history = self.violation_history[-self.convergence_window:]
                    self.step_history = self.step_history[-self.convergence_window:]

                # Log ALM metrics to TensorBoard
                if trainer.logger is not None:
                    # Primary ALM metrics
                    trainer.logger.log_metrics({
                        'alm/lambda_gc': float(lambda_gc),
                        'alm/rho_adaptive': float(rho_adaptive),
                        'alm/constraint_violation': float(constraint_violation),
                        'alm/gc_deviation': float(gc_deviation),
                    }, step=current_step)

                    # Convergence analysis (if we have sufficient history)
                    if len(self.violation_history) >= 10:
                        recent_violations = self.violation_history[-10:]
                        violation_trend = recent_violations[-1] - recent_violations[0]
                        violation_stability = max(recent_violations) - min(recent_violations)

                        trainer.logger.log_metrics({
                            'alm/violation_trend': violation_trend,
                            'alm/violation_stability': violation_stability,
                            'alm/rho_growth_rate': self.rho_history[-1] / max(self.rho_history[0], 1e-8),
                        }, step=current_step)

    def on_train_epoch_end(self, trainer, pl_module):
        """Comprehensive ALM system analysis at epoch end."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_epoch = trainer.current_epoch

            # Only analyze after curriculum warm-up period
            if current_epoch >= pl_module.curriculum_epochs:
                # Get current ALM state
                lambda_gc = trainer.logged_metrics.get('lambda_gc', 0)
                rho_adaptive = trainer.logged_metrics.get('rho_adaptive', 0)
                constraint_violation = trainer.logged_metrics.get('constraint_violation', 0)
                mean_gc = trainer.logged_metrics.get('mean_gc_window', 0)

                # ALM convergence assessment
                converged = float(constraint_violation) <= pl_module.alm_dual_tolerance

                # Detailed epoch summary
                print(f"\n{'='*60}")
                print(f"ALM System Analysis - Epoch {current_epoch}")
                print(f"{'='*60}")
                print("Current State:")
                print(f"   • GC Content: {float(mean_gc):.4f} (target: {pl_module.gc_target:.4f})")
                print(f"   • Constraint Violation: {float(constraint_violation):.2e}")
                print(f"   • Lambda (Multiplier): {float(lambda_gc):.4f}")
                print(f"   • Rho (Penalty): {float(rho_adaptive):.2e}")
                print(f"   • Converged: {'Yes' if converged else 'No'}")

                # Convergence diagnostics
                if len(self.violation_history) >= 5:
                    recent_violations = self.violation_history[-5:]
                    improvement_rate = (recent_violations[0] - recent_violations[-1]) / max(recent_violations[0], 1e-8)

                    print("Convergence Diagnostics:")
                    print(f"   • Recent Improvement Rate: {improvement_rate:.2%}")
                    print(f"   • Penalty Growth: {self.rho_history[-1] / max(self.rho_history[0], 1e-8):.2f}x")
                    print(f"   • Stability: {'Good' if max(recent_violations) - min(recent_violations) < 1e-3 else 'Improving'}")

                print(f"{'='*60}\n")

                # TensorBoard epoch summary
                if trainer.logger is not None:
                    trainer.logger.log_metrics({
                        'alm_epoch/converged': 1.0 if converged else 0.0,
                        'alm_epoch/final_lambda': float(lambda_gc),
                        'alm_epoch/final_rho': float(rho_adaptive),
                        'alm_epoch/final_violation': float(constraint_violation),
                    }, step=current_epoch)


class GCValidationHook(pl.Callback):
    """Validation hook to monitor GC content during training."""

    def __init__(self, gc_target=0.4361, tolerance=0.02):
        super().__init__()
        self.gc_target = gc_target
        self.tolerance = tolerance
        self.gc_target_min = gc_target - tolerance
        self.gc_target_max = gc_target + tolerance

    def on_train_epoch_end(self, trainer, pl_module):
        """Check GC content at the end of each epoch."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_epoch = trainer.current_epoch

            # Only validate after curriculum warm-up period
            if current_epoch >= pl_module.curriculum_epochs:
                # Get the logged mean GC content from the last step
                if 'mean_gc_window' in trainer.logged_metrics:
                    current_gc = trainer.logged_metrics.get('mean_gc_window', None)

                    if current_gc is not None:
                        current_gc_val = float(current_gc)

                        # Log validation status
                        within_target = self.gc_target_min <= current_gc_val <= self.gc_target_max

                        if within_target:
                            print(
                                f"Epoch {current_epoch}: GC content {current_gc_val:.3f} is within target range "
                                f"[{self.gc_target_min:.3f}, {self.gc_target_max:.3f}]"
                            )
                        else:
                            print(
                                f"Epoch {current_epoch}: GC content {current_gc_val:.3f} is outside target range "
                                f"[{self.gc_target_min:.3f}, {self.gc_target_max:.3f}]"
                            )

                        # Log lambda value if available
                        if 'lambda_gc' in trainer.logged_metrics:
                            lambda_val = float(trainer.logged_metrics.get('lambda_gc', 0))
                            print(f"   Lambda: {lambda_val:.4f}")




def main(args):
    """Finetune the CodonTransformer model."""
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("gupta-reyansh123/Rice_Optimizer")
    harnessed_model = plTrainHarness(
        model, args.learning_rate, args.warmup_fraction, args.gc_penalty_weight, tokenizer,
        gc_target=args.gc_target, use_lagrangian=args.use_lagrangian,
        lagrangian_rho=args.lagrangian_rho, curriculum_epochs=args.curriculum_epochs,
        alm_tolerance=args.alm_tolerance, alm_dual_tolerance=args.alm_dual_tolerance,
        alm_penalty_update_factor=args.alm_penalty_update_factor,
        alm_initial_penalty_factor=args.alm_initial_penalty_factor,
        alm_tolerance_update_factor=args.alm_tolerance_update_factor,
        alm_rel_penalty_increase_threshold=args.alm_rel_penalty_increase_threshold,
        alm_max_penalty=args.alm_max_penalty, alm_min_penalty=args.alm_min_penalty,
        gc3_target=args.gc3_target, gc_tolerance=args.gc_tolerance,
        gc_temperature=args.gc_temperature,
    )

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
    gc_validation = GCValidationHook(
        gc_target=args.gc_target,
        tolerance=0.02  # 2% tolerance around target
    )

    # Enhanced ALM monitoring callback for comprehensive system analysis
    alm_monitor = ALMMonitoringCallback(
        log_every_n_steps=args.log_every_n_steps,
        convergence_window=50  # Track last 50 steps for convergence analysis
    )

    callbacks = [save_checkpoint, gc_validation, alm_monitor]

    # Determine accelerator and device configuration dynamically
    if args.num_gpus > 0:
        accelerator = "gpu"
        devices = args.num_gpus
    else:
        # Fallback to CPU training when --num_gpus 0
        accelerator = "cpu"
        devices = 1  # Lightning expects at least one device

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy=("ddp_find_unused_parameters_true" if accelerator == "gpu" and devices > 1 else "auto"),
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
    )

    # Finetune the model
    trainer.fit(harnessed_model, data_loader)


def build_parser():
    """Argument parser for finetune.py (also used by finetune_species_model.py)."""
    parser = argparse.ArgumentParser(description="Fine-tune CodonTransformer for rice codon optimization.")
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
        "--num_workers", type=int, default=3, help="Number of workers for data loading"
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
    parser.add_argument(
        "--gc_penalty_weight",
        type=float,
        default=10,
        help="Weight for the GC content penalty in the loss function",
    )
    parser.add_argument(
        "--gc_target",
        type=float,
        default=0.4361,
        help="Target GC content as a fraction (default: 0.4361, i.e. 43.61%% for rice)",
    )
    parser.add_argument(
        "--gc3_target",
        type=float,
        default=None,
        help="Optional target for GC at the third codon position, as a fraction (default: not enforced)",
    )
    parser.add_argument(
        "--gc_tolerance",
        type=float,
        default=0.005,
        help="No-penalty band around the GC targets, as a fraction (default: 0.005, i.e. 0.5%% GC)",
    )
    parser.add_argument(
        "--gc_temperature",
        type=float,
        default=1.0,
        help="Temperature for measuring GC in the loss. Use <1 (e.g. 0.5) if you decode with argmax, "
             "so the penalty sees the sharpened distribution argmax picks from (default: 1.0)",
    )
    parser.add_argument(
        "--use_lagrangian",
        action="store_true",
        help="Use Augmented-Lagrangian method for GC control",
    )
    parser.add_argument(
        "--lagrangian_rho",
        type=float,
        default=10.0,
        help="Penalty coefficient for Augmented-Lagrangian method",
    )
    parser.add_argument(
        "--curriculum_epochs",
        type=int,
        default=3,
        help="Number of warm-up epochs before enforcing GC constraints",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="How often to log metrics (in training steps)",
    )

    # Enhanced ALM parameters for self-tuning GC constraint system
    parser.add_argument(
        "--alm_tolerance",
        type=float,
        default=1e-5,
        help="Primal tolerance for ALM inner solver stopping criterion",
    )
    parser.add_argument(
        "--alm_dual_tolerance",
        type=float,
        default=1e-5,
        help="Dual tolerance for ALM constraint violation",
    )
    parser.add_argument(
        "--alm_penalty_update_factor",
        type=float,
        default=10.0,
        help="Factor for updating ALM penalty parameters (rho_update_factor)",
    )
    parser.add_argument(
        "--alm_initial_penalty_factor",
        type=float,
        default=20.0,
        help="Factor for automatic ALM penalty initialization (init_rho)",
    )
    parser.add_argument(
        "--alm_tolerance_update_factor",
        type=float,
        default=0.1,
        help="Factor for updating ALM primal tolerance",
    )
    parser.add_argument(
        "--alm_rel_penalty_increase_threshold",
        type=float,
        default=0.1,
        help="Relative threshold for ALM penalty increases (gc_tolerance)",
    )
    parser.add_argument(
        "--alm_max_penalty",
        type=float,
        default=1e6,
        help="Maximum ALM penalty value to prevent ill-conditioning",
    )
    parser.add_argument(
        "--alm_min_penalty",
        type=float,
        default=1e-6,
        help="Minimum ALM penalty value",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)

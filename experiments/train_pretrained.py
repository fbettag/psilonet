"""
Training script for pre-trained psychedelic SmolLM2.

Two-stage fine-tuning:
1. Stage 1: Freeze baseline, train skip layers only (5 epochs)
2. Stage 2: Unfreeze all, full fine-tuning (10 epochs)

Usage:
    # Stage 1
    python experiments/train_pretrained.py --stage 1 --epochs 5 --freeze_baseline

    # Stage 2
    python experiments/train_pretrained.py --stage 2 --epochs 10 --load_checkpoint stage1
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm
import time
import json

from modules.pretrained_psychedelic import create_pretrained_psychedelic_model
from modules.psychedelic_smollm import load_tokenizer
from experiments.utils import (
    ExperimentTracker,
    compute_perplexity,
    print_training_header,
    print_epoch_stats,
    batch_iterator,
    estimate_tokens_per_second,
)


def prepare_dataset(dataset_name: str = "wikitext", split: str = "train[:5000]", tokenizer=None):
    """Prepare dataset for training."""
    print(f"📚 Loading dataset: {dataset_name} ({split})")

    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    print(f"✅ Loaded {len(dataset)} examples")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="np"
        )

    print("🔤 Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    print(f"✅ Dataset ready: {len(tokenized)} examples")
    return tokenized


def train_epoch(model, dataset, optimizer, batch_size: int = 16):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0
    num_tokens = 0
    start_time = time.time()

    # Create batches
    dataset_list = list(dataset)
    batches = list(batch_iterator(dataset_list, batch_size))

    for batch in tqdm(batches, desc="Training"):
        # Prepare batch
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids

        # Forward pass
        def loss_fn(model):
            logits, _ = model(input_ids)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            # Cross-entropy loss
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')
            return loss

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(model, loss_fn)(model)

        # Clip gradients
        from mlx.utils import tree_flatten, tree_unflatten
        max_norm = 1.0
        flat_grads = tree_flatten(grads)
        total_norm = mx.sqrt(sum((g * g).sum() for g in flat_grads if isinstance(g, mx.array)))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            flat_grads = [g * clip_coef if isinstance(g, mx.array) else g for g in flat_grads]
            grads = tree_unflatten(flat_grads)

        # Update parameters
        optimizer.update(model, grads)

        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        num_tokens += input_ids.size

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    tokens_per_sec = estimate_tokens_per_second(num_tokens, elapsed_time)

    return avg_loss, tokens_per_sec, elapsed_time


def evaluate(model, dataset, batch_size: int = 16):
    """Evaluate model on validation set."""
    total_loss = 0.0
    num_batches = 0

    dataset_list = list(dataset)
    batches = list(batch_iterator(dataset_list, batch_size, shuffle=False))

    for batch in tqdm(batches, desc="Evaluating"):
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids

        # Forward pass (no gradients)
        logits, _ = model(input_ids)

        # Compute loss
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, config, checkpoint_path):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = checkpoint_path / f"weights_epoch_{epoch}.npz"
    print(f"💾 Saving checkpoint to {weights_path}...")

    # Convert MLX arrays to numpy for saving
    weights = {}
    for name, param in model.parameters().items():
        if isinstance(param, mx.array):
            weights[name] = param.__array__()

    mx.savez(str(weights_path), **weights)

    # Save config
    config_path = checkpoint_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Checkpoint saved")


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Find latest weights file
    weights_files = list(checkpoint_path.glob("weights_epoch_*.npz"))
    if not weights_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    latest_weights = max(weights_files, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"📥 Loading checkpoint from {latest_weights}...")

    # Load weights
    weights = mx.load(str(latest_weights))
    model.update(weights)

    print(f"✅ Checkpoint loaded")


def train_pretrained(args):
    """Main training function for pre-trained psychedelic model."""
    print_training_header()
    print(f"\n{'='*80}")
    print(f"STAGE {args.stage}: {'SKIP-ONLY TRAINING' if args.freeze_baseline else 'FULL FINE-TUNING'}".center(80))
    print(f"{'='*80}\n")

    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("✅ Tokenizer loaded")

    # Prepare datasets
    train_dataset = prepare_dataset(
        args.dataset,
        split=f"train[:{args.train_samples}]",
        tokenizer=tokenizer
    )
    val_dataset = prepare_dataset(
        args.dataset,
        split=f"validation[:{args.val_samples}]",
        tokenizer=tokenizer
    )

    # Create model
    print("\n🧠 Creating pre-trained psychedelic model...")
    model = create_pretrained_psychedelic_model(
        skip_distance=args.skip_distance,
        skip_alpha=args.skip_alpha,
        skip_start_layer=args.skip_start_layer
    )

    # Load checkpoint if specified
    if args.load_checkpoint:
        load_checkpoint(model, args.load_checkpoint)

    # Freeze/unfreeze based on stage
    if args.freeze_baseline:
        model.freeze_baseline_weights()
    else:
        model.unfreeze_all_weights()

    stats = model.get_architecture_stats()

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name=f"pretrained_stage{args.stage}_{args.experiment_name}",
        config={
            "model": "pretrained_psychedelic",
            "stage": args.stage,
            "freeze_baseline": args.freeze_baseline,
            **vars(args),
            **stats
        },
        use_wandb=args.use_wandb
    )

    # Create optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    print(f"\n⚙️  Configuration:")
    print(f"   Stage: {args.stage}")
    print(f"   Freeze baseline: {args.freeze_baseline}")
    print(f"   Learning rate: {args.learning_rate:.2e}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print()

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 80 + "\n")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}".center(80))
        print(f"{'='*80}\n")

        # Train
        print("🍄 Training...")
        train_loss, tokens_per_sec, elapsed_time = train_epoch(
            model,
            train_dataset,
            optimizer,
            args.batch_size
        )

        # Evaluate
        val_loss = evaluate(model, val_dataset, args.batch_size)

        print_epoch_stats(
            epoch,
            train_loss,
            val_loss,
            tokens_per_sec=tokens_per_sec,
            elapsed_time=elapsed_time
        )

        tracker.log_metrics({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_perplexity': compute_perplexity(train_loss),
            'val_perplexity': compute_perplexity(val_loss),
            'tokens_per_sec': tokens_per_sec,
        }, step=epoch)

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = Path("checkpoints") / f"stage{args.stage}_{args.experiment_name}"
            save_checkpoint(model, optimizer, epoch, vars(args), checkpoint_dir)

    # Finish experiment
    tracker.finish()

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!".center(80))
    print("=" * 80 + "\n")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📁 Checkpoints saved to: checkpoints/stage{args.stage}_{args.experiment_name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train pre-trained psychedelic SmolLM2")

    # Stage configuration
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                       help="Training stage (1: skip-only, 2: full fine-tuning)")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--train_samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=500, help="Number of validation samples")

    # Training args
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # Psychedelic args
    parser.add_argument("--skip_distance", type=int, default=3, help="Skip-layer distance")
    parser.add_argument("--skip_alpha", type=float, default=0.5, help="Skip connection weight")
    parser.add_argument("--skip_start_layer", type=int, default=3, help="First layer with skip connections")

    # Freezing
    parser.add_argument("--freeze_baseline", action="store_true", help="Freeze baseline weights")
    parser.add_argument("--no_freeze", dest="freeze_baseline", action="store_false")
    parser.set_defaults(freeze_baseline=True)

    # Checkpointing
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")

    # Experiment args
    parser.add_argument("--experiment_name", type=str, default="pretrained_psychedelic",
                       help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases tracking")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=False)

    args = parser.parse_args()

    train_pretrained(args)


if __name__ == "__main__":
    main()

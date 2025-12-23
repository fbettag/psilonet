"""
Training script comparing baseline vs psychedelic SmolLM2 models.

This script trains both models on the same dataset and tracks:
1. Loss and perplexity
2. Attention entropy (psychedelic metric)
3. Convergence speed
4. Parameter efficiency

Usage:
    python experiments/train_comparison.py --epochs 10 --batch_size 16
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

from modules import (
    create_baseline_model,
    create_psychedelic_model,
    load_tokenizer,
)
from experiments.utils import (
    ExperimentTracker,
    compute_perplexity,
    print_training_header,
    print_epoch_stats,
    batch_iterator,
    estimate_tokens_per_second,
)


def prepare_dataset(dataset_name: str = "wikitext", split: str = "train[:5000]", tokenizer=None):
    """
    Prepare dataset for training.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split specification
        tokenizer: Tokenizer for encoding text

    Returns:
        Tokenized dataset
    """
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
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
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
        labels = input_ids  # MLX arrays are immutable, no need to copy

        # Forward pass
        def loss_fn(model):
            logits, _ = model(input_ids)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            # Cross-entropy loss (need to take mean for scalar loss)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')
            return loss

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(model, loss_fn)(model)

        # Clip gradients to prevent explosion (flatten nested dicts)
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
    """
    Evaluate model on validation set.

    Returns:
        Average loss
    """
    total_loss = 0.0
    num_batches = 0

    dataset_list = list(dataset)
    batches = list(batch_iterator(dataset_list, batch_size, shuffle=False))

    for batch in tqdm(batches, desc="Evaluating"):
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids  # MLX arrays are immutable

        # Forward pass (no gradients)
        logits, _ = model(input_ids)

        # Compute loss
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train_comparison(args):
    """Main training comparison function"""
    print_training_header()

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

    # Create models
    print("\n🧠 Creating models...")
    baseline_model = create_baseline_model()
    psychedelic_model = create_psychedelic_model(
        skip_distance=args.skip_distance,
        skip_alpha=args.skip_alpha,
        skip_start_layer=args.skip_start_layer
    )

    baseline_stats = baseline_model.get_architecture_stats()
    psychedelic_stats = psychedelic_model.get_architecture_stats()

    print(f"\n✅ Baseline model: {baseline_stats['total_params_millions']:.1f}M parameters")
    print(f"✅ Psychedelic model: {psychedelic_stats['total_params_millions']:.1f}M parameters")

    # Initialize experiment trackers
    baseline_tracker = ExperimentTracker(
        experiment_name=f"baseline_{args.experiment_name}",
        config={"model": "baseline", **vars(args), **baseline_stats},
        use_wandb=args.use_wandb
    )

    psychedelic_tracker = ExperimentTracker(
        experiment_name=f"psychedelic_{args.experiment_name}",
        config={"model": "psychedelic", **vars(args), **psychedelic_stats},
        use_wandb=args.use_wandb
    )

    baseline_tracker.log_model_comparison(baseline_stats, psychedelic_stats)

    # Create optimizers with different learning rates
    # Psychedelic model is bigger (185M vs 150M) so needs gentler learning rate
    baseline_optimizer = optim.Adam(learning_rate=args.learning_rate)
    psychedelic_lr = args.learning_rate * 0.4  # 40% of baseline LR for bigger model
    psychedelic_optimizer = optim.Adam(learning_rate=psychedelic_lr)

    print(f"\n⚙️  Learning Rates:")
    print(f"   Baseline:    {args.learning_rate:.2e}")
    print(f"   Psychedelic: {psychedelic_lr:.2e} (gentler for bigger model)")
    print()

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 80 + "\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}".center(80))
        print(f"{'='*80}\n")

        # Train baseline
        print("🧠 Training BASELINE model...")
        baseline_train_loss, baseline_tokens_sec, baseline_time = train_epoch(
            baseline_model,
            train_dataset,
            baseline_optimizer,
            args.batch_size
        )

        baseline_val_loss = evaluate(baseline_model, val_dataset, args.batch_size)

        print_epoch_stats(
            epoch,
            baseline_train_loss,
            baseline_val_loss,
            tokens_per_sec=baseline_tokens_sec,
            elapsed_time=baseline_time
        )

        baseline_tracker.log_metrics({
            'epoch': epoch,
            'train_loss': baseline_train_loss,
            'val_loss': baseline_val_loss,
            'train_perplexity': compute_perplexity(baseline_train_loss),
            'val_perplexity': compute_perplexity(baseline_val_loss),
            'tokens_per_sec': baseline_tokens_sec,
        }, step=epoch)

        # Train psychedelic
        print("\n🍄 Training PSYCHEDELIC model...")
        psychedelic_train_loss, psychedelic_tokens_sec, psychedelic_time = train_epoch(
            psychedelic_model,
            train_dataset,
            psychedelic_optimizer,
            args.batch_size
        )

        psychedelic_val_loss = evaluate(psychedelic_model, val_dataset, args.batch_size)

        print_epoch_stats(
            epoch,
            psychedelic_train_loss,
            psychedelic_val_loss,
            tokens_per_sec=psychedelic_tokens_sec,
            elapsed_time=psychedelic_time
        )

        psychedelic_tracker.log_metrics({
            'epoch': epoch,
            'train_loss': psychedelic_train_loss,
            'val_loss': psychedelic_val_loss,
            'train_perplexity': compute_perplexity(psychedelic_train_loss),
            'val_perplexity': compute_perplexity(psychedelic_val_loss),
            'tokens_per_sec': psychedelic_tokens_sec,
        }, step=epoch)

        # Compare
        improvement = ((baseline_val_loss - psychedelic_val_loss) / baseline_val_loss) * 100
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON (Epoch {epoch})")
        print(f"{'='*80}")
        print(f"Baseline val loss:     {baseline_val_loss:.4f}")
        print(f"Psychedelic val loss:  {psychedelic_val_loss:.4f}")
        print(f"Improvement:           {improvement:+.2f}%")
        print(f"{'='*80}\n")

    # Finish experiments
    baseline_tracker.finish()
    psychedelic_tracker.finish()

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!".center(80))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train and compare baseline vs psychedelic SmolLM2")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")

    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")

    # Psychedelic args
    parser.add_argument("--skip_distance", type=int, default=3, help="Skip-layer distance")
    parser.add_argument("--skip_alpha", type=float, default=0.5, help="Skip connection weight")
    parser.add_argument("--skip_start_layer", type=int, default=3, help="First layer with skip connections")

    # Experiment args
    parser.add_argument("--experiment_name", type=str, default="smollm_comparison", help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases tracking")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)

    args = parser.parse_args()

    train_comparison(args)


if __name__ == "__main__":
    main()

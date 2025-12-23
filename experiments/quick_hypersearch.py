"""
Quick hyperparameter search - tests only the most promising configurations.

Tests 15 experiments:
- Skip distance: [3, 4, 5, 6, 7] (skip 2 - too short, causes degradation)
- Skip alpha: [0.3, 0.5, 0.7] (skip 1.0 - too aggressive)
- Dataset: WikiText-2 (same as Stage 1 for fair comparison)

Goal: Find a better configuration than current best (distance=3, alpha=0.5, 9.4% improvement)
Estimated time: ~4 hours
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import itertools
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from modules.psychedelic_smollm import load_tokenizer
from experiments.utils import batch_iterator, estimate_tokens_per_second, compute_perplexity
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm


def prepare_dataset(num_samples: int = 2000, split: str = "train", seed: int = 42):
    """Prepare WikiText-2 dataset for training (same as Stage 1)."""
    print(f"📚 Loading WikiText-2 ({split}, {num_samples} samples)...")

    # Load WikiText-2 (same as Stage 1)
    if split == "train":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_samples}]")

    print(f"✅ Loaded {len(dataset)} examples")

    # Tokenize (same as Stage 1)
    tokenizer = load_tokenizer()

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

    # Convert to list format for our training loop
    dataset_list = []
    for item in tokenized:
        # HuggingFace datasets already returns lists
        input_ids = item["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        dataset_list.append({"input_ids": input_ids})

    print(f"✅ Dataset ready: {len(dataset_list)} examples")
    return dataset_list


def train_epoch(model, dataset, optimizer, batch_size: int = 4, max_norm: float = 1.0):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0
    num_tokens = 0
    start_time = time.time()

    batches = list(batch_iterator(dataset, batch_size))

    for batch in tqdm(batches, desc="Training"):
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids

        # Forward pass with gradients
        def loss_fn(model):
            logits, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)

        # Gradient clipping
        from mlx.utils import tree_flatten, tree_unflatten
        flat_grads = tree_flatten(grads)
        total_norm = mx.sqrt(sum((g * g).sum() for g in flat_grads if isinstance(g, mx.array)))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            flat_grads = [g * clip_coef if isinstance(g, mx.array) else g for g in flat_grads]
            grads = tree_unflatten(flat_grads)

        optimizer.update(model, grads)

        total_loss += loss.item()
        num_batches += 1
        num_tokens += input_ids.size

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    tokens_per_sec = estimate_tokens_per_second(num_tokens, elapsed_time)
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity, tokens_per_sec


def evaluate(model, dataset, batch_size: int = 4):
    """Evaluate model on validation set."""
    total_loss = 0.0
    num_batches = 0

    batches = list(batch_iterator(dataset, batch_size, shuffle=False))

    for batch in tqdm(batches, desc="Evaluating"):
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids

        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = compute_perplexity(avg_loss)
    return avg_loss, perplexity


def run_experiment(config: dict, train_data, val_data):
    """Run a single experiment and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: distance={config['skip_distance']}, alpha={config['skip_alpha']}")
    print(f"{'='*80}")

    start_time = time.time()

    # Create model
    model = create_pretrained_psychedelic_model(
        skip_distance=config['skip_distance'],
        skip_alpha=config['skip_alpha'],
        skip_start_layer=3,
    )
    model.freeze_baseline_weights()

    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=1e-4)

    # Training
    best_val_loss = float('inf')
    best_epoch = 0
    metrics = []

    for epoch in range(1, 6):  # 5 epochs like Stage 1
        train_loss, train_ppl, tokens_per_sec = train_epoch(
            model=model,
            dataset=train_data,
            optimizer=optimizer,
            batch_size=4,
            max_norm=1.0,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            dataset=val_data,
            batch_size=4,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        metrics.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_ppl": float(train_ppl),
            "val_ppl": float(val_ppl),
        })

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")

        # Memory cleanup after each epoch
        mx.metal.clear_cache()
        gc.collect()

    # Calculate improvement
    baseline = 0.5634
    improvement = ((baseline - best_val_loss) / baseline) * 100

    result = {
        "config": config,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "improvement_pct": float(improvement),
        "time_minutes": (time.time() - start_time) / 60,
        "metrics": metrics,
    }

    print(f"\n✓ Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"✓ Improvement: {improvement:+.2f}%")
    print(f"✓ Time: {result['time_minutes']:.1f} min\n")

    # Cleanup model and optimizer to free memory
    del model
    del optimizer
    mx.metal.clear_cache()
    gc.collect()

    return result


def main():
    print("="*80)
    print("QUICK HYPERPARAMETER SEARCH")
    print("Testing 15 configurations: distance=[3,4,5,6,7] × alpha=[0.3,0.5,0.7]")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train_data = prepare_dataset(num_samples=2000, split="train")
    val_data = prepare_dataset(num_samples=500, split="validation")

    # Test grid
    skip_distances = [3, 4, 5, 6, 7]
    skip_alphas = [0.3, 0.5, 0.7]

    # Load existing results if available
    results_file = Path("logs/quick_hypersearch_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        completed_configs = {(r['config']['skip_distance'], r['config']['skip_alpha']) for r in results}
        print(f"✓ Loaded {len(results)} existing results, skipping completed experiments\n")
    else:
        results = []
        completed_configs = set()

    total = len(skip_distances) * len(skip_alphas)
    current = 0

    for distance, alpha in itertools.product(skip_distances, skip_alphas):
        current += 1

        # Skip if already completed
        if (distance, alpha) in completed_configs:
            print(f"\n[{current}/{total}] Skipping distance={distance}, alpha={alpha} (already completed)")
            continue

        print(f"\n[{current}/{total}] " + "="*60)

        config = {
            "skip_distance": distance,
            "skip_alpha": alpha,
        }

        result = run_experiment(config, train_data, val_data)
        results.append(result)

        # Save intermediate results
        with open("logs/quick_hypersearch_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Memory cleanup
        gc.collect()
        mx.metal.clear_cache()

    # Find best
    best = max(results, key=lambda x: x['improvement_pct'])

    print("\n" + "="*80)
    print("🏆 RESULTS SUMMARY")
    print("="*80)

    # Print all results sorted by improvement
    results_sorted = sorted(results, key=lambda x: x['improvement_pct'], reverse=True)

    print("\nAll Configurations (sorted by improvement):")
    print(f"{'Rank':<6} {'Distance':<10} {'Alpha':<8} {'Val Loss':<12} {'Improvement':<12} {'Time':<8}")
    print("-" * 80)

    for i, r in enumerate(results_sorted, 1):
        print(f"{i:<6} {r['config']['skip_distance']:<10} {r['config']['skip_alpha']:<8.1f} "
              f"{r['best_val_loss']:<12.4f} {r['improvement_pct']:+<12.2f}% {r['time_minutes']:<8.1f}min")

    print("\n" + "="*80)
    print("🎯 BEST CONFIGURATION")
    print("="*80)
    print(f"Skip Distance: {best['config']['skip_distance']}")
    print(f"Skip Alpha: {best['config']['skip_alpha']}")
    print(f"Val Loss: {best['best_val_loss']:.4f}")
    print(f"Improvement: {best['improvement_pct']:+.2f}%")
    print(f"Best Epoch: {best['best_epoch']}")

    # Compare to current best (Stage 1: distance=3, alpha=0.5, 9.4%)
    current_best_improvement = 9.4
    if best['improvement_pct'] > current_best_improvement:
        delta = best['improvement_pct'] - current_best_improvement
        print(f"\n✨ NEW RECORD! {delta:+.2f}% better than previous best!")
    else:
        delta = current_best_improvement - best['improvement_pct']
        print(f"\n📊 Current best still holds ({delta:.2f}% ahead)")

    print(f"\n✓ Full results saved to logs/quick_hypersearch_results.json")


if __name__ == "__main__":
    main()

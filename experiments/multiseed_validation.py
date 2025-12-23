"""
Multi-seed validation of top hyperparameter configurations.

Tests the top 5 configurations from quick_hypersearch with 5 different random seeds each.
This confirms whether observed differences (e.g., α=0.7 vs α=0.5) are statistically
significant or just seed noise.

Total experiments: 5 configs × 5 seeds = 25 runs
Expected time: ~7 hours (17 min/run)
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import gc
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from modules.psychedelic_smollm import load_tokenizer
from experiments.utils import batch_iterator, estimate_tokens_per_second, compute_perplexity
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm


def prepare_dataset(
    num_samples: int = 2000,
    split: str = "train",
    seed: int = 42,
    max_length: int = 512,
):
    """Prepare WikiText-2 dataset for training (same as hyperparameter search)."""
    print(f"📚 Loading WikiText-2 ({split}, {num_samples} samples, seed={seed})...")

    # Load WikiText-2
    if split == "train":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_samples}]")

    print(f"✅ Loaded {len(dataset)} examples")

    # Tokenize
    tokenizer = load_tokenizer()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np"
        )

    print("🔤 Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Convert to list format
    dataset_list = []
    for item in tokenized:
        input_ids = item["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        dataset_list.append({"input_ids": input_ids})

    print(f"✅ Dataset ready: {len(dataset_list)} examples")
    return dataset_list


def train_epoch(model, dataset, optimizer, batch_size: int = 4, max_norm: float = 1.0, seed: int = 42):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0
    num_tokens = 0
    start_time = time.time()

    batches = list(batch_iterator(dataset, batch_size, shuffle=True, seed=seed))

    for batch in tqdm(batches, desc="Training", leave=False):
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

        # Mitigate Metal allocator bloat on long epochs
        # Metal allocator can leak within long training loops; flush every batch for stability
        mx.clear_cache()
        gc.collect()

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

    for batch in tqdm(batches, desc="Evaluating", leave=False):
        input_ids = mx.array([item['input_ids'] for item in batch])
        labels = input_ids

        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='mean')

        total_loss += loss.item()
        num_batches += 1

        mx.clear_cache()
        gc.collect()

    avg_loss = total_loss / num_batches
    perplexity = compute_perplexity(avg_loss)
    return avg_loss, perplexity


def run_experiment(config: dict, seed: int, train_data, val_data):
    """Run a single experiment with given config and seed."""
    print(f"\n{'='*80}")
    print(f"Config: distance={config['skip_distance']}, alpha={config['skip_alpha']}, seed={seed}")
    print(f"{'='*80}")

    start_time = time.time()

    # Set MLX random seed
    mx.random.seed(seed)

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

    for epoch in range(1, 6):  # 5 epochs
        train_loss, train_ppl, tokens_per_sec = train_epoch(
            model=model,
            dataset=train_data,
            optimizer=optimizer,
            batch_size=4,
            max_norm=1.0,
            seed=seed + epoch,  # Different shuffle each epoch
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
        mx.clear_cache()
        gc.collect()

    # Calculate improvement
    baseline = 0.5634
    improvement = ((baseline - best_val_loss) / baseline) * 100

    result = {
        "config": config,
        "seed": seed,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "improvement_pct": float(improvement),
        "time_minutes": (time.time() - start_time) / 60,
        "metrics": metrics,
    }

    print(f"✓ Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"✓ Improvement: {improvement:+.2f}%")

    # Cleanup
    del model
    del optimizer
    mx.clear_cache()
    gc.collect()

    return result


def compute_statistics(results):
    """Compute mean, std, and confidence intervals for each config."""
    # Group by config
    configs = {}
    for r in results:
        key = (r['config']['skip_distance'], r['config']['skip_alpha'])
        if key not in configs:
            configs[key] = []
        configs[key].append(r['best_val_loss'])

    # Compute statistics
    stats_results = []
    for (distance, alpha), val_losses in configs.items():
        val_losses = np.array(val_losses)
        mean_loss = np.mean(val_losses)
        std_loss = np.std(val_losses, ddof=1)

        # 95% confidence interval
        n = len(val_losses)
        sem = std_loss / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_loss, scale=sem)

        # Improvement
        baseline = 0.5634
        mean_improvement = ((baseline - mean_loss) / baseline) * 100

        stats_results.append({
            "distance": distance,
            "alpha": alpha,
            "mean_val_loss": float(mean_loss),
            "std_val_loss": float(std_loss),
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "mean_improvement_pct": float(mean_improvement),
            "num_seeds": n,
            "individual_losses": val_losses.tolist(),
        })

    # Sort by mean val loss
    stats_results.sort(key=lambda x: x['mean_val_loss'])

    return stats_results


def significance_tests(stats_results):
    """Perform pairwise t-tests between configurations."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    # Get individual results grouped by config
    config_groups = {}
    for s in stats_results:
        key = (s['distance'], s['alpha'])
        config_groups[key] = np.array(s['individual_losses'])

    # Compare top config with others
    top_key = (stats_results[0]['distance'], stats_results[0]['alpha'])
    top_losses = config_groups[top_key]

    print(f"\nComparing top config (d={top_key[0]}, α={top_key[1]}) with others:")
    print(f"{'Config':<15} {'Mean Δ':<12} {'t-stat':<10} {'p-value':<10} {'Significant?':<12}")
    print("-" * 80)

    for s in stats_results[1:]:
        key = (s['distance'], s['alpha'])
        other_losses = config_groups[key]

        # Paired t-test (since same seeds used)
        t_stat, p_value = stats.ttest_ind(top_losses, other_losses)

        mean_delta = s['mean_val_loss'] - stats_results[0]['mean_val_loss']
        significant = "YES" if p_value < 0.05 else "NO"

        print(f"d={key[0]}, α={key[1]:<5.1f} {mean_delta:+.5f}     {t_stat:<10.3f} {p_value:<10.4f} {significant}")


def main():
    print("="*80)
    print("MULTI-SEED VALIDATION")
    print("Testing top 5 configurations with 5 seeds each (25 total experiments)")
    print("="*80)

    # Top configurations from hyperparameter search
    configs = [
        {"skip_distance": 3, "skip_alpha": 0.7, "name": "d3_a0.7 (current best)"},
        {"skip_distance": 4, "skip_alpha": 0.7, "name": "d4_a0.7 (runner-up)"},
        {"skip_distance": 3, "skip_alpha": 0.5, "name": "d3_a0.5 (Stage 1 baseline)"},
        {"skip_distance": 3, "skip_alpha": 0.65, "name": "d3_a0.65 (interpolate)"},
        {"skip_distance": 4, "skip_alpha": 0.5, "name": "d4_a0.5 (alternative)"},
    ]

    seeds = [42, 123, 456, 789, 1024]

    # Load data once
    print("\nLoading data...")
    train_data = prepare_dataset(num_samples=2000, split="train", seed=42)
    val_data = prepare_dataset(num_samples=500, split="validation", seed=42)

    # Load existing results if available
    results_file = Path("logs/multiseed_validation_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        completed = {(r['config']['skip_distance'], r['config']['skip_alpha'], r['seed']) for r in results}
        print(f"✓ Loaded {len(results)} existing results\n")
    else:
        results = []
        completed = set()

    total = len(configs) * len(seeds)
    current = 0

    for config in configs:
        for seed in seeds:
            current += 1

            # Skip if already completed
            key = (config['skip_distance'], config['skip_alpha'], seed)
            if key in completed:
                print(f"\n[{current}/{total}] Skipping {config['name']}, seed={seed} (already completed)")
                continue

            print(f"\n[{current}/{total}] " + "="*60)
            print(f"Config: {config['name']}, Seed: {seed}")

            result = run_experiment(config, seed, train_data, val_data)
            results.append(result)

            # Save intermediate results
            with open("logs/multiseed_validation_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            # Memory cleanup
            gc.collect()
            mx.clear_cache()

    # Compute statistics
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    stats_results = compute_statistics(results)

    print("\nResults (sorted by mean validation loss):")
    print(f"{'Rank':<6} {'Config':<12} {'Mean Val Loss':<15} {'Std':<10} {'95% CI':<25} {'Mean Improv.':<12}")
    print("-" * 90)

    for i, s in enumerate(stats_results, 1):
        ci_str = f"[{s['ci_95_lower']:.4f}, {s['ci_95_upper']:.4f}]"
        print(f"{i:<6} d={s['distance']}, α={s['alpha']:<5.1f} "
              f"{s['mean_val_loss']:<15.5f} ±{s['std_val_loss']:<9.5f} {ci_str:<25} "
              f"{s['mean_improvement_pct']:+.2f}%")

    # Significance tests
    significance_tests(stats_results)

    # Save statistics
    stats_file = Path("logs/multiseed_validation_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "statistics": stats_results,
            "baseline_val_loss": 0.5634,
            "num_seeds": len(seeds),
            "seeds": seeds,
        }, f, indent=2)

    print(f"\n✓ Full results saved to logs/multiseed_validation_results.json")
    print(f"✓ Statistics saved to logs/multiseed_validation_statistics.json")


if __name__ == "__main__":
    main()

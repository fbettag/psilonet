"""
Generalization Test: Scale to SmolLM2-360M + Multiple Datasets

Goal: Validate that psychedelic skip-layer mechanism generalizes to:
1. Larger models (360M params vs 135M)
2. Different datasets (creative + factual)

This is a "smoke test" - short runs to validate generalization, not full optimization.

Expected outcome: Similar relative improvement on larger model and different data.
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datasets import load_dataset
import gc
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from experiments.multiseed_validation import prepare_dataset, train_epoch, evaluate
from experiments.utils import compute_perplexity
import mlx.core as mx
import mlx.optimizers as optim
from mlx.core import metal
from transformers import AutoTokenizer
import os


# Optionally lift MLX's internal memory ceiling so Metal allocations match the
# system-level wired limit (set via `sysctl iogpu.wired_limit_mb`).
_mlx_limit_bytes = None
if "MLX_MEMORY_LIMIT_MB" in os.environ:
    try:
        _mlx_limit_bytes = int(os.environ["MLX_MEMORY_LIMIT_MB"]) * 1024 * 1024
    except ValueError:
        pass
elif "MLX_MEMORY_LIMIT_GB" in os.environ:
    try:
        _mlx_limit_bytes = int(os.environ["MLX_MEMORY_LIMIT_GB"]) * 1024 * 1024 * 1024
    except ValueError:
        pass

if _mlx_limit_bytes:
    try:
        metal.set_memory_limit(_mlx_limit_bytes)
        print(f"⚙️  MLX memory limit set to {_mlx_limit_bytes / (1024 ** 3):.2f} GB")
    except Exception as err:
        print(f"⚠️  Unable to set MLX memory limit: {err}")


def prepare_dataset_custom(
    dataset_name: str,
    split: str,
    num_samples: int,
    seed: int = 42,
    max_length: int = 256,
) -> List[Dict]:
    """
    Prepare custom datasets for generalization testing.

    Supported datasets:
    - wikitext: WikiText-2 (factual, encyclopedic)
    - tinystories: TinyStories (creative, narrative)
    - openwebtext: OpenWebText (mixed, web text)
    """
    print(f"📚 Loading {dataset_name} ({split}, {num_samples} samples, seed={seed})...")

    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_column = "text"
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        text_column = "text"
    elif dataset_name == "openwebtext":
        # OpenWebText is large, so we use a subset
        dataset = load_dataset("Skylion007/openwebtext", split="train")
        text_column = "text"
        # For validation, we'll use a different slice
        if split == "validation":
            dataset = dataset.select(range(100000, 110000))
        else:
            dataset = dataset.select(range(0, 100000))
    elif dataset_name == "ag_news":
        mapped_split = "test" if split == "validation" else "train"
        dataset = load_dataset("ag_news", split=mapped_split)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 50)

    # Sample
    if len(dataset) > num_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        dataset = dataset.select(indices)

    print(f"✅ Loaded {len(dataset)} examples")

    # Tokenize
    # Note: For 360M, we should load the appropriate tokenizer
    # But for simplicity, we'll use the 135M tokenizer (same family)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    print(f"🔤 Tokenizing and padding...")
    tokenized = []
    for example in dataset:
        text = example[text_column]
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Skip very short sequences
        if len(tokens) < 10:
            continue

        # Truncate to max length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Pad to max length for uniform batching
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokens = tokens + [pad_token_id] * (max_length - len(tokens))

        tokenized.append({
            'input_ids': tokens,
            'text': text,
        })

    print(f"✅ Dataset ready: {len(tokenized)} examples (padded to {max_length} tokens)")
    return tokenized


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    skip_distance: int,
    skip_alpha: float,
    epochs: int = 5,
    batch_size: int = 4,
    train_samples: int = 1000,
    val_samples: int = 200,
    seed: int = 42,
    max_length: int = 256,
    freeze_baseline: bool = False,
) -> Dict:
    """
    Run a single generalization experiment.

    Args:
        model_name: "SmolLM2-135M" or "SmolLM2-360M"
        dataset_name: "wikitext", "tinystories", or "openwebtext"
        skip_distance: Skip distance (0 for baseline)
        skip_alpha: Skip alpha (ignored if distance=0)
        epochs: Number of training epochs
        batch_size: Batch size
        train_samples: Number of training samples
        val_samples: Number of validation samples
        seed: Random seed

    Returns:
        Dict with results
    """
    config_str = f"{model_name}, {dataset_name}, d={skip_distance}, α={skip_alpha}"
    print(f"\n{'='*80}")
    print(f"Experiment: {config_str}")
    print(f"{'='*80}")

    is_large_model = "360m" in model_name.lower()

    effective_train_samples = train_samples
    effective_val_samples = val_samples
    if is_large_model:
        effective_train_samples = min(train_samples, 100)
        effective_val_samples = min(val_samples, 25)
        print(f"⚙️  Adjusting sample counts for {model_name}: train={effective_train_samples}, val={effective_val_samples}")

    max_seq_len = 64 if is_large_model else max_length

    # Prepare dataset
    if dataset_name == "wikitext":
        # Use existing prepare_dataset for WikiText with shorter context for memory efficiency
        train_dataset = prepare_dataset(
            num_samples=effective_train_samples,
            split="train",
            seed=seed,
            max_length=max_seq_len,
        )
        val_dataset = prepare_dataset(
            num_samples=effective_val_samples,
            split="validation",
            seed=seed,
            max_length=max_seq_len,
        )
    else:
        # Use custom loader for other datasets
        train_dataset = prepare_dataset_custom(
            dataset_name=dataset_name,
            split="train",
            num_samples=effective_train_samples,
            seed=seed,
            max_length=max_seq_len,
        )
        val_dataset = prepare_dataset_custom(
            dataset_name=dataset_name,
            split="validation",
            num_samples=effective_val_samples,
            seed=seed,
            max_length=max_seq_len,
        )

    # Map to HuggingFace identifiers (default to provided name if already an HF id)
    hf_model_id = {
        "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
        "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
    }.get(model_name, model_name)

    if skip_distance == 0:
        # Baseline (no skip connections)
        print(f"🔽 Loading baseline model: {model_name} (no skip connections)")
        # Disable skips by starting after the final layer; keep skip_distance=1 to avoid degenerate d=0 paths
        # We read num_layers from the HF config so the start marker is always ≥ depth.
        from modules.psychedelic_smollm import load_smollm2_config

        num_layers = load_smollm2_config(hf_model_id).get("num_hidden_layers", 30)
        model = create_pretrained_psychedelic_model(
            model_name=hf_model_id,
            skip_distance=1,
            skip_alpha=0.0,
            skip_start_layer=num_layers,  # ensures zero skip-enabled layers
        )
    else:
        # Psychedelic skip model
        print(f"🔽 Loading psychedelic skip model: {model_name} (d={skip_distance}, α={skip_alpha})")
        model = create_pretrained_psychedelic_model(
            model_name=hf_model_id,
            skip_distance=skip_distance,
            skip_alpha=skip_alpha,
            skip_start_layer=3,
        )

    hidden_size = getattr(model, "hidden_size", 0)
    large_model = hidden_size >= 800 or is_large_model

    if large_model:
        partial_layers = 2  # number of upper transformer layers to train
        partial_start = max(0, model.num_layers - partial_layers)
        model.unfreeze_layers(from_layer=partial_start)
        if skip_distance == 0:
            print(f"🔓 Partial baseline fine-tune: layers ≥ {partial_start} (out of {model.num_layers}) remain trainable")
        else:
            print(f"🍄 Partial skip fine-tune: baseline layers ≥ {partial_start} (out of {model.num_layers}) unfrozen + skip heads")
    else:
        if freeze_baseline:
            model.freeze_baseline_weights()
            print("🧊 Baseline frozen for memory efficiency")
        else:
            if skip_distance == 0:
                print("🔓 Baseline fine-tuning: keeping all pre-trained weights trainable")
            else:
                model.freeze_baseline_weights()  # CRITICAL: Only train skip connections!

    # Unified optimizer settings (ensure baseline and skip share the same schedule)
    if large_model:
        learning_rate = 5e-5
        optimizer = optim.Adafactor(learning_rate=learning_rate)
        optimizer_name = "Adafactor"
    else:
        learning_rate = 1e-4
        optimizer = optim.AdamW(learning_rate=learning_rate)
        optimizer_name = "AdamW"
    print(f"⚙️  Optimizer: {optimizer_name} (lr={learning_rate:.1e})")

    # Train
    print(f"🏋️  Training for {epochs} epochs...")
    # Adjust batch size dynamically for larger models to fit GPU memory
    effective_batch_size = 1 if large_model else batch_size
    if effective_batch_size != batch_size:
        print(f"⚙️  Adjusting batch size to {effective_batch_size} for memory safety")

    # Report token budget for transparency
    sample_token_length = len(train_dataset[0]['input_ids'])
    train_tokens = len(train_dataset) * sample_token_length * epochs
    val_tokens = len(val_dataset) * sample_token_length
    print(f"   ➤ Token budget (train): {train_tokens:,} (seq_len={sample_token_length}, epochs={epochs})")
    print(f"   ➤ Token budget (val):   {val_tokens:,}")

    for epoch in range(epochs):
        train_loss, perplexity, tokens_per_sec = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            batch_size=effective_batch_size,
            max_norm=1.0,
            seed=seed + epoch,
        )
        print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, ppl={perplexity:.2f}")

    # Evaluate
    print(f"📊 Evaluating...")
    val_loss, val_ppl = evaluate(model, val_dataset, batch_size=effective_batch_size)

    print(f"✅ Results: val_loss={val_loss:.4f}, perplexity={val_ppl:.2f}")

    # Aggressive memory cleanup
    del model
    del optimizer
    del train_dataset
    del val_dataset
    mx.clear_cache()
    gc.collect()

    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'skip_distance': skip_distance,
        'skip_alpha': skip_alpha,
        'val_loss': float(val_loss),
        'val_perplexity': float(val_ppl),
        'seed': seed,
    }


def generalization_test(
    output_dir: Path,
    model_names: List[str] = ["SmolLM2-135M", "SmolLM2-360M"],
    dataset_names: List[str] = ["wikitext", "tinystories"],
    epochs: int = 5,
    batch_size: int = 4,
    train_samples: int = 1000,
    val_samples: int = 200,
    seed: int = 42,
    max_length: int = 256,
    freeze_baseline: bool = False,
    baseline_epochs: int = None,
    baseline_train_samples: int = None,
    baseline_val_samples: int = None,
):
    """
    Run generalization tests across models and datasets.

    Tests:
    1. Baseline (no skip) vs Psychedelic skip (d=3, α=0.65)
    2. Multiple models (135M, 360M)
    3. Multiple datasets (wikitext, tinystories)
    """
    print("=" * 80)
    print("GENERALIZATION TEST")
    print("=" * 80)
    print(f"Models: {model_names}")
    print(f"Datasets: {dataset_names}")
    print(f"Training: {epochs} epochs, {train_samples} samples")
    print(f"Seed: {seed}")
    print("=" * 80)

    results = []

    # Test configurations
    configs = [
        {'skip_distance': 0, 'skip_alpha': 0.0, 'name': 'Baseline'},
        {'skip_distance': 3, 'skip_alpha': 0.65, 'name': 'Psychedelic (d=3, α=0.65)'},
    ]

    total_experiments = len(model_names) * len(dataset_names) * len(configs)
    experiment_count = 0

    for model_name in model_names:
        for dataset_name in dataset_names:
            for config in configs:
                experiment_count += 1
                is_baseline = config['skip_distance'] == 0
                cfg_epochs = baseline_epochs if (is_baseline and baseline_epochs is not None) else epochs
                cfg_train_samples = baseline_train_samples if (is_baseline and baseline_train_samples is not None) else train_samples
                cfg_val_samples = baseline_val_samples if (is_baseline and baseline_val_samples is not None) else val_samples

                print(f"\n[{experiment_count}/{total_experiments}] {model_name} + {dataset_name} + {config['name']}")
                if is_baseline and (baseline_epochs or baseline_train_samples or baseline_val_samples):
                    print(f"   ↳ Baseline overrides: epochs={cfg_epochs}, train_samples={cfg_train_samples}, val_samples={cfg_val_samples}")

                result = run_single_experiment(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    skip_distance=config['skip_distance'],
                    skip_alpha=config['skip_alpha'],
                    epochs=cfg_epochs,
                    batch_size=batch_size,
                    train_samples=cfg_train_samples,
                    val_samples=cfg_val_samples,
                    seed=seed,
                    max_length=max_length,
                    freeze_baseline=freeze_baseline,
                )

                result['config_name'] = config['name']
                results.append(result)

                # Additional memory cleanup between experiments
                mx.clear_cache()
                gc.collect()

    return results


def plot_generalization_results(results: List[Dict], output_path: Path):
    """Plot generalization test results."""
    # Group results by model and dataset
    datasets = sorted(set(r['dataset_name'] for r in results))
    models = sorted(set(r['model_name'] for r in results))

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 6 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Filter results for this dataset
        dataset_results = [r for r in results if r['dataset_name'] == dataset]

        # Group by model and config
        x_labels = []
        baseline_losses = []
        psychedelic_losses = []

        for model in models:
            model_results = [r for r in dataset_results if r['model_name'] == model]

            baseline = [r for r in model_results if r['skip_distance'] == 0]
            psychedelic = [r for r in model_results if r['skip_distance'] > 0]

            if baseline and psychedelic:
                x_labels.append(model.replace("SmolLM2-", ""))
                baseline_losses.append(baseline[0]['val_loss'])
                psychedelic_losses.append(psychedelic[0]['val_loss'])

        # Plot bars
        x = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_losses, width, label='Baseline',
                      color='#95B8D1', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, psychedelic_losses, width, label='Psychedelic Skip',
                      color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)

        # Add improvement percentages
        for i, (baseline, psychedelic) in enumerate(zip(baseline_losses, psychedelic_losses)):
            improvement = (baseline - psychedelic) / baseline * 100
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, max(baseline, psychedelic) * 1.05,
                   f'{improvement:+.1f}%',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color)

        ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Generalization Test: {dataset.title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Generalization plot saved: {output_path}")
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Psychedelic generalization test harness")
    parser.add_argument("--model_names", type=str, default="SmolLM2-135M,SmolLM2-360M",
                        help="Comma-separated model list (e.g., 'SmolLM2-135M' or 'SmolLM2-135M,SmolLM2-360M')")
    parser.add_argument("--dataset_names", type=str, default="wikitext,tinystories",
                        help="Comma-separated datasets: wikitext,tinystories,openwebtext")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--val_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256,
                        help="Sequence length for truncation/padding (lower to reduce Metal memory)")
    parser.add_argument("--freeze_baseline", action="store_true",
                        help="Freeze baseline weights to cut memory (applies to both baseline and skip runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="logs/generalization_test",
                        help="Where to store results + plots")
    parser.add_argument("--baseline_epochs", type=int, default=None,
                        help="Override epochs for the baseline run only (keep skip config at --epochs)")
    parser.add_argument("--baseline_train_samples", type=int, default=None,
                        help="Override train_samples for the baseline run only")
    parser.add_argument("--baseline_val_samples", type=int, default=None,
                        help="Override val_samples for the baseline run only")
    return parser.parse_args()


def _split_arg_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def main():
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_names = _split_arg_list(args.model_names)
    dataset_names = _split_arg_list(args.dataset_names)

    # Run generalization tests
    results = generalization_test(
        output_dir=output_dir,
        model_names=model_names,
        dataset_names=dataset_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        max_length=args.max_length,
        freeze_baseline=args.freeze_baseline,
        baseline_epochs=args.baseline_epochs,
        baseline_train_samples=args.baseline_train_samples,
        baseline_val_samples=args.baseline_val_samples,
    )

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("GENERALIZATION TEST SUMMARY")
    print("=" * 80)

    # Group by dataset
    datasets = sorted(set(r['dataset_name'] for r in results))
    for dataset in datasets:
        print(f"\n📊 {dataset.upper()}")
        print("-" * 80)

        dataset_results = [r for r in results if r['dataset_name'] == dataset]

        baseline = [r for r in dataset_results if r['skip_distance'] == 0]
        psychedelic = [r for r in dataset_results if r['skip_distance'] > 0]

        if baseline and psychedelic:
            baseline_loss = baseline[0]['val_loss']
            psychedelic_loss = psychedelic[0]['val_loss']
            improvement = (baseline_loss - psychedelic_loss) / baseline_loss * 100

            print(f"  Baseline loss:     {baseline_loss:.4f}")
            print(f"  Psychedelic loss:  {psychedelic_loss:.4f}")
            print(f"  Improvement:       {improvement:+.2f}%")

            if improvement > 0:
                print(f"  ✅ Psychedelic skip generalizes to {dataset}")
            else:
                print(f"  ⚠️  No improvement on {dataset}")

    # Save results
    results_file = output_dir / "generalization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"\n✅ Results saved: {results_file}")

    # Generate plot
    plot_generalization_results(results, output_dir / "generalization_comparison.png")

    print("\n" + "=" * 80)
    print("✅ GENERALIZATION TEST COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"  - generalization_results.json")
    print(f"  - generalization_comparison.png")


if __name__ == "__main__":
    main()

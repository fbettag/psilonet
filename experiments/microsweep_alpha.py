"""
Micro-sweep around optimal alpha (α ∈ [0.60, 0.68]).

Goal: Fine-grained dose-response curve to pinpoint exact optimal α.

Based on multi-seed validation findings:
- d=3, α=0.65: 0.5137 ± 0.0036 (winner)
- d=3, α=0.70: 0.5151 ± 0.0058 (close second)

Test points: [0.60, 0.62, 0.64, 0.65, 0.66, 0.68]
Seeds: [42, 123, 456] (3 seeds for statistical reliability)
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from experiments.multiseed_validation import prepare_dataset, train_epoch, evaluate
from experiments.utils import estimate_tokens_per_second, compute_perplexity
import mlx.optimizers as optim
import mlx.core as mx


def microsweep_alpha(
    alpha_values: List[float],
    skip_distance: int = 3,
    seeds: List[int] = [42, 123, 456],
    epochs: int = 10,
    batch_size: int = 8,
    train_samples: int = 2000,
    val_samples: int = 500,
):
    """
    Run micro-sweep around optimal alpha with paired seeds.

    Returns:
        Dict with results for each alpha value
    """
    print("=" * 80)
    print("MICRO-SWEEP: α ∈ [0.60, 0.68]")
    print("=" * 80)
    print(f"Testing {len(alpha_values)} alpha values with {len(seeds)} seeds each")
    print(f"Total experiments: {len(alpha_values) * len(seeds)}")
    print(f"Distance: {skip_distance} (fixed)")
    print(f"Alpha values: {alpha_values}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    # Prepare datasets once
    print(f"📚 Loading datasets...")
    train_dataset = prepare_dataset(num_samples=train_samples, split="train", seed=42)
    val_dataset = prepare_dataset(num_samples=val_samples, split="validation", seed=42)
    print(f"✅ Datasets ready: {len(train_dataset)} train, {len(val_dataset)} val")

    results = []
    experiment_count = 0
    total_experiments = len(alpha_values) * len(seeds)

    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"ALPHA = {alpha:.2f}")
        print(f"{'='*80}")

        alpha_results = {
            'alpha': alpha,
            'distance': skip_distance,
            'seed_results': [],
        }

        for seed in seeds:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] α={alpha:.2f}, seed={seed}")

            # Create model
            model = create_pretrained_psychedelic_model(
                skip_distance=skip_distance,
                skip_alpha=alpha,
                skip_start_layer=3,
            )
            model.freeze_baseline_weights()  # CRITICAL: Only train skip connections!

            # Set up optimizer
            optimizer = optim.AdamW(learning_rate=1e-4)  # Match multiseed_validation

            # Train for epochs
            for epoch in range(epochs):
                train_loss, perplexity, tokens_per_sec = train_epoch(
                    model=model,
                    dataset=train_dataset,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    max_norm=1.0,
                    seed=seed + epoch,  # Different seed per epoch
                )

                if epoch % 2 == 0:  # Print every 2 epochs
                    print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, ppl={perplexity:.2f}")

            # Evaluate
            val_loss, val_ppl = evaluate(model, val_dataset, batch_size=batch_size)

            alpha_results['seed_results'].append({
                'seed': seed,
                'val_loss': float(val_loss),
            })

            print(f"  ✓ Val loss: {val_loss:.4f}, perplexity: {val_ppl:.2f}")

            # Cleanup
            del model
            del optimizer
            mx.clear_cache()

        # Compute statistics for this alpha
        losses = [r['val_loss'] for r in alpha_results['seed_results']]
        alpha_results['mean_val_loss'] = float(np.mean(losses))
        alpha_results['std_val_loss'] = float(np.std(losses))
        alpha_results['min_val_loss'] = float(np.min(losses))
        alpha_results['max_val_loss'] = float(np.max(losses))

        # 95% confidence interval
        n = len(losses)
        margin = 1.96 * (np.std(losses) / np.sqrt(n))
        alpha_results['ci_95_lower'] = float(np.mean(losses) - margin)
        alpha_results['ci_95_upper'] = float(np.mean(losses) + margin)

        results.append(alpha_results)

        print(f"\n📊 α={alpha:.2f} Summary:")
        print(f"  Mean: {alpha_results['mean_val_loss']:.4f} ± {alpha_results['std_val_loss']:.4f}")
        print(f"  95% CI: [{alpha_results['ci_95_lower']:.4f}, {alpha_results['ci_95_upper']:.4f}]")
        print(f"  Range: [{alpha_results['min_val_loss']:.4f}, {alpha_results['max_val_loss']:.4f}]")

    return results


def plot_dose_response(results: List[Dict], output_path: Path):
    """Plot fine-grained dose-response curve."""
    alphas = [r['alpha'] for r in results]
    means = [r['mean_val_loss'] for r in results]
    stds = [r['std_val_loss'] for r in results]
    ci_lower = [r['ci_95_lower'] for r in results]
    ci_upper = [r['ci_95_upper'] for r in results]

    plt.figure(figsize=(12, 7))

    # Main curve with confidence intervals
    plt.plot(alphas, means, 'o-', linewidth=2, markersize=8,
             color='#2E86AB', label='Mean validation loss')
    plt.fill_between(alphas, ci_lower, ci_upper,
                     alpha=0.2, color='#2E86AB', label='95% confidence interval')

    # Mark optimal point
    min_idx = np.argmin(means)
    optimal_alpha = alphas[min_idx]
    optimal_loss = means[min_idx]
    plt.plot(optimal_alpha, optimal_loss, 'r*', markersize=20,
             label=f'Optimal: α={optimal_alpha:.2f}', zorder=10)

    # Styling
    plt.xlabel('Skip Alpha (α)', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
    plt.title('Fine-Grained Dose-Response Curve (d=3, Micro-Sweep)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add annotations
    for alpha, mean, std in zip(alphas, means, stds):
        plt.annotate(f'{mean:.4f}',
                    xy=(alpha, mean),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='#1A1A1A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Dose-response curve saved: {output_path}")
    plt.close()


def plot_all_seeds(results: List[Dict], output_path: Path):
    """Plot individual seed results to show variance."""
    plt.figure(figsize=(12, 7))

    # Plot each seed as a separate line
    alphas = [r['alpha'] for r in results]
    seeds = results[0]['seed_results']
    num_seeds = len(seeds)

    colors = sns.color_palette("husl", num_seeds)

    for seed_idx in range(num_seeds):
        seed_losses = [r['seed_results'][seed_idx]['val_loss'] for r in results]
        seed_num = results[0]['seed_results'][seed_idx]['seed']
        plt.plot(alphas, seed_losses, 'o-', alpha=0.6,
                linewidth=1.5, markersize=6,
                color=colors[seed_idx],
                label=f'Seed {seed_num}')

    # Plot mean
    means = [r['mean_val_loss'] for r in results]
    plt.plot(alphas, means, 'k-', linewidth=3, markersize=10,
             marker='D', label='Mean', zorder=10)

    plt.xlabel('Skip Alpha (α)', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
    plt.title('Seed-by-Seed Dose-Response (d=3, Micro-Sweep)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Seed-by-seed plot saved: {output_path}")
    plt.close()


def main():
    # Configuration
    alpha_values = [0.60, 0.62, 0.64, 0.65, 0.66, 0.68]
    skip_distance = 3
    seeds = [42, 123, 456]

    # Output directory
    output_dir = Path("logs/microsweep_alpha")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run micro-sweep
    results = microsweep_alpha(
        alpha_values=alpha_values,
        skip_distance=skip_distance,
        seeds=seeds,
        epochs=5,  # Match multiseed_validation
        batch_size=4,  # Match multiseed_validation
        train_samples=2000,
        val_samples=500,
    )

    # Find optimal
    means = [r['mean_val_loss'] for r in results]
    min_idx = np.argmin(means)
    optimal = results[min_idx]

    # Summary
    print("\n" + "=" * 80)
    print("MICRO-SWEEP SUMMARY")
    print("=" * 80)
    print(f"\n{'Alpha':<10} {'Mean Loss':<15} {'Std':<10} {'95% CI':<25}")
    print("-" * 70)
    for r in results:
        ci = f"[{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}]"
        print(f"{r['alpha']:<10.2f} {r['mean_val_loss']:<15.4f} "
              f"{r['std_val_loss']:<10.4f} {ci:<25}")

    print("\n" + "=" * 80)
    print("🏆 OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"Alpha: {optimal['alpha']:.2f}")
    print(f"Mean validation loss: {optimal['mean_val_loss']:.4f} ± {optimal['std_val_loss']:.4f}")
    print(f"95% CI: [{optimal['ci_95_lower']:.4f}, {optimal['ci_95_upper']:.4f}]")
    print(f"Improvement over baseline (0.5634): {(1 - optimal['mean_val_loss']/0.5634)*100:.2f}%")

    # Save results
    results_file = output_dir / "microsweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'alpha_values': alpha_values,
                'skip_distance': skip_distance,
                'seeds': seeds,
                'num_experiments': len(alpha_values) * len(seeds),
            },
            'results': results,
            'optimal': {
                'alpha': optimal['alpha'],
                'mean_val_loss': optimal['mean_val_loss'],
                'std_val_loss': optimal['std_val_loss'],
            },
        }, f, indent=2)

    print(f"\n✅ Results saved: {results_file}")

    # Generate plots
    plot_dose_response(results, output_dir / "dose_response_curve.png")
    plot_all_seeds(results, output_dir / "seed_by_seed.png")

    print("\n" + "=" * 80)
    print("✅ MICRO-SWEEP COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"  - microsweep_results.json")
    print(f"  - dose_response_curve.png")
    print(f"  - seed_by_seed.png")


if __name__ == "__main__":
    main()

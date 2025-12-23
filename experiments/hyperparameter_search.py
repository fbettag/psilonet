"""
Comprehensive hyperparameter search for psychedelic skip-layer architecture.

Systematically explores:
- Skip distance (2, 3, 4, 5 layers)
- Skip alpha (0.3, 0.5, 0.7, 1.0)
- Learning rate (5e-5, 1e-4, 2e-4, 3e-4)
- Skip start layer (2, 3, 4, 5)
- Multiple skip connections

Goal: Find the optimal configuration for maximum validation loss improvement.
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import itertools

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from experiments.utils import load_fineweb_edu, train_epoch, evaluate
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def run_single_experiment(config: dict, train_data, val_data):
    """
    Run a single hyperparameter configuration.

    Args:
        config: Dict with keys: skip_distance, skip_alpha, learning_rate,
                skip_start_layer, epochs, batch_size
        train_data: Training dataset
        val_data: Validation dataset

    Returns:
        Dict with results: best_val_loss, final_train_loss, training_time, etc.
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(config, indent=2)}")

    start_time = time.time()

    # Create model with specified configuration
    model = create_pretrained_psychedelic_model(
        skip_distance=config['skip_distance'],
        skip_alpha=config['skip_alpha'],
        skip_start_layer=config['skip_start_layer'],
    )

    # Freeze baseline weights (we know this works best from Stage 1/2)
    model.freeze_baseline_weights()

    # Setup optimizer
    optimizer = optim.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=0.0,
    )

    # Training metrics
    metrics = []
    best_val_loss = float('inf')
    best_epoch = 0

    # Train
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n📊 Epoch {epoch}/{config['epochs']}")

        # Train epoch
        train_loss, train_ppl, tokens_per_sec = train_epoch(
            model=model,
            optimizer=optimizer,
            data=train_data,
            batch_size=config['batch_size'],
            max_norm=1.0,
        )

        # Evaluate
        val_loss, val_ppl = evaluate(
            model=model,
            data=val_data,
            batch_size=config['batch_size'],
        )

        # Track metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_perplexity": float(train_ppl),
            "val_perplexity": float(val_ppl),
            "tokens_per_sec": float(tokens_per_sec),
        }
        metrics.append(epoch_metrics)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train PPL: {train_ppl:.3f} | Val PPL: {val_ppl:.3f}")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")
        print(f"  Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    training_time = time.time() - start_time

    # Compute improvement over baseline
    # Baseline from Stage 1 starting point: 0.5634
    baseline_val_loss = 0.5634
    improvement_pct = ((baseline_val_loss - best_val_loss) / baseline_val_loss) * 100

    results = {
        "config": config,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "final_train_loss": float(metrics[-1]['train_loss']),
        "final_val_loss": float(metrics[-1]['val_loss']),
        "improvement_pct": float(improvement_pct),
        "training_time_sec": training_time,
        "all_metrics": metrics,
    }

    print(f"\n{'='*80}")
    print(f"RESULTS: {config['name']}")
    print(f"{'='*80}")
    print(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Improvement: {improvement_pct:+.2f}%")
    print(f"Training Time: {training_time/60:.1f} minutes")

    return results


def phase1_architecture_sweep(train_data, val_data, base_config: dict):
    """
    Phase 1: Sweep skip_distance and skip_alpha.

    This is the most important phase - these parameters define the
    psychedelic connectivity pattern.
    """
    print("\n" + "="*80)
    print("PHASE 1: ARCHITECTURE SWEEP (Skip Distance × Alpha)")
    print("="*80)

    skip_distances = [2, 3, 4, 5]
    skip_alphas = [0.3, 0.5, 0.7, 1.0]

    results = []
    total_experiments = len(skip_distances) * len(skip_alphas)
    current = 0

    for distance, alpha in itertools.product(skip_distances, skip_alphas):
        current += 1
        config = base_config.copy()
        config.update({
            "name": f"phase1_dist{distance}_alpha{alpha:.1f}",
            "skip_distance": distance,
            "skip_alpha": alpha,
            "phase": "1_architecture",
        })

        print(f"\n[{current}/{total_experiments}] Testing: distance={distance}, alpha={alpha}")

        result = run_single_experiment(config, train_data, val_data)
        results.append(result)

        # Save intermediate results
        save_path = Path("logs") / f"hypersearch_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def phase2_learning_rate_sweep(train_data, val_data, base_config: dict, best_arch: dict):
    """
    Phase 2: Optimize learning rate using best architecture from Phase 1.
    """
    print("\n" + "="*80)
    print("PHASE 2: LEARNING RATE OPTIMIZATION")
    print(f"Using best architecture: distance={best_arch['skip_distance']}, alpha={best_arch['skip_alpha']}")
    print("="*80)

    learning_rates = [5e-5, 1e-4, 2e-4, 3e-4]
    batch_sizes = [4, 8]

    results = []
    total_experiments = len(learning_rates) * len(batch_sizes)
    current = 0

    for lr, bs in itertools.product(learning_rates, batch_sizes):
        current += 1
        config = base_config.copy()
        config.update({
            "name": f"phase2_lr{lr:.0e}_bs{bs}",
            "skip_distance": best_arch['skip_distance'],
            "skip_alpha": best_arch['skip_alpha'],
            "learning_rate": lr,
            "batch_size": bs,
            "phase": "2_learning_rate",
        })

        print(f"\n[{current}/{total_experiments}] Testing: lr={lr:.0e}, batch_size={bs}")

        result = run_single_experiment(config, train_data, val_data)
        results.append(result)

        # Save intermediate results
        save_path = Path("logs") / f"hypersearch_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def phase3_advanced_architectures(train_data, val_data, base_config: dict, best_config: dict):
    """
    Phase 3: Test advanced architecture variations.
    - Different skip start layers
    - (Future: Multiple skip connections)
    """
    print("\n" + "="*80)
    print("PHASE 3: ADVANCED ARCHITECTURE VARIATIONS")
    print(f"Using best config so far: {best_config['name']}")
    print("="*80)

    skip_start_layers = [2, 3, 4, 5]

    results = []

    for start_layer in skip_start_layers:
        config = base_config.copy()
        config.update({
            "name": f"phase3_start{start_layer}",
            "skip_distance": best_config['skip_distance'],
            "skip_alpha": best_config['skip_alpha'],
            "learning_rate": best_config['learning_rate'],
            "batch_size": best_config['batch_size'],
            "skip_start_layer": start_layer,
            "phase": "3_advanced",
        })

        print(f"\nTesting: skip_start_layer={start_layer}")

        result = run_single_experiment(config, train_data, val_data)
        results.append(result)

        # Save intermediate results
        save_path = Path("logs") / f"hypersearch_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    """
    Run complete hyperparameter search.
    """
    print("="*80)
    print("PSYCHEDELIC LLM HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data (same as Stage 1)
    print("Loading FineWeb-Edu dataset...")
    train_data = load_fineweb_edu(split="train", num_samples=2000, seed=42)
    val_data = load_fineweb_edu(split="validation", num_samples=500, seed=42)
    print(f"✓ Loaded {len(train_data)} train samples, {len(val_data)} val samples\n")

    # Base configuration (defaults)
    base_config = {
        "epochs": 5,  # Same as successful Stage 1
        "batch_size": 4,
        "learning_rate": 1e-4,
        "skip_distance": 3,
        "skip_alpha": 0.5,
        "skip_start_layer": 3,
        "train_samples": 2000,
        "val_samples": 500,
    }

    # Phase 1: Architecture sweep (most important)
    phase1_results = phase1_architecture_sweep(train_data, val_data, base_config)

    # Find best from Phase 1
    best_phase1 = min(phase1_results, key=lambda x: x['best_val_loss'])
    print(f"\n🏆 PHASE 1 WINNER: {best_phase1['config']['name']}")
    print(f"   Val Loss: {best_phase1['best_val_loss']:.4f}")
    print(f"   Improvement: {best_phase1['improvement_pct']:+.2f}%")

    # Phase 2: Learning rate optimization
    phase2_results = phase2_learning_rate_sweep(
        train_data, val_data, base_config, best_phase1['config']
    )

    # Find best from Phase 2
    best_phase2 = min(phase2_results, key=lambda x: x['best_val_loss'])
    print(f"\n🏆 PHASE 2 WINNER: {best_phase2['config']['name']}")
    print(f"   Val Loss: {best_phase2['best_val_loss']:.4f}")
    print(f"   Improvement: {best_phase2['improvement_pct']:+.2f}%")

    # Phase 3: Advanced architectures
    phase3_results = phase3_advanced_architectures(
        train_data, val_data, base_config, best_phase2['config']
    )

    # Find best from Phase 3
    best_phase3 = min(phase3_results, key=lambda x: x['best_val_loss'])
    print(f"\n🏆 PHASE 3 WINNER: {best_phase3['config']['name']}")
    print(f"   Val Loss: {best_phase3['best_val_loss']:.4f}")
    print(f"   Improvement: {best_phase3['improvement_pct']:+.2f}%")

    # Overall best
    all_results = phase1_results + phase2_results + phase3_results
    overall_best = min(all_results, key=lambda x: x['best_val_loss'])

    print("\n" + "="*80)
    print("🎉 OVERALL BEST CONFIGURATION")
    print("="*80)
    print(json.dumps(overall_best['config'], indent=2))
    print(f"\nBest Val Loss: {overall_best['best_val_loss']:.4f}")
    print(f"Improvement: {overall_best['improvement_pct']:+.2f}%")
    print(f"Training Time: {overall_best['training_time_sec']/60:.1f} minutes")

    # Save final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "phase1_results": phase1_results,
        "phase2_results": phase2_results,
        "phase3_results": phase3_results,
        "best_overall": overall_best,
        "baseline_val_loss": 0.5634,
    }

    save_path = Path("logs") / "hypersearch_final_results.json"
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved to {save_path}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

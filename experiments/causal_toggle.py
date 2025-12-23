"""
Causal Toggle Experiment: Ablation Study at Inference

Goal: Prove causal contribution of skip-layer attention by toggling it on/off at inference.

Method:
1. Load trained model with optimal config (d=3, α=0.65)
2. Run inference with skip connections ON (normal)
3. Run inference with skip connections OFF (ablation)
4. Measure perplexity degradation → proves causal contribution

Optional: Selective layer-wise ablation to identify most critical skip connections.
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model
from experiments.multiseed_validation import prepare_dataset, evaluate
import mlx.core as mx
import mlx.optimizers as optim


def evaluate_with_toggle(
    model,
    dataset,
    batch_size: int = 8,
    skip_enabled: bool = True,
) -> float:
    """
    Evaluate model with skip connections toggled on/off.

    Args:
        model: PsychedelicSkipModel instance
        dataset: Validation dataset
        batch_size: Batch size for evaluation
        skip_enabled: Whether to enable skip connections

    Returns:
        Validation loss
    """
    # Store original skip_alpha values
    original_alphas = []
    for layer in model.layers:
        if hasattr(layer, 'skip_alpha'):
            original_alphas.append(float(layer.skip_alpha))

    # Toggle skip connections
    if not skip_enabled:
        for layer in model.layers:
            if hasattr(layer, 'skip_alpha'):
                # Set alpha to 0 to disable skip connections
                layer.skip_alpha = mx.array(0.0)

    # Evaluate
    val_loss, _ = evaluate(model, dataset, batch_size=batch_size)

    # Restore original alphas
    if not skip_enabled:
        layer_idx = 0
        for layer in model.layers:
            if hasattr(layer, 'skip_alpha'):
                layer.skip_alpha = mx.array(original_alphas[layer_idx])
                layer_idx += 1

    return val_loss


def layer_wise_ablation(
    model,
    dataset,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Ablate skip connections layer-by-layer to find most critical layers.

    Returns:
        List of dicts with layer_idx and val_loss when that layer is ablated
    """
    results = []

    # Find all layers with skip connections
    skip_layers = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'skip_alpha'):
            skip_layers.append(idx)

    print(f"Found {len(skip_layers)} layers with skip connections")
    print(f"Layer indices: {skip_layers}")

    # Ablate each layer individually
    for layer_idx in skip_layers:
        # Store original alpha
        layer = model.layers[layer_idx]
        original_alpha = float(layer.skip_alpha)

        # Disable this layer's skip connection
        layer.skip_alpha = mx.array(0.0)

        # Evaluate
        val_loss, _ = evaluate(model, dataset, batch_size=batch_size)

        results.append({
            'layer_idx': layer_idx,
            'val_loss': float(val_loss),
        })

        print(f"  Layer {layer_idx}: val_loss={val_loss:.4f} (skip disabled)")

        # Restore alpha
        layer.skip_alpha = mx.array(original_alpha)

    return results


def train_model_once(
    skip_distance: int = 3,
    skip_alpha: float = 0.65,
    epochs: int = 10,
    batch_size: int = 8,
    train_samples: int = 2000,
    seed: int = 42,
):
    """
    Train a single model to be used for ablation experiments.

    Returns:
        Trained model
    """
    print(f"🔧 Training model: d={skip_distance}, α={skip_alpha}, seed={seed}")

    # Load dataset
    train_dataset = prepare_dataset(num_samples=train_samples, split="train", seed=seed)

    # Create model
    model = create_pretrained_psychedelic_model(
        skip_distance=skip_distance,
        skip_alpha=skip_alpha,
        skip_start_layer=3,
    )
    model.freeze_baseline_weights()  # CRITICAL: Only train skip connections!

    # Set up optimizer
    optimizer = optim.AdamW(learning_rate=1e-4)  # Match multiseed_validation

    # Import train_epoch
    from experiments.multiseed_validation import train_epoch

    # Train
    for epoch in range(epochs):
        train_loss, perplexity, tokens_per_sec = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            batch_size=batch_size,
            max_norm=1.0,
            seed=seed + epoch,
        )

        if epoch % 2 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, ppl={perplexity:.2f}")

    print(f"✅ Training complete")

    # Cleanup optimizer but keep model
    del optimizer
    mx.clear_cache()

    return model


def causal_toggle_experiment(
    output_dir: Path,
    skip_distance: int = 3,
    skip_alpha: float = 0.65,
    epochs: int = 10,
    batch_size: int = 8,
    train_samples: int = 2000,
    val_samples: int = 500,
    seed: int = 42,
    do_layerwise: bool = True,
):
    """
    Run causal toggle experiment.

    Steps:
    1. Train model with optimal config
    2. Evaluate with skip ON
    3. Evaluate with skip OFF
    4. Compare perplexity degradation
    5. Optional: Layer-wise ablation
    """
    print("=" * 80)
    print("CAUSAL TOGGLE EXPERIMENT")
    print("=" * 80)
    print(f"Configuration: d={skip_distance}, α={skip_alpha}")
    print(f"Training: {epochs} epochs, {train_samples} samples")
    print(f"Seed: {seed}")
    print("=" * 80)

    # Load validation dataset
    val_dataset = prepare_dataset(num_samples=val_samples, split="validation", seed=seed)

    # Train model
    print("\n📚 Step 1: Training model...")
    model = train_model_once(
        skip_distance=skip_distance,
        skip_alpha=skip_alpha,
        epochs=epochs,
        batch_size=batch_size,
        train_samples=train_samples,
        seed=seed,
    )

    # Evaluate with skip connections ON
    print("\n📊 Step 2: Evaluating with skip connections ENABLED...")
    loss_skip_on = evaluate_with_toggle(
        model=model,
        dataset=val_dataset,
        batch_size=batch_size,
        skip_enabled=True,
    )
    ppl_skip_on = np.exp(loss_skip_on)
    print(f"✅ Skip ON:  loss={loss_skip_on:.4f}, perplexity={ppl_skip_on:.2f}")

    # Evaluate with skip connections OFF
    print("\n📊 Step 3: Evaluating with skip connections DISABLED...")
    loss_skip_off = evaluate_with_toggle(
        model=model,
        dataset=val_dataset,
        batch_size=batch_size,
        skip_enabled=False,
    )
    ppl_skip_off = np.exp(loss_skip_off)
    print(f"✅ Skip OFF: loss={loss_skip_off:.4f}, perplexity={ppl_skip_off:.2f}")

    # Compute degradation
    loss_delta = loss_skip_off - loss_skip_on
    ppl_delta = ppl_skip_off - ppl_skip_on
    ppl_pct_increase = (ppl_skip_off / ppl_skip_on - 1) * 100

    print("\n" + "=" * 80)
    print("📈 CAUSAL CONTRIBUTION")
    print("=" * 80)
    print(f"Loss increase:       {loss_delta:+.4f} ({loss_delta/loss_skip_on*100:+.2f}%)")
    print(f"Perplexity increase: {ppl_delta:+.2f} ({ppl_pct_increase:+.2f}%)")
    print("=" * 80)

    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'skip_distance': skip_distance,
            'skip_alpha': skip_alpha,
            'epochs': epochs,
            'batch_size': batch_size,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'seed': seed,
        },
        'toggle_results': {
            'skip_on': {
                'loss': float(loss_skip_on),
                'perplexity': float(ppl_skip_on),
            },
            'skip_off': {
                'loss': float(loss_skip_off),
                'perplexity': float(ppl_skip_off),
            },
            'degradation': {
                'loss_delta': float(loss_delta),
                'loss_pct': float(loss_delta / loss_skip_on * 100),
                'ppl_delta': float(ppl_delta),
                'ppl_pct': float(ppl_pct_increase),
            },
        },
    }

    # Layer-wise ablation (optional)
    if do_layerwise:
        print("\n" + "=" * 80)
        print("📊 Step 4: Layer-wise ablation analysis...")
        print("=" * 80)
        layerwise_results = layer_wise_ablation(
            model=model,
            dataset=val_dataset,
            batch_size=batch_size,
        )

        # Add baseline (all skip ON) for comparison
        layerwise_results.insert(0, {
            'layer_idx': -1,  # -1 means "all layers ON"
            'val_loss': float(loss_skip_on),
        })

        results['layerwise_ablation'] = layerwise_results

        # Find most critical layers
        ablated_losses = [r['val_loss'] for r in layerwise_results[1:]]  # Skip baseline
        most_critical_idx = np.argmax(ablated_losses)
        most_critical = layerwise_results[most_critical_idx + 1]

        print(f"\n🎯 Most critical layer: {most_critical['layer_idx']}")
        print(f"   Loss when ablated: {most_critical['val_loss']:.4f}")
        print(f"   Degradation: {most_critical['val_loss'] - loss_skip_on:.4f}")

    # Cleanup
    del model
    mx.clear_cache()

    return results


def plot_causal_contribution(results: Dict, output_path: Path):
    """Plot causal toggle results."""
    skip_on = results['toggle_results']['skip_on']
    skip_off = results['toggle_results']['skip_off']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Loss comparison
    conditions = ['Skip ON\n(Normal)', 'Skip OFF\n(Ablated)']
    losses = [skip_on['loss'], skip_off['loss']]
    colors = ['#2E86AB', '#E63946']

    bars1 = ax1.bar(conditions, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Causal Toggle: Loss Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add degradation annotation
    degradation_pct = results['toggle_results']['degradation']['loss_pct']
    ax1.annotate('', xy=(0.5, skip_off['loss']), xytext=(0.5, skip_on['loss']),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(0.5, (skip_on['loss'] + skip_off['loss'])/2,
            f'+{degradation_pct:.1f}%',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2),
            fontsize=11, fontweight='bold', color='red')

    # Plot 2: Perplexity comparison
    ppls = [skip_on['perplexity'], skip_off['perplexity']]
    bars2 = ax2.bar(conditions, ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Causal Toggle: Perplexity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels
    for bar, ppl in zip(bars2, ppls):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add degradation annotation
    ppl_pct = results['toggle_results']['degradation']['ppl_pct']
    ax2.annotate('', xy=(0.5, skip_off['perplexity']), xytext=(0.5, skip_on['perplexity']),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(0.5, (skip_on['perplexity'] + skip_off['perplexity'])/2,
            f'+{ppl_pct:.1f}%',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2),
            fontsize=11, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Causal toggle plot saved: {output_path}")
    plt.close()


def plot_layerwise_ablation(results: Dict, output_path: Path):
    """Plot layer-wise ablation results."""
    if 'layerwise_ablation' not in results:
        return

    layerwise = results['layerwise_ablation']
    baseline = layerwise[0]  # All skip ON
    ablated = layerwise[1:]  # Individual layers ablated

    layer_indices = [r['layer_idx'] for r in ablated]
    losses = [r['val_loss'] for r in ablated]
    degradations = [loss - baseline['val_loss'] for loss in losses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Absolute losses
    ax1.axhline(y=baseline['val_loss'], color='green', linestyle='--', linewidth=2,
                label=f"Baseline (all skip ON): {baseline['val_loss']:.4f}", zorder=1)
    ax1.bar(layer_indices, losses, color='#E63946', alpha=0.7, edgecolor='black', linewidth=1, zorder=2)
    ax1.set_xlabel('Layer Index (Ablated)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Layer-wise Ablation: Absolute Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Plot 2: Degradation from baseline
    colors_grad = ['#E63946' if d > 0 else '#2E86AB' for d in degradations]
    ax2.bar(layer_indices, degradations, color=colors_grad, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer Index (Ablated)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Degradation (Δ from baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Layer-wise Ablation: Degradation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Mark most critical layer
    most_critical_idx = np.argmax(degradations)
    ax2.bar(layer_indices[most_critical_idx], degradations[most_critical_idx],
           color='red', alpha=0.9, edgecolor='black', linewidth=2, label='Most critical')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Layer-wise ablation plot saved: {output_path}")
    plt.close()


def main():
    # Configuration
    output_dir = Path("logs/causal_toggle")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run experiment with optimal α=0.66 from micro-sweep
    results = causal_toggle_experiment(
        output_dir=output_dir,
        skip_distance=3,
        skip_alpha=0.66,  # Optimal: 8.41% improvement, CI=[0.5128, 0.5193]
        epochs=5,  # Match multiseed_validation
        batch_size=4,  # Match multiseed_validation
        train_samples=2000,
        val_samples=500,
        seed=42,
        do_layerwise=True,
    )

    # Save results
    results_file = output_dir / "causal_toggle_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {results_file}")

    # Generate plots
    plot_causal_contribution(results, output_dir / "causal_toggle_comparison.png")
    plot_layerwise_ablation(results, output_dir / "layerwise_ablation.png")

    print("\n" + "=" * 80)
    print("✅ CAUSAL TOGGLE EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"  - causal_toggle_results.json")
    print(f"  - causal_toggle_comparison.png")
    print(f"  - layerwise_ablation.png")


if __name__ == "__main__":
    main()

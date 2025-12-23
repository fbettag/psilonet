"""
CKA (Centered Kernel Alignment) analysis of psychedelic skip-layer representations.

Goal: Understand why skip_distance=3 is optimal by analyzing layer similarity patterns.

Analysis:
1. Compute CKA between all layer pairs for baseline and psychedelic models
2. Visualize similarity matrices
3. Measure how skip connections change representation similarity
4. Identify "sweet spot" distance where novelty is maximal without diffusion

Based on: "Similarity of Neural Network Representations Revisited" (Kornblith et al., 2019)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import create_pretrained_psychedelic_model, create_baseline_model
from modules.psychedelic_smollm import load_tokenizer
from experiments.utils import batch_iterator
import mlx.core as mx
from datasets import load_dataset
from tqdm import tqdm


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two sets of representations.

    Args:
        X: (n_samples, dim_x) - representations from layer X
        Y: (n_samples, dim_y) - representations from layer Y

    Returns:
        CKA similarity score (0 to 1, higher = more similar)

    CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    where HSIC is Hilbert-Schmidt Independence Criterion
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute kernel matrices (linear kernel = dot product)
    K = X @ X.T  # (n, n)
    L = Y @ Y.T  # (n, n)

    # HSIC computation
    hsic_xy = np.trace(K @ L)
    hsic_xx = np.trace(K @ K)
    hsic_yy = np.trace(L @ L)

    # CKA
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)

    return cka


def extract_layer_representations(model, dataset, batch_size: int = 8, max_samples: int = 500):
    """
    Extract hidden representations from all layers for a given dataset.

    Returns:
        List of (n_samples, seq_len, hidden_dim) arrays, one per layer
    """
    print(f"🔍 Extracting representations from {len(model.layers)} layers...")

    # Storage for all layer representations
    all_layer_reps = [[] for _ in range(len(model.layers))]

    # Whether skip paths are active for this model
    enable_skip = getattr(model, "enable_skip_layers", False)
    skip_distance = getattr(model, "skip_distance", 0)
    skip_start = getattr(model, "skip_start_layer", 10**9)

    num_samples = 0
    batches = list(batch_iterator(dataset, batch_size, shuffle=False))

    for batch in tqdm(batches[:max_samples // batch_size], desc="Extracting reps"):
        input_ids = mx.array([item['input_ids'] for item in batch])
        seq_len = input_ids.shape[1]

        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)

        # Forward pass with hooks to capture all layer outputs
        hidden_states = model.embed_tokens(input_ids)
        history = [hidden_states]

        for layer_idx, layer in enumerate(model.layers):
            skip_x = None
            if enable_skip and layer_idx >= skip_start and layer_idx >= skip_distance:
                skip_x = history[layer_idx - skip_distance]

            hidden_states = layer(hidden_states, skip_x=skip_x, mask=mask)
            history.append(hidden_states)

            # Store flattened representation (batch_size * seq_len, hidden_dim)
            rep = hidden_states.reshape(-1, hidden_states.shape[-1])
            all_layer_reps[layer_idx].append(np.array(rep))

        num_samples += len(batch)
        if num_samples >= max_samples:
            break

    # Concatenate batches
    layer_reps = []
    for layer_idx in range(len(model.layers)):
        reps = np.concatenate(all_layer_reps[layer_idx], axis=0)
        layer_reps.append(reps)
        print(f"  Layer {layer_idx}: {reps.shape}")

    return layer_reps


def compute_cka_matrix(layer_reps: List[np.ndarray], subsample: int = 2000):
    """
    Compute CKA similarity matrix between all layer pairs (memory-efficient).

    Args:
        layer_reps: List of (n_samples, hidden_dim) arrays
        subsample: Number of samples to use (for computational efficiency)

    Returns:
        (n_layers, n_layers) CKA matrix
    """
    n_layers = len(layer_reps)
    cka_matrix = np.zeros((n_layers, n_layers))

    print(f"\n📊 Computing {n_layers}x{n_layers} CKA matrix...")

    for i in tqdm(range(n_layers), desc="CKA computation"):
        for j in range(i, n_layers):
            # Subsample for efficiency
            X = layer_reps[i][:subsample]
            Y = layer_reps[j][:subsample]

            cka = linear_CKA(X, Y)
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka  # Symmetric

    return cka_matrix


def plot_cka_matrix(cka_matrix: np.ndarray, title: str, save_path: Path):
    """Plot CKA similarity matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cka_matrix,
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'CKA Similarity'},
        xticklabels=range(len(cka_matrix)),
        yticklabels=range(len(cka_matrix))
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_cka_difference(cka_baseline: np.ndarray, cka_psychedelic: np.ndarray,
                       skip_distance: int, save_path: Path):
    """Plot difference between psychedelic and baseline CKA matrices."""
    difference = cka_psychedelic - cka_baseline

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        difference,
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        square=True,
        cbar_kws={'label': 'CKA Difference (Psychedelic - Baseline)'},
        xticklabels=range(len(difference)),
        yticklabels=range(len(difference))
    )
    plt.title(f'CKA Difference: Skip Distance = {skip_distance}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def analyze_skip_novelty(cka_matrix: np.ndarray, skip_distance: int):
    """
    Measure how much "novelty" skip connections introduce.

    Novelty = 1 - CKA(layer_i, layer_{i-skip_distance})

    Higher novelty = layer i-d provides different information than layer i would naturally see
    """
    n_layers = len(cka_matrix)
    novelties = []

    for i in range(skip_distance, n_layers):
        cka_similarity = cka_matrix[i, i - skip_distance]
        novelty = 1 - cka_similarity
        novelties.append(novelty)

    mean_novelty = np.mean(novelties)
    std_novelty = np.std(novelties)

    return {
        'skip_distance': skip_distance,
        'mean_novelty': float(mean_novelty),
        'std_novelty': float(std_novelty),
        'per_layer_novelty': novelties,
    }


def prepare_dataset(num_samples: int = 200, seed: int = 42):
    """Prepare WikiText-2 validation set for analysis (memory-efficient)."""
    print(f"📚 Loading WikiText-2 validation ({num_samples} samples)...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_samples}]")
    tokenizer = load_tokenizer()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="np"
        )

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    dataset_list = []
    for item in tokenized:
        input_ids = item["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        dataset_list.append({"input_ids": input_ids})

    print(f"✅ Dataset ready: {len(dataset_list)} examples")
    return dataset_list


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CKA representation analysis")
    parser.add_argument("--distance", type=int, default=None, help="Skip distance to analyze (overrides defaults)")
    parser.add_argument("--alpha", type=float, default=None, help="Skip alpha to analyze (used with --distance)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned checkpoint for the skip model")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of validation samples for CKA")
    args = parser.parse_args()
    print("="*80)
    print("CKA REPRESENTATION ANALYSIS")
    print("="*80)

    # Create output directory
    output_dir = Path("logs/cka_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data (memory-efficient: reduced from 500)
    dataset = prepare_dataset(num_samples=args.num_samples, seed=42)

    # Configurations to analyze
    if args.distance is not None and args.alpha is not None:
        configs = [{
            'distance': args.distance,
            'alpha': args.alpha,
            'name': f'd{args.distance}_a{args.alpha} (custom)',
            'checkpoint': args.checkpoint,
        }]
    else:
        configs = [
            {'distance': 2, 'alpha': 0.5, 'name': 'd2_a0.5 (unstable)'},
            {'distance': 3, 'alpha': 0.5, 'name': 'd3_a0.5 (Stage 1)'},
            {'distance': 3, 'alpha': 0.65, 'name': 'd3_a0.65 (multi-seed winner)'},
            {'distance': 3, 'alpha': 0.7, 'name': 'd3_a0.7'},
            {'distance': 4, 'alpha': 0.7, 'name': 'd4_a0.7 (runner-up)'},
            {'distance': 5, 'alpha': 0.5, 'name': 'd5_a0.5 (degraded)'},
        ]

    # Baseline model (no skip connections)
    print("\n" + "="*80)
    print("BASELINE MODEL (no skip connections)")
    print("="*80)

    baseline_model = create_baseline_model()
    baseline_reps = extract_layer_representations(baseline_model, dataset, batch_size=4, max_samples=200)
    baseline_cka = compute_cka_matrix(baseline_reps, subsample=2000)

    # Save baseline CKA
    plot_cka_matrix(
        baseline_cka,
        "Baseline Model: CKA Similarity Matrix",
        output_dir / "cka_baseline.png"
    )

    # Cleanup baseline model to free memory
    del baseline_model
    del baseline_reps
    mx.clear_cache()
    gc.collect()
    print("🧹 Baseline model cleanup complete\n")

    # Analyze each psychedelic configuration
    all_results = []

    for config in configs:
        print("\n" + "="*80)
        print(f"PSYCHEDELIC MODEL: {config['name']}")
        print("="*80)

        # Create psychedelic model
        model = create_pretrained_psychedelic_model(
            skip_distance=config['distance'],
            skip_alpha=config['alpha'],
            skip_start_layer=3,
        )

        # If a finetuned checkpoint is provided, load it
        if config.get('checkpoint'):
            ckpt_path = Path(config['checkpoint'])
            weights_files = list(ckpt_path.glob("weights_epoch_*.npz"))
            if not weights_files:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
            latest_weights = max(weights_files, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"📥 Loading finetuned checkpoint: {latest_weights}")
            weights = mx.load(str(latest_weights))
            model.update(weights)

        # Note: Using pre-trained (not fine-tuned) to see architectural effects
        # Future: Can compare pre-trained vs fine-tuned

        # Extract representations (memory-efficient)
        psychedelic_reps = extract_layer_representations(model, dataset, batch_size=4, max_samples=200)
        psychedelic_cka = compute_cka_matrix(psychedelic_reps, subsample=2000)

        # Plot psychedelic CKA
        plot_cka_matrix(
            psychedelic_cka,
            f"Psychedelic Model ({config['name']}): CKA Similarity",
            output_dir / f"cka_psychedelic_{config['distance']}_{config['alpha']}.png"
        )

        # Plot difference
        plot_cka_difference(
            baseline_cka,
            psychedelic_cka,
            config['distance'],
            output_dir / f"cka_diff_{config['distance']}_{config['alpha']}.png"
        )

        # Analyze skip novelty
        novelty = analyze_skip_novelty(psychedelic_cka, config['distance'])
        novelty['config'] = config
        all_results.append(novelty)

        print(f"\n📊 Novelty Analysis:")
        print(f"  Mean novelty: {novelty['mean_novelty']:.4f} ± {novelty['std_novelty']:.4f}")
        print(f"  Interpretation: Higher = skip provides more distinct information")

        # Aggressive memory cleanup
        del model
        del psychedelic_reps
        del psychedelic_cka
        mx.clear_cache()
        gc.collect()
        print(f"🧹 Model cleanup complete ({config['name']})\n")

    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY: NOVELTY ACROSS CONFIGURATIONS")
    print("="*80)

    print(f"{'Config':<25} {'Mean Novelty':<15} {'Std':<10}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['config']['name']:<25} "
              f"{result['mean_novelty']:<15.4f} "
              f"{result['std_novelty']:<10.4f}")

    # Plot novelty comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    configs_names = [r['config']['name'] for r in all_results]
    novelties = [r['mean_novelty'] for r in all_results]
    stds = [r['std_novelty'] for r in all_results]

    ax.bar(range(len(configs_names)), novelties, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(range(len(configs_names)))
    ax.set_xticklabels(configs_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Novelty (1 - CKA)', fontsize=12)
    ax.set_title('Skip Connection Novelty Across Configurations', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "novelty_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / 'novelty_comparison.png'}")
    plt.close()

    # Save results
    results_file = output_dir / "cka_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'baseline_cka': baseline_cka.tolist(),
            'configurations': all_results,
        }, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")
    print(f"✅ Figures saved to {output_dir}/")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("1. Baseline CKA shows progressive similarity decay with distance")
    print("2. Skip connections should show:")
    print("   - d=2: Too similar (low novelty) → unstable/redundant")
    print("   - d=3-4: Optimal novelty → meaningful new information")
    print("   - d=5+: High novelty but potentially too different → diffusion")
    print("3. Alpha controls strength of skip influence on similarity patterns")
    print("\nCheck the CKA heatmaps and difference plots for visual confirmation!")


if __name__ == "__main__":
    main()

"""
Experiment utilities for tracking and evaluation
"""

import mlx.core as mx
import wandb
import time
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class ExperimentTracker:
    """
    Tracks experiments with wandb and local logging.

    Metrics tracked:
    - Training loss & perplexity
    - Attention entropy (psychedelic metric!)
    - Convergence speed
    - Parameter count
    """

    def __init__(
        self,
        project_name: str = "psychedelic-lm",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        log_dir: str = "./logs"
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = use_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Local metrics storage
        self.metrics_history = []
        self.start_time = time.time()

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config or {}
            )
            print(f"📊 Wandb tracking initialized: {wandb.run.url}")
        else:
            print(f"📊 Local tracking only (wandb disabled)")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to wandb and local storage"""
        metrics['timestamp'] = time.time() - self.start_time

        if step is not None:
            metrics['step'] = step

        self.metrics_history.append(metrics)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_model_comparison(self, baseline_stats: Dict, psychedelic_stats: Dict):
        """Log comparison between baseline and psychedelic models"""
        comparison = {
            'baseline_params': baseline_stats['total_params'],
            'psychedelic_params': psychedelic_stats['total_params'],
            'additional_params': psychedelic_stats['total_params'] - baseline_stats['total_params'],
            'skip_layers': psychedelic_stats.get('skip_layers_enabled', 0),
            'skip_distance': psychedelic_stats.get('skip_distance', 0),
            'skip_alpha': psychedelic_stats.get('skip_alpha', 0.0),
        }

        if self.use_wandb:
            wandb.config.update(comparison)

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        for key, value in comparison.items():
            print(f"{key:.<40} {value}")
        print("=" * 60 + "\n")

    def save_metrics(self):
        """Save metrics history to JSON"""
        output_path = self.log_dir / f"{self.experiment_name}_metrics.json"

        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"💾 Metrics saved to: {output_path}")

    def finish(self):
        """Finish tracking and save results"""
        self.save_metrics()

        if self.use_wandb:
            wandb.finish()

        print(f"✅ Experiment completed: {self.experiment_name}")


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss"""
    import math
    return math.exp(loss)


def create_causal_mask(seq_len: int) -> mx.array:
    """Create causal attention mask for autoregressive generation"""
    return mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)


def batch_iterator(data: List, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
    """Create batches from data"""
    import random

    if shuffle:
        data = data.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def estimate_tokens_per_second(num_tokens: int, elapsed_time: float) -> float:
    """Estimate tokens processed per second"""
    return num_tokens / elapsed_time if elapsed_time > 0 else 0


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_training_header():
    """Print training header"""
    print("\n" + "=" * 80)
    print("🍄 PSYCHEDELIC LLM TRAINING".center(80))
    print("Exploring psilocybin-inspired neural architectures".center(80))
    print("=" * 80 + "\n")


def print_epoch_stats(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    attention_entropy: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    elapsed_time: Optional[float] = None
):
    """Print epoch statistics in a nice format"""
    print(f"\n📊 Epoch {epoch}")
    print("-" * 60)
    print(f"Train Loss:     {train_loss:.4f}  |  Perplexity: {compute_perplexity(train_loss):.2f}")

    if val_loss is not None:
        print(f"Val Loss:       {val_loss:.4f}  |  Perplexity: {compute_perplexity(val_loss):.2f}")

    if attention_entropy is not None:
        print(f"Attn Entropy:   {attention_entropy:.4f}  🍄 (higher = more psychedelic)")

    if tokens_per_sec is not None:
        print(f"Speed:          {tokens_per_sec:.0f} tokens/sec")

    if elapsed_time is not None:
        print(f"Time:           {format_time(elapsed_time)}")

    print("-" * 60)

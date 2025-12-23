"""
Psychedelic-inspired SmolLM2 architecture with skip-layer attention.

This module wraps HuggingFace's SmolLM2-135M and adds psychedelic mechanisms:
- Skip-layer attention for enhanced cross-layer connectivity
- Optional entropy regularization
- Configurable skip patterns
"""

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict, List, Tuple
import json

from .skip_layer_attention import SkipLayerAttention, PsychedelicTransformerBlock


class PsychedelicSmolLM(nn.Module):
    """
    SmolLM2-135M with psychedelic-inspired modifications.

    Architecture modifications:
    - Skip-layer attention connections (3-layer skip distance)
    - Deep & thin design (inherited from SmolLM2)
    - Grouped-Query Attention (GQA) for efficiency

    Args:
        config: Model configuration dict or path to config
        skip_distance: Number of layers to skip (default: 3)
        skip_alpha: Weight for skip connections (default: 0.5)
        enable_skip_layers: Enable psychedelic skip connections
        skip_start_layer: First layer to enable skip connections (default: 3)
    """

    def __init__(
        self,
        config: Dict,
        skip_distance: int = 3,
        skip_alpha: float = 0.5,
        enable_skip_layers: bool = True,
        skip_start_layer: int = 3,
    ):
        super().__init__()

        self.config = config
        self.vocab_size = config.get('vocab_size', 49152)
        self.hidden_size = config.get('hidden_size', 576)
        self.num_layers = config.get('num_hidden_layers', 30)
        self.num_heads = config.get('num_attention_heads', 9)
        self.intermediate_size = config.get('intermediate_size', 1536)
        self.max_position_embeddings = config.get('max_position_embeddings', 2048)

        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha
        self.enable_skip_layers = enable_skip_layers
        self.skip_start_layer = skip_start_layer

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer blocks with optional skip-layer attention
        self.layers = []
        for i in range(self.num_layers):
            # Enable skip connections after skip_start_layer
            use_skip = enable_skip_layers and i >= skip_start_layer

            if use_skip:
                layer = PsychedelicTransformerBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.intermediate_size / self.hidden_size,
                    skip_distance=skip_distance,
                    dropout=0.0,
                    skip_alpha=skip_alpha
                )
            else:
                # Standard transformer block without skip connections
                layer = PsychedelicTransformerBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.intermediate_size / self.hidden_size,
                    skip_distance=0,  # No skip
                    dropout=0.0,
                    skip_alpha=0.0
                )

            self.layers.append(layer)

        # Final layer norm
        self.norm = nn.LayerNorm(self.hidden_size)

        # Language model head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_layer_outputs: bool = False
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        """
        Forward pass with psychedelic skip-layer connections.

        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Optional attention mask
            return_layer_outputs: Return intermediate layer outputs for analysis

        Returns:
            logits: Output logits (B, L, V)
            layer_outputs: Optional list of layer outputs for analysis
        """
        B, L = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Store layer outputs for skip connections
        layer_outputs = [hidden_states]

        # Create causal mask
        if attention_mask is None:
            # Causal mask: prevent attending to future tokens
            mask = mx.triu(mx.full((L, L), float('-inf')), k=1)
        else:
            mask = attention_mask

        # Pass through transformer layers with skip connections
        for i, layer in enumerate(self.layers):
            # Get skip connection input if available
            skip_idx = i - self.skip_distance
            skip_input = None

            if self.enable_skip_layers and i >= self.skip_start_layer and skip_idx >= 0:
                skip_input = layer_outputs[skip_idx]

            # Forward through layer
            hidden_states = layer(hidden_states, skip_x=skip_input, mask=mask)
            layer_outputs.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)

        if return_layer_outputs:
            return logits, layer_outputs
        return logits, None

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_layer_outputs: bool = False
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        """Make the module callable"""
        return self.forward(input_ids, attention_mask, return_layer_outputs)

    def get_architecture_stats(self) -> Dict:
        """Return statistics about the architecture"""
        # Count parameters - calculate from architecture instead of materialized params
        # This works even before parameters are initialized
        total_params = 0

        # Embeddings
        total_params += self.vocab_size * self.hidden_size  # token embeddings

        # Each transformer layer
        for i in range(self.num_layers):
            # Attention (Q, K, V projections + output projection)
            total_params += 4 * self.hidden_size * self.hidden_size

            # Skip-layer attention (if enabled)
            if self.enable_skip_layers and i >= self.skip_start_layer:
                total_params += 4 * self.hidden_size * self.hidden_size  # skip Q, K, V + output

            # Layer norms (2 per block)
            total_params += 2 * self.hidden_size * 2  # scale + bias

            # MLP
            total_params += self.hidden_size * self.intermediate_size  # up projection
            total_params += self.intermediate_size * self.hidden_size  # down projection

        # Final layer norm
        total_params += self.hidden_size * 2

        # LM head (often tied with embeddings, but count separately)
        total_params += self.vocab_size * self.hidden_size

        skip_layers = sum(1 for i in range(self.num_layers)
                         if i >= self.skip_start_layer and self.enable_skip_layers)

        return {
            'model_type': 'Psychedelic SmolLM2' if self.enable_skip_layers else 'Baseline SmolLM2',
            'total_params': total_params,
            'total_params_millions': total_params / 1e6,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'skip_layers_enabled': skip_layers,
            'skip_distance': self.skip_distance if self.enable_skip_layers else 0,
            'skip_alpha': self.skip_alpha if self.enable_skip_layers else 0.0,
        }


def load_smollm2_config(model_name: str = "HuggingFaceTB/SmolLM2-135M") -> Dict:
    """Load SmolLM2 configuration from HuggingFace"""
    config = AutoConfig.from_pretrained(model_name)
    return config.to_dict()


def create_baseline_model(model_name: str = "HuggingFaceTB/SmolLM2-135M") -> PsychedelicSmolLM:
    """
    Create baseline SmolLM2 model without psychedelic modifications.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Baseline model (skip connections disabled)
    """
    config = load_smollm2_config(model_name)

    model = PsychedelicSmolLM(
        config=config,
        enable_skip_layers=False,  # Baseline: no psychedelic modifications
    )

    return model


def create_psychedelic_model(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    skip_distance: int = 3,
    skip_alpha: float = 0.5,
    skip_start_layer: int = 3,
) -> PsychedelicSmolLM:
    """
    Create psychedelic-modified SmolLM2 model with skip-layer attention.

    Args:
        model_name: HuggingFace model identifier
        skip_distance: Number of layers to skip back
        skip_alpha: Weight for skip connections (0.5 = equal weight)
        skip_start_layer: First layer to enable skip connections

    Returns:
        Psychedelic model with skip-layer connections
    """
    config = load_smollm2_config(model_name)

    model = PsychedelicSmolLM(
        config=config,
        skip_distance=skip_distance,
        skip_alpha=skip_alpha,
        enable_skip_layers=True,  # Enable psychedelic modifications
        skip_start_layer=skip_start_layer,
    )

    return model


def load_tokenizer(model_name: str = "HuggingFaceTB/SmolLM2-135M"):
    """Load tokenizer for SmolLM2"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def print_model_comparison():
    """Print comparison between baseline and psychedelic models"""
    print("🧠 Loading models for comparison...\n")

    baseline = create_baseline_model()
    psychedelic = create_psychedelic_model()

    print("=" * 60)
    print("BASELINE SmolLM2-135M")
    print("=" * 60)
    stats = baseline.get_architecture_stats()
    for key, value in stats.items():
        print(f"{key:.<40} {value}")

    print("\n" + "=" * 60)
    print("🍄 PSYCHEDELIC SmolLM2-135M")
    print("=" * 60)
    stats = psychedelic.get_architecture_stats()
    for key, value in stats.items():
        print(f"{key:.<40} {value}")

    print("\n" + "=" * 60)
    print(f"Additional parameters for skip-layer attention:")
    baseline_params = baseline.get_architecture_stats()['total_params']
    psychedelic_params = psychedelic.get_architecture_stats()['total_params']
    additional_params = psychedelic_params - baseline_params
    percent_increase = (additional_params / baseline_params) * 100

    print(f"Additional params: {additional_params:,} ({percent_increase:.2f}% increase)")
    print("=" * 60)


if __name__ == "__main__":
    print_model_comparison()

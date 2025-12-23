"""
Pre-trained SmolLM2 with injected skip-layer connections.

This module loads HuggingFace's pre-trained SmolLM2-135M and adds
psychedelic skip-layer attention mechanisms on top of the frozen weights.

Two-stage training approach:
1. Freeze baseline, train skip layers only
2. Unfreeze all, co-adapt layers together
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, Optional, List


def load_pretrained_smollm(model_name: str = "HuggingFaceTB/SmolLM2-135M"):
    """
    Load pre-trained SmolLM2 from HuggingFace.

    Returns:
        HuggingFace model with pre-trained weights
    """
    print(f"🔽 Loading pre-trained model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✅ Loaded pre-trained weights")
    return model


def convert_hf_to_mlx_weights(hf_model) -> Dict[str, mx.array]:
    """
    Convert HuggingFace PyTorch weights to MLX format.

    Args:
        hf_model: HuggingFace model with PyTorch weights

    Returns:
        Dictionary of MLX arrays
    """
    print("🔄 Converting PyTorch weights to MLX format...")
    mlx_weights = {}

    for name, param in hf_model.named_parameters():
        # Convert PyTorch tensor to numpy, then to MLX
        numpy_param = param.detach().cpu().numpy()
        mlx_weights[name] = mx.array(numpy_param)

    print(f"✅ Converted {len(mlx_weights)} weight tensors")
    return mlx_weights


class PretrainedPsychedelicBlock(nn.Module):
    """
    Transformer block with pre-trained weights + new skip-layer connections.

    Architecture:
    - Baseline attention/MLP: Pre-trained weights (can be frozen)
    - Skip-layer attention: New weights (Xavier init, always trainable)
    """

    def __init__(
        self,
        hidden_size: int = 576,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        intermediate_size: int = 1536,
        skip_distance: int = 3,
        skip_alpha: float = 0.5,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha
        self.layer_idx = layer_idx

        # Baseline components (will be loaded with pre-trained weights)
        # SmolLM2 uses Grouped Query Attention (GQA) - separate Q, K, V
        self.kv_head_dim = (hidden_size // num_heads) * self.num_kv_heads

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_k = nn.Linear(hidden_size, self.kv_head_dim, bias=False)
        self.self_attn_v = nn.Linear(hidden_size, self.kv_head_dim, bias=False)
        self.self_attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size, bias=False)

        # NEW: Skip-layer attention components (Xavier initialization)
        # Use same architecture for simplicity (can optimize later)
        self.skip_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.skip_k = nn.Linear(hidden_size, self.kv_head_dim, bias=False)
        self.skip_v = nn.Linear(hidden_size, self.kv_head_dim, bias=False)
        self.skip_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize skip weights with Xavier (better than random)
        self._init_skip_weights()

    def _init_skip_weights(self):
        """Initialize skip-layer weights with Xavier initialization."""
        # Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        for module in [self.skip_q, self.skip_k, self.skip_v, self.skip_out]:
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                fan_out = module.weight.shape[0]
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                module.weight = mx.random.uniform(
                    -limit, limit, module.weight.shape
                )

    def _compute_attention(self, q, k, v, mask=None):
        """Standard scaled dot-product attention."""
        # q, k, v: (batch, num_heads, seq_len, head_dim)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ v
        return output

    def forward(self, x, skip_x=None, mask=None):
        """
        Forward pass with optional skip-layer connection.

        Args:
            x: Current layer input (batch, seq_len, hidden_size)
            skip_x: Skip layer input from N-skip_distance (same shape)
            mask: Attention mask

        Returns:
            Output hidden states
        """
        batch_size, seq_len, _ = x.shape

        # ===== BASELINE ATTENTION (pre-trained) =====
        residual = x
        x_norm = self.input_layernorm(x)

        # GQA: Separate Q, K, V projections
        q = self.self_attn_q(x_norm)
        k = self.self_attn_k(x_norm)
        v = self.self_attn_v(x_norm)

        # Reshape for multi-head attention
        # Q: (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # K, V: (batch, seq_len, num_kv_heads, head_dim) - then repeat to match num_heads
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Repeat K and V to match num_heads (for GQA)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        attn_output = self._compute_attention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.self_attn_out(attn_output)

        # ===== PSYCHEDELIC SKIP-LAYER ATTENTION (new) =====
        if skip_x is not None:
            # Cross-attention: Q from current layer, K/V from skip layer
            sq = self.skip_q(x_norm)

            skip_norm = self.input_layernorm(skip_x)
            sk = self.skip_k(skip_norm)
            sv = self.skip_v(skip_norm)

            # Reshape
            sq = sq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            sk = sk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            sv = sv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Repeat K and V for GQA
            if repeats > 1:
                sk = mx.repeat(sk, repeats, axis=1)
                sv = mx.repeat(sv, repeats, axis=1)

            skip_attn_output = self._compute_attention(sq, sk, sv, mask)
            skip_attn_output = skip_attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
            skip_attn_output = self.skip_out(skip_attn_output)

            # Combine baseline + skip with alpha weighting
            attn_output = attn_output + self.skip_alpha * skip_attn_output

        # Residual connection
        x = residual + attn_output

        # ===== MLP (pre-trained) =====
        residual = x
        x_norm = self.post_attention_layernorm(x)

        # SwiGLU activation (SmolLM2 uses this)
        gate = self.mlp_gate(x_norm)
        up = self.mlp_up(x_norm)
        x = self.mlp_down(nn.silu(gate) * up)

        x = residual + x

        return x

    def __call__(self, x, skip_x=None, mask=None):
        return self.forward(x, skip_x, mask)


class PretrainedPsychedelicSmolLM(nn.Module):
    """
    Full SmolLM2 model with pre-trained weights + psychedelic skip connections.
    """

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 576,
        num_layers: int = 30,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        intermediate_size: int = 1536,
        skip_distance: int = 3,
        skip_alpha: float = 0.5,
        skip_start_layer: int = 3,
        checkpoint_every: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha
        self.skip_start_layer = skip_start_layer
        self.checkpoint_every = checkpoint_every

        # Embeddings (will load pre-trained)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = []
        for i in range(num_layers):
            layer = PretrainedPsychedelicBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                skip_distance=skip_distance,
                skip_alpha=skip_alpha,
                layer_idx=i,
            )
            self.layers.append(layer)

        # Final layer norm and LM head (will load pre-trained)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def load_pretrained_weights(self, hf_model):
        """
        Load pre-trained weights from HuggingFace model.

        Only loads baseline components, skip-layer weights stay Xavier initialized.
        """
        print("📥 Injecting pre-trained weights into MLX model...")

        state_dict = hf_model.state_dict()

        # Load embeddings
        self.embed_tokens.weight = mx.array(state_dict['model.embed_tokens.weight'].cpu().numpy())

        # Load each transformer layer
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}'

            # Layer norms
            layer.input_layernorm.weight = mx.array(state_dict[f'{prefix}.input_layernorm.weight'].cpu().numpy())
            layer.post_attention_layernorm.weight = mx.array(state_dict[f'{prefix}.post_attention_layernorm.weight'].cpu().numpy())

            # Attention - separate Q, K, V for GQA
            layer.self_attn_q.weight = mx.array(state_dict[f'{prefix}.self_attn.q_proj.weight'].cpu().numpy())
            layer.self_attn_k.weight = mx.array(state_dict[f'{prefix}.self_attn.k_proj.weight'].cpu().numpy())
            layer.self_attn_v.weight = mx.array(state_dict[f'{prefix}.self_attn.v_proj.weight'].cpu().numpy())
            layer.self_attn_out.weight = mx.array(state_dict[f'{prefix}.self_attn.o_proj.weight'].cpu().numpy())

            # MLP
            layer.mlp_gate.weight = mx.array(state_dict[f'{prefix}.mlp.gate_proj.weight'].cpu().numpy())
            layer.mlp_up.weight = mx.array(state_dict[f'{prefix}.mlp.up_proj.weight'].cpu().numpy())
            layer.mlp_down.weight = mx.array(state_dict[f'{prefix}.mlp.down_proj.weight'].cpu().numpy())

        # Final layer norm and LM head
        self.norm.weight = mx.array(state_dict['model.norm.weight'].cpu().numpy())
        self.lm_head.weight = mx.array(state_dict['lm_head.weight'].cpu().numpy())

        print("✅ Pre-trained weights loaded successfully")
        print("🎯 Skip-layer weights remain Xavier initialized (ready to train)")

    def freeze_baseline_weights(self):
        """
        Freeze all baseline (pre-trained) parameters.
        Only skip-layer parameters will be trainable.
        """
        self.freeze_layers(upto=self.num_layers)

    def unfreeze_all_weights(self):
        """
        Unfreeze all parameters for full fine-tuning (Stage 2).
        """
        print("🔥 Unfreezing all weights for full fine-tuning...")

        # Unfreeze embeddings
        self.embed_tokens.unfreeze()

        # Unfreeze all layer components
        for layer in self.layers:
            layer.input_layernorm.unfreeze()
            layer.self_attn_q.unfreeze()
            layer.self_attn_k.unfreeze()
            layer.self_attn_v.unfreeze()
            layer.self_attn_out.unfreeze()
            layer.post_attention_layernorm.unfreeze()
            layer.mlp_gate.unfreeze()
            layer.mlp_up.unfreeze()
            layer.mlp_down.unfreeze()
            # Skip weights already trainable

        # Unfreeze final norm and LM head
        self.norm.unfreeze()
        self.lm_head.unfreeze()

        print("✅ All weights unfrozen")

    def freeze_layers(self, upto: int):
        """
        Freeze all baseline parameters up to (but not including) layer `upto`.
        Skip-layer parameters remain trainable.

        Args:
            upto: Number of initial layers to freeze (0 => freeze nothing).
        """
        upto = max(0, min(upto, self.num_layers))

        # Freeze embeddings regardless
        self.embed_tokens.freeze()

        # Freeze selected layers
        for idx, layer in enumerate(self.layers):
            if idx < upto:
                layer.input_layernorm.freeze()
                layer.self_attn_q.freeze()
                layer.self_attn_k.freeze()
                layer.self_attn_v.freeze()
                layer.self_attn_out.freeze()
                layer.post_attention_layernorm.freeze()
                layer.mlp_gate.freeze()
                layer.mlp_up.freeze()
                layer.mlp_down.freeze()
            else:
                layer.input_layernorm.unfreeze()
                layer.self_attn_q.unfreeze()
                layer.self_attn_k.unfreeze()
                layer.self_attn_v.unfreeze()
                layer.self_attn_out.unfreeze()
                layer.post_attention_layernorm.unfreeze()
                layer.mlp_gate.unfreeze()
                layer.mlp_up.unfreeze()
                layer.mlp_down.unfreeze()

        # Final norm + LM head should match top layer behaviour
        if upto >= self.num_layers:
            self.norm.freeze()
            self.lm_head.freeze()
        else:
            self.norm.unfreeze()
            self.lm_head.unfreeze()

    def unfreeze_layers(self, from_layer: int):
        """
        Unfreeze all layers starting from `from_layer`.
        """
        self.freeze_layers(from_layer)
        # freeze_layers handles embeddings + final head consistently

    def forward(self, input_ids, mask=None, return_layer_outputs=False):
        """
        Forward pass through the model.

        Args:
            input_ids: (batch, seq_len) token indices
            mask: Optional attention mask

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden_states: List of layer outputs
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask if not provided
        if mask is None:
            mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)

        # Embeddings
        x = self.embed_tokens(input_ids)

        # Store hidden states for skip connections
        hidden_states = [x]

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            # Determine if we should use skip connection
            if i >= self.skip_start_layer:
                skip_idx = i - self.skip_distance
                if skip_idx >= 0:
                    skip_x = hidden_states[skip_idx]
                else:
                    skip_x = None
            else:
                skip_x = None

            if self.checkpoint_every > 0 and i % self.checkpoint_every == 0:
                x = checkpoint(layer)(x, skip_x, mask)
            else:
                x = layer(x, skip_x=skip_x, mask=mask)
            hidden_states.append(x)

        # Final layer norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x)

        if return_layer_outputs:
            return logits, hidden_states
        return logits, None

    def __call__(self, input_ids, mask=None, return_layer_outputs=False):
        return self.forward(input_ids, mask, return_layer_outputs)

    def get_architecture_stats(self) -> Dict:
        """Get model statistics."""
        # Calculate KV dimension for GQA
        num_kv_heads = self.num_kv_heads
        head_dim = self.hidden_size // self.num_heads
        kv_dim = head_dim * num_kv_heads

        # Baseline parameters (pre-trained)
        baseline_params = 0
        baseline_params += self.vocab_size * self.hidden_size  # Embeddings
        baseline_params += self.vocab_size * self.hidden_size  # LM head
        baseline_params += self.hidden_size  # Final layer norm

        for i in range(self.num_layers):
            # GQA attention: Q (full), K (reduced), V (reduced), O (full)
            baseline_params += self.hidden_size * self.hidden_size  # Q proj
            baseline_params += self.hidden_size * kv_dim  # K proj
            baseline_params += self.hidden_size * kv_dim  # V proj
            baseline_params += self.hidden_size * self.hidden_size  # O proj
            baseline_params += self.hidden_size  # Input layer norm
            baseline_params += self.hidden_size  # Post-attention layer norm
            baseline_params += self.hidden_size * self.intermediate_size  # MLP gate
            baseline_params += self.hidden_size * self.intermediate_size  # MLP up
            baseline_params += self.intermediate_size * self.hidden_size  # MLP down

        # Skip-layer parameters (new)
        skip_params = 0
        num_skip_layers = max(0, self.num_layers - self.skip_start_layer)
        for i in range(num_skip_layers):
            skip_params += self.hidden_size * self.hidden_size  # Skip Q
            skip_params += self.hidden_size * kv_dim  # Skip K
            skip_params += self.hidden_size * kv_dim  # Skip V
            skip_params += self.hidden_size * self.hidden_size  # Skip out

        total_params = baseline_params + skip_params

        return {
            'total_params': total_params,
            'baseline_params': baseline_params,
            'skip_params': skip_params,
            'total_params_millions': total_params / 1e6,
            'baseline_params_millions': baseline_params / 1e6,
            'skip_params_millions': skip_params / 1e6,
            'num_layers': self.num_layers,
            'num_skip_layers': num_skip_layers,
            'skip_distance': self.skip_distance,
            'skip_alpha': self.skip_alpha,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
        }


def create_pretrained_psychedelic_model(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    skip_distance: int = 3,
    skip_alpha: float = 0.5,
    skip_start_layer: int = 3,
    checkpoint_every: int = 0,
) -> PretrainedPsychedelicSmolLM:
    """
    Create a psychedelic model with pre-trained SmolLM2 weights.

    Args:
        model_name: HuggingFace model identifier
        skip_distance: Number of layers to skip
        skip_alpha: Weight for skip connections
        skip_start_layer: First layer to add skip connections
        checkpoint_every: Enable gradient checkpointing every N layers

    Returns:
        PretrainedPsychedelicSmolLM with loaded weights
    """
    # Load HuggingFace model
    hf_model = load_pretrained_smollm(model_name)
    hf_config = hf_model.config

    # Create MLX model
    print("🏗️  Creating MLX model architecture...")
    vocab_size = getattr(hf_config, "vocab_size", 49152)
    hidden_size = getattr(hf_config, "hidden_size", 576)
    num_layers = getattr(hf_config, "num_hidden_layers", 30)
    num_heads = getattr(hf_config, "num_attention_heads", 9)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    intermediate_size = getattr(hf_config, "intermediate_size", 1536)

    mlx_model = PretrainedPsychedelicSmolLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        skip_distance=skip_distance,
        skip_alpha=skip_alpha,
        skip_start_layer=skip_start_layer,
        checkpoint_every=checkpoint_every,
    )

    # Inject pre-trained weights
    mlx_model.load_pretrained_weights(hf_model)

    # Get statistics
    stats = mlx_model.get_architecture_stats()
    print(f"\n📊 Model Statistics:")
    print(f"   Total params: {stats['total_params_millions']:.1f}M")
    print(f"   Baseline (pre-trained): {stats['baseline_params_millions']:.1f}M")
    print(f"   Skip layers (new): {stats['skip_params_millions']:.1f}M")
    print(f"   Skip-enabled layers: {stats['num_skip_layers']}/{stats['num_layers']}")

    return mlx_model

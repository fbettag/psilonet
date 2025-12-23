
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
import numpy as np
from typing import Dict, Optional, List, Tuple
from .pretrained_psychedelic import PretrainedPsychedelicBlock, PretrainedPsychedelicSmolLM, create_pretrained_psychedelic_model

class MultiTapPsychedelicBlock(PretrainedPsychedelicBlock):
    """
    Transformer block that can attend to multiple skip distances simultaneously
    using learnable weights (softmax).
    """
    def __init__(
        self,
        hidden_size: int = 576,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        intermediate_size: int = 1536,
        distances: List[int] = [3, 4, 5],
        skip_alpha: float = 0.65,
        layer_idx: int = 0,
    ):
        # We call super with the first distance just to satisfy the constructor
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            skip_distance=distances[0],
            skip_alpha=skip_alpha,
            layer_idx=layer_idx
        )
        self.distances = distances
        # Learnable logits for tap weights
        self.tap_logits = mx.zeros((len(distances),))

    def forward(self, x, skip_xs: List[Optional[mx.array]] = None, mask=None):
        """
        Args:
            x: Current layer input
            skip_xs: List of hidden states from different skip distances
            mask: Attention mask
        """
        batch_size, seq_len, _ = x.shape

        # ===== BASELINE ATTENTION (pre-trained) =====
        residual = x
        x_norm = self.input_layernorm(x)

        q = self.self_attn_q(x_norm)
        k = self.self_attn_k(x_norm)
        v = self.self_attn_v(x_norm)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        attn_output = self._compute_attention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.self_attn_out(attn_output)

        # ===== MULTI-TAP SKIP-LAYER ATTENTION =====
        if skip_xs is not None and any(s is not None for s in skip_xs):
            tap_weights = mx.softmax(self.tap_logits)
            
            sq = self.skip_q(x_norm)
            sq = sq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Blend K and V from all available taps
            blended_k = None
            blended_v = None
            
            active_taps = 0
            for i, (dist, s_x) in enumerate(zip(self.distances, skip_xs)):
                if s_x is None:
                    continue
                
                active_taps += 1
                s_norm = self.input_layernorm(s_x)
                sk = self.skip_k(s_norm)
                sv = self.skip_v(s_norm)
                
                sk = sk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
                sv = sv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
                
                if blended_k is None:
                    blended_k = tap_weights[i] * sk
                    blended_v = tap_weights[i] * sv
                else:
                    blended_k = blended_k + tap_weights[i] * sk
                    blended_v = blended_v + tap_weights[i] * sv

            if active_taps > 0:
                # Normalize weights if some taps were missing (e.g. early layers)
                if active_taps < len(self.distances):
                    # Re-normalize sum to 1.0 roughly or just let it be? 
                    # Better to re-normalize the used weights.
                    used_weight_sum = sum(tap_weights[i] for i, s in enumerate(skip_xs) if s is not None)
                    blended_k = blended_k / (used_weight_sum + 1e-8)
                    blended_v = blended_v / (used_weight_sum + 1e-8)

                if repeats > 1:
                    blended_k = mx.repeat(blended_k, repeats, axis=1)
                    blended_v = mx.repeat(blended_v, repeats, axis=1)

                skip_attn_output = self._compute_attention(sq, blended_k, blended_v, mask)
                skip_attn_output = skip_attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
                skip_attn_output = self.skip_out(skip_attn_output)

                attn_output = attn_output + self.skip_alpha * skip_attn_output

        x = residual + attn_output

        # ===== MLP =====
        residual = x
        x_norm = self.post_attention_layernorm(x)
        gate = self.mlp_gate(x_norm)
        up = self.mlp_up(x_norm)
        x = self.mlp_down(nn.silu(gate) * up)
        x = residual + x

        return x

    def __call__(self, x, skip_xs=None, mask=None):
        return self.forward(x, skip_xs, mask)

class MultiTapPsychedelicSmolLM(PretrainedPsychedelicSmolLM):
    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 576,
        num_layers: int = 30,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        intermediate_size: int = 1536,
        distances: List[int] = [3, 4, 5],
        skip_alpha: float = 0.65,
        skip_start_layer: int = 3,
        checkpoint_every: int = 0,
    ):
        # We call super with the first distance
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            skip_distance=distances[0],
            skip_alpha=skip_alpha,
            skip_start_layer=skip_start_layer,
            checkpoint_every=checkpoint_every
        )
        self.distances = distances
        
        # Replace standard layers with MultiTap layers
        self.layers = []
        for i in range(num_layers):
            layer = MultiTapPsychedelicBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                distances=distances,
                skip_alpha=skip_alpha,
                layer_idx=i,
            )
            self.layers.append(layer)

    def forward(self, input_ids, mask=None, return_layer_outputs=False):
        batch_size, seq_len = input_ids.shape
        if mask is None:
            mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)

        x = self.embed_tokens(input_ids)
        hidden_states = [x]

        for i, layer in enumerate(self.layers):
            # Collect multiple skip inputs
            skip_xs = []
            if i >= self.skip_start_layer:
                for d in self.distances:
                    skip_idx = i - d
                    if skip_idx >= 0:
                        skip_xs.append(hidden_states[skip_idx])
                    else:
                        skip_xs.append(None)
            else:
                skip_xs = None

            if self.checkpoint_every > 0 and i % self.checkpoint_every == 0:
                x = checkpoint(layer)(x, skip_xs, mask)
            else:
                x = layer(x, skip_xs=skip_xs, mask=mask)
            hidden_states.append(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        if return_layer_outputs:
            return logits, hidden_states
        return logits, None

def create_multitap_psychedelic_model(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    distances: List[int] = [3, 4, 5],
    skip_alpha: float = 0.65,
    skip_start_layer: int = 3,
    checkpoint_every: int = 0,
) -> MultiTapPsychedelicSmolLM:
    from .pretrained_psychedelic import load_pretrained_smollm
    hf_model = load_pretrained_smollm(model_name)
    hf_config = hf_model.config

    vocab_size = getattr(hf_config, "vocab_size", 49152)
    hidden_size = getattr(hf_config, "hidden_size", 576)
    num_layers = getattr(hf_config, "num_hidden_layers", 30)
    num_heads = getattr(hf_config, "num_attention_heads", 9)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    intermediate_size = getattr(hf_config, "intermediate_size", 1536)

    mlx_model = MultiTapPsychedelicSmolLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        distances=distances,
        skip_alpha=skip_alpha,
        skip_start_layer=skip_start_layer,
        checkpoint_every=checkpoint_every,
    )
    mlx_model.load_pretrained_weights(hf_model)
    return mlx_model

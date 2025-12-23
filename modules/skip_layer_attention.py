"""
Skip-Layer Attention: Psychedelic-inspired mechanism for enhanced cross-layer connectivity

Based on neuroscience research showing that psilocybin increases functional connectivity
between distant brain regions, reducing hierarchical modularity and enhancing global integration.

References:
- Chen et al. (2024): Skip-layer attention shows 13.3% improvement
- Daws et al. (2022): Psilocybin reduces network modularity
- Herzog et al. (2023): Entropy increases from 2.15 to 2.25 nat
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


class SkipLayerAttention(nn.Module):
    """
    Attention mechanism with skip connections to distant layers.

    Psychedelic mechanism: Enables direct communication between non-adjacent layers,
    mimicking psilocybin's effect of connecting previously segregated brain regions.

    Args:
        dim: Hidden dimension size
        num_heads: Number of attention heads
        skip_distance: Number of layers to skip back (default: 3)
        dropout: Dropout probability
        skip_alpha: Weight for skip connection (default: 0.5)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        skip_distance: int = 3,
        dropout: float = 0.0,
        skip_alpha: float = 0.5
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha
        self.scale = self.head_dim ** -0.5

        # Standard attention projections
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Skip-layer attention projections (psychedelic enhancement)
        self.skip_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.skip_out = nn.Linear(dim, dim, bias=False)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def _reshape_for_heads(self, x: mx.array) -> mx.array:
        """Reshape tensor for multi-head attention: (B, L, D) -> (B, H, L, D/H)"""
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _compute_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Compute scaled dot-product attention"""
        # Attention scores: (B, H, L, L)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (B, H, L, D/H)
        out = attn_weights @ v

        return out

    def forward(
        self,
        x: mx.array,
        skip_x: Optional[mx.array] = None,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass with optional skip-layer input.

        Args:
            x: Current layer input (B, L, D)
            skip_x: Input from layer at distance skip_distance (B, L, D)
            mask: Optional attention mask

        Returns:
            Output tensor (B, L, D)
        """
        B, L, D = x.shape

        # Standard self-attention
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        out = self._compute_attention(q, k, v, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.out_proj(out)

        # PSYCHEDELIC MECHANISM: Skip-layer connection
        if skip_x is not None:
            skip_qkv = self.skip_qkv(skip_x)
            sq, sk, sv = mx.split(skip_qkv, 3, axis=-1)

            sq = self._reshape_for_heads(sq)
            sk = self._reshape_for_heads(sk)
            sv = self._reshape_for_heads(sv)

            # Use queries from current layer with keys/values from distant layer
            skip_out = self._compute_attention(q, sk, sv, mask)
            skip_out = skip_out.transpose(0, 2, 1, 3).reshape(B, L, D)
            skip_out = self.skip_out(skip_out)

            # Combine: This is the psychedelic magic - mixing distant layers
            out = out + self.skip_alpha * skip_out

        return out

    def __call__(self, x: mx.array, skip_x: Optional[mx.array] = None, mask: Optional[mx.array] = None) -> mx.array:
        """Make the module callable"""
        return self.forward(x, skip_x, mask)


class PsychedelicTransformerBlock(nn.Module):
    """
    Transformer block with psychedelic skip-layer attention.

    Integrates skip-layer connectivity inspired by psilocybin's effect on
    brain network topology: reduced modularity, increased global integration.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        skip_distance: int = 3,
        dropout: float = 0.0,
        skip_alpha: float = 0.5
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = SkipLayerAttention(
            dim=dim,
            num_heads=num_heads,
            skip_distance=skip_distance,
            dropout=dropout,
            skip_alpha=skip_alpha
        )

        self.ln2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(
        self,
        x: mx.array,
        skip_x: Optional[mx.array] = None,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor (B, L, D)
            skip_x: Skip connection from distant layer
            mask: Attention mask

        Returns:
            Output tensor (B, L, D)
        """
        # Attention block with residual
        x = x + self.attn(self.ln1(x), skip_x=skip_x, mask=mask)

        # MLP block with residual
        x = x + self.mlp(self.ln2(x))

        return x

    def __call__(self, x: mx.array, skip_x: Optional[mx.array] = None, mask: Optional[mx.array] = None) -> mx.array:
        """Make the module callable"""
        return self.forward(x, skip_x, mask)


def compute_attention_entropy(attn_weights: mx.array, eps: float = 1e-10) -> float:
    """
    Compute Shannon entropy of attention weights.

    Higher entropy = more diverse attention patterns (psychedelic effect).

    Args:
        attn_weights: Attention weights (B, H, L, L)
        eps: Small constant for numerical stability

    Returns:
        Mean entropy across all attention distributions
    """
    # Normalize to ensure proper probability distribution
    attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + eps)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -(attn_weights * mx.log(attn_weights + eps)).sum(axis=-1)

    # Average over batch, heads, queries
    return entropy.mean().item()

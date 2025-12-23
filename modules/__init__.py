"""
Psychedelic-inspired neural network modules
"""

from .skip_layer_attention import SkipLayerAttention, PsychedelicTransformerBlock, compute_attention_entropy
from .psychedelic_smollm import (
    PsychedelicSmolLM,
    create_baseline_model,
    create_psychedelic_model,
    load_tokenizer,
    load_smollm2_config
)
from .pretrained_psychedelic import (
    PretrainedPsychedelicSmolLM,
    create_pretrained_psychedelic_model,
    load_pretrained_smollm,
)

__all__ = [
    'SkipLayerAttention',
    'PsychedelicTransformerBlock',
    'compute_attention_entropy',
    'PsychedelicSmolLM',
    'create_baseline_model',
    'create_psychedelic_model',
    'load_tokenizer',
    'load_smollm2_config',
    'PretrainedPsychedelicSmolLM',
    'create_pretrained_psychedelic_model',
    'load_pretrained_smollm',
]

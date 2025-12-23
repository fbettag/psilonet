"""
Quick test to verify pre-trained model loading works.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import create_pretrained_psychedelic_model
import mlx.core as mx

def test_pretrained_loading():
    """Test that we can load pre-trained weights."""
    print("=" * 80)
    print("Testing Pre-Trained Psychedelic Model Loading")
    print("=" * 80 + "\n")

    try:
        # Create model with pre-trained weights
        model = create_pretrained_psychedelic_model(
            skip_distance=3,
            skip_alpha=0.5,
            skip_start_layer=3,
        )

        print("\n✅ Model created successfully!")

        # Test forward pass with dummy input
        print("\n🧪 Testing forward pass...")
        batch_size = 2
        seq_len = 10
        dummy_input = mx.random.randint(0, 49152, (batch_size, seq_len))

        logits, hidden_states = model(dummy_input)

        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Number of hidden states: {len(hidden_states)}")

        # Test freezing
        print("\n❄️  Testing weight freezing...")
        model.freeze_baseline_weights()
        print("✅ Baseline weights frozen")

        print("\n🔥 Testing weight unfreezing...")
        model.unfreeze_all_weights()
        print("✅ All weights unfrozen")

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pretrained_loading()
    sys.exit(0 if success else 1)

"""
Quick test to verify the setup works correctly.

This script:
1. Tests Skip-Layer Attention implementation
2. Creates baseline and psychedelic models
3. Runs a forward pass with dummy data
4. Verifies architecture statistics

Run this BEFORE starting full training to catch any issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from modules import (
    SkipLayerAttention,
    create_baseline_model,
    create_psychedelic_model,
    load_tokenizer,
)


def test_skip_layer_attention():
    """Test Skip-Layer Attention module"""
    print("\n" + "="*60)
    print("TEST 1: Skip-Layer Attention Module")
    print("="*60)

    hidden_dim = 256
    num_heads = 4
    seq_len = 128
    batch_size = 2

    # Create test data
    x = mx.random.normal((batch_size, seq_len, hidden_dim))
    skip_x = mx.random.normal((batch_size, seq_len, hidden_dim))

    # Test standard attention (no skip)
    print("Testing standard attention...")
    attn = SkipLayerAttention(hidden_dim, num_heads)
    out_standard = attn(x, skip_x=None)
    print(f"✅ Standard attention output shape: {out_standard.shape}")
    assert out_standard.shape == (batch_size, seq_len, hidden_dim), "Shape mismatch!"

    # Test skip-layer attention
    print("Testing skip-layer attention...")
    out_skip = attn(x, skip_x=skip_x)
    print(f"✅ Skip-layer attention output shape: {out_skip.shape}")
    assert out_skip.shape == (batch_size, seq_len, hidden_dim), "Shape mismatch!"

    print("✅ Skip-Layer Attention module works correctly!\n")
    return True


def test_models():
    """Test model creation and forward pass"""
    print("="*60)
    print("TEST 2: Model Creation and Forward Pass")
    print("="*60)

    # Create models
    print("\nCreating baseline model...")
    baseline = create_baseline_model()
    baseline_stats = baseline.get_architecture_stats()
    print(f"✅ Baseline: {baseline_stats['total_params_millions']:.1f}M parameters")

    print("\nCreating psychedelic model...")
    psychedelic = create_psychedelic_model(skip_distance=3, skip_alpha=0.5)
    psychedelic_stats = psychedelic.get_architecture_stats()
    print(f"✅ Psychedelic: {psychedelic_stats['total_params_millions']:.1f}M parameters")

    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy data...")
    batch_size = 2
    seq_len = 32
    dummy_input = mx.random.randint(0, 1000, (batch_size, seq_len))

    print("  Baseline forward pass...")
    baseline_logits, _ = baseline(dummy_input)
    print(f"  ✅ Baseline output shape: {baseline_logits.shape}")
    assert baseline_logits.shape == (batch_size, seq_len, baseline.vocab_size)

    print("  Psychedelic forward pass...")
    psychedelic_logits, _ = psychedelic(dummy_input)
    print(f"  ✅ Psychedelic output shape: {psychedelic_logits.shape}")
    assert psychedelic_logits.shape == (batch_size, seq_len, psychedelic.vocab_size)

    # Compare parameter counts
    additional_params = psychedelic_stats['total_params'] - baseline_stats['total_params']
    if baseline_stats['total_params'] > 0:
        percent_increase = (additional_params / baseline_stats['total_params']) * 100
    else:
        percent_increase = 0.0

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Baseline params:        {baseline_stats['total_params']:,}")
    print(f"Psychedelic params:     {psychedelic_stats['total_params']:,}")
    print(f"Additional params:      {additional_params:,} (+{percent_increase:.2f}%)")
    print(f"Skip layers enabled:    {psychedelic_stats['skip_layers_enabled']}/{psychedelic_stats['num_layers']}")
    print(f"Skip distance:          {psychedelic_stats['skip_distance']}")
    print(f"Skip alpha:             {psychedelic_stats['skip_alpha']}")
    print("="*60 + "\n")

    return True


def test_tokenizer():
    """Test tokenizer loading"""
    print("="*60)
    print("TEST 3: Tokenizer")
    print("="*60)

    print("Loading tokenizer...")
    try:
        tokenizer = load_tokenizer()
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")

        # Test encoding
        test_text = "The psychedelic experience transforms neural connectivity."
        tokens = tokenizer.encode(test_text)
        print(f"✅ Test encoding: {len(tokens)} tokens")
        print(f"   Text: '{test_text}'")
        print(f"   Tokens: {tokens[:10]}...")

        return True
    except Exception as e:
        print(f"⚠️  Tokenizer test failed (this is OK if model not downloaded yet)")
        print(f"   Error: {e}")
        print(f"   Run setup.sh to download the model")
        return False


def main():
    print("\n" + "="*60)
    print("🍄 PSYCHEDELIC LLM SETUP TEST")
    print("="*60)
    print("Testing all components before training...")
    print("="*60 + "\n")

    results = []

    # Run tests
    try:
        results.append(("Skip-Layer Attention", test_skip_layer_attention()))
    except Exception as e:
        print(f"❌ Skip-Layer Attention test failed: {e}")
        results.append(("Skip-Layer Attention", False))

    try:
        results.append(("Models", test_models()))
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        results.append(("Models", False))

    try:
        results.append(("Tokenizer", test_tokenizer()))
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        results.append(("Tokenizer", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Run: source venv/bin/activate  (if not activated)")
        print("2. Run: python experiments/train_comparison.py --epochs 5 --batch_size 8")
        print("3. Watch the psychedelic magic happen! 🍄\n")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("If tokenizer test failed, run: ./setup.sh\n")


if __name__ == "__main__":
    main()

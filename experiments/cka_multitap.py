
import argparse, json, gc, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.multitap_psychedelic import create_multitap_psychedelic_model
from modules.psychedelic_smollm import load_smollm2_config
from experiments.per_head_cka import collect_tokens, numpy_cka, load_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["wikitext", "tinystories"], required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out-suffix", default="_multitap_scaled")
    p.add_argument("--tokens-per-layer", type=int, default=2000)
    args = p.parse_args()

    mx.set_default_device(mx.gpu)
    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, val_ds = load_data(args.dataset, tokenizer, 1, 100, 48)

    cfg = load_smollm2_config(MODEL_ID)
    num_layers = cfg.get("num_hidden_layers", 30)
    num_heads = cfg.get("num_attention_heads", 9)
    head_dim = cfg.get("hidden_size", 576) // num_heads

    baseline = create_multitap_psychedelic_model(
        model_name=MODEL_ID, distances=[3], skip_start_layer=num_layers
    )

    multitap = create_multitap_psychedelic_model(
        model_name=MODEL_ID, distances=[3, 4, 5]
    )
    multitap.load_weights(args.weights, strict=False)

    print(f"🔬 Collecting tokens for {args.dataset}...")
    base_tokens = collect_tokens(
        baseline, val_ds, num_layers, 2, 48, tokenizer.pad_token_id,
        args.tokens_per_layer, head_dim, num_heads
    )
    multi_tokens = collect_tokens(
        multitap, val_ds, num_layers, 2, 48, tokenizer.pad_token_id,
        args.tokens_per_layer, head_dim, num_heads
    )

    heatmap = []
    for i in range(num_layers):
        row = []
        for h in range(num_heads):
            b = base_tokens[i][h]
            m = multi_tokens[i][h]
            if b.shape[0] == 0 or m.shape[0] == 0:
                row.append(0.0)
                continue
            n = min(b.shape[0], m.shape[0])
            row.append(float(numpy_cka(b[:n], m[:n])))
        heatmap.append(row)

    out_json = Path(f"logs/cka/cka_multitap_{args.dataset}{args.out_suffix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cka_per_head": heatmap, "mean": float(np.mean(heatmap))}, indent=2))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='CKA')
    plt.title(f'Multi-tap CKA Alignment: {args.dataset}')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.savefig(out_json.with_suffix(".png"), dpi=200)
    print(f"Saved results to {out_json}")

if __name__ == "__main__":
    main()

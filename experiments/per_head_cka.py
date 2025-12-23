"""
Per-head CKA between baseline SmolLM2-135M and a skip model after a small
skip-only finetune. Uses token-level representations and splits the hidden
dimension into heads to get a head-wise CKA heatmap (layers x heads).

Outputs (suffix-aware):
  - logs/cka/cka_per_head{suffix}.json
  - logs/cka/cka_per_head{suffix}.png
"""
import argparse, json, gc, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import create_pretrained_psychedelic_model
from modules.psychedelic_smollm import load_smollm2_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["wikitext", "tinystories"], default="wikitext")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--train-samples", type=int, default=400)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--skip-distance", type=int, default=3)
    p.add_argument("--skip-alpha", type=float, default=0.65)
    p.add_argument("--skip-start-layer", type=int, default=3)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens-per-layer", type=int, default=3000)
    return p.parse_args()


def numpy_cka(x, y):
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xxt = x.T @ x
    yyt = y.T @ y
    hsic = (xxt * yyt).sum()
    denom = np.sqrt((xxt * xxt).sum() * (yyt * yyt).sum() + 1e-8)
    return hsic / denom


def load_data(dataset, tokenizer, train_samples, val_samples, seq_len):
    if dataset == "wikitext":
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{train_samples}]")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{val_samples}]")
        text_col = "text"
    elif dataset == "tinystories":
        train = load_dataset("roneneldan/TinyStories", split=f"train[:{train_samples}]")
        val = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_samples}]")
        text_col = "text"
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    def tokenize(batch):
        ids = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        return {"input_ids": ids["input_ids"]}

    train = train.map(tokenize, remove_columns=train.column_names)
    val = val.map(tokenize, remove_columns=val.column_names)
    return train, val


def batches(ds, batch_size, seq_len, pad_id):
    for i in range(0, len(ds), batch_size):
        chunk = ds[i : i + batch_size]["input_ids"]
        flat = []
        for x in chunk:
            while isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
                x = x[0]
            x = list(x)
            if len(x) < seq_len:
                x = x + [pad_id] * (seq_len - len(x))
            else:
                x = x[:seq_len]
            flat.append(x)
        yield np.array(flat, dtype=np.int32)


def collect_tokens(model, ds, num_layers, batch_size, seq_len, pad_id, max_tokens, head_dim, num_heads):
    rng = np.random.default_rng()
    layer_head_tokens = [[[] for _ in range(num_heads)] for _ in range(num_layers)]
    for np_batch in batches(ds, batch_size, seq_len, pad_id):
        x = mx.array(np_batch)
        _, hiddens = model(x, return_layer_outputs=True)
        mask = (np_batch != pad_id)
        for i in range(num_layers):
            hi = np.array(hiddens[i + 1].tolist(), dtype=np.float32)  # [b, s, d]
            m = mask
            flat = hi[m]  # [tokens, d]
            if flat.shape[0] == 0:
                continue
            flat = flat.reshape(flat.shape[0], num_heads, head_dim)  # [tokens, heads, dim]
            for h in range(num_heads):
                layer_head_tokens[i][h].append(flat[:, h, :])
        mx.clear_cache(); gc.collect()
    out = [[None for _ in range(num_heads)] for _ in range(num_layers)]
    for i in range(num_layers):
        for h in range(num_heads):
            if len(layer_head_tokens[i][h]) == 0:
                out[i][h] = np.zeros((0, head_dim), dtype=np.float32)
                continue
            all_tokens = np.concatenate(layer_head_tokens[i][h], axis=0)
            if all_tokens.shape[0] > max_tokens:
                idx = rng.choice(all_tokens.shape[0], size=max_tokens, replace=False)
                all_tokens = all_tokens[idx]
            out[i][h] = all_tokens
    return out


if __name__ == "__main__":
    args = parse_args()
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    OUT_DIR = Path("logs/cka")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON = OUT_DIR / f"cka_per_head{args.out_suffix}.json"
    OUT_PNG = OUT_DIR / f"cka_per_head{args.out_suffix}.png"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_data(args.dataset, tokenizer, args.train_samples, args.val_samples, args.seq_len)

    cfg = load_smollm2_config(MODEL_ID)
    num_layers = cfg.get("num_hidden_layers", 30)
    num_heads = cfg.get("num_attention_heads", 9)
    head_dim = cfg.get("hidden_size", 576) // num_heads

    baseline = create_pretrained_psychedelic_model(
        model_name=MODEL_ID, skip_distance=1, skip_alpha=0.0, skip_start_layer=num_layers
    )

    skip_model = create_pretrained_psychedelic_model(
        model_name=MODEL_ID,
        skip_distance=args.skip_distance,
        skip_alpha=args.skip_alpha,
        skip_start_layer=args.skip_start_layer,
    )
    skip_model.freeze_baseline_weights()
    opt = optim.AdamW(learning_rate=args.lr)

    # Finetune skip-only
    for _ in range(args.epochs):
        for np_batch in batches(train_ds, args.batch, args.seq_len, tokenizer.pad_token_id):
            x = mx.array(np_batch)
            labels = x
            def loss_fn(m):
                logits, _ = m(x)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = labels[:, 1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
            loss, grads = nn.value_and_grad(skip_model, loss_fn)(skip_model)
            opt.update(skip_model, grads)
            mx.clear_cache(); gc.collect()

    base_tokens = collect_tokens(
        baseline, val_ds, num_layers, args.batch, args.seq_len, tokenizer.pad_token_id,
        args.tokens_per_layer, head_dim, num_heads
    )
    skip_tokens = collect_tokens(
        skip_model, val_ds, num_layers, args.batch, args.seq_len, tokenizer.pad_token_id,
        args.tokens_per_layer, head_dim, num_heads
    )

    heatmap = []
    for i in range(num_layers):
        row = []
        for h in range(num_heads):
            b = base_tokens[i][h]
            s = skip_tokens[i][h]
            if b.shape[0] == 0 or s.shape[0] == 0:
                row.append(0.0)
                continue
            n = min(b.shape[0], s.shape[0])
            row.append(float(numpy_cka(b[:n], s[:n])))
        heatmap.append(row)

    out = {
        "model": MODEL_ID,
        "skip_cfg": dict(skip_distance=args.skip_distance, skip_alpha=args.skip_alpha, skip_start_layer=args.skip_start_layer),
        "seq_len": args.seq_len,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "tokens_per_layer": args.tokens_per_layer,
        "cka_per_head": heatmap,
        "cka_mean": float(np.mean(heatmap)),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='CKA')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title('Per-head CKA (baseline vs skip)')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved", OUT_JSON, "and", OUT_PNG)

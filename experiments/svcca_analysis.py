"""
Layerwise SVCCA between baseline SmolLM2-135M and skip model after a small
skip-only finetune (same budget as cka_analysis by default).

Outputs (suffix-aware):
  - logs/svcca/svcca_results{suffix}.json
  - logs/svcca/svcca_plot{suffix}.png

SVCCA implementation is numpy-only (no scipy). We whiten each view then take
the SVD of the cross-covariance; singular values are the canonical correlations.
Reported score is the mean canonical correlation per layer.
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
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--skip-distance", type=int, default=3)
    p.add_argument("--skip-alpha", type=float, default=0.65)
    p.add_argument("--skip-start-layer", type=int, default=3)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pca-components", type=int, default=128, help="PCA cap before SVCCA")
    p.add_argument("--tokens-per-layer", type=int, default=4096, help="Number of token vectors sampled per layer for SVCCA.")
    return p.parse_args()


def pca_reduce(x, k):
    x = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(k, vt.shape[0])
    return (x @ vt[:k].T), vt[:k]


def whiten(x, eps=1e-5):
    # x: [n, d]
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0]))
    vals = np.clip(vals, eps, None)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return x @ w


def svcca_score(x, y, k=128):
    # x, y: [n, d]
    x_red, _ = pca_reduce(x, k)
    y_red, _ = pca_reduce(y, k)
    xw = whiten(x_red)
    yw = whiten(y_red)
    k_mat = xw.T @ yw / max(xw.shape[0] - 1, 1)
    s = np.linalg.svd(k_mat, compute_uv=False)
    return float(np.mean(s))


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


def batches(ds, batch_size, seq_len):
    for i in range(0, len(ds), batch_size):
        chunk = ds[i : i + batch_size]["input_ids"]
        flat = []
        for x in chunk:
            while isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
                x = x[0]
            x = list(x)
            if len(x) < seq_len:
                x = x + [tokenizer.eos_token_id] * (seq_len - len(x))
            else:
                x = x[:seq_len]
            flat.append(x)
        yield np.array(flat, dtype=np.int32)


def collect_layer_tokens(model, ds, num_layers, batch_size, seq_len, pad_id, max_tokens):
    rng = np.random.default_rng()
    layer_tokens = [[] for _ in range(num_layers)]
    for np_batch in batches(ds, batch_size, seq_len):
        x = mx.array(np_batch)
        _, hiddens = model(x, return_layer_outputs=True)
        mask = (np_batch != pad_id)
        for i in range(num_layers):
            hi = np.array(hiddens[i + 1].tolist(), dtype=np.float32)  # [b, s, d]
            m = mask
            flat = hi[m]
            if flat.shape[0] == 0:
                continue
            layer_tokens[i].append(flat)
        mx.clear_cache(); gc.collect()
    out = []
    for i in range(num_layers):
        if len(layer_tokens[i]) == 0:
            out.append(np.zeros((0, 1), dtype=np.float32))
            continue
        all_tokens = np.concatenate(layer_tokens[i], axis=0)
        if all_tokens.shape[0] > max_tokens:
            idx = rng.choice(all_tokens.shape[0], size=max_tokens, replace=False)
            all_tokens = all_tokens[idx]
        out.append(all_tokens)
    return out


if __name__ == "__main__":
    args = parse_args()
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    OUT_DIR = Path("logs/svcca")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON = OUT_DIR / f"svcca_results{args.out_suffix}.json"
    OUT_PNG = OUT_DIR / f"svcca_plot{args.out_suffix}.png"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_data(args.dataset, tokenizer, args.train_samples, args.val_samples, args.seq_len)

    cfg = load_smollm2_config(MODEL_ID)
    num_layers = cfg.get("num_hidden_layers", 30)

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
        for np_batch in batches(train_ds, args.batch, args.seq_len):
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

    base_reps = collect_layer_tokens(
        baseline, val_ds, num_layers, args.batch, args.seq_len, tokenizer.pad_token_id, args.tokens_per_layer
    )
    skip_reps = collect_layer_tokens(
        skip_model, val_ds, num_layers, args.batch, args.seq_len, tokenizer.pad_token_id, args.tokens_per_layer
    )

    svcca_scores = []
    for i in range(num_layers):
        svcca_scores.append(svcca_score(base_reps[i], skip_reps[i], k=args.pca_components))

    out = {
        "model": MODEL_ID,
        "skip_cfg": dict(skip_distance=args.skip_distance, skip_alpha=args.skip_alpha, skip_start_layer=args.skip_start_layer),
        "seq_len": args.seq_len,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "svcca_per_layer": svcca_scores,
        "svcca_mean": float(np.mean(svcca_scores)),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(svcca_scores, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('SVCCA (baseline vs skip)')
    plt.title('Layerwise SVCCA after skip-only finetune')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved", OUT_JSON, "and", OUT_PNG)

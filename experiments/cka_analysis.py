"""
Layerwise CKA between baseline SmolLM2-135M and a skip model after a small
skip-only finetune (baseline frozen). Tunable finetune budget and dataset via
CLI. Outputs (with optional suffix):
  - logs/cka/cka_results{suffix}.json
  - logs/cka/cka_plot{suffix}.png
"""
import sys, gc, json, argparse, numpy as np
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
    return p.parse_args()

args = parse_args()

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
SEQ_LEN = args.seq_len
BATCH = args.batch
TRAIN_SAMPLES = args.train_samples
VAL_SAMPLES = args.val_samples
LR = args.lr
EPOCHS = args.epochs
DATASET = args.dataset  # or tinystories
SKIP_CFG = dict(skip_distance=args.skip_distance, skip_alpha=args.skip_alpha, skip_start_layer=args.skip_start_layer)
OUT_DIR = Path("logs/cka")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / f"cka_results{args.out_suffix}.json"
OUT_PNG = OUT_DIR / f"cka_plot{args.out_suffix}.png"

mx.set_default_device(mx.gpu)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

mx.random.seed(args.seed)
np.random.seed(args.seed)

def load_data():
    if DATASET == "wikitext":
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{TRAIN_SAMPLES}]")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{VAL_SAMPLES}]")
        text_col = "text"
    elif DATASET == "tinystories":
        train = load_dataset("roneneldan/TinyStories", split=f"train[:{TRAIN_SAMPLES}]")
        val = load_dataset("roneneldan/TinyStories", split=f"validation[:{VAL_SAMPLES}]")
        text_col = "text"
    else:
        raise ValueError(f"Unsupported dataset {DATASET}")

    def tokenize(batch):
        ids = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
        return {"input_ids": ids["input_ids"]}

    train = train.map(tokenize, remove_columns=train.column_names)
    val = val.map(tokenize, remove_columns=val.column_names)
    return train, val

train_ds, val_ds = load_data()

def batches(ds):
    for i in range(0, len(ds), BATCH):
        chunk = ds[i : i + BATCH]["input_ids"]
        flat = []
        for x in chunk:
            while isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
                x = x[0]
            x = list(x)
            if len(x) < SEQ_LEN:
                x = x + [tokenizer.eos_token_id] * (SEQ_LEN - len(x))
            else:
                x = x[:SEQ_LEN]
            flat.append(x)
        yield np.array(flat, dtype=np.int32)

cfg = load_smollm2_config(MODEL_ID)
num_layers = cfg.get("num_hidden_layers", 30)

baseline = create_pretrained_psychedelic_model(
    model_name=MODEL_ID, skip_distance=1, skip_alpha=0.0, skip_start_layer=num_layers
)

skip_model = create_pretrained_psychedelic_model(model_name=MODEL_ID, **SKIP_CFG)
skip_model.freeze_baseline_weights()
opt = optim.AdamW(learning_rate=LR)

def forward_hidden(model, x):
    logits, hiddens = model(x, return_layer_outputs=True)
    return logits, hiddens

def train_skip():
    for _ in range(EPOCHS):
        for np_batch in batches(train_ds):
            x = mx.array(np_batch)
            labels = x
            def loss_fn(m):
                logits, _ = forward_hidden(m, x)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = labels[:, 1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
            loss, grads = nn.value_and_grad(skip_model, loss_fn)(skip_model)
            opt.update(skip_model, grads)
            mx.clear_cache(); gc.collect()

def collect_layer_means(model):
    layer_means = [[] for _ in range(num_layers)]
    for np_batch in batches(val_ds):
        x = mx.array(np_batch)
        _, hiddens = forward_hidden(model, x)
        for i in range(num_layers):
            hi = hiddens[i + 1]
            layer_means[i].append(mx.mean(hi, axis=1).astype(mx.float32))
        mx.clear_cache(); gc.collect()
    layer_means = [mx.concatenate(chunk, axis=0) for chunk in layer_means]
    return layer_means

def numpy_cka(x, y):
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xxt = x.T @ x
    yyt = y.T @ y
    hsic = (xxt * yyt).sum()
    denom = np.sqrt((xxt * xxt).sum() * (yyt * yyt).sum() + 1e-8)
    return hsic / denom

def main():
    train_skip()
    base_reps = collect_layer_means(baseline)
    skip_reps = collect_layer_means(skip_model)
    cka_scores = []
    for i in range(num_layers):
        xi = np.array(base_reps[i].tolist(), dtype=np.float32)
        yi = np.array(skip_reps[i].tolist(), dtype=np.float32)
        cka_scores.append(float(numpy_cka(xi, yi)))
    out = {
        "model": MODEL_ID,
        "skip_cfg": SKIP_CFG,
        "seq_len": SEQ_LEN,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "cka_per_layer": cka_scores,
        "cka_mean": float(np.mean(cka_scores)),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(cka_scores, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('CKA (baseline vs skip)')
    plt.title('Layerwise CKA after skip-only finetune')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved", OUT_JSON, "and", OUT_PNG)

if __name__ == "__main__":
    main()

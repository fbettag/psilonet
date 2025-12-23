"""
Quick alpha ablation to probe the therapeutic window.
Defaults are a tiny budget (train 300 / val 150, 1 epoch, bs=1) but all
settings are overridable via CLI.
Supports optional gradient accumulation and SGD to ease Metal memory.
Outputs: logs/alpha_ablation/results.json (or suffixed when requested).
"""
import sys, gc, json, argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import utils as mx_utils
import mlx.optimizers as optim
from mlx import utils as mx_utils
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import create_pretrained_psychedelic_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+", default=[0.55, 0.65, 0.75])
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--val-samples", type=int, default=150)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--grad-accum", type=int, default=1, help="microbatch accumulation steps")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--checkpoint-every", type=int, default=0, help="enable gradient checkpointing every N layers (0=off)")
    return p.parse_args()

args = parse_args()

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_DIR = ".tokenizers/smollm2-pad"
ALPHAS = args.alphas
TRAIN_SAMPLES = args.train_samples
VAL_SAMPLES = args.val_samples
SEQ_LEN = args.seq_len
BATCH = args.batch
LR = args.lr
EPOCHS = args.epochs
OUT = Path(f"logs/alpha_ablation/results{args.out_suffix}.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
GRAD_ACCUM = max(1, args.grad_accum)
OPT_CHOICE = args.optimizer
CKPT_EVERY = args.checkpoint_every

mx.set_default_device(mx.gpu)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def tokenize(ds):
    def tok(batch):
        ids = tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
        return {"input_ids": ids["input_ids"]}

    return ds.map(tok, remove_columns=ds.column_names)


def batches(ds):
    for i in range(0, len(ds), BATCH):
        yield np.array(ds[i : i + BATCH]["input_ids"], dtype=np.int32)


def run_alpha(alpha):
    model = create_pretrained_psychedelic_model(
        model_name=MODEL_ID, skip_distance=3, skip_alpha=alpha, skip_start_layer=3,
        checkpoint_every=CKPT_EVERY
    )
    model.freeze_baseline_weights()
    opt = optim.SGD(learning_rate=LR) if OPT_CHOICE == "sgd" else optim.AdamW(learning_rate=LR)

    # train
    for _ in range(EPOCHS):
        accum_grads = None
        step = 0
        for np_batch in batches(train_ds):
            x = mx.array(np_batch)
            labels = x

            def loss_fn(m):
                logits, _ = m(x)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = labels[:, 1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            accum_grads = grads if accum_grads is None else nn.add(accum_grads, grads)
            step += 1
            if step % GRAD_ACCUM == 0:
                scaled = mx_utils.tree_map(lambda g: g / GRAD_ACCUM, accum_grads)
                opt.update(model, scaled)
                accum_grads = None
                mx.clear_cache(); gc.collect()
        if accum_grads is not None:
            scaled = mx_utils.tree_map(lambda g: g / GRAD_ACCUM, accum_grads)
            opt.update(model, scaled)
            mx.clear_cache(); gc.collect()

    # val
    total_loss = 0.0
    steps = 0
    for np_batch in batches(val_ds):
        x = mx.array(np_batch)
        labels = x
        logits, _ = model(x)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
        total_loss += loss.item()
        steps += 1
        mx.clear_cache(); gc.collect()

    return total_loss / steps


if __name__ == "__main__":
    mx.random.seed(42)
    np.random.seed(42)
    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{TRAIN_SAMPLES}]")
    val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{VAL_SAMPLES}]")
    train_ds = tokenize(train_ds)
    val_ds = tokenize(val_ds)

    results = []
    for alpha in ALPHAS:
        val_loss = run_alpha(alpha)
        results.append({"alpha": alpha, "val_loss": val_loss})
        print(f"alpha={alpha}: val_loss={val_loss:.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print("Saved", OUT)

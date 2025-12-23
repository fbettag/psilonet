
import sys, gc, json, argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import utils as mx_utils
import mlx.optimizers as optim
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.multitap_psychedelic import create_multitap_psychedelic_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distances", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--val-samples", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=36)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--out-suffix", default="_multitap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-weights", type=str, default="", help="Path to save weights .npz")
    p.add_argument("--dataset", choices=["wikitext", "tinystories"], default="wikitext")
    return p.parse_args()

def main():
    args = parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    
    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ds):
        def tok(batch):
            ids = tokenizer(batch["text"], truncation=True, max_length=args.seq_len, padding="max_length")
            return {"input_ids": ids["input_ids"]}
        return ds.map(tok, remove_columns=ds.column_names)

    if args.dataset == "wikitext":
        train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{args.train_samples}]")
        val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{args.val_samples}]")
    else:
        train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{args.train_samples}]")
        val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{args.val_samples}]")
        
    train_ds = tokenize(train_ds)
    val_ds = tokenize(val_ds)

    def batches(ds):
        for i in range(0, len(ds), args.batch):
            yield np.array(ds[i : i + args.batch]["input_ids"], dtype=np.int32)

    model = create_multitap_psychedelic_model(
        model_name=MODEL_ID,
        distances=args.distances,
        checkpoint_every=args.checkpoint_every
    )
    model.freeze_baseline_weights()
    
    # We want to track tap_logits, which are trainable parameters
    opt = optim.AdamW(learning_rate=args.lr)

    print(f"🚀 Training Multi-tap model on {args.dataset} with distances {args.distances}...")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        steps = 0
        for np_batch in batches(train_ds):
            x = mx.array(np_batch)
            labels = x

            def loss_fn(m):
                logits, _ = m(x)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = labels[:, 1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            opt.update(model, grads)
            mx.eval(model.parameters())
            total_train_loss += loss.item()
            steps += 1
            if steps % 50 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            mx.clear_cache(); gc.collect()
        
        print(f"Epoch {epoch} complete. Average Train Loss: {total_train_loss/steps:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    val_steps = 0
    for np_batch in batches(val_ds):
        x = mx.array(np_batch)
        labels = x
        logits, _ = model(x)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
        total_val_loss += loss.item()
        val_steps += 1
        mx.clear_cache(); gc.collect()

    mean_val_loss = total_val_loss / val_steps
    print(f"✅ Final Validation Loss: {mean_val_loss:.4f}")

    if args.save_weights:
        save_path = Path(args.save_weights)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save trainable parameters (skip layers + tap_logits)
        weights = dict(mx_utils.tree_flatten(model.trainable_parameters()))
        mx.savez(str(save_path), **weights)
        print(f"Saved trainable weights to {save_path}")

    # Inspect tap weights (softmax of tap_logits)
    tap_weights_summary = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'tap_logits'):
            w = mx.softmax(layer.tap_logits).tolist()
            tap_weights_summary.append({"layer": i, "weights": w})
    
    results = {
        "dataset": args.dataset,
        "distances": args.distances,
        "val_loss": mean_val_loss,
        "tap_weights": tap_weights_summary
    }
    
    OUT = Path(f"logs/multitap/results_{args.dataset}{args.out_suffix}.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved results and tap weights to {OUT}")

if __name__ == "__main__":
    main()

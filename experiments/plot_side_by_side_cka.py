
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_side_by_side():
    wiki_path = Path("logs/cka/cka_per_head_wikitext_seed0.json")
    tiny_path = Path("logs/cka/cka_per_head_tinystories_seed0.json")
    out_path = Path("logs/cka/cka_per_head_side_by_side.png")

    with open(wiki_path, "r") as f:
        wiki_data = json.load(f)["cka_per_head"]
    with open(tiny_path, "r") as f:
        tiny_data = json.load(f)["cka_per_head"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    im1 = ax1.imshow(wiki_data, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    ax1.set_title('WikiText-2 (Head-selective collapse)')
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Layer Index')

    im2 = ax2.imshow(tiny_data, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    ax2.set_title('TinyStories (Uniform high alignment)')
    ax2.set_xlabel('Head Index')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label='CKA Similarity')

    plt.suptitle('Per-Head Representation Alignment: Baseline vs Psychedelic Skip-Layer', fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved side-by-side plot to {out_path}")

if __name__ == "__main__":
    plot_side_by_side()

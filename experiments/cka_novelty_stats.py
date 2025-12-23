"""
Utility: print novelty-window stats from saved CKA JSON files.
Usage:
  python experiments/cka_novelty_stats.py logs/cka/cka_results_wikitext.json ...
"""
import json
import sys

START = 9
END = 28  # slice end (exclusive)


def stats(path: str):
    data = json.loads(open(path).read())
    cka = data["cka_per_layer"]
    window = cka[START:END]
    return {
        "file": path,
        "mean": sum(window) / len(window),
        "min": min(window),
        "max": max(window),
        "layers": f"{START}-{END-1}",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/cka_novelty_stats.py <cka_json...>")
        sys.exit(1)
    for f in sys.argv[1:]:
        s = stats(f)
        print(f"{s['file']}: window {s['layers']} mean={s['mean']:.3f} min={s['min']:.3f} max={s['max']:.3f}")

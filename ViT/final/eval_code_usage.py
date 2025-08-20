import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_metrics(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                step = int(row[0])
                key = row[1]
                val = float(row[2])
                rows.append((step, key, val))
            except:
                continue
    return rows

def pivot_metrics(rows, keys=("train/perplexity", "train/used_codes", "val/mse")):
    d = {k: defaultdict(list) for k in keys}
    for step, key, val in rows:
        if key in d:
            d[key][step].append(val)

    xs = sorted({s for k in keys for s in d[k].keys()})
    out = {k: (xs, [np.mean(d[k].get(x, [np.nan])) for x in xs]) for k in keys}
    return out

def plot_curves(metrics_dir):
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    rows = load_metrics(csv_path)
    metrics = pivot_metrics(rows)

    for key, (xs, ys) in metrics.items():
        if len(xs) == 0:
            continue
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.title(key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(metrics_dir, key.replace("/", "_") + ".png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_dir", type=str)
    args = ap.parse_args()
    plot_curves(args.metrics_dir)

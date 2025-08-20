import os, re, json, argparse, math
import numpy as np

def list_shards(out_dir, prefix="codes_", suffix=".npy"):
    """List and sort shard files like codes_00000.npy, codes_00001.npy, ..."""
    files = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith(suffix)]
    files = sorted(files, key=lambda x: int(re.findall(r"(\d+)", x)[-1]))
    return [os.path.join(out_dir, f) for f in files]

def load_stats(out_dir):
    """Load export stats.json if present."""
    p = os.path.join(out_dir, "stats.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def safe_log(p):
    return np.log(p + 1e-12)

def compute_global_usage(shard_paths, codebook_size):
    """Compute global histogram, used/dead, and perplexity."""
    hist = np.zeros(codebook_size, dtype=np.int64)
    total = 0
    for p in shard_paths:
        codes = np.load(p)  # [T, N]
        flat = codes.reshape(-1)
        h, _ = np.histogram(flat, bins=np.arange(codebook_size + 1))
        hist += h
        total += flat.size
    probs = hist / max(total, 1)
    used = int((hist > 0).sum())
    dead = int((hist == 0).sum())
    entropy = -np.sum(probs * safe_log(probs))
    perplexity = float(np.exp(entropy))
    return {
        "hist": hist,
        "used_codes": used,
        "dead_codes": dead,
        "perplexity": perplexity,
        "total_tokens": int(total),
    }

def compute_per_position_stats(shard_paths, codebook_size, num_patches, sample_limit=None):
    """
    Entropy and dead-rate per position i in [0..N-1].
    Aggregates counts position-wise in streaming fashion.
    """
    counts = np.zeros((num_patches, codebook_size), dtype=np.int64)
    total_rows = 0
    for p in shard_paths:
        codes = np.load(p)  # [T, N]
        if sample_limit is not None and total_rows + codes.shape[0] > sample_limit:
            codes = codes[: max(sample_limit - total_rows, 0)]
        Tcur, N = codes.shape
        for i in range(N):
            h, _ = np.histogram(codes[:, i], bins=np.arange(codebook_size + 1))
            counts[i] += h
        total_rows += Tcur
        if sample_limit is not None and total_rows >= sample_limit:
            break

    totals = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(totals, 1)
    ent = -np.sum(probs * safe_log(probs), axis=1)
    perp = np.exp(ent)
    used = (counts > 0).sum(axis=1)
    dead_rate = 1.0 - (used / float(codebook_size))
    return {
        "pos_entropy": ent,
        "pos_perplexity": perp,
        "pos_used_codes": used,
        "pos_dead_rate": dead_rate,
    }

def compute_per_sample_unique_counts(shard_paths, sample_limit=None):
    """Distribution of unique code counts per sample (across N positions)."""
    uniques = []
    seen = 0
    for p in shard_paths:
        codes = np.load(p)  # [T, N]
        if sample_limit is not None and seen + codes.shape[0] > sample_limit:
            codes = codes[: max(sample_limit - seen, 0)]
        for row in codes:
            uniques.append(len(np.unique(row)))
        seen += codes.shape[0]
        if sample_limit is not None and seen >= sample_limit:
            break
    uniques = np.array(uniques, dtype=np.int32)
    return {
        "per_sample_unique_counts": uniques,
        "mean_unique": float(uniques.mean()) if uniques.size else 0.0,
        "std_unique": float(uniques.std()) if uniques.size else 0.0,
        "num_samples": int(uniques.size),
    }

def compute_bigrams_sparse(shard_paths, codebook_size, num_patches, sample_limit=None, keep_topk=2000000):
    """
    Approximate bigram counts across adjacent positions (i->i+1), pooled over i.
    Returns top-K most frequent pairs to avoid huge memory.
    """
    from collections import Counter
    ctr = Counter()
    seen = 0
    for p in shard_paths:
        codes = np.load(p)  # [T, N]
        if sample_limit is not None and seen + codes.shape[0] > sample_limit:
            codes = codes[: max(sample_limit - seen, 0)]
        if codes.shape[1] >= 2:
            left = codes[:, :-1].reshape(-1)
            right = codes[:, 1:].reshape(-1)
            for a, b in zip(left, right):
                ctr[(int(a), int(b))] += 1
        seen += codes.shape[0]
        if sample_limit is not None and seen >= sample_limit:
            break
        if len(ctr) > keep_topk * 1.2:
            ctr = Counter(dict(ctr.most_common(keep_topk)))

    top = ctr.most_common(keep_topk)
    pairs = [[a, b, c] for (a, b), c in top]
    return {"bigrams_top": pairs, "total_pairs": int(sum(c for _, c in top))}

def save_array(path, arr):
    np.save(path, arr)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def parse_args():
    ap = argparse.ArgumentParser("Analyze exported discrete tokens (codes_*.npy).")
    ap.add_argument("--tokens_dir", type=str, required=True, help="Directory containing codes_*.npy and stats.json.")
    ap.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples for heavy stats.")
    ap.add_argument("--analyze_bigrams", action="store_true", help="Compute top bigrams (may be heavy).")
    ap.add_argument("--compare_dir", type=str, default=None, help="Optional: another tokens dir (e.g., VQ-VAE) to compare.")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = args.tokens_dir
    shards = list_shards(out_dir)
    assert len(shards) > 0, f"No codes_*.npy found in {out_dir}"

    base_stats = load_stats(out_dir)
    K = int(base_stats.get("codebook_size", 1024))
    N = int(base_stats.get("num_patches", 0)) or int(np.load(shards[0]).shape[1])

    g = compute_global_usage(shards, K)
    save_array(os.path.join(out_dir, "global_hist.npy"), g["hist"])
    save_json(os.path.join(out_dir, "global_usage.json"), {k:int(v) if k in ["used_codes","dead_codes","total_tokens"] else float(v) for k,v in g.items() if k!="hist"})

    pos = compute_per_position_stats(shards, K, N, sample_limit=args.sample_limit)
    save_array(os.path.join(out_dir, "pos_entropy.npy"), pos["pos_entropy"])
    save_array(os.path.join(out_dir, "pos_perplexity.npy"), pos["pos_perplexity"])
    save_array(os.path.join(out_dir, "pos_used_codes.npy"), pos["pos_used_codes"])
    save_array(os.path.join(out_dir, "pos_dead_rate.npy"), pos["pos_dead_rate"])
    save_json(os.path.join(out_dir, "pos_summary.json"), {
        "entropy_mean": float(pos["pos_entropy"].mean()),
        "entropy_std": float(pos["pos_entropy"].std()),
        "perplexity_mean": float(pos["pos_perplexity"].mean()),
        "perplexity_std": float(pos["pos_perplexity"].std()),
        "used_codes_mean": float(pos["pos_used_codes"].mean()),
        "dead_rate_mean": float(pos["pos_dead_rate"].mean())
    })

    uniq = compute_per_sample_unique_counts(shards, sample_limit=args.sample_limit)
    save_array(os.path.join(out_dir, "per_sample_unique.npy"), uniq["per_sample_unique_counts"])
    save_json(os.path.join(out_dir, "per_sample_unique_summary.json"), {
        "mean_unique": uniq["mean_unique"],
        "std_unique": uniq["std_unique"],
        "num_samples": uniq["num_samples"]
    })

    if args.analyze_bigrams:
        b = compute_bigrams_sparse(shards, K, N, sample_limit=args.sample_limit)
        save_json(os.path.join(out_dir, "bigrams_top.json"), b)

    if args.compare_dir is not None:
        other = args.compare_dir
        other_shards = list_shards(other)
        K2 = K
        g2 = compute_global_usage(other_shards, K2)
        cmp = {
            "this_perplexity": float(g["perplexity"]),
            "other_perplexity": float(g2["perplexity"]),
            "this_used": int(g["used_codes"]),
            "other_used": int(g2["used_codes"]),
            "this_dead": int(g["dead_codes"]),
            "other_dead": int(g2["dead_codes"]),
            "this_total_tokens": int(g["total_tokens"]),
            "other_total_tokens": int(g2["total_tokens"]),
        }
        save_json(os.path.join(out_dir, "compare_summary.json"), cmp)

    print("[Analyze] Done.")

if __name__ == "__main__":
    main()

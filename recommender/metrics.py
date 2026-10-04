"""metrics.py -- accuracy metrics now; ranking metrics (P@K, R@K, NDCG, MAP, coverage...) added in Part 5."""
import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ======================= ranking metrics (Phase 7) =======================
# `ranked` = list of item ids in recommended order (best first); `relevant` = set of item ids the
# user liked in the held-out data (rating >= 4). All per-user; average over users outside.

def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k


def recall_at_k(ranked, relevant, k):
    relevant = set(relevant)
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked, relevant, k):
    """Binary-gain NDCG: DCG = sum rel_i / log2(i+1) over ranks i=1..k; ideal puts min(k,|R|) hits first."""
    relevant = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(ranked, relevant, k):
    """AP@K = (1 / min(K, |R|)) * sum_{i<=K} P@i * rel_i   (MAP@K = mean of this over users)."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, x in enumerate(ranked[:k]):
        if x in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(k, len(relevant))


def coverage(rec_lists, n_items):
    """Share of the catalogue that appears in at least one user's recommendation list."""
    seen = set()
    for lst in rec_lists:
        seen.update(lst)
    return len(seen) / n_items


def novelty(rec_lists, item_pop):
    """Mean of -log2(popularity) over all recommended items (higher = more novel / less popular).
    item_pop[i] = fraction of users who rated item i in the training data (must be > 0)."""
    vals = [-np.log2(item_pop[i]) for lst in rec_lists for i in lst]
    return float(np.mean(vals)) if vals else 0.0


def popularity_buckets(item_counts, n_buckets=5):
    """Assign each item to a popularity bucket 0..n_buckets-1 (0 = least popular) by rank, ties broken by order."""
    counts = np.asarray(item_counts)
    order = np.argsort(np.argsort(counts, kind="stable"), kind="stable")      # rank 0..n-1
    return np.minimum((order * n_buckets) // len(counts), n_buckets - 1)


def intra_list_diversity(lst, dist):
    """Mean pairwise distance between items in one list. dist(a, b) -> number in [0, 1]."""
    n = len(lst)
    if n < 2:
        return 0.0
    tot = sum(dist(lst[a], lst[b]) for a in range(n) for b in range(a + 1, n))
    return tot / (n * (n - 1) / 2)


def bucket_distance(buckets):
    """Simple diversity metric: |bucket_a - bucket_b| / (n_buckets - 1). Same bucket -> 0, head vs tail -> 1."""
    span = max(int(np.max(buckets)), 1)
    return lambda a, b: abs(int(buckets[a]) - int(buckets[b])) / span


def cosine_distance(E):
    """For later: E = item embedding matrix (e.g. from Alphin's text model). dist = (1 - cos) / 2 in [0, 1]."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return lambda a, b: float((1.0 - En[a] @ En[b]) / 2.0)

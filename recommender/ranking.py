"""ranking.py -- the ranking protocol (Phase 7.6).

For each sampled test user: score ALL items, drop the items the user already has in training
(for the test split also the validation items -- they are known data, not candidates), take the top K,
and compare with the items the user rated >= 4 in the held-out split.
Users are a random sample (default 1000, seed 42) of held-out users with >= 1 relevant item.
"""
import numpy as np
import pandas as pd
from . import data, metrics


def recommend_lists(model, d, users, K, known):
    """Top-K item lists (best first) for `users`; items in the `known` rating table are never recommended."""
    seen = data.build_csr(known, d["n_users"], d["n_items"])[users].toarray() > 0
    S = np.asarray(model.score_all(users), dtype=float)
    S[seen] = -np.inf
    top = np.argpartition(-S, K - 1, axis=1)[:, :K]                          # unordered top K
    rows = np.arange(len(users))[:, None]
    top = top[rows, np.argsort(-S[rows, top], axis=1)]                       # sort those K best-first
    return [row.tolist() for row in top]


def evaluate_ranking(model, d, split="test", K=10, n_users=1000, seed=42, rel_threshold=4.0, n_buckets=5):
    held = d[split]
    rel_rows = held[held.rating >= rel_threshold]
    relevant = rel_rows.groupby("u")["i"].apply(set).to_dict()
    eligible = np.array(sorted(relevant))
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))

    known = pd.concat([d["train"], d["val"]]) if split == "test" else d["train"]
    lists = recommend_lists(model, d, users, K, known)

    counts = np.bincount(d["train"].i.values, minlength=d["n_items"]).astype(float)
    pop = np.maximum(counts, 1) / d["n_users"]                               # fraction of users who rated the item
    dist = metrics.bucket_distance(metrics.popularity_buckets(counts, n_buckets))
    per = lambda f: float(np.mean([f(l, relevant[u], K) for l, u in zip(lists, users)]))
    return {
        f"P@{K}": per(metrics.precision_at_k), f"R@{K}": per(metrics.recall_at_k),
        f"NDCG@{K}": per(metrics.ndcg_at_k), f"MAP@{K}": per(metrics.average_precision_at_k),
        f"coverage@{K}": metrics.coverage(lists, d["n_items"]),
        f"novelty@{K}": metrics.novelty(lists, pop),
        f"diversity@{K}": float(np.mean([metrics.intra_list_diversity(l, dist) for l in lists])),
        "n_eval_users": int(len(users)),
    }

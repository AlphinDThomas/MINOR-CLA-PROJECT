"""data.py -- Akhila's recommender: loading, ID mapping, sparse matrix, splits.

Shared schema: user_id, item_id, rating, timestamp (+ review_id). Seed 42 everywhere.
RULE: missing rating = unknown, never zero. Everything here is sparse.
"""
import numpy as np
import pandas as pd
from scipy import sparse

SEED = 42


def make_synthetic_ratings(n_users=12000, n_items=3000, k=5, density_target=0.006, seed=SEED):
    """Stand-in data with real low-rank structure (so models can beat baselines).
    Observed pattern is popularity-skewed, ratings are clipped/rounded to 1..5."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n_users, k)); Q = rng.normal(0, 1, (n_items, k))
    bu = rng.normal(0, 0.3, n_users); bi = rng.normal(0, 0.4, n_items)
    pop = rng.pareto(1.2, n_items) + 1; pop = pop / pop.sum()
    act = rng.pareto(1.5, n_users) + 1; act = act / act.sum()
    n = int(density_target * n_users * n_items)
    u = rng.choice(n_users, n, p=act); i = rng.choice(n_items, n, p=pop)
    pairs = pd.DataFrame({"u": u, "i": i}).drop_duplicates()
    u, i = pairs.u.values, pairs.i.values
    raw = 3.7 + bu[u] + bi[i] + 0.45 * (P[u] * Q[i]).sum(1) + rng.normal(0, 0.5, len(u))
    df = pd.DataFrame({
        "user_id": ["U%05d" % x for x in u], "item_id": ["I%05d" % x for x in i],
        "rating": np.clip(np.round(raw), 1, 5).astype(float),
        "timestamp": rng.integers(1_000_000_000, 1_400_000_000, len(u))})
    return df


def load_ratings(path=None, min_count=5, sample_n=None, seed=SEED, synthetic_kwargs=None, has_header=False):
    """Load ratings; fall back to synthetic if `path` is None/missing.
    Real Kaggle file has NO header: userId, productId, Rating, timestamp."""
    import os
    if path and os.path.exists(path):
        if has_header:
            df = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
        else:
            df = pd.read_csv(path, names=["user_id", "item_id", "rating", "timestamp"],
                             dtype={"user_id": str, "item_id": str})
        source = "real"
    else:
        df = make_synthetic_ratings(**(synthetic_kwargs or {}), seed=seed)
        source = "synthetic"
    df = k_core(df, min_count)
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)
        df = k_core(df, min_count)          # re-apply after sampling
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", np.arange(len(df)))
    df.attrs["source"] = source
    return df


def k_core(df, min_count=5):
    """Iteratively keep users and items with >= min_count ratings."""
    while True:
        uc = df.groupby("user_id")["item_id"].transform("size")
        n0 = len(df)
        df = df[uc >= min_count]
        ic = df.groupby("item_id")["user_id"].transform("size")
        df = df[ic >= min_count]
        if len(df) == n0:
            return df


def encode_ids(df):
    """Map raw IDs to 0..n-1 ints. Returns df with u, i columns + the two lookup dicts."""
    ucat = pd.Categorical(df["user_id"]); icat = pd.Categorical(df["item_id"])
    out = df.copy()
    out["u"] = ucat.codes.astype(np.int64); out["i"] = icat.codes.astype(np.int64)
    user_map = {x: n for n, x in enumerate(ucat.categories)}
    item_map = {x: n for n, x in enumerate(icat.categories)}
    return out, user_map, item_map


def build_csr(df, n_users, n_items, value_col="rating"):
    """Users x items CSR matrix. Only observed entries are stored."""
    return sparse.csr_matrix((df[value_col].values, (df["u"].values, df["i"].values)),
                             shape=(n_users, n_items))


def split_80_10_10(df, seed=SEED):
    """Shuffle with seed 42 -> train/val/test. Returns df with a `split` column
    (same format as the shared split file: review_id, split)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df)); n = len(df)
    split = np.empty(n, dtype=object)
    split[perm[: int(.8 * n)]] = "train"
    split[perm[int(.8 * n): int(.9 * n)]] = "val"
    split[perm[int(.9 * n):]] = "test"
    out = df.copy(); out["split"] = split
    return out


def apply_shared_split(df, split_df):
    """Use Ashil's split file (review_id, split) once available."""
    return df.merge(split_df[["review_id", "split"]], on="review_id", how="inner")


def describe(df, n_users, n_items):
    nnz = len(df); dens = nnz / (n_users * n_items)
    print(f"users={n_users:,}  items={n_items:,}  ratings={nnz:,}  density={dens:.5f} ({dens*100:.3f}%)")
    return dens


def prepare(path=None, min_count=5, sample_n=None, seed=SEED, has_header=False, **kw):
    """One-call pipeline -> dict with df (with u,i,split), train CSR, sizes, maps."""
    df = load_ratings(path, min_count, sample_n, seed, kw or None, has_header)
    df = split_80_10_10(df, seed)
    df, umap, imap = encode_ids(df)
    nU, nI = len(umap), len(imap)
    train = df[df.split == "train"]
    return dict(df=df, train=train, val=df[df.split == "val"], test=df[df.split == "test"],
                R_train=build_csr(train, nU, nI), R_all=build_csr(df, nU, nI),
                n_users=nU, n_items=nI, user_map=umap, item_map=imap,
                source=df.attrs.get("source", "?"))

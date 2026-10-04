"""api.py -- the stable interface for the rest of the team (Phase 9).

    from recommender import train
    model = train(ratings_df, weights=None, params=None)       # -> RecommenderModel
    model.predict(user_id, item_id)                            # -> predicted rating in [1, 5]
    model.recommend(user_id, n=10, exclude_seen=True)          # -> ranked list of item ids (best first)
    model.save("model.pkl");  RecommenderModel.load("model.pkl")

* ratings_df  : DataFrame with user_id, item_id, rating (+ review_id if weights are given as a DataFrame).
* weights     : None (all ones) | array/Series aligned to ratings_df rows | DataFrame (review_id, w).
                w in [0, 1], high = trusted. w = 0 behaves exactly like deleting that rating.
* params      : dict, any of:  model 'funksvd' (default) | 'als';  k, reg, seed;
                funksvd: lr, epochs, patience;   als: reg_bias, n_iters, patience;
                val_frac (default 0): hold out this fraction of rows for early stopping.
IDs are the original (raw) user/item ids; integer indexing is handled inside.
Unknown users/items are handled: predict falls back to mu (+ the known biases); recommend falls back to the
most-rated items.
"""
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from . import funksvd, als, weights as wts

DEFAULTS = {
    "funksvd": dict(k=10, reg=0.2, lr=0.005, epochs=20, patience=3, seed=42),
    "als": dict(k=20, reg=1.0, reg_bias=None, n_iters=15, patience=3, seed=42),
}


class RecommenderModel:
    def __init__(self, core, users, items, seen, counts, kind, params):
        self._core, self._kind, self.params = core, kind, params
        self._users, self._items = pd.Index(users), pd.Index(items)     # position = internal integer id
        self._seen, self._counts = seen, counts                          # CSR (users x items) of rated pairs, item rating counts

    # ---- predict -----------------------------------------------------------
    def predict(self, user_id, item_id):
        """Predicted rating (clipped to [1, 5]). Scalars -> float; lists/arrays -> numpy array."""
        scalar = np.isscalar(user_id) and np.isscalar(item_id)
        u = self._users.get_indexer(np.atleast_1d(user_id)); i = self._items.get_indexer(np.atleast_1d(item_id))
        if len(u) != len(i):
            if len(u) == 1: u = np.repeat(u, len(i))
            elif len(i) == 1: i = np.repeat(i, len(u))
            else: raise ValueError("user_id and item_id must have the same length")
        c = self._core
        ku, ki = u >= 0, i >= 0
        out = np.full(len(u), c.mu)
        out[ku] += c.bu[u[ku]]
        out[ki] += c.bi[i[ki]]
        both = ku & ki
        out[both] += np.einsum("nk,nk->n", c.P[u[both]], c.Q[i[both]])
        out = np.clip(out, 1.0, 5.0)
        return float(out[0]) if scalar else out

    # ---- recommend ---------------------------------------------------------
    def recommend(self, user_id, n=10, exclude_seen=True):
        """Ranked list of the top-n item ids (best first). Unknown user -> the n most-rated items."""
        u = self._users.get_indexer([user_id])[0]
        if u < 0:
            order = np.argsort(-self._counts, kind="stable")[:n]
            return self._items[order].tolist()
        scores = np.asarray(self._core.score_all([u])[0], dtype=float)
        if exclude_seen:
            scores[self._seen[u].indices] = -np.inf
        n = min(n, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, n - 1)[:n] if n > 0 else np.array([], dtype=int)
        top = top[np.argsort(-scores[top], kind="stable")]
        return self._items[top].tolist()

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"RecommenderModel({self._kind}, users={len(self._users)}, items={len(self._items)}, params={self.params})"


def _align_weights(ratings_df, weights):
    if weights is None:
        return np.ones(len(ratings_df))
    if isinstance(weights, pd.DataFrame):
        if "review_id" not in ratings_df.columns:
            raise ValueError("a weights DataFrame needs ratings_df to have a review_id column")
        w = wts.align_weights(ratings_df, weights)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(ratings_df):
            raise ValueError(f"weights has {len(w)} entries but ratings_df has {len(ratings_df)} rows")
    if np.isnan(w).any() or w.min() < 0 or w.max() > 1:
        raise ValueError("weights must be numbers in [0, 1]")
    return np.asarray(w, dtype=float)


def train(ratings_df, weights=None, params=None):
    """Train a recommender on ratings_df (see module docstring). Returns a RecommenderModel."""
    params = dict(params or {})
    kind = str(params.pop("model", "funksvd")).lower()
    if kind not in DEFAULTS:
        raise ValueError(f"params['model'] must be one of {list(DEFAULTS)}")
    val_frac = float(params.pop("val_frac", 0.0))
    cfg = {**DEFAULTS[kind], **params}
    df = ratings_df.reset_index(drop=True)
    w = _align_weights(df, weights)

    ucat, icat = pd.Categorical(df["user_id"]), pd.Categorical(df["item_id"])
    u, i = ucat.codes.astype(np.int64), icat.codes.astype(np.int64)
    r = df["rating"].values.astype(float)
    nU, nI = len(ucat.categories), len(icat.categories)

    val, fit_idx = None, np.arange(len(df))
    if val_frac > 0:
        rng = np.random.default_rng(cfg["seed"]); perm = rng.permutation(len(df)); nv = int(val_frac * len(df))
        vi, fit_idx = np.sort(perm[:nv]), np.sort(perm[nv:])
        val = (u[vi], i[vi], r[vi])

    if kind == "funksvd":
        core = funksvd.FunkSVD(k=cfg["k"], lr=cfg["lr"], reg=cfg["reg"], epochs=cfg["epochs"],
                               patience=cfg["patience"], seed=cfg["seed"], verbose=False)
    else:
        core = als.ALS(k=cfg["k"], reg=cfg["reg"], reg_bias=cfg["reg_bias"], n_iters=cfg["n_iters"],
                       patience=cfg["patience"], seed=cfg["seed"], verbose=False)
    core.fit(u[fit_idx], i[fit_idx], r[fit_idx], nU, nI, weights=w[fit_idx], val=val)

    seen = sparse.csr_matrix((np.ones(len(df)), (u, i)), shape=(nU, nI))      # every rated pair counts as seen (even w = 0)
    seen.sum_duplicates(); seen.sort_indices()
    counts = np.bincount(i, minlength=nI).astype(float)
    return RecommenderModel(core, ucat.categories, icat.categories, seen, counts, kind, {"model": kind, **cfg})

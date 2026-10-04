"""baselines.py -- three simple predictors every real model must beat.
All statistics are computed from TRAIN ratings only (no leakage into val/test).
Inputs are integer arrays u, i (from data.encode_ids) and ratings r."""
import numpy as np
from . import metrics


class GlobalMean:
    def fit(self, u, i, r, n_users, n_items):
        self.mu = float(np.mean(r)); return self

    def predict(self, u, i):
        return np.full(len(u), self.mu)


class ItemMean:
    """Item's average rating; global mean for items never seen in train."""
    def fit(self, u, i, r, n_users, n_items):
        self.mu = float(np.mean(r))
        s = np.bincount(i, weights=r, minlength=n_items); c = np.bincount(i, minlength=n_items)
        self.item_mean = np.where(c > 0, s / np.maximum(c, 1), self.mu); return self

    def predict(self, u, i):
        return self.item_mean[i]


class BiasBaseline:
    """mu + b_i + b_u with shrinkage (damping):
         b_i = sum_{u in I_i}(r - mu) / (lam_i + |I_i|)
         b_u = sum_{i in I_u}(r - mu - b_i) / (lam_u + |I_u|)
    Shrinkage pulls biases of rarely-rated users/items toward 0."""
    def __init__(self, lam_i=10.0, lam_u=15.0):
        self.lam_i, self.lam_u = lam_i, lam_u

    def fit(self, u, i, r, n_users, n_items):
        self.mu = float(np.mean(r))
        d = r - self.mu
        ci = np.bincount(i, minlength=n_items)
        self.b_i = np.bincount(i, weights=d, minlength=n_items) / (self.lam_i + ci)
        d2 = d - self.b_i[i]
        cu = np.bincount(u, minlength=n_users)
        self.b_u = np.bincount(u, weights=d2, minlength=n_users) / (self.lam_u + cu)
        return self

    def predict(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i]

    def score_all(self, users):
        users = np.asarray(users)
        return self.mu + self.b_u[users][:, None] + self.b_i[None, :]


class Popularity:
    """Ranking-only reference: recommend the items with the most training ratings (same list for everyone).
    No rating prediction (RMSE/MAE are not defined)."""
    def fit(self, u, i, r, n_users, n_items):
        self.count = np.bincount(i, minlength=n_items).astype(float); return self

    def score_all(self, users):
        return np.tile(self.count, (len(users), 1))


def evaluate(model, u, i, r, lo=1.0, hi=5.0):
    p = np.clip(model.predict(u, i), lo, hi)
    return metrics.rmse(r, p), metrics.mae(r, p)


def baseline_table(d):
    """Three-row table on the validation split. d = data.prepare(...) dict."""
    import pandas as pd
    tr, va = d["train"], d["val"]
    fit_args = (tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"])
    rows = []
    for name, m in [("Global mean", GlobalMean()), ("Item mean", ItemMean()), ("User+item bias", BiasBaseline())]:
        m.fit(*fit_args)
        rm, ma = evaluate(m, va.u.values, va.i.values, va.rating.values)
        rows.append({"model": name, "val_RMSE": round(rm, 4), "val_MAE": round(ma, 4)})
    return pd.DataFrame(rows)

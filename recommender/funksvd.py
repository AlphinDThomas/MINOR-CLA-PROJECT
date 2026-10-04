"""funksvd.py -- FunkSVD from scratch (NumPy, SGD, biases, L2), optional numba speed-up.

Model:   r_hat(u,i) = mu + b_u + b_i + p_u . q_i
Per observed rating (weight w, default 1), all updates use the values from BEFORE this step:
    e   = r - r_hat
    b_u += lr*w*(e - lam*b_u)            b_i += lr*w*(e - lam*b_i)
    p_u += lr*w*(e*q_i - lam*p_u)        q_i += lr*w*(e*p_u - lam*q_i)      (old p_u, old q_i)
Only observed ratings are visited -- missing is unknown, never zero.
Rows with w == 0 are skipped entirely and the shuffle is over active rows only, so a zero weight
behaves EXACTLY like deleting that row.
"""
import numpy as np
from . import metrics

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                   # numba not installed -> pure NumPy fallback
    HAVE_NUMBA = False


def _epoch_py(order, u, i, r, w, mu, bu, bi, P, Q, lr, reg):
    for idx in order:
        a, b, wt = u[idx], i[idx], w[idx]
        pu, qi = P[a].copy(), Q[b].copy()           # copies: use OLD values for both updates
        e = r[idx] - (mu + bu[a] + bi[b] + pu @ qi)
        s = lr * wt
        bu[a] += s * (e - reg * bu[a])
        bi[b] += s * (e - reg * bi[b])
        P[a] = pu + s * (e * qi - reg * pu)
        Q[b] = qi + s * (e * pu - reg * qi)


if HAVE_NUMBA:
    @njit(cache=True)
    def _epoch_nb(order, u, i, r, w, mu, bu, bi, P, Q, lr, reg):
        k = P.shape[1]
        for n in range(order.shape[0]):
            idx = order[n]
            a, b, wt = u[idx], i[idx], w[idx]
            dot = 0.0
            for f in range(k):
                dot += P[a, f] * Q[b, f]
            e = r[idx] - (mu + bu[a] + bi[b] + dot)
            s = lr * wt
            bu[a] += s * (e - reg * bu[a])
            bi[b] += s * (e - reg * bi[b])
            for f in range(k):
                pf, qf = P[a, f], Q[b, f]
                P[a, f] = pf + s * (e * qf - reg * pf)
                Q[b, f] = qf + s * (e * pf - reg * qf)


class FunkSVD:
    def __init__(self, k=20, lr=0.005, reg=0.02, epochs=20, init_std=0.1,
                 patience=2, seed=42, clip=(1.0, 5.0), verbose=True, use_numba=True):
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        self.init_std, self.patience, self.seed = init_std, patience, seed
        self.clip, self.verbose = clip, verbose
        self.use_numba = use_numba and HAVE_NUMBA
        self.history = []

    def fit(self, u, i, r, n_users, n_items, weights=None, val=None):
        """u,i int arrays, r ratings. val=(u,i,r) optional -> early stopping on val RMSE."""
        rng = np.random.default_rng(self.seed)
        u = np.asarray(u, dtype=np.int64); i = np.asarray(i, dtype=np.int64); r = np.asarray(r, float)
        w = np.ones(len(r)) if weights is None else np.asarray(weights, float)
        self.n_users, self.n_items = n_users, n_items
        active = np.flatnonzero(w > 0)                       # zero-weight rows are ignored completely
        self.mu = float(np.average(r, weights=w)) if w.sum() > 0 else float(r.mean())
        self.bu = np.zeros(n_users); self.bi = np.zeros(n_items)
        self.P = rng.normal(0, self.init_std, (n_users, self.k))
        self.Q = rng.normal(0, self.init_std, (n_items, self.k))
        epoch = _epoch_nb if self.use_numba else _epoch_py
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for ep in range(1, self.epochs + 1):
            order = active[rng.permutation(len(active))].astype(np.int64)   # shuffle every epoch
            epoch(order, u, i, r, w, self.mu, self.bu, self.bi, self.P, self.Q, self.lr, self.reg)
            tr = metrics.rmse(r[active], self.predict(u[active], i[active]))
            vr = metrics.rmse(val[2], self.predict(val[0], val[1])) if val is not None else np.nan
            self.history.append((ep, tr, vr))
            if self.verbose:
                print(f"epoch {ep:2d}  train RMSE {tr:.4f}  val RMSE {vr:.4f}")
            if val is not None:
                if vr < best - 1e-5:
                    best, bad = vr, 0
                    best_state = (self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy())
                else:
                    bad += 1
                    if bad >= self.patience:
                        if self.verbose: print(f"early stop; best val RMSE {best:.4f}")
                        break
        if best_state is not None:
            self.bu, self.bi, self.P, self.Q = best_state
        return self

    def predict(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        raw = self.mu + self.bu[u] + self.bi[i] + np.einsum("nk,nk->n", self.P[u], self.Q[i])
        return np.clip(raw, *self.clip) if self.clip else raw

    def score_all(self, users):
        """Raw (unclipped) predicted ratings of every item for the given users -> (len(users), n_items)."""
        users = np.asarray(users)
        return self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T

"""als.py -- Alternating Least Squares from scratch (NumPy + SciPy sparse), with biases and weights.

Model:      r_hat(u,i) = mu + b_u + b_i + p_u . q_i
Objective:  J = sum_{(u,i) in Omega} w_ui (r_ui - mu - b_u - b_i - p_u.q_i)^2
                + lam * ( sum_u n_u (|p_u|^2 + b_u^2) + sum_i m_i (|q_i|^2 + b_i^2) )
  n_u / m_i = number of ratings of user u / item i with w > 0   ("weighted-lambda", ALS-WR).
  Optional: biases can get their own strength  reg_bias  (default = reg):  lam*n_u*|p_u|^2 + reg_bias*n_u*b_u^2.
  Biases are treated as extra factors: user unknown x_u=[p_u; b_u] sees item features z_i=[q_i; 1],
  item unknown y_i=[q_i; b_i] sees user features [p_u; 1].  Each half-step is an EXACT minimiser
  of J over one block, so J never increases.
Only observed ratings are used -- missing is unknown, never zero.
"""
import numpy as np
from scipy import sparse
from . import metrics


def _positional_csr(rows, cols, n_rows, n_cols, order="csr"):
    """Sparse matrix whose data = position of each rating in the input arrays (so r and w stay aligned)."""
    pos = np.arange(len(rows)) + 1                       # +1 so no stored value is 0
    M = sparse.csr_matrix((pos, (rows, cols)), shape=(n_rows, n_cols))
    if order == "csc":
        M = M.tocsc()
    M.sort_indices()
    return M.indptr, M.indices, M.data - 1


def _half_step(indptr, indices, pos, r, w, F, other_bias, mu, lam, k, lam_b=None):
    """Solve every row-block exactly. F = fixed factors of the OTHER side (n_other x k).
    Returns (new_factors n_rows x k, new_bias n_rows)."""
    n_rows = len(indptr) - 1
    out = np.zeros((n_rows, k)); bias = np.zeros(n_rows)
    lam_b = lam if lam_b is None else lam_b
    reg_vec = np.r_[np.full(k, lam), lam_b]          # penalty per coordinate: k factors + 1 bias
    for a in range(n_rows):
        s, e = indptr[a], indptr[a + 1]
        if s == e:
            continue
        p = pos[s:e]; idx = indices[s:e]
        wa = w[p]
        n_a = np.count_nonzero(wa > 0)
        if n_a == 0:
            continue                                      # no usable ratings -> keep zeros
        X = np.empty((e - s, k + 1)); X[:, :k] = F[idx]; X[:, k] = 1.0
        t = r[p] - mu - other_bias[idx]
        Xw = X * wa[:, None]
        A = X.T @ Xw + n_a * np.diag(reg_vec)            # (k+1)x(k+1), symmetric positive definite
        v = Xw.T @ t
        x = np.linalg.solve(A, v)
        out[a], bias[a] = x[:k], x[k]
    return out, bias


class ALS:
    def __init__(self, k=20, reg=1.0, n_iters=15, init_std=0.1, patience=2,
                 seed=42, clip=(1.0, 5.0), verbose=True, reg_bias=None):
        self.k, self.reg, self.n_iters, self.init_std = k, reg, n_iters, init_std
        self.reg_bias = reg if reg_bias is None else reg_bias
        self.patience, self.seed, self.clip, self.verbose = patience, seed, clip, verbose
        self.history = []     # rows: (half_step, which, objective, train_rmse, val_rmse)

    # ---- objective -------------------------------------------------------
    def objective(self, u, i, r, w):
        res = r - (self.mu + self.bu[u] + self.bi[i] + np.einsum("nk,nk->n", self.P[u], self.Q[i]))
        data = float(np.sum(w * res ** 2))
        active = w > 0
        n_u = np.bincount(u[active], minlength=self.n_users); m_i = np.bincount(i[active], minlength=self.n_items)
        reg = (self.reg * (np.sum(n_u * np.sum(self.P ** 2, 1)) + np.sum(m_i * np.sum(self.Q ** 2, 1))) +
               self.reg_bias * (np.sum(n_u * self.bu ** 2) + np.sum(m_i * self.bi ** 2)))
        return data + float(reg)

    # ---- training --------------------------------------------------------
    def fit(self, u, i, r, n_users, n_items, weights=None, val=None):
        rng = np.random.default_rng(self.seed)
        u, i, r = np.asarray(u), np.asarray(i), np.asarray(r, float)
        w = np.ones(len(r)) if weights is None else np.asarray(weights, float)
        self.n_users, self.n_items = n_users, n_items
        self.mu = float(np.average(r, weights=w)) if w.sum() > 0 else float(r.mean())
        self.P = np.zeros((n_users, self.k)); self.bu = np.zeros(n_users); self.bi = np.zeros(n_items)
        self.Q = rng.normal(0, self.init_std, (n_items, self.k))
        U = _positional_csr(u, i, n_users, n_items, "csr")        # rows = users
        V = _positional_csr(i, u, n_items, n_users, "csr")        # rows = items (CSC-equivalent of the same matrix)
        best, best_state, bad, self.history = np.inf, None, 0, []
        hs = 0
        for it in range(1, self.n_iters + 1):
            # user step: fix Q, bi
            self.P, self.bu = _half_step(*U[:2], U[2], r, w, self.Q, self.bi, self.mu, self.reg, self.k, self.reg_bias)
            hs += 1; self._log(hs, "user", u, i, r, w, val)
            # item step: fix P, bu
            self.Q, self.bi = _half_step(*V[:2], V[2], r, w, self.P, self.bu, self.mu, self.reg, self.k, self.reg_bias)
            hs += 1; self._log(hs, "item", u, i, r, w, val)
            vr = self.history[-1][4]
            if self.verbose:
                h = self.history[-1]
                print(f"iter {it:2d}  objective {h[2]:.2f}  train RMSE {h[3]:.4f}  val RMSE {vr:.4f}")
            if val is not None:
                if vr < best - 1e-5:
                    best, bad = vr, 0
                    best_state = tuple(x.copy() for x in (self.P, self.Q, self.bu, self.bi))
                else:
                    bad += 1
                    if bad >= self.patience:
                        if self.verbose: print(f"early stop; best val RMSE {best:.4f}")
                        break
        if best_state is not None:
            self.P, self.Q, self.bu, self.bi = best_state
        return self

    def _log(self, hs, which, u, i, r, w, val):
        obj = self.objective(u, i, r, w)
        tr = metrics.rmse(r, self.predict(u, i))
        vr = metrics.rmse(val[2], self.predict(val[0], val[1])) if val is not None else np.nan
        self.history.append((hs, which, obj, tr, vr))

    # ---- prediction ------------------------------------------------------
    def predict(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        raw = self.mu + self.bu[u] + self.bi[i] + np.einsum("nk,nk->n", self.P[u], self.Q[i])
        return np.clip(raw, *self.clip) if self.clip else raw

    def score_all(self, users):
        """Raw (unclipped) predicted ratings of every item for the given users -> (len(users), n_items)."""
        users = np.asarray(users)
        return self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T

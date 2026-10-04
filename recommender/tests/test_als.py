import numpy as np
from scipy.optimize import minimize
from recommender import als, data, baselines, metrics


def tiny(seed=0, nu=15, ni=12, k=2, keep=0.7):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (nu, k)); Q = rng.normal(0, 1, (ni, k))
    R = 3 + 0.5 * P @ Q.T
    mask = rng.random((nu, ni)) < keep
    u, i = np.nonzero(mask)
    return u, i, R[u, i], nu, ni


def test_objective_never_increases_every_half_step():
    d = data.prepare(n_users=4000, n_items=2500, density_target=0.001)
    tr = d["train"]
    m = als.ALS(k=10, reg=0.1, n_iters=8, verbose=False, patience=99)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"])
    obj = np.array([h[2] for h in m.history])
    assert (np.diff(obj) <= 1e-8 * obj[:-1]).all(), np.diff(obj).max()


def test_user_step_zero_gradient_and_matches_bruteforce():
    u, i, r, nu, ni = tiny()
    m = als.ALS(k=3, reg=0.2, n_iters=3, verbose=False, patience=99, clip=None).fit(u, i, r, nu, ni)
    w = np.ones(len(r)); a = 4
    sel = u == a; n_a = sel.sum()
    def user_obj(x):                                    # J restricted to user a's block [p_a; b_a]
        p, b = x[:3], x[3]
        res = r[sel] - (m.mu + b + m.bi[i[sel]] + m.Q[i[sel]] @ p)
        return np.sum(res ** 2) + 0.2 * n_a * (p @ p + b * b)
    # one fresh user half step from the stored Q, bi
    from recommender.als import _positional_csr, _half_step
    U = _positional_csr(u, i, nu, ni, "csr")
    P, bu = _half_step(U[0], U[1], U[2], r, w, m.Q, m.bi, m.mu, 0.2, 3)
    x_als = np.r_[P[a], bu[a]]
    g = np.zeros(4)
    eps = 1e-6
    for j in range(4):
        e = np.zeros(4); e[j] = eps
        g[j] = (user_obj(x_als + e) - user_obj(x_als - e)) / (2 * eps)
    assert np.abs(g).max() < 1e-5                         # gradient = 0 at ALS solution
    x_bf = minimize(user_obj, np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    assert np.allclose(x_als, x_bf, atol=1e-4)            # independent optimiser agrees


def test_normal_matrix_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    X = np.c_[rng.normal(size=(6, 3)), np.ones(6)]; w = rng.random(6)
    A = X.T @ (X * w[:, None]) + 0.1 * 6 * np.eye(4)
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A).min() >= 0.1 * 6 - 1e-12  # >= lam * n_u > 0
    np.linalg.cholesky(A)                                  # succeeds only for SPD


def test_als_fits_tiny_lowrank():
    u, i, r, nu, ni = tiny(keep=1.0)
    m = als.ALS(k=3, reg=1e-4, n_iters=40, verbose=False, patience=99, clip=None).fit(u, i, r, nu, ni)
    assert metrics.rmse(r, m.predict(u, i)) < 0.02


def test_unseen_user_and_item_are_safe():
    u = np.array([0, 0, 1, 1]); i = np.array([0, 1, 0, 1]); r = np.array([4., 2., 3., 5.])
    m = als.ALS(k=2, reg=0.1, n_iters=3, verbose=False).fit(u, i, r, 3, 3)   # user 2, item 2 never rated
    p = m.predict([2], [2]); assert np.isfinite(p).all() and 1 <= p[0] <= 5


def test_funksvd_objective_is_half_of_als_objective_for_01_weights():
    """derivation_funksvd.md section 5: J_F = 0.5 * J_ALS when w in {0,1}."""
    u, i, r, nu, ni = tiny(seed=3, nu=20, ni=15, keep=0.6)
    rng = np.random.default_rng(5); k, lam = 3, 0.37
    w = (rng.random(len(r)) < 0.8).astype(float)
    m = als.ALS(k=k, reg=lam)
    m.n_users, m.n_items, m.mu = nu, ni, 3.1
    m.P, m.Q = rng.normal(size=(nu, k)), rng.normal(size=(ni, k))
    m.bu, m.bi = rng.normal(size=nu), rng.normal(size=ni)
    J_als = m.objective(u, i, r, w)
    e = r - (m.mu + m.bu[u] + m.bi[i] + np.einsum("nk,nk->n", m.P[u], m.Q[i]))
    reg = m.bu[u] ** 2 + m.bi[i] ** 2 + (m.P[u] ** 2).sum(1) + (m.Q[i] ** 2).sum(1)
    J_f = np.sum(0.5 * w * (e ** 2 + lam * reg))
    assert abs(J_f - 0.5 * J_als) < 1e-9 * max(1.0, abs(J_als))

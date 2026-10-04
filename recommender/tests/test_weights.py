"""Phase 6 sanity tests, run for BOTH models:
   (1) all weights = 1  -> identical to unweighted
   (2) weight 0 on chosen rows -> identical to deleting those rows
   (3) random weights run without error and give finite predictions
"""
import numpy as np, pytest
from recommender import data, funksvd, als, weights


@pytest.fixture(scope="module")
def d():
    return data.prepare(n_users=9000, n_items=2500, density_target=0.004)


def args(d, rows=None):
    tr = d["train"] if rows is None else d["train"].iloc[rows]
    return tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"]


def mk(model):
    if model == "funk":
        return funksvd.FunkSVD(k=6, epochs=4, patience=99, verbose=False, use_numba=False)
    return als.ALS(k=6, reg=0.5, n_iters=4, patience=99, verbose=False)


@pytest.mark.parametrize("model", ["funk", "als"])
def test_all_ones_equals_unweighted(d, model):
    a = args(d); ones = np.ones(len(a[0]))
    p0 = mk(model).fit(*a).predict(d["val"].u.values, d["val"].i.values)
    p1 = mk(model).fit(*a, weights=ones).predict(d["val"].u.values, d["val"].i.values)
    assert np.array_equal(p0, p1)                                  # exact


@pytest.mark.parametrize("model", ["funk", "als"])
def test_zero_weight_equals_deleting_rows(d, model):
    n = len(d["train"]); rng = np.random.default_rng(7)
    drop = rng.choice(n, size=n // 10, replace=False)
    w = np.ones(n); w[drop] = 0.0
    keep = np.setdiff1d(np.arange(n), drop)                        # keeps original relative order
    pw = mk(model).fit(*args(d), weights=w).predict(d["val"].u.values, d["val"].i.values)
    pd_ = mk(model).fit(*args(d, keep)).predict(d["val"].u.values, d["val"].i.values)
    assert np.allclose(pw, pd_, atol=1e-8), np.abs(pw - pd_).max()


@pytest.mark.parametrize("model", ["funk", "als"])
def test_random_weights_run(d, model):
    a = args(d); w = np.random.default_rng(0).random(len(a[0]))
    p = mk(model).fit(*a, weights=w).predict(d["val"].u.values, d["val"].i.values)
    assert np.isfinite(p).all() and p.min() >= 1 and p.max() <= 5


def test_filter_modes():
    p = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    assert (weights.make_weights(p, "none") == 1).all()
    assert (weights.make_weights(p, "hard", threshold=0.5) == [1, 1, 0, 0, 0]).all()
    assert np.allclose(weights.make_weights(p, "soft", gamma=1), 1 - p)
    assert np.allclose(weights.make_weights(p, "soft", gamma=2), (1 - p) ** 2)
    w = weights.make_weights(np.random.default_rng(1).random(100), "soft", gamma=3)
    assert ((w >= 0) & (w <= 1)).all()
    with pytest.raises(ValueError): weights.make_weights(p, "bogus")


def test_als_separate_bias_lambda_monotone(d):
    tr = d["train"]
    m = als.ALS(k=6, reg=1.0, reg_bias=0.05, n_iters=6, patience=99, verbose=False)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"])
    obj = np.array([h[2] for h in m.history])
    assert (np.diff(obj) <= 1e-8 * obj[:-1]).all()

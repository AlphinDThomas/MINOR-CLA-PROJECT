import numpy as np
from recommender import baselines, funksvd, metrics, data

def test_metrics_known_values():
    assert metrics.rmse([1, 2, 3], [1, 2, 3]) == 0
    assert abs(metrics.rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-12
    assert metrics.mae([1, 2, 3], [2, 2, 5]) == 1.0

def test_baselines_hand_example():
    u = np.array([0, 0, 1]); i = np.array([0, 1, 0]); r = np.array([4., 2., 4.])
    g = baselines.GlobalMean().fit(u, i, r, 2, 2); assert abs(g.predict([0], [0])[0] - 10 / 3) < 1e-12
    im = baselines.ItemMean().fit(u, i, r, 2, 3)
    assert im.predict([0], [0])[0] == 4.0 and im.predict([0], [1])[0] == 2.0
    assert abs(im.predict([0], [2])[0] - 10 / 3) < 1e-12          # unseen item -> global mean

def test_itemmean_no_leakage():
    u = np.array([0, 1]); i = np.array([0, 0]); r = np.array([5., 5.])
    im = baselines.ItemMean().fit(u, i, r, 2, 2)
    assert im.predict([0], [1])[0] == 5.0                          # item 1 unseen in train

def test_funksvd_fits_tiny_lowrank():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 1, (12, 2)); Q = rng.normal(0, 1, (10, 2))
    R = 3 + P @ Q.T * 0.5
    uu, ii = np.meshgrid(np.arange(12), np.arange(10), indexing="ij")
    u, i, r = uu.ravel(), ii.ravel(), R.ravel()
    m = funksvd.FunkSVD(k=4, lr=0.02, reg=1e-4, epochs=300, patience=999, clip=None, verbose=False)
    m.fit(u, i, r, 12, 10)
    assert m.history[-1][1] < 0.05                                 # near-zero train error

def test_funksvd_beats_bias_on_synthetic():
    d = data.prepare()
    tr, va = d["train"], d["val"]
    b = baselines.BiasBaseline().fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"])
    bias_rmse = baselines.evaluate(b, va.u.values, va.i.values, va.rating.values)[0]
    m = funksvd.FunkSVD(k=20, epochs=15, verbose=False)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"],
          val=(va.u.values, va.i.values, va.rating.values))
    assert m.history and min(h[2] for h in m.history) < bias_rmse

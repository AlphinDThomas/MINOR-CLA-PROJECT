import numpy as np, pandas as pd, pytest, os
from recommender import train, RecommenderModel, data


@pytest.fixture(scope="module")
def df():
    d = data.k_core(data.make_synthetic_ratings(n_users=4000, n_items=1500, density_target=0.006), 5).reset_index(drop=True)
    d["review_id"] = np.arange(len(d)); return d


@pytest.fixture(scope="module", params=["funksvd", "als"])
def model(request, df):
    return train(df, params={"model": request.param, "k": 6, "epochs": 4, "n_iters": 4})


def test_interface_basics(model, df):
    r = df.iloc[0]
    p = model.predict(r.user_id, r.item_id)
    assert isinstance(p, float) and 1 <= p <= 5
    arr = model.predict(df.user_id.iloc[:5].tolist(), df.item_id.iloc[:5].tolist())
    assert arr.shape == (5,) and abs(arr[0] - p) < 1e-12
    assert model.predict(r.user_id, ["nope", r.item_id]).shape == (2,)           # broadcast one user over many items


def test_recommend_ranked_unseen_raw_ids(model, df):
    u = df.user_id.iloc[0]
    rec = model.recommend(u, n=10)
    assert len(rec) == 10 and len(set(rec)) == 10
    assert not set(rec) & set(df[df.user_id == u].item_id)                         # exclude_seen=True
    full = model._core.score_all([model._users.get_loc(u)])[0]
    ranked = [full[model._items.get_loc(x)] for x in rec]
    assert (np.diff(ranked) <= 1e-12).all()                                         # best first by raw score
    with_seen = model.recommend(u, n=10, exclude_seen=False)
    assert len(with_seen) == 10


def test_unknown_user_and_item_fallbacks(model, df):
    rec = model.recommend("brand-new-user", n=5)
    cnt = df.item_id.value_counts()
    assert len(rec) == 5 and sorted(cnt[rec].values, reverse=True) == sorted(cnt.values, reverse=True)[:5]   # the 5 most-rated items
    assert 1 <= model.predict("brand-new-user", "brand-new-item") <= 5
    assert 1 <= model.predict("brand-new-user", df.item_id.iloc[0]) <= 5


def test_weights_all_ones_and_zero_equals_delete(df):
    P = {"model": "als", "k": 4, "n_iters": 3, "reg": 0.5}
    m0 = train(df, params=P); m1 = train(df, weights=np.ones(len(df)), params=P)
    us, its = df.user_id.iloc[:200].tolist(), df.item_id.iloc[:200].tolist()
    assert np.array_equal(m0.predict(us, its), m1.predict(us, its))                 # all ones == unweighted, exact
    rng = np.random.default_rng(1); drop = rng.choice(len(df), len(df) // 10, replace=False)
    w = np.ones(len(df)); w[drop] = 0
    mw = train(df, weights=w, params=P)
    md = train(df.drop(index=drop).reset_index(drop=True), params=P)
    # same users/items must exist in both for a fair comparison
    ok = [(a, b) for a, b in zip(us, its) if a in md._users and b in md._items]
    a_, b_ = zip(*ok)
    assert np.allclose(mw.predict(list(a_), list(b_)), md.predict(list(a_), list(b_)), atol=1e-6)


def test_weights_as_dataframe_and_validation(df):
    wdf = pd.DataFrame({"review_id": df.review_id, "w": np.linspace(0, 1, len(df))})
    train(df, weights=wdf, params={"k": 4, "epochs": 2})
    with pytest.raises(ValueError): train(df, weights=np.ones(len(df) - 1))
    with pytest.raises(ValueError): train(df, weights=np.full(len(df), 1.5))
    with pytest.raises(ValueError): train(df, params={"model": "svdpp"})


def test_save_load_roundtrip(model, df, tmp_path):
    p = tmp_path / "m.pkl"; model.save(p); m2 = RecommenderModel.load(p)
    us, its = df.user_id.iloc[:100].tolist(), df.item_id.iloc[:100].tolist()
    assert np.array_equal(model.predict(us, its), m2.predict(us, its))
    assert model.recommend(df.user_id.iloc[0], 5) == m2.recommend(df.user_id.iloc[0], 5)


def test_val_frac_early_stopping_runs(df):
    m = train(df, params={"model": "funksvd", "epochs": 6, "val_frac": 0.1, "patience": 1})
    assert 1 <= m.predict(df.user_id.iloc[0], df.item_id.iloc[0]) <= 5

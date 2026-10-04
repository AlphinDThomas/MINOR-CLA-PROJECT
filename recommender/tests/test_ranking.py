import numpy as np, pandas as pd
from recommender import ranking, data


class Oracle:
    """Scores = the true held-out rating, so a correct protocol must reach perfect NDCG."""
    def __init__(self, d, split): self.M = data.build_csr(d[split], d["n_users"], d["n_items"]).toarray()
    def score_all(self, users): return self.M[users]


class Random:
    def __init__(self, n_items): self.n, self.rng = n_items, np.random.default_rng(0)
    def score_all(self, users): return self.rng.random((len(users), self.n))


def make():
    return data.prepare(n_users=6000, n_items=2000, density_target=0.004)


def test_oracle_gets_perfect_ndcg_and_random_is_low():
    d = make()
    r = ranking.evaluate_ranking(Oracle(d, "test"), d, split="test", K=5, n_users=300)
    # an oracle that ranks held-out ratings by value puts relevant (>=4) items first -> perfect for users with <=K relevant
    assert r["NDCG@5"] > 0.99 and r["P@5"] > 0.0
    rr = ranking.evaluate_ranking(Random(d["n_items"]), d, split="test", K=5, n_users=300)
    assert rr["NDCG@5"] < 0.05


def test_seen_items_are_never_recommended():
    d = make()
    class LovesSeen:                                   # scores seen items far above everything else
        def score_all(self, users):
            return data.build_csr(d["train"], d["n_users"], d["n_items"])[users].toarray() * 100.0 + 1.0
    users = np.arange(200)
    lists = ranking.recommend_lists(LovesSeen(), d, users, K=10, known=d["train"])
    seen = d["train"].groupby("u")["i"].apply(set).to_dict()
    for u, lst in zip(users, lists):
        assert not (set(lst) & seen.get(u, set())), u
        assert len(lst) == 10 and len(set(lst)) == 10  # K distinct items


def test_reproducible_and_keys():
    d = make(); m = Random(d["n_items"])
    a = ranking.evaluate_ranking(Oracle(d, "test"), d, K=10, n_users=100, seed=1)
    b = ranking.evaluate_ranking(Oracle(d, "test"), d, K=10, n_users=100, seed=1)
    assert a == b
    for k in ["P@10", "R@10", "NDCG@10", "MAP@10", "coverage@10", "novelty@10", "diversity@10"]:
        assert k in a

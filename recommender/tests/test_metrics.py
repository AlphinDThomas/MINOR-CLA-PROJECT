"""Every ranking metric is checked against a hand-computed answer."""
import numpy as np
from recommender import metrics as m


def test_precision_recall_hand():
    ranked = ["a", "b", "c", "d"]; rel = {"a", "c", "z"}
    assert m.precision_at_k(ranked, rel, 3) == 2 / 3          # hits a, c in top 3
    assert m.recall_at_k(ranked, rel, 3) == 2 / 3             # 2 of 3 relevant found
    assert m.precision_at_k(ranked, rel, 1) == 1.0
    assert m.recall_at_k(ranked, set(), 3) == 0.0             # no relevant items -> 0, no crash


def test_ndcg_hand():
    # ranked [hit, miss, hit], relevant = 2 items:  DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3)
    v = m.ndcg_at_k(["a", "x", "c"], {"a", "c"}, 3)
    assert abs(v - 1.5 / (1 + 1 / np.log2(3))) < 1e-12        # = 0.9197...
    assert m.ndcg_at_k(["a", "c", "x"], {"a", "c"}, 3) == 1.0  # perfect ranking
    assert m.ndcg_at_k(["x", "y", "z"], {"a"}, 3) == 0.0


def test_average_precision_hand():
    # ranked [hit, miss, hit], |R|=2, K=3 -> (1/1 + 2/3) / 2 = 0.8333...
    assert abs(m.average_precision_at_k(["a", "x", "c"], {"a", "c"}, 3) - (1 + 2 / 3) / 2) < 1e-12
    # K smaller than |R|: relevant {a,b,c}, ranked [a,b,x], K=2 -> (1/1 + 2/2)/min(2,3) = 1
    assert m.average_precision_at_k(["a", "b", "x"], {"a", "b", "c"}, 2) == 1.0


def test_coverage_hand():
    assert m.coverage([[0, 1], [1, 2], [2, 3]], n_items=10) == 0.4     # items {0,1,2,3} of 10


def test_novelty_hand():
    pop = np.array([0.5, 0.25, 0.125])                         # -log2 = 1, 2, 3
    assert abs(m.novelty([[0, 1], [2]], pop) - (1 + 2 + 3) / 3) < 1e-12


def test_buckets_and_diversity_hand():
    counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])          # 10 items, 5 buckets -> 2 per bucket
    b = m.popularity_buckets(counts, 5)
    assert list(b) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    dist = m.bucket_distance(b)
    assert dist(0, 1) == 0.0 and dist(0, 9) == 1.0 and dist(0, 4) == 0.5
    # list [0, 4, 9]: pairs (0,4)=0.5, (0,9)=1.0, (4,9)=0.5 -> mean 2/3
    assert abs(m.intra_list_diversity([0, 4, 9], dist) - 2 / 3) < 1e-12
    assert m.intra_list_diversity([3], dist) == 0.0


def test_cosine_distance_hand():
    E = np.array([[1.0, 0], [0, 1.0], [-1.0, 0]])
    d = m.cosine_distance(E)
    assert abs(d(0, 0)) < 1e-12 and abs(d(0, 1) - 0.5) < 1e-12 and abs(d(0, 2) - 1.0) < 1e-12

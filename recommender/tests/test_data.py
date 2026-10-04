import numpy as np
from recommender import data

def test_kcore_and_sparse_not_zero_filled():
    d = data.prepare()
    df = d["df"]
    assert df.groupby("u").size().min() >= 5 and df.groupby("i").size().min() >= 5
    assert d["R_all"].nnz == len(df)                       # only observed stored
    assert d["R_all"].data.min() >= 1                      # no zero ratings stored

def test_split_sizes_disjoint_and_seeded():
    d = data.prepare(); n = len(d["df"])
    assert abs(len(d["train"]) / n - .8) < .01 and abs(len(d["val"]) / n - .1) < .01
    a = data.prepare()["df"].split.values; b = data.prepare()["df"].split.values
    assert (a == b).all()                                  # seed 42 reproducible

def test_csr_matches_df():
    d = data.prepare(); r = d["df"].iloc[0]
    assert d["R_all"][r.u, r.i] == r.rating

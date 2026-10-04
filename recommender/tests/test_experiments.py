"""Tests for the attack-condition runner. The fixture (_attack_fixture.py) is ONLY a test helper,
NOT the team's attack generator (that is Ashil's job)."""
import numpy as np, pandas as pd, pytest
from recommender import experiments as ex
from recommender.tests._attack_fixture import make_attacked

M = ("ALS (k=20, lam=0.5)",)


@pytest.fixture(scope="module")
def att():
    return make_attacked()


def test_fakes_only_in_train(att):
    a, _ = att
    d = ex.prepare_attacked(a, seed=42)
    assert d["train"].is_fake.sum() > 0
    assert not d["val"].is_fake.any() and not d["test"].is_fake.any()
    assert len(d["targets"]) == 1 and len(d["genuine_users"]) < d["n_users"]


def test_conditions_behave(att):
    a, s = att
    r = ex.run_attack(a, "t", scores_df=s, models=M, seeds=(42,)).set_index("condition")
    assert r.loc["oracle", "prediction_shift"] == 0.0                         # oracle IS the reference
    assert r.loc["none", "prediction_shift"] > 0.3                            # the attack works when unfiltered
    assert r.loc["hard", "prediction_shift"] < 0.5 * r.loc["none", "prediction_shift"]   # informative filter helps
    assert r.loc["hard", "mean_w_fake"] < 0.1 and r.loc["hard", "mean_w_genuine"] > 0.95
    assert r.loc["none", "n_zero_weight"] == 0 and r.loc["oracle", "n_zero_weight"] == r.loc["oracle", "n_fake"]
    assert r.loc["soft", "mean_w_fake"] < r.loc["soft", "mean_w_genuine"]


def test_weights_file_instead_of_scores(att):
    a, s = att
    w = pd.DataFrame({"review_id": s.review_id, "w": 1 - s.p_fake})
    r = ex.run_attack(a, "t", weights_df=w, models=M, seeds=(42,), conditions=("none", "soft")).set_index("condition")
    assert r.loc["soft", "prediction_shift"] < r.loc["none", "prediction_shift"]


def test_clean_data_cost_path_has_no_oracle_and_no_harness(att):
    a, s = att
    clean = a[~a.is_fake]
    r = ex.run_attack(clean, "clean", scores_df=s, models=M, seeds=(42,))
    assert set(r.condition) == {"none", "hard", "soft"}                       # oracle skipped
    assert "prediction_shift" not in r.columns
    assert (r.n_fake == 0).all() and r.RMSE.notna().all()


def test_summary_and_paired_helpers():
    df = pd.DataFrame({"model": ["a"] * 3 + ["b"] * 3, "seed": [1, 2, 3] * 2,
                       "RMSE": [1.0, 1.1, 1.2, 1.1, 1.2, 1.3], "NDCG@10": [.1, .2, .3, .1, .1, .1]})
    s = ex.summarize(df).set_index(["model", "metric"])
    assert abs(s.loc[("a", "RMSE"), "mean"] - 1.1) < 1e-12 and abs(s.loc[("a", "RMSE"), "std"] - 0.1) < 1e-12
    assert abs(s.loc[("a", "RMSE"), "ci95"] - 4.302652729911275 * 0.1 / np.sqrt(3)) < 1e-9   # t(0.975, df=2)
    assert "-0.1000" in ex.paired(df, "a", "b", "RMSE")

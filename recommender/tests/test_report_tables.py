import numpy as np, pandas as pd
from recommender import make_report_tables as mrt


def test_clean_tables_exact_ci_and_attack_table(tmp_path, monkeypatch):
    # summary rows as produced by experiments.summarize
    rows = []
    for m, rmse in [("A", 1.0), ("B", 1.1)]:
        for met, mean in [("RMSE", rmse), ("MAE", .8), ("P@10", .01), ("R@10", .05), ("NDCG@10", .02), ("MAP@10", .01),
                          ("coverage@10", .1), ("novelty@10", 7.0), ("diversity@10", .2)]:
            rows.append(dict(model=m, metric=met, mean=mean, std=.01, ci95=.0124, n=5))
    summ = pd.DataFrame(rows)
    t = mrt.clean_tables(summ)
    assert len(t) == 3 and "1.0000 ± 0.0124" in t[0] and "| model | RMSE | MAE |" in t[0] and "NDCG@10" in t[1]
    att = pd.DataFrame({"setting": "s1", "model": "A", "condition": ["none"] * 3 + ["hard"] * 3, "seed": [1, 2, 3] * 2,
                        "RMSE": [1.0, 1.1, 1.2, 1.0, 1.0, 1.0], "prediction_shift": [.9, 1.0, 1.1, .1, .1, .1]})
    out = mrt.attack_table(att)[0]
    half = 4.302652729911275 * 0.1 / np.sqrt(3)
    assert f"1.000 ± {half:.3f}" in out and "Attack setting: s1" in out        # t(0.975, df=2) * sd / sqrt(n)

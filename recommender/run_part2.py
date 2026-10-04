"""run_part2.py -- Phase 2 baseline table + Phase 3 FunkSVD vs scikit-surprise SVD (same split).
Real data:   python -m recommender.run_part2 recommender/data/ratings_sample.csv
Stand-in:    python -m recommender.run_part2
"""
import sys, time, numpy as np, pandas as pd
from . import data, baselines, funksvd, metrics


def surprise_svd(d, k=20, lr=0.005, reg=0.02, epochs=20, seed=42):
    from surprise import SVD, Dataset, Reader
    tr, va = d["train"], d["val"]
    reader = Reader(rating_scale=(1, 5))
    ts = Dataset.load_from_df(tr[["u", "i", "rating"]], reader).build_full_trainset()
    algo = SVD(n_factors=k, lr_all=lr, reg_all=reg, n_epochs=epochs, random_state=seed, init_std_dev=0.1)
    algo.fit(ts)
    p = np.array([algo.predict(int(a), int(b)).est for a, b in zip(va.u.values, va.i.values)])
    return metrics.rmse(va.rating.values, p), metrics.mae(va.rating.values, p)


def main(path=None):
    if path:
        d = data.prepare(path, has_header=True)
    else:  # real-like sparsity stand-in
        d = data.prepare(n_users=45000, n_items=30000, density_target=0.00022)
    print("source:", d["source"]); data.describe(d["df"], d["n_users"], d["n_items"])
    tr, va = d["train"], d["val"]
    print("\n== Phase 2: baselines (validation) =="); bt = baselines.baseline_table(d); print(bt.to_string(index=False))

    print("\n== Phase 3: FunkSVD (k=20, lr=0.005, lam=0.02, 20 epochs) ==")
    t = time.time()
    m = funksvd.FunkSVD(k=20, lr=0.005, reg=0.02, epochs=20, patience=3)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"],
          val=(va.u.values, va.i.values, va.rating.values))
    print(f"train time {time.time()-t:.1f}s")
    f_rm = metrics.rmse(va.rating.values, m.predict(va.u.values, va.i.values))
    f_ma = metrics.mae(va.rating.values, m.predict(va.u.values, va.i.values))
    s_rm, s_ma = surprise_svd(d)
    out = pd.concat([bt, pd.DataFrame([
        {"model": "FunkSVD (mine)", "val_RMSE": round(f_rm, 4), "val_MAE": round(f_ma, 4)},
        {"model": "Surprise SVD (reference)", "val_RMSE": round(s_rm, 4), "val_MAE": round(s_ma, 4)}])])
    print("\n== Summary =="); print(out.to_string(index=False))
    out.to_csv("recommender/results_part2.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

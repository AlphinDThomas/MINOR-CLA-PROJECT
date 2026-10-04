"""als_quick_sweep.py -- small lambda x k grid for ALS, validation only (never the test split).
Run: python -m recommender.als_quick_sweep recommender/data/ratings_sample.csv
Takes a few minutes on the real sample. Saves recommender/results_als_quick.csv.
"""
import sys, time, numpy as np, pandas as pd
from . import data, als


def main(path=None, lams=(0.1, 0.3, 1.0, 3.0, 10.0), ks=(5, 20)):
    d = data.prepare(path, has_header=True) if path else data.prepare(n_users=6000, n_items=4000, density_target=0.0006)
    tr, va = d["train"], d["val"]
    print("source:", d["source"], "| train ratings:", len(tr))
    rows = []
    for k in ks:
        for lam in lams:
            t = time.time()
            m = als.ALS(k=k, reg=lam, n_iters=12, patience=2, verbose=False)
            m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"],
                  val=(va.u.values, va.i.values, va.rating.values))
            v = [h[4] for h in m.history if h[1] == "item"]
            rows.append({"k": k, "lambda": lam, "best_val_RMSE": round(min(v), 4),
                         "best_iter": int(np.argmin(v)) + 1, "train_RMSE": round(m.history[-1][3], 4),
                         "secs": round(time.time() - t, 1)})
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv("recommender/results_als_quick.csv", index=False)
    print("\n", out.to_string(index=False))
    b = out.loc[out.best_val_RMSE.idxmin()]
    print(f"\nBest on validation: k={int(b.k)}, lambda={b['lambda']}, val RMSE {b.best_val_RMSE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

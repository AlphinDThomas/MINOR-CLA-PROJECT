"""run_part3.py -- Phase 4: train ALS, print table next to Part 2 results, save convergence plot.
Real data:  python -m recommender.run_part3 recommender/data/ratings_sample.csv
Stand-in:   python -m recommender.run_part3
"""
import os, sys, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import data, als, metrics


def main(path=None, k=20, reg=1.0, n_iters=15):
    d = data.prepare(path, has_header=True) if path else data.prepare(n_users=45000, n_items=30000, density_target=0.00022)
    print("source:", d["source"]); data.describe(d["df"], d["n_users"], d["n_items"])
    tr, va = d["train"], d["val"]
    t = time.time()
    m = als.ALS(k=k, reg=reg, n_iters=n_iters, patience=99)   # run all iterations for the plot; model still restores best-val iteration
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"],
          val=(va.u.values, va.i.values, va.rating.values))
    secs = time.time() - t
    pv = m.predict(va.u.values, va.i.values)
    rm, ma = metrics.rmse(va.rating.values, pv), metrics.mae(va.rating.values, pv)
    print(f"\nALS train time {secs:.1f}s")

    # convergence check: objective must not increase at any half-step
    obj = np.array([h[2] for h in m.history]); worst = np.max(np.diff(obj) / obj[:-1])
    print(f"max relative change in objective over a half-step: {worst:.2e}  (must be <= 0)")

    # table
    rows = [{"model": f"ALS (mine, k={k}, lam={reg})", "val_RMSE": round(rm, 4), "val_MAE": round(ma, 4)}]
    out = pd.DataFrame(rows)
    p2 = "recommender/results_part2.csv"
    full = pd.concat([pd.read_csv(p2), out]) if os.path.exists(p2) else out
    print("\n== Summary =="); print(full.to_string(index=False))
    out.to_csv("recommender/results_part3.csv", index=False)

    # plot
    os.makedirs("recommender/notebooks", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hs = [h[0] for h in m.history]
    ax[0].plot(hs, obj, "o-", ms=3); ax[0].set_yscale("log")
    ax[0].set_xlabel("half-step (odd = user, even = item)"); ax[0].set_ylabel("objective J (log scale)")
    ax[0].set_title("ALS objective decreases at every half-step")
    itv = [(h[0] / 2, h[3], h[4]) for h in m.history if h[1] == "item"]
    ax[1].plot([a for a, _, _ in itv], [b for _, b, _ in itv], "o-", label="train RMSE")
    ax[1].plot([a for a, _, _ in itv], [c for _, _, c in itv], "s-", label="validation RMSE")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("RMSE"); ax[1].legend(); ax[1].set_title("ALS RMSE per iteration")
    plt.tight_layout(); plt.savefig("recommender/notebooks/als_convergence.png", dpi=150)
    print("saved recommender/notebooks/als_convergence.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

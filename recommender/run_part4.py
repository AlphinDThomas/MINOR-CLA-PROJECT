"""run_part4.py -- Phase 5 studies: spectrum, lambda sweeps (both models), rank sweep (both models).
Tuned on the VALIDATION split only. Lambda is tuned first at k=20, then k is swept with the best lambda.
Real data:   python -m recommender.run_part4 recommender/data/ratings_sample.csv
Smoke test:  python -m recommender.run_part4 --quick
"""
import sys, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import data, funksvd, als, baselines, metrics, spectrum

NB = "recommender/notebooks/"


def evaluate_model(m, d):
    va = d["val"]; p = m.predict(va.u.values, va.i.values)
    return metrics.rmse(va.rating.values, p), metrics.mae(va.rating.values, p)


def run_funk(d, k, reg, epochs):
    tr, va = d["train"], d["val"]
    m = funksvd.FunkSVD(k=k, reg=reg, epochs=epochs, patience=3, verbose=False)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"], val=(va.u.values, va.i.values, va.rating.values))
    return evaluate_model(m, d)


def run_als(d, k, reg, reg_bias, iters):
    tr, va = d["train"], d["val"]
    m = als.ALS(k=k, reg=reg, reg_bias=reg_bias, n_iters=iters, patience=2, verbose=False)
    m.fit(tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"], val=(va.u.values, va.i.values, va.rating.values))
    return evaluate_model(m, d)


def main(path=None, quick=False):
    d = data.prepare(path, has_header=True) if path else data.prepare(n_users=9000, n_items=2500, density_target=0.004)
    print("source:", d["source"]); data.describe(d["df"], d["n_users"], d["n_items"])
    tr = d["train"]
    bias_rmse = baselines.baseline_table(d).set_index("model").loc["User+item bias", "val_RMSE"]
    f_epochs, a_iters = (4, 3) if quick else (20, 10)

    # ---- 1. spectrum -------------------------------------------------------
    t = time.time()
    s = spectrum.top_singular_values(d["R_train"], k=20 if quick else 50)
    pd.DataFrame({"j": np.arange(1, len(s) + 1), "sigma": s}).to_csv("recommender/singular_values.csv", index=False)
    spectrum.plot_spectrum({"clean (train)": s}, NB + "spectrum.png")
    print(f"[spectrum] top5 = {np.round(s[:5], 2)}  ({time.time()-t:.0f}s)")

    # ---- 2. lambda sweeps (k = 20) ------------------------------------------
    rows = []
    f_regs = [0.01, 0.05] if quick else [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    a_lams = [0.5, 2.0] if quick else [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0]
    a_bias = [None, 0.1] if quick else [None, 0.1, 0.5]          # None = same lambda for biases and factors
    for reg in f_regs:
        rm, ma = run_funk(d, 20, reg, f_epochs); rows.append(dict(model="FunkSVD", k=20, reg=reg, reg_bias="-", val_RMSE=rm, val_MAE=ma)); print(rows[-1])
    for lam in a_lams:
        for rb in a_bias:
            rm, ma = run_als(d, 20, lam, rb, a_iters)
            rows.append(dict(model="ALS", k=20, reg=lam, reg_bias=("= reg" if rb is None else rb), val_RMSE=rm, val_MAE=ma)); print(rows[-1])
    L = pd.DataFrame(rows); L.to_csv("recommender/sweep_lambda.csv", index=False)
    bf = L[L.model == "FunkSVD"].sort_values("val_RMSE").iloc[0]
    ba = L[L.model == "ALS"].sort_values("val_RMSE").iloc[0]
    best_f_reg = float(bf.reg); best_a_reg = float(ba.reg); best_a_rb = None if ba.reg_bias == "= reg" else float(ba.reg_bias)
    print(f"\nBEST lambda (validation): FunkSVD reg={best_f_reg} (RMSE {bf.val_RMSE:.4f}) | ALS reg={best_a_reg}, reg_bias: {ba.reg_bias} (RMSE {ba.val_RMSE:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    f = L[L.model == "FunkSVD"]; ax[0].semilogx(f.reg, f.val_RMSE, "o-")
    ax[0].axhline(bias_rmse, ls="--", c="gray", label="bias baseline"); ax[0].set_title("FunkSVD: validation RMSE vs $\\lambda$ (k=20)")
    ax[0].set_xlabel("$\\lambda$ (log scale)"); ax[0].set_ylabel("validation RMSE"); ax[0].legend()
    for rb, g in L[L.model == "ALS"].groupby("reg_bias", sort=False):
        ax[1].semilogx(g.reg, g.val_RMSE, "o-", label=f"bias $\\lambda$ {rb}")
    ax[1].axhline(bias_rmse, ls="--", c="gray", label="bias baseline"); ax[1].set_title("ALS: validation RMSE vs $\\lambda$ (k=20)")
    ax[1].set_xlabel("$\\lambda$ (log scale)"); ax[1].legend(); plt.tight_layout(); plt.savefig(NB + "sweep_lambda.png", dpi=150); plt.close()

    # ---- 3. rank sweep with best lambda --------------------------------------
    ks = [5, 10] if quick else [5, 10, 20, 50, 100]
    rows = []
    for k in ks:
        t = time.time()
        rm, ma = run_funk(d, k, best_f_reg, f_epochs); rows.append(dict(model="FunkSVD", k=k, val_RMSE=rm, val_MAE=ma)); print(rows[-1])
        rm, ma = run_als(d, k, best_a_reg, best_a_rb, a_iters); rows.append(dict(model="ALS", k=k, val_RMSE=rm, val_MAE=ma)); print(rows[-1], f"({time.time()-t:.0f}s for both)")
    K = pd.DataFrame(rows); K.to_csv("recommender/sweep_k.csv", index=False)
    plt.figure(figsize=(6.5, 4.2))
    for name, g in K.groupby("model"):
        plt.plot(g.k, g.val_RMSE, "o-", label=name)
    plt.axhline(bias_rmse, ls="--", c="gray", label="bias baseline"); plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks]); plt.minorticks_off()
    plt.xlabel("rank k"); plt.ylabel("validation RMSE"); plt.title("Validation RMSE vs rank k"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(NB + "sweep_k.png", dpi=150); plt.close()
    print("\n== k sweep =="); print(K.round(4).to_string(index=False))
    print("\nsaved: sweep_lambda.csv, sweep_k.csv, singular_values.csv, notebooks/{spectrum,sweep_lambda,sweep_k}.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, quick="--quick" in sys.argv)

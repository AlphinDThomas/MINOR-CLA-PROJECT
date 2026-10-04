"""experiments.py -- Phase 8.

  clean :  python -m recommender.experiments clean  [ratings_sample.csv]
  attack:  python -m recommender.experiments attack attacked.csv [--scores scores.csv] [--weights weights.csv]
                                                        [--split split.csv] [--setting NAME]
  (attack with a file that has NO fake rows = Phase 8.3 "cost of the filters on clean data")

Shared formats: ratings (review_id,user_id,item_id,rating,timestamp), split (review_id,split),
scores (review_id,p_fake), weights (review_id,w), attacked = ratings + is_fake, attack_type, target_item.
"""
import sys, time
import numpy as np
import pandas as pd
from scipy import stats
from . import data, baselines, funksvd, als, metrics, ranking, weights as wts

MODEL_NAMES = ["Popularity", "Bias baseline", "FunkSVD (k=10, reg=0.2)", "ALS (k=20, lam=1.0)", "ALS (k=20, lam=0.5)"]
FACTOR_MODELS = MODEL_NAMES[2:]
STANDIN = dict(n_users=45000, n_items=30000, density_target=0.00022)      # real-like sparsity stand-in


def make_model(name, seed):
    if name == "Popularity": return baselines.Popularity()
    if name == "Bias baseline": return baselines.BiasBaseline()
    if name.startswith("FunkSVD"): return funksvd.FunkSVD(k=10, reg=0.2, epochs=20, patience=3, seed=seed, verbose=False)
    if name == "ALS (k=20, lam=1.0)": return als.ALS(k=20, reg=1.0, n_iters=15, patience=3, seed=seed, verbose=False)
    if name == "ALS (k=20, lam=0.5)": return als.ALS(k=20, reg=0.5, n_iters=15, patience=3, seed=seed, verbose=False)
    raise ValueError(name)


def fit_model(name, d, seed, w=None):
    tr, va = d["train"], d["val"]
    m = make_model(name, seed); t = time.time()
    args = (tr.u.values, tr.i.values, tr.rating.values, d["n_users"], d["n_items"])
    if name in FACTOR_MODELS:
        m.fit(*args, weights=w, val=(va.u.values, va.i.values, va.rating.values))
    else:
        m.fit(*args)
    return m, time.time() - t


def test_metrics(m, d, K=10, seed=42):
    te = d["test"]; out = {}
    if hasattr(m, "predict"):
        p = np.clip(m.predict(te.u.values, te.i.values), 1, 5)
        out["RMSE"], out["MAE"] = metrics.rmse(te.rating.values, p), metrics.mae(te.rating.values, p)
    else:
        out["RMSE"], out["MAE"] = np.nan, np.nan
    out.update(ranking.evaluate_ranking(m, d, split="test", K=K, seed=seed))
    return out


# ----------------------------------------------------------------- clean comparison
def run_clean(path=None, seeds=(42, 43, 44, 45, 46), K=10, models=MODEL_NAMES, out="recommender/results_clean.csv"):
    rows = []
    for s in seeds:
        d = data.prepare(path, has_header=True, seed=s) if path else data.prepare(seed=s, **STANDIN)
        if s == seeds[0]:
            print("source:", d["source"]); data.describe(d["df"], d["n_users"], d["n_items"])
        for name in models:
            m, secs = fit_model(name, d, s)
            r = dict(model=name, seed=s, **test_metrics(m, d, K, s), train_secs=round(secs, 1))
            rows.append(r); print(f"seed {s} | {name:26s} RMSE {r['RMSE']:.4f}  P@{K} {r[f'P@{K}']:.4f}  NDCG@{K} {r[f'NDCG@{K}']:.4f}")
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    summ = summarize(df); summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print("\n== mean +- std over seeds (95% CI half-width in the summary csv) =="); print(pretty(summ).to_string())
    for a, b in [("FunkSVD (k=10, reg=0.2)", "ALS (k=20, lam=1.0)"), ("FunkSVD (k=10, reg=0.2)", "ALS (k=20, lam=0.5)"),
                 ("FunkSVD (k=10, reg=0.2)", "Bias baseline")]:
        for met in ["RMSE", f"NDCG@{K}"]:
            print(paired(df, a, b, met))
    return df


def summarize(df):
    cols = [c for c in df.columns if c not in ("model", "seed", "n_eval_users", "train_secs")]
    rows = []
    for name, g in df.groupby("model", sort=False):
        n = len(g); tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        for c in cols:
            v = g[c].dropna().values
            if len(v) == 0: continue
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            rows.append(dict(model=name, metric=c, mean=v.mean(), std=sd, ci95=tcrit * sd / np.sqrt(len(v)) if len(v) > 1 else np.nan, n=len(v)))
    return pd.DataFrame(rows)


def pretty(summ):
    t = summ.assign(txt=summ.apply(lambda r: f"{r['mean']:.4f} +- {r['std']:.4f}", axis=1))
    return t.pivot(index="model", columns="metric", values="txt").loc[list(dict.fromkeys(summ.model))]


def paired(df, a, b, metric):
    """Same seeds = same splits, so compare per seed: mean(a - b) with a 95% CI (t distribution)."""
    x = df[df.model == a].set_index("seed")[metric]; y = df[df.model == b].set_index("seed")[metric]
    diff = (x - y).dropna().values
    if len(diff) < 2: return f"{metric}: {a} - {b}: need >= 2 seeds"
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    return f"paired {metric:9s}: {a} - {b} = {diff.mean():+.4f}  (95% CI {diff.mean()-half:+.4f} .. {diff.mean()+half:+.4f})"


# ----------------------------------------------------------------- attack conditions
def prepare_attacked(attacked, split_df=None, seed=42):
    """Genuine ratings are split 80/10/10; injected (fake) rows go to TRAIN only. val/test stay genuine."""
    df = attacked.copy()
    df["is_fake"] = df["is_fake"].astype(bool) if "is_fake" in df else False
    if split_df is not None:
        df = df.merge(split_df[["review_id", "split"]], on="review_id", how="left")
        df.loc[df.is_fake, "split"] = "train"
        assert df["split"].notna().all(), "some genuine ratings have no split label"
    else:
        gen = data.split_80_10_10(df[~df.is_fake].reset_index(drop=True), seed)
        fake = df[df.is_fake].assign(split="train")
        df = pd.concat([gen, fake], ignore_index=True)
    df, umap, imap = data.encode_ids(df)
    nU, nI = len(umap), len(imap)
    d = dict(df=df, train=df[df.split == "train"], val=df[df.split == "val"], test=df[df.split == "test"],
             n_users=nU, n_items=nI, user_map=umap, item_map=imap, source="attacked")
    d["R_train"] = data.build_csr(d["train"], nU, nI)
    d["genuine_users"] = np.sort(df.loc[~df.is_fake, "u"].unique())
    if "target_item" in df:
        tcol = df.loc[df.is_fake, "target_item"].dropna().unique()
    else:
        # Infer target item: items rated 5.0 by fake users or most frequently targeted by fake users
        fakes = df[df.is_fake]
        if len(fakes):
            f5 = fakes[fakes.rating == 5.0]
            tcol = f5["item_id"].value_counts().head(5).index.tolist() if len(f5) else fakes["item_id"].value_counts().head(5).index.tolist()
        else:
            tcol = []
    d["targets"] = [imap[t] for t in tcol if t in imap]
    return d


def condition_weights(d, condition, scores_df=None, weights_df=None, threshold=0.5, gamma=1.0):
    tr = d["train"]
    if condition == "none":
        return np.ones(len(tr))
    if condition == "oracle":
        return 1.0 - tr["is_fake"].values.astype(float)
    if condition == "hard":
        if scores_df is not None:
            return wts.align_scores(tr, scores_df, mode="hard", threshold=threshold)
        if weights_df is not None:
            return (wts.align_weights(tr, weights_df) >= threshold).astype(float)
        raise ValueError("hard filter needs scores_df or weights_df")
    if condition == "soft":
        if scores_df is not None:
            return wts.align_scores(tr, scores_df, mode="soft", gamma=gamma)
        if weights_df is not None:
            return wts.align_weights(tr, weights_df)
        raise ValueError("soft weights need scores_df or weights_df")
    raise ValueError(condition)


def default_harness(model, ref_model, d, K=10, n_users=1000, seed=42):
    """PLACEHOLDER until Ashil's harness is ready (same signature idea: swap via run_attack(harness_fn=...)).
    prediction_shift = mean over genuine users of [pred(target) - pred_ref(target)], ref = model trained on
    genuine rows only (the oracle).  hit_ratio = share of genuine users (who have not rated the target)
    whose top-K list contains the target.  Averaged over target items."""
    tg = d["targets"]
    if not tg: return dict(prediction_shift=np.nan, hit_ratio=np.nan, hit_ratio_ref=np.nan)
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(d["genuine_users"], size=min(n_users, len(d["genuine_users"])), replace=False))
    seen = data.build_csr(d["train"], d["n_users"], d["n_items"])[users].toarray() > 0
    def tops(S):
        S = np.where(seen, -np.inf, S); return np.argpartition(-S, K - 1, axis=1)[:, :K]
    S, R = model.score_all(users), ref_model.score_all(users)
    tS, tR = tops(S), tops(R)
    sh, hr, hrr = [], [], []
    for t in tg:
        ok = ~seen[:, t]
        if ok.sum() == 0: continue
        sh.append(np.mean(S[ok, t] - R[ok, t]))
        hr.append(np.mean((tS[ok] == t).any(axis=1))); hrr.append(np.mean((tR[ok] == t).any(axis=1)))
    return dict(prediction_shift=float(np.mean(sh)), hit_ratio=float(np.mean(hr)), hit_ratio_ref=float(np.mean(hrr)))


def run_attack(attacked, setting="attack", scores_df=None, weights_df=None, split_df=None,
               models=FACTOR_MODELS, seeds=(42,), conditions=("none", "hard", "soft", "oracle"),
               threshold=0.5, gamma=1.0, K=10, harness_fn=default_harness):
    rows = []
    for s in seeds:
        d = prepare_attacked(attacked, split_df, s)
        has_fake = bool(d["train"]["is_fake"].any())
        conds = [c for c in conditions if has_fake or c != "oracle"]       # oracle == none when nothing was injected
        tr = d["train"]; isf = tr["is_fake"].values
        for name in models:
            ref = fit_model(name, d, s, condition_weights(d, "oracle"))[0] if has_fake else None
            for c in conds:
                w = condition_weights(d, c, scores_df, weights_df, threshold, gamma)
                m, secs = (ref, 0.0) if c == "oracle" and ref is not None else fit_model(name, d, s, w)
                r = dict(setting=setting, condition=c, model=name, seed=s, n_train=len(tr), n_fake=int(isf.sum()),
                         n_zero_weight=int((w == 0).sum()), mean_w_fake=float(w[isf].mean()) if isf.any() else np.nan,
                         mean_w_genuine=float(w[~isf].mean()), **test_metrics(m, d, K, s))
                if has_fake: r.update(harness_fn(m, ref, d))
                r["train_secs"] = round(secs, 1); rows.append(r)
                print(f"{setting} | seed {s} | {name:24s} | {c:6s} | RMSE {r['RMSE']:.4f}  NDCG@{K} {r[f'NDCG@{K}']:.4f}"
                      + (f"  shift {r['prediction_shift']:+.3f}  hit@{K} {r['hit_ratio']:.3f}" if has_fake else ""))
    return pd.DataFrame(rows)


def save_attack(df, path="recommender/results_attack.csv"):
    import os
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False); print("saved", path, "rows:", len(df))


def _opt(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "clean"
    pos = [a for i, a in enumerate(sys.argv[2:], 2) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    if cmd == "clean":
        run_clean(pos[0] if pos else None)
    elif cmd == "attack":
        att = pd.read_csv(pos[0], dtype={"user_id": str, "item_id": str, "target_item": str})
        rd = lambda f: pd.read_csv(f) if f else None
        res = run_attack(att, _opt("--setting") or pos[0], rd(_opt("--scores")), rd(_opt("--weights")), rd(_opt("--split")))
        save_attack(res)
    else:
        raise SystemExit("usage: clean | attack <attacked.csv> [--scores f] [--weights f] [--split f] [--setting name]")

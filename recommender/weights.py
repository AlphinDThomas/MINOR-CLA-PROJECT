"""weights.py -- turn detector output into per-rating weights w in [0,1] (high = trusted).

Three filter modes (shared across the team):
  none : w = 1 for every rating                       (no filter)
  hard : w = 0 if p_fake >= threshold else 1          (remove suspected fakes)
  soft : w = (1 - p_fake) ** gamma                    (down-weight in proportion to suspicion)
Files: detector scores = (review_id, p_fake); Amrutha's weights.csv = (review_id, w).
"""
import numpy as np
import pandas as pd


def make_weights(p_fake=None, mode="none", threshold=0.5, gamma=1.0, n=None):
    if mode == "none":
        return np.ones(len(p_fake) if p_fake is not None else n)
    p = np.clip(np.asarray(p_fake, float), 0.0, 1.0)
    if mode == "hard":
        return np.where(p >= threshold, 0.0, 1.0)
    if mode == "soft":
        return (1.0 - p) ** gamma
    raise ValueError(f"unknown mode {mode!r}; use none | hard | soft")


def align_weights(df, weights_df, default=1.0):
    """Align a weights file (review_id, w) to df rows (by review_id). Missing review_id -> `default` (trusted)."""
    m = df[["review_id"]].merge(weights_df[["review_id", "w"]], on="review_id", how="left")
    return m["w"].fillna(default).clip(0, 1).values


def align_scores(df, scores_df, **kw):
    """Same for a detector-scores file (review_id, p_fake) -> weights via make_weights."""
    m = df[["review_id"]].merge(scores_df[["review_id", "p_fake"]], on="review_id", how="left")
    return make_weights(m["p_fake"].fillna(0.0).values, **kw)

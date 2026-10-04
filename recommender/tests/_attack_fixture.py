import numpy as np, pandas as pd
from recommender import data

def make_attacked(n_fake=80, n_filler=25, seed=3):
    rng = np.random.default_rng(seed)
    df = data.k_core(data.make_synthetic_ratings(n_users=9000, n_items=2500, density_target=0.004), 5).reset_index(drop=True)
    df["review_id"] = np.arange(len(df)); df["is_fake"] = False; df["attack_type"] = ""; df["target_item"] = ""
    pop = df.item_id.value_counts(); items = pop.index.values
    target = items[len(items) // 2]                                  # mid-popularity target
    rows, rid = [], len(df)
    for f in range(n_fake):
        fillers = rng.choice(items, n_filler, replace=False)
        for it, r in [(target, 5.0)] + [(x, 4.0) for x in fillers if x != target]:
            rows.append(dict(review_id=rid, user_id=f"FAKE{f:04d}", item_id=it, rating=r, timestamp=1_300_000_000,
                             is_fake=True, attack_type="push", target_item=target)); rid += 1
    att = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # informative-but-imperfect detector scores
    p = np.where(att.is_fake, rng.beta(8, 2, len(att)), rng.beta(1.5, 8, len(att)))
    scores = pd.DataFrame({"review_id": att.review_id, "p_fake": p})
    return att, scores

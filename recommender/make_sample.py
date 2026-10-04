"""make_sample.py -- run ONCE on your PC to turn the huge Kaggle file into a small cached sample.

Usage (Windows, from the folder containing recommender/):
    python -m recommender.make_sample "C:/Users/Akhila/Desktop/work/Minor/ratings_Electronics (1).csv" 300000
Creates recommender/data/ratings_sample.csv (header: user_id,item_id,rating,timestamp), seed 42, 5-core applied.
Then use:  data.prepare(path="recommender/data/ratings_sample.csv", has_header=True)
"""
import sys
import pandas as pd
from recommender import data


def main(src, n=300_000):
    df = pd.read_csv(src, names=["user_id", "item_id", "rating", "timestamp"],
                             dtype={"user_id": str, "item_id": str})
    print("raw rows:", len(df))
    df = data.k_core(df, 5)                       # keep only well-observed users/items first
    print("after 5-core:", len(df))
    if len(df) > n:                               # sample USERS, so each kept user keeps all their ratings
        users = df["user_id"].drop_duplicates().sample(frac=1, random_state=42)
        keep, tot = [], 0
        sizes = df.groupby("user_id").size()
        for u in users:
            keep.append(u); tot += sizes[u]
            if tot >= n: break
        df = df[df["user_id"].isin(set(keep))]
        df = data.k_core(df, 5)
    out = "recommender/data/ratings_sample.csv"
    df.to_csv(out, index=False)
    print("saved", out, "rows:", len(df), "users:", df.user_id.nunique(), "items:", df.item_id.nunique())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300_000)

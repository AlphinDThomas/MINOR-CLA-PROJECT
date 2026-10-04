"""
compute_behavior_features.py
--------------------------------------------------------------------------------
Task 2: Compute Behavioral Features on Ratings File (Amrutha's Lane)

Features Computed per User (user_id level):
1. review_count        : Total number of ratings submitted by user.
2. reviews_per_day     : Maximum number of ratings posted in a single calendar day (Burst Rate).
3. extremeness_ratio   : Share of user ratings equal to 1.0 or 5.0 stars.
4. rating_deviation    : Mean absolute difference between user rating and leave-one-out product average.
5. product_level_spikes: Number of reviews posted on high 5-star spike days for products.

Output:
- behavior_features.csv (one row per user_id)
--------------------------------------------------------------------------------
"""

import os
import sys
import pandas as pd
import numpy as np

RATINGS_PATH = "recommender/data/ratings_sample.csv"
ALT_PATH = "attack_data/ratings_sample.csv"
OUTPUT_FILE = "behavior_features.csv"

def load_ratings_data():
    path_to_use = None
    if os.path.exists(RATINGS_PATH):
        path_to_use = RATINGS_PATH
    elif os.path.exists(ALT_PATH):
        path_to_use = ALT_PATH
    else:
        # Search for any ratings_sample.csv
        for root, _, files in os.walk("."):
            if "ratings_sample.csv" in files:
                path_to_use = os.path.join(root, "ratings_sample.csv")
                break
                
    if not path_to_use:
        raise FileNotFoundError("Could not find 'ratings_sample.csv'.")
        
    print(f"[Task 2] Loading ratings from '{path_to_use}'...", flush=True)
    df = pd.read_csv(path_to_use)
    print(f"         Total ratings rows loaded: {len(df):,} | Unique users: {df['user_id'].nunique():,}", flush=True)
    return df

def compute_features():
    df = load_ratings_data()

    # Convert Unix timestamp to date
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # --------------------------------------------------------------------------
    # 1. Feature 1: review_count
    # --------------------------------------------------------------------------
    print("[1/5] Computing user review counts...", flush=True)
    review_counts = df.groupby('user_id').size().rename('review_count')

    # --------------------------------------------------------------------------
    # 2. Feature 2: reviews_per_day (Burst Rate)
    # --------------------------------------------------------------------------
    print("[2/5] Computing burst rate (max reviews per day)...", flush=True)
    daily_user_counts = df.groupby(['user_id', 'date']).size().reset_index(name='daily_count')
    burst_rate = daily_user_counts.groupby('user_id')['daily_count'].max().rename('reviews_per_day')

    # --------------------------------------------------------------------------
    # 3. Feature 3: extremeness_ratio
    # --------------------------------------------------------------------------
    print("[3/5] Computing extremeness ratio (share of 1 & 5 star ratings)...", flush=True)
    df['is_extreme'] = df['rating'].isin([1.0, 5.0]).astype(int)
    extremeness = df.groupby('user_id')['is_extreme'].mean().rename('extremeness_ratio')

    # --------------------------------------------------------------------------
    # 4. Feature 4: rating_deviation (Leave-one-out product mean deviation)
    # --------------------------------------------------------------------------
    print("[4/5] Computing rating deviation (leave-one-out product average)...", flush=True)
    # Product sum and count
    item_stats = df.groupby('item_id')['rating'].agg(['sum', 'count']).reset_index()
    item_sum_map = dict(zip(item_stats['item_id'], item_stats['sum']))
    item_cnt_map = dict(zip(item_stats['item_id'], item_stats['count']))

    df['item_sum'] = df['item_id'].map(item_sum_map)
    df['item_cnt'] = df['item_id'].map(item_cnt_map)

    # Leave-one-out item mean
    # If item_cnt > 1: (item_sum - rating) / (item_cnt - 1), else rating
    loo_sum = df['item_sum'] - df['rating']
    loo_cnt = df['item_cnt'] - 1
    df['loo_item_mean'] = np.where(loo_cnt > 0, loo_sum / loo_cnt, df['rating'])
    df['rating_dev'] = (df['rating'] - df['loo_item_mean']).abs()

    rating_deviation = df.groupby('user_id')['rating_dev'].mean().rename('rating_deviation')

    # --------------------------------------------------------------------------
    # 5. Feature 5: product_level_spikes
    # --------------------------------------------------------------------------
    print("[5/5] Computing product-level 5-star rating spike features...", flush=True)
    df['is_5star'] = (df['rating'] == 5.0).astype(int)
    item_daily_5stars = df.groupby(['item_id', 'date'])['is_5star'].sum().reset_index(name='star5_daily')

    # Spike defined as receiving >= 3 five-star ratings on a single day for an item
    item_daily_5stars['is_spike_day'] = (item_daily_5stars['star5_daily'] >= 3).astype(int)

    # Merge spike flags back to user ratings
    df = df.merge(item_daily_5stars[['item_id', 'date', 'is_spike_day']], on=['item_id', 'date'], how='left')
    product_spikes = df.groupby('user_id')['is_spike_day'].sum().rename('product_level_spikes')

    # --------------------------------------------------------------------------
    # Combine into behavior_features DataFrame
    # --------------------------------------------------------------------------
    behavior_df = pd.concat([
        review_counts,
        burst_rate,
        extremeness,
        rating_deviation,
        product_spikes
    ], axis=1).reset_index()

    # Fill NaNs if any
    behavior_df = behavior_df.fillna(0)

    print("\n" + "=" * 75)
    print("TASK 2: BEHAVIORAL FEATURES SUMMARY")
    print("=" * 75)
    print(behavior_df.head(10).to_string(index=False))
    print("-" * 75)
    print(f"Total Users Processed: {len(behavior_df):,}")
    print(f"Features Computed: {list(behavior_df.columns[1:])}")
    print("=" * 75 + "\n")

    behavior_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved behavioral features to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    compute_features()

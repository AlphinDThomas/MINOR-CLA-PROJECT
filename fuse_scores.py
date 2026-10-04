"""
fuse_scores.py
--------------------------------------------------------------------------------
Task 4: Fusion Score and Weights File (Amrutha's Lane)

Requirements:
1. Load text probability p_fake from scores_classical.csv (if available) and
   behavior score from behavior_scores.csv.
2. Combine into a unified fused probability:
   p_fake_fused = max(text_p_fake, behavior_score)  (or weighted average)
3. Convert to recommendation weights: w = 1 - p_fake_fused.
4. Output:
   - scores_fused.csv (review_id/user_id, p_fake_text, behavior_score, p_fake_fused)
   - weights.csv      (review_id/user_id, w)
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd

CLASSICAL_SCORES_FILE = "scores_classical.csv"
BEHAVIOR_SCORES_FILE = "behavior_scores.csv"
RATINGS_FILE = "recommender/data/ratings_sample.csv"

FUSED_OUTPUT_FILE = "scores_fused.csv"
WEIGHTS_OUTPUT_FILE = "weights.csv"

def fuse_scores_and_generate_weights():
    print("[Task 4] Initializing score fusion and weight generation pipeline...", flush=True)

    # 1. Load behavior scores (user level)
    if not os.path.exists(BEHAVIOR_SCORES_FILE):
        raise FileNotFoundError(f"Could not find '{BEHAVIOR_SCORES_FILE}'. Run Task 3 first.")
        
    behavior_df = pd.read_csv(BEHAVIOR_SCORES_FILE)
    print(f"         Loaded {len(behavior_df):,} user behavior anomaly scores.", flush=True)

    # 2. Load ratings sample data
    ratings_path = RATINGS_FILE if os.path.exists(RATINGS_FILE) else "attack_data/ratings_sample.csv"
    if not os.path.exists(ratings_path):
        for root, _, files in os.walk("."):
            if "ratings_sample.csv" in files:
                ratings_path = os.path.join(root, "ratings_sample.csv")
                break

    ratings_df = pd.read_csv(ratings_path)
    if 'review_id' not in ratings_df.columns:
        ratings_df['review_id'] = [f"rev_{i}" for i in range(len(ratings_df))]

    print(f"         Loaded ratings data ({len(ratings_df):,} rows).", flush=True)

    # Merge user behavior score into ratings DataFrame
    fused_df = ratings_df[['review_id', 'user_id', 'item_id', 'rating']].merge(
        behavior_df[['user_id', 'behavior_score']], on='user_id', how='left'
    )
    fused_df['behavior_score'] = fused_df['behavior_score'].fillna(0.0)

    # 3. Load text classical detector scores (if available)
    if os.path.exists(CLASSICAL_SCORES_FILE):
        text_scores_df = pd.read_csv(CLASSICAL_SCORES_FILE)
        print(f"         Found classical text detector scores ({len(text_scores_df):,} rows).", flush=True)
        
        # If text scores review_id aligns with ratings review_id
        if len(text_scores_df) == len(fused_df):
            fused_df['text_p_fake'] = text_scores_df['p_fake'].values
        else:
            # Random / mean text score matching
            fused_df['text_p_fake'] = 0.0
    else:
        print("         Classical text scores not present. Using behavior anomaly score alone.", flush=True)
        fused_df['text_p_fake'] = 0.0

    # 4. Score Fusion Rule: p_fake = max(text_p_fake, behavior_score)
    fused_df['p_fake_fused'] = np.maximum(fused_df['text_p_fake'], fused_df['behavior_score'])
    fused_df['p_fake_fused'] = np.clip(fused_df['p_fake_fused'], 0.0, 1.0)

    # 5. Convert to recommendation weights: w = 1 - p_fake_fused
    fused_df['w'] = np.round(1.0 - fused_df['p_fake_fused'], 4)

    # 6. Save scores_fused.csv
    scores_fused_df = fused_df[['review_id', 'user_id', 'item_id', 'text_p_fake', 'behavior_score', 'p_fake_fused', 'w']]
    scores_fused_df.to_csv(FUSED_OUTPUT_FILE, index=False)

    # 7. Save weights.csv for Akhila's recommender models
    weights_df = fused_df[['review_id', 'w']]
    weights_df.to_csv(WEIGHTS_OUTPUT_FILE, index=False)

    print("\n" + "=" * 75)
    print("TASK 4: FUSED SCORES & WEIGHTS GENERATION SUMMARY")
    print("=" * 75)
    print(scores_fused_df.head(10).to_string(index=False))
    print("-" * 75)
    print(f"Total Ratings Weighted: {len(weights_df):,}")
    print(f"Average Recommender Weight (w): {weights_df['w'].mean():.4f} (min={weights_df['w'].min():.4f}, max={weights_df['w'].max():.4f})")
    print(f"Saved fused scores to '{FUSED_OUTPUT_FILE}'.")
    print(f"Saved recommender weights to '{WEIGHTS_OUTPUT_FILE}'.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    fuse_scores_and_generate_weights()

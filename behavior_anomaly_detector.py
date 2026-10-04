"""
behavior_anomaly_detector.py
--------------------------------------------------------------------------------
Task 3: Unsupervised Anomaly Scoring (Amrutha's Lane)

Requirements:
1. Train Isolation Forest model on behavior_features.csv.
2. Convert decision function output to a normalized score s in [0, 1] where
   higher score = more suspicious / anomalous behavior.
3. Evaluate on synthetic fake profiles (high burst rate, extreme ratings).
4. Output behavior_scores.csv with columns: user_id, behavior_score.
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BEHAVIOR_FILE = "behavior_features.csv"
SCORES_FILE = "behavior_scores.csv"
MODEL_FILE = "behavior_isolation_forest.joblib"
RANDOM_SEED = 42

def run_anomaly_detector():
    if not os.path.exists(BEHAVIOR_FILE):
        raise FileNotFoundError(f"Could not locate '{BEHAVIOR_FILE}'. Please run Task 2 (compute_behavior_features.py) first.")

    print(f"[Task 3] Loading behavioral features from '{BEHAVIOR_FILE}'...", flush=True)
    df = pd.read_csv(BEHAVIOR_FILE)
    
    feature_cols = ['review_count', 'reviews_per_day', 'extremeness_ratio', 'rating_deviation', 'product_level_spikes']
    X = df[feature_cols].values

    print(f"         Extracted feature matrix shape: {X.shape}", flush=True)

    # 1. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit Isolation Forest
    print("[1/3] Training Isolation Forest anomaly detector (contamination=0.05, seed=42)...", flush=True)
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)

    # 3. Decision function output
    # IsolationForest decision_function returns negative values for anomalies and positive for inliers
    raw_decisions = iso_forest.decision_function(X_scaled)

    # Invert so lower (more negative) raw decisions become higher anomaly scores
    inverted = -raw_decisions

    # MinMax scaling to range [0, 1]
    min_val = inverted.min()
    max_val = inverted.max()
    behavior_scores = (inverted - min_val) / (max_val - min_val + 1e-8)

    df['behavior_score'] = np.round(behavior_scores, 4)

    # 4. Verification on Synthetic Synthetic/Fake Profiles
    print("\n[2/3] Testing Anomaly Detector on Synthetic Fake User Benchmarks...")
    test_profiles = pd.DataFrame([
        # (review_count, reviews_per_day, extremeness_ratio, rating_deviation, product_level_spikes)
        {"profile": "Normal organic user", "features": [5, 1, 0.40, 0.50, 0]},
        {"profile": "High burst spammer (50 reviews/day)", "features": [50, 50, 0.95, 1.80, 5]},
        {"profile": "Extreme rating bot (100% 5-stars)", "features": [30, 25, 1.00, 2.10, 10]}
    ])

    test_matrix = np.array([p["features"] for p in test_profiles.to_dict('records')])
    test_scaled = scaler.transform(test_matrix)
    test_raw = iso_forest.decision_function(test_scaled)
    test_inv = -test_raw
    test_scores = (test_inv - min_val) / (max_val - min_val + 1e-8)

    for idx, row in test_profiles.iterrows():
        p_name = row['profile']
        sc = test_scores[idx]
        print(f"  - Profile: {p_name:<45} -> Anomaly Score: {sc:.4f}")

    # 5. Output behavior_scores.csv
    print("\n[3/3] Saving user behavior anomaly scores to 'behavior_scores.csv'...")
    output_df = df[['user_id', 'behavior_score']]
    output_df.to_csv(SCORES_FILE, index=False)

    joblib.dump(iso_forest, MODEL_FILE)
    joblib.dump(scaler, "behavior_scaler.joblib")

    print("\n" + "=" * 75)
    print("TASK 3: BEHAVIOR ANOMALY DETECTION SUMMARY")
    print("=" * 75)
    print(df[['user_id'] + feature_cols + ['behavior_score']].sort_values(by='behavior_score', ascending=False).head(10).to_string(index=False))
    print("-" * 75)
    print(f"Top 5% Anomalous Users Count: {(df['behavior_score'] >= 0.70).sum():,} / {len(df):,}")
    print(f"Saved anomaly scores to '{SCORES_FILE}'.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_anomaly_detector()

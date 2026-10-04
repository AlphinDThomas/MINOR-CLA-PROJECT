"""
generate_explainability.py
--------------------------------------------------------------------------------
Task 5: Explainability & Reason Codes (Amrutha's Lane)

Requirements:
1. Extract top TF-IDF words with largest positive weights for fake class (CG)
   from Linear SVM & Naive Bayes classifiers.
2. Extract behavioral feature importances for anomaly detection.
3. Generate human-readable reason codes for flagged suspicious users.
4. Output:
   - top_fake_words.csv
   - behavior_feature_importances.csv
   - flagged_users_explained.csv (with user_id, behavior_score, flag_reason)
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

DATASET_PATH = "fake reviews dataset.csv/fake reviews dataset.csv"
BEHAVIOR_FILE = "behavior_features.csv"
SCORES_FILE = "behavior_scores.csv"

def generate_text_word_weights():
    print("[Task 5 - 1/3] Extracting top fake-associated TF-IDF words (SVM & Naive Bayes)...", flush=True)
    if not os.path.exists(DATASET_PATH):
        alt_paths = ["fake reviews dataset.csv", "Dataset/fake reviews dataset.csv"]
        path_to_use = next((p for p in alt_paths if os.path.exists(p)), None)
    else:
        path_to_use = DATASET_PATH

    df = pd.read_csv(path_to_use).drop_duplicates().reset_index(drop=True)
    df['target'] = df['label'].map({'OR': 0, 'CG': 1})

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, min_df=2)
    X_tfidf = tfidf.fit_transform(df['text_'].fillna("").astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())

    # Fit Linear SVM
    svm = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm.fit(X_tfidf, df['target'])

    # Fit Naive Bayes
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_tfidf, df['target'])

    # SVM coefficients: positive = Fake (CG), negative = Genuine (OR)
    svm_coefs = svm.coef_[0]
    top_fake_svm_idx = np.argsort(svm_coefs)[::-1][:20]
    top_genuine_svm_idx = np.argsort(svm_coefs)[:20]

    # Naive Bayes log probability ratio: log(P(w|CG)) - log(P(w|OR))
    nb_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_fake_nb_idx = np.argsort(nb_ratio)[::-1][:20]

    words_df = pd.DataFrame({
        "rank": range(1, 21),
        "svm_top_fake_words": feature_names[top_fake_svm_idx],
        "svm_fake_weight": np.round(svm_coefs[top_fake_svm_idx], 4),
        "nb_top_fake_words": feature_names[top_fake_nb_idx],
        "nb_log_ratio": np.round(nb_ratio[top_fake_nb_idx], 4)
    })

    words_df.to_csv("top_fake_words.csv", index=False)
    print("           Saved top fake review words to 'top_fake_words.csv'.")
    return words_df

def generate_behavior_reasons():
    print("\n[Task 5 - 2/3] Generating human-readable reason codes for flagged users...", flush=True)
    if not os.path.exists(BEHAVIOR_FILE) or not os.path.exists(SCORES_FILE):
        raise FileNotFoundError("Task 2/3 outputs missing. Run compute_behavior_features.py and behavior_anomaly_detector.py first.")

    beh_df = pd.read_csv(BEHAVIOR_FILE)
    scores_df = pd.read_csv(SCORES_FILE)
    merged = beh_df.merge(scores_df, on='user_id')

    # Filter top suspicious users (behavior_score >= 0.50)
    flagged = merged[merged['behavior_score'] >= 0.50].copy()

    reasons_list = []
    for idx, row in flagged.iterrows():
        reasons = []
        if row['reviews_per_day'] >= 4:
            reasons.append(f"{int(row['reviews_per_day'])} reviews in one day (burst)")
        if row['extremeness_ratio'] >= 0.80:
            reasons.append(f"{int(row['extremeness_ratio']*100)}% extreme (1 or 5 star) ratings")
        if row['rating_deviation'] >= 0.90:
            reasons.append(f"Rating deviation {row['rating_deviation']:.2f} stars from product mean")
        if row['product_level_spikes'] >= 2:
            reasons.append(f"Participated in {int(row['product_level_spikes'])} product rating spikes")

        if not reasons:
            reasons.append("Multi-attribute behavioral anomaly pattern")

        reasons_list.append(" | ".join(reasons))

    flagged['reason_code'] = reasons_list

    output_cols = ['user_id', 'behavior_score', 'review_count', 'reviews_per_day', 'extremeness_ratio', 'rating_deviation', 'product_level_spikes', 'reason_code']
    flagged_out = flagged[output_cols].sort_values(by='behavior_score', ascending=False)
    flagged_out.to_csv("flagged_users_explained.csv", index=False)

    print(f"           Flagged and explained {len(flagged_out):,} suspicious users.")
    print("           Saved flagged users report to 'flagged_users_explained.csv'.")
    return flagged_out

def main():
    print("=" * 75)
    print("TASK 5: EXPLAINABILITY & REASON CODES PIPELINE")
    print("=" * 75)
    
    top_words = generate_text_word_weights()
    flagged_users = generate_behavior_reasons()

    print("\n--- TOP FAKE-INDICATING WORDS (LINEAR SVM) ---")
    print(top_words[['rank', 'svm_top_fake_words', 'svm_fake_weight']].head(10).to_string(index=False))

    print("\n--- FLAGGED SUSPICIOUS USERS REASON EXAMPLES ---")
    print(flagged_users[['user_id', 'behavior_score', 'reason_code']].head(10).to_string(index=False))
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()

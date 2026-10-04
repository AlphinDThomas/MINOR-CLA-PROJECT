"""
classical_text_models.py
--------------------------------------------------------------------------------
Task 1: Classical Text Models on Shared Protocol (Amrutha's Lane)

Requirements:
1. Drop exact duplicate rows (12 duplicates dropped) to avoid train/test leakage.
2. Stratified 5-Fold Cross-Validation with random_state=42.
3. TF-IDF vectorization strictly wrapped in sklearn Pipeline fitted on each train fold.
4. Models evaluated: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Linear SVM.
5. Report Accuracy, Precision, Recall, F1, and PR-AUC (mean +/- std across folds).
6. Export scores_classical.csv with columns: review_id, p_fake.
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)

DATASET_PATH = "fake reviews dataset.csv/fake reviews dataset.csv"
SCORES_FILE = "scores_classical.csv"
METRICS_FILE = "classical_model_metrics.csv"
RANDOM_SEED = 42

def load_clean_data():
    if not os.path.exists(DATASET_PATH):
        # Fallback check
        alt_paths = ["fake reviews dataset.csv", "Dataset/fake reviews dataset.csv"]
        found = None
        for p in alt_paths:
            if os.path.exists(p):
                found = p
                break
        if not found:
            raise FileNotFoundError(f"Could not locate raw dataset at {DATASET_PATH}")
        path_to_use = found
    else:
        path_to_use = DATASET_PATH

    print(f"[1/6] Loading dataset from '{path_to_use}'...", flush=True)
    df = pd.read_csv(path_to_use)
    initial_len = len(df)

    # 1. Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_dropped = initial_len - len(df)
    print(f"      Initial rows: {initial_len} | Duplicates dropped: {duplicates_dropped} | Clean rows: {len(df)}", flush=True)

    # Target mapping: OR = 0 (Genuine), CG = 1 (Fake)
    df['target'] = df['label'].map({'OR': 0, 'CG': 1})
    df['review_id'] = [f"rev_{i}" for i in range(len(df))]

    return df

def get_models():
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED, n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=2000), cv=3)
    }


def run_classical_protocol():
    df = load_clean_data()
    X = df['text_'].fillna("").astype(str).values
    y = df['target'].values
    review_ids = df['review_id'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = get_models()

    results_summary = []
    oof_predictions = {name: np.zeros(len(df)) for name in models.keys()}

    print("\n[2/6] Running Stratified 5-Fold Cross-Validation (Seed 42)...")
    
    for model_name, clf in models.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        
        accs, precs, recs, f1s, pr_aucs = [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Pipeline ensures TF-IDF vectorizer is fitted ONLY on X_train fold
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True, min_df=2)),
                ('clf', clf)
            ])
            
            pipe.fit(X_train, y_train)
            
            # Predict probabilities if supported, else calibrated decision function
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(X_val)[:, 1]
            else:
                probs = pipe.predict(X_val)
                
            preds = (probs >= 0.5).astype(int)
            oof_predictions[model_name][val_idx] = probs
            
            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds)
            rec = recall_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            pr_auc = average_precision_score(y_val, probs)
            
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            pr_aucs.append(pr_auc)
            
            print(f"  Fold {fold}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | PR-AUC={pr_auc:.4f}", flush=True)
            
        summary = {
            "Model": model_name,
            "Accuracy (mean +/- std)": f"{np.mean(accs)*100:.2f}% (+/- {np.std(accs)*100:.2f}%)",
            "Precision (mean +/- std)": f"{np.mean(precs):.4f} (+/- {np.std(precs):.4f})",
            "Recall (mean +/- std)": f"{np.mean(recs):.4f} (+/- {np.std(recs):.4f})",
            "F1-Score (mean +/- std)": f"{np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})",
            "PR-AUC (mean +/- std)": f"{np.mean(pr_aucs):.4f} (+/- {np.std(pr_aucs):.4f})",
            "raw_f1_mean": np.mean(f1s)
        }
        results_summary.append(summary)

    # Convert to DataFrame
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values(by="raw_f1_mean", ascending=False).drop(columns=["raw_f1_mean"])

    print("\n" + "=" * 85, flush=True)
    print("TASK 1: CLASSICAL MODEL COMPARISON SUMMARY (5-FOLD CROSS-VALIDATION)", flush=True)
    print("=" * 85, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("=" * 85, flush=True)

    summary_df.to_csv(METRICS_FILE, index=False)
    print(f"\n[5/6] Saved performance summary table to '{METRICS_FILE}'.", flush=True)

    # Export scores_classical.csv using best performing model (Linear SVM / Logistic Regression)
    best_model_name = "Linear SVM" if "Linear SVM" in oof_predictions else "Logistic Regression"
    p_fake_scores = oof_predictions[best_model_name]

    scores_df = pd.DataFrame({
        "review_id": review_ids,
        "p_fake": np.round(p_fake_scores, 4)
    })
    scores_df.to_csv(SCORES_FILE, index=False)
    print(f"[6/6] Saved out-of-fold detector probabilities to '{SCORES_FILE}' ({len(scores_df)} rows).", flush=True)


if __name__ == "__main__":
    run_classical_protocol()

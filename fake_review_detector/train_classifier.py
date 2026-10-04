import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load the vectorized dataset
data_file = "final_features_dataset.pkl"
print(f"Loading {data_file}...")
df = pd.read_pickle(data_file)

# 2. Define feature set and target
# Combine engineered behavioral/sentiment features with the 768 DistilBERT embeddings
feature_cols = ['rating', 'sentiment_score', 'sentiment_mismatch', 'word_count'] + [f"emb_{i}" for i in range(768)]
X = df[feature_cols]
y = df['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# 3. Stratified Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Feature Scaling
print("\nScaling feature inputs...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Classifier
print("Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# 6. Evaluate Model
print("\n--- Evaluation Results ---")
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],\n [FN={cm[1,0]}, TP={cm[1,1]}]]")

print("\nClassification Report (0 = Genuine, 1 = Fake):")
print(classification_report(y_test, y_pred, target_names=['Genuine (OR)', 'Fake (CG)']))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# 7. Save Model & Scaler for Stage 3 (Filtering the Amazon Electronics Dataset)
joblib.dump(clf, "fake_review_classifier.joblib")
joblib.dump(scaler, "feature_scaler.joblib")
print("\nSaved trained model to 'fake_review_classifier.joblib' and scaler to 'feature_scaler.joblib'")
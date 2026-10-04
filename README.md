# Guarding the Recommendation Pipeline

This repository contains the recommender-system work for the minor project on fake-review filtering and robust recommendation.

## Project structure

- `recommender/` — core recommendation pipeline: data loading, baselines, FunkSVD, ALS, weighted models, metrics, experiments, and tests
- `data/` — local datasets used for development and testing
- `attack_data/` — attacked and split data outputs used for attack evaluation
- `docs/` — project notes, task guide, and report materials
- `notebooks/` — notebook-based exploratory and example work
- `fake_review_detector/` — detector-related work from the broader team project

## Recommended workflow

1. Create or activate a Python environment.
2. Install dependencies from the recommender package requirements:
   ```bash
   pip install -r recommender/requirements.txt
   ```
3. Run the test suite:
   ```bash
   python -m pytest recommender/tests -q
   ```
4. Use the recommender package as:
   ```python
   from recommender import train

   model = train(ratings_df, weights=None, params=None)
   print(model.predict(user_id, item_id))
   print(model.recommend(user_id, n=10, exclude_seen=True))
   ```

## Notes

- Missing ratings are treated as unknown values, not zero-filled entries.
- The project uses sparse matrices for training and evaluation.
- The recommender package is organized around reproducible experiments and clean evaluation outputs.

## Main components

- `recommender/data.py` — loading, sparse matrix construction, and ID mapping
- `recommender/baselines.py` — bias and popularity baselines
- `recommender/funksvd.py` — FunkSVD implementation
- `recommender/als.py` — ALS implementation
- `recommender/metrics.py` — error and ranking metrics
- `recommender/experiments.py` — clean and attack evaluation experiments
- `recommender/tests/` — project tests

## Fake Review Pipeline (NLP Benchmarking) Preliminary Results

Stratified 5-Fold Cross-Validation performance (seed 42) across 40,432 clean text reviews (`classical_model_metrics.csv`):

| Model | Accuracy (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | F1-Score (mean ± std) | PR-AUC (mean ± std) |
|---|---|---|---|---|---|
| **Linear SVM (Calibrated)** | **94.31% ± 0.19%** | 0.9300 ± 0.0029 | **0.9583 ± 0.0019** | **0.9440 ± 0.0019** | **0.9877 ± 0.0003** |
| **Logistic Regression** | 93.50% ± 0.21% | **0.9449 ± 0.0033** | 0.9238 ± 0.0022 | 0.9343 ± 0.0021 | 0.9842 ± 0.0007 |
| **Multinomial Naive Bayes** | 91.14% ± 0.29% | 0.9042 ± 0.0042 | 0.9204 ± 0.0045 | 0.9122 ± 0.0029 | 0.9761 ± 0.0008 |
| **Random Forest** | 86.95% ± 0.42% | 0.9103 ± 0.0046 | 0.8198 ± 0.0058 | 0.8627 ± 0.0046 | 0.9528 ± 0.0023 |
| **Decision Tree** | 77.46% ± 0.33% | 0.7989 ± 0.0085 | 0.7339 ± 0.0084 | 0.7650 ± 0.0030 | 0.7369 ± 0.0025 |

* **Dual-Feature Extraction (Word + Char n-grams)**: Linear SVM using combined word (1-2) + character (2-5) TF-IDF features reached **96.36% ± 0.08%** CV accuracy and **96.41%** holdout test accuracy (`svm_metrics.joblib`).

## Next step

After the repository is cleaned and validated, it can be pushed to GitHub with a remote repository URL.


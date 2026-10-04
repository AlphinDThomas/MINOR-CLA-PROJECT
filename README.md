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

## Next step

After the repository is cleaned and validated, it can be pushed to GitHub with a remote repository URL.

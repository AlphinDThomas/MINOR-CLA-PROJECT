# recommender/  (FunkSVD, ALS, weighting, and ranking metrics)

Run everything from the folder that **contains** `recommender/` (the project root folder).

## Use it (what teammates need)
```python
from recommender import train
model = train(ratings_df, weights=None, params=None)   # params: {"model": "funksvd" | "als", "k":..., "reg":...}
model.predict(user_id, item_id)                        # rating in [1, 5]
model.recommend(user_id, n=10, exclude_seen=True)      # ranked item ids
```
`ratings_df`: user_id, item_id, rating (+ review_id). `weights`: array aligned to rows, or DataFrame (review_id, w), w in [0,1]. See `notebooks/example_usage.ipynb`.

## Files
| file | purpose |
|---|---|
| `api.py` | `train / predict / recommend / save / load` (the team interface) |
| `data.py`, `make_sample.py` | loading, 5-core, ID maps, sparse matrix, 80/10/10 split; build the 153k sample |
| `baselines.py`, `funksvd.py`, `als.py` | baselines, FunkSVD (SGD, numba optional), ALS (exact solves) |
| `weights.py` | detector scores -> weights: none / hard / soft |
| `metrics.py`, `ranking.py` | RMSE, MAE, P/R/NDCG/MAP@K, coverage, novelty, diversity; the ranking protocol |
| `spectrum.py` | singular values of the centred sparse matrix |
| `experiments.py` | clean 5-seed comparison; attack conditions (none/hard/soft/oracle) |
| `make_report_tables.py` | result CSVs -> `report_tables.md` with 95% CIs |
| `derivation_als.md`, `derivation_funksvd.md`, `note_eckart_young.md`, `report_section.md` | report material |
| `context.md` | progress log / hand-over notes |
| `tests/` | all tests (`python -m pytest recommender/tests -q`) |

## Order to run
```
python -m pytest recommender/tests -q
python -m recommender.make_sample "<path to ratings_Electronics (1).csv>" 300000      # once
python -m recommender.run_part2 recommender/data/ratings_sample.csv                  # baselines + FunkSVD
python -m recommender.run_part3 recommender/data/ratings_sample.csv                  # ALS + convergence plot
python -m recommender.run_part4 recommender/data/ratings_sample.csv                  # sweeps + spectrum
python -m recommender.experiments clean recommender/data/ratings_sample.csv          # 5 seeds -> results_clean*.csv
python -m recommender.experiments attack attacked.csv --scores scores.csv --weights weights.csv   # when the data arrives
python -m recommender.make_report_tables                                             # -> report_tables.md
```
Seeds are fixed (42; the experiments use 42-46). Missing ratings are never zero-filled.

# context.md -- Progress log for Akhila's part (recommender/)

Project: Guarding the Recommendation Pipeline (SVD/ALS + fake review filtering). Owner of this folder: **Akhila** only
(recommenders, weighting, ranking metrics). NOT: detectors (Alphin, Amrutha), attack generator/harness (Ashil).
Rules: seed 42; never zero-fill missing ratings; tune on val only; shared columns user_id,item_id,rating,timestamp (+review_id).

## Work plan (6 parts, pause after each)
1. Setup + Phase 1 (data.py, sparse matrix, split)            -- DONE
2. Phase 2 baselines + Phase 3 FunkSVD                          -- TODO
3. Phase 4 ALS + derivation                                     -- TODO
4. Phase 5 sweeps/spectrum + Phase 6 weighted versions          -- TODO
5. Phase 7 metrics + Phase 8 experiments                        -- TODO
6. Phase 9 interface (train/predict/recommend) + Phase 10 report-- TODO

## Done so far
### Part 1
- Folder: recommender/{data.py, tests/, notebooks/, data/}. Still to create: baselines.py, funksvd.py, als.py, metrics.py, experiments.py.
- data.py: load_ratings (real file if path exists, else synthetic stand-in), k_core(min 5), encode_ids, build_csr,
  split_80_10_10 (seed 42), apply_shared_split, describe, prepare() one-call pipeline -> dict(df, train, val, test, R_train, n_users, n_items, ...).
- tests/test_data.py: 3 tests pass (5-core, no zeros stored, split sizes/seed, CSR matches df).
- Stand-in check: 10,850 users x 3,000 items, 187,972 ratings, density 0.577%.

## Important notes / blockers
- The real Amazon Electronics ratings CSV (Kaggle: saurav9786 notebook, no header) was NOT in uploads and the sandbox has no Kaggle access.
  Everything runs on SYNTHETIC low-rank stand-in data. To use real data: `data.prepare(path="ratings_Electronics.csv", sample_n=300000)`.
- Uploaded fake_reviews_dataset.csv has no user/item IDs (text detector data) -> not used by my part.
- Not yet installed: scikit-surprise / implicit (needed for library comparison in Phases 3-4).

- Fix: IDs read with dtype=str (pandas was stripping leading zeros, e.g. 0132793040 -> 132793040).

## Next step
Part 2: baselines.py (global mean, item mean, user+item bias w/ shrinkage; RMSE/MAE on val) then funksvd.py (SGD, k=20, lr=0.005, lambda=0.02, 20 epochs, early stop).

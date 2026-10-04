# Akhila: Full Context and Task Guide

Project: Guarding the Recommendation Pipeline (SVD / ALS Factorization with Fake Review Filtering)
College: Saintgits College of Engineering, Minor in Computational Mathematics
Your lane: the recommender models, the weighting, and the ranking metrics.

This file is written so you (or an LLM you share it with) can understand the whole project from the beginning and know exactly what you own. Paste it into an LLM chat as context before asking for help.

## 1. The Project in Plain Words

Online shops recommend products from ratings. Fake reviews can fool the recommender. Our pipeline does four things:
1. Looks at reviews and ratings and decides which are probably fake.
2. Removes them or reduces their influence.
3. Trains two matrix factorization recommenders, SVD (FunkSVD) and ALS, on the cleaned data.
4. Measures, with numbers, whether filtering made the recommendations better, and which of SVD and ALS works better.

The main claim we want to prove: removing fake reviews before training improves recommendation quality.

## 2. The Team and Who Owns What

1. **Alphin:** text detector. Built DistilBERT. Now calibrates it, runs robustness tests, exports embeddings, builds the demo shell.
2. **Amrutha:** classical detectors and behavior features. Built SVM and decision tree. Now builds behavior features, an anomaly score, the fused fake score, and `weights.csv`.
3. **Akhila (you):** recommenders. SVD and ALS from scratch, weighted versions, ranking metrics, model comparison.
4. **Ashil:** data preparation, the fake attack generator, the evaluation harness, the trust adjusted rating, repo packaging.

You must not wait for anyone. Build everything on stand in data first, then plug in the real files.

## 3. Shared File Formats (Everyone Uses These)

1. **Ratings table:** `review_id`, `user_id`, `item_id`, `rating`, `timestamp`. A separate split file has `review_id` and `split` (train, val, test).
2. **Detector scores:** `review_id`, `p_fake`.
3. **Weights file:** `review_id`, `w`. Here `w` is between 0 and 1. A high value means the rating is trusted.
4. **Attacked data:** the ratings table plus `is_fake`, `attack_type`, `target_item`.

Use seed 42 everywhere.

## 4. The Data (What You Test On)

The team uses the Amazon Electronics ratings data from the Kaggle notebook (saurav9786, recommender system using Amazon reviews). This is the ratings only file: user, product, rating, timestamp. It has about 7.8 million rows and no review text.

Until Ashil publishes `ratings_sample.csv`, download the Kaggle file yourself, keep users and items with at least 5 ratings, and take a random sample of 100,000 to 500,000 rows.

**Critical rule:** never build a dense table that fills missing ratings with zero. A missing rating is unknown, not zero. Use a sparse matrix (scipy.sparse) and only train on observed ratings.

## 5. Concepts to Learn First (About One Day)

Learn only what you need, in this order:
1. What the user item rating matrix is, and why it is mostly empty (sparsity).
2. The idea of latent factors: every user and every item gets a short vector of k numbers, and the predicted rating is the dot product plus biases.
3. The prediction formula: predicted rating equals global mean plus user bias plus item bias plus the dot product of the user vector and the item vector.
4. RMSE and MAE as accuracy measures.
5. Regularization (lambda): why it prevents overfitting.
6. Precision at K, Recall at K, NDCG at K for ranking quality.

## 6. Setup

1. Use Python 3.10 or newer in a Jupyter notebook or Colab.
2. Libraries: numpy, scipy, pandas, matplotlib, scikit learn, and one reference library such as scikit surprise (for FunkSVD) or implicit (for ALS).
3. Create a folder in the team repo called `recommender` with these files:
   1. `data.py`: loading and building the sparse matrix.
   2. `baselines.py`: simple baseline predictors.
   3. `funksvd.py`: your FunkSVD.
   4. `als.py`: your ALS.
   5. `metrics.py`: all metric functions.
   6. `experiments.py`: runs and saves results.
   7. `notebooks`: figures and exploration.

## 7. Step by Step Tasks

### Phase 1: Load the data and build the matrix

1. Load the ratings file. Rename columns to `user_id`, `item_id`, `rating`, `timestamp`.
2. Map user IDs and item IDs to integers starting at 0 (use a dictionary or pandas categorical codes).
3. Build a sparse CSR matrix with rows as users, columns as items, values as ratings.
4. Print the number of users, items, ratings and the density (ratings divided by users times items).
5. Use the shared split file when available. Until then, shuffle with seed 42 and split 80, 10, 10.

Done when: you can print the shape, density and the first few ratings of a user from the sparse matrix.

### Phase 2: Baselines (an hour of work, very useful for the report)

1. Global mean predictor: predict the mean rating for everything.
2. Item mean predictor: predict each item's average (use the global mean for unseen items).
3. User plus item bias baseline: global mean plus user bias plus item bias, with simple shrinkage.
4. Compute RMSE and MAE on the validation split for each.

Done when: you have a three row table of baseline errors. Any real model must beat these.

### Phase 3: FunkSVD from scratch

Model: predicted rating for user u and item i equals mu plus b_u plus b_i plus the dot product of p_u and q_i.

Training uses stochastic gradient descent over the observed training ratings:
1. Compute the error: the rating minus the prediction.
2. Update the biases and both factor vectors by a small step in the direction that reduces the error, with a pull toward zero controlled by lambda.
3. Repeat for every rating, for several epochs, shuffling each epoch.

Details to decide:
1. Number of factors k: start with 20.
2. Learning rate: start with 0.005.
3. Lambda (regularization): start with 0.02.
4. Epochs: 20, and stop when the validation error stops improving.
5. Initialize factors with small random numbers (normal, standard deviation 0.1).

Do these checks:
1. Print training RMSE and validation RMSE every epoch.
2. Compare your final validation RMSE with the reference library's SVD on the same split. They should be close.

Done when: your FunkSVD beats the bias baseline and is within a small margin of the library.

### Phase 4: ALS from scratch (your main math contribution)

Idea: fix all item vectors and solve for every user vector exactly, then fix all user vectors and solve for every item vector, and repeat.

Derivation to write in your report:
1. Start with the regularized squared error objective over observed ratings only.
2. Fix the item vectors. For one user, set the gradient with respect to that user's vector equal to zero.
3. This gives the normal equations: a k by k matrix times the user vector equals a right hand side vector.
4. Show the k by k matrix is symmetric and positive definite when lambda is greater than zero, so a unique solution exists.
5. The cost per user is about the number of that user's ratings times k squared, plus k cubed for the solve.
6. The item step is exactly symmetric.
7. Explain why the objective never increases: each half step is an exact minimization over one block.

Implementation outline:
1. Keep the matrix in CSR format (for user steps) and CSC format (for item steps).
2. For each user, gather the vectors of the items that user rated, build the k by k matrix and the right hand side, and call `np.linalg.solve`.
3. Use mean centering and biases, or include them as extra factors.
4. Scale lambda by the number of ratings per user (the weighted lambda variant) for better stability.
5. After every half step, compute and store the objective value and the validation RMSE.

Done when: the objective curve decreases at every step, validation RMSE is close to a library ALS, and you have the convergence plot.

### Phase 5: Rank, lambda and spectrum studies

1. Sweep k over 5, 10, 20, 50, 100 and plot validation RMSE for both models.
2. Sweep lambda over several values on a log scale and plot validation RMSE.
3. Compute the top 50 singular values of the mean centered sparse matrix with `scipy.sparse.linalg.svds` and plot them. This shows how much structure is in the data and helps justify your choice of k.
4. Prepare for the attack study: when Ashil's attacked data arrives, plot the top singular values for clean versus attacked data. Coordinated fake profiles can add extra large singular values.
5. Write a short note explaining why truncated SVD of a fully observed matrix is optimal (Eckart Young theorem), and why that does not directly apply when most entries are missing. This is why FunkSVD and ALS minimize error over observed entries only.

### Phase 6: Weighted versions (the key link to the detector)

Idea: each rating gets a weight w between 0 and 1 from Amrutha's `weights.csv`. A suspected fake rating has a small weight and so influences the factors less.

1. Weighted objective: each squared error is multiplied by the rating's weight.
2. Weighted ALS: the matrix and the right hand side for each user are built with the weights included. One line of the matrix product changes.
3. Weighted FunkSVD: multiply each rating's gradient step by its weight.
4. Until real weights exist, test with random weights, with all weights equal to one (must reproduce the unweighted results exactly), and with weights of zero on a few chosen rows (must behave like deleting those rows).
5. Support three modes: no filter (all ones), hard filter (weights of 0 or 1 using a threshold on `p_fake`), and soft weights (w equals one minus p_fake, raised to a power gamma).

Done when: the three tests above pass.

### Phase 7: The metrics module

Write these as small, tested functions:
1. RMSE and MAE.
2. Precision at K, Recall at K, NDCG at K, MAP at K. Treat ratings of 4 and 5 as relevant.
3. Coverage: the share of the catalog that appears in any user's recommendations.
4. Novelty: the average of negative log popularity of recommended items.
5. Diversity: the average difference between recommended items (start with a simple version using item popularity buckets, improve later with embeddings from Alphin).
6. Ranking protocol: for each test user, rank all items they have not rated in training and take the top K. Use a random sample of 1,000 users to keep it fast.

Test each function on a tiny hand made example where you know the answer.

### Phase 8: Experiments

1. **Clean data comparison:** SVD versus ALS on the clean data, with 5 seeds each. Save all metrics to a CSV.
2. **Attack comparison (after Ashil's data arrives):** for each attack setting, train and evaluate under four conditions: no filter, hard filter, soft weights, and the oracle (remove exactly the injected users). Ashil's evaluation harness computes prediction shift and hit ratio. Use his function when it is ready.
3. **Cost on clean data:** run the filters on data with no attack, to show they do not hurt.
4. Save results as `results_clean.csv` and `results_attack.csv`, one row per run.

### Phase 9: The interface you hand to the team

Keep the interface simple and stable, so others can plug in without reading your code:
1. `train(ratings_df, weights=None, params=None)` returns a trained model object.
2. `model.predict(user_id, item_id)` returns a predicted rating.
3. `model.recommend(user_id, n=10, exclude_seen=True)` returns a ranked list of item IDs.

Provide one example notebook that shows how to call these three functions.

### Phase 10: What you write for the report

1. A derivation section for FunkSVD and ALS, including the weighted version.
2. The convergence plot and the singular value spectrum plot.
3. The rank sweep and lambda sweep plots.
4. The results tables for clean and attacked data, with confidence intervals across seeds.
5. A discussion of SVD versus ALS and of hard versus soft filtering, saying honestly where each wins.

## 8. Order of Work and Checkpoints

1. Days 1 to 2: concepts, setup, Phase 1, Phase 2.
2. Days 3 to 5: Phase 3 (FunkSVD).
3. Days 6 to 9: Phase 4 (ALS) and the derivation.
4. Days 10 to 11: Phase 5 (sweeps and spectrum).
5. Days 12 to 13: Phase 6 (weights) and Phase 7 (metrics).
6. When the other files arrive: Phase 8 (experiments).
7. Final: Phase 9 and Phase 10.

These are rough. If something takes twice as long, tell the team early.

## 9. How to Stay Aligned with the Team

1. Use the agreed column names and seed 42.
2. Never wait for others. Use random weights and the sample ratings until the real files arrive.
3. Do not build detectors (Alphin, Amrutha) or the attack generator (Ashil).
4. Ask Ashil for his evaluation function signature early, so your outputs match it.
5. Ask Amrutha how `weights.csv` is produced, so you know what the weight values mean.
6. Post a short update to the group every time a module is done.

## 10. Common Pitfalls

1. Filling missing ratings with zero (breaks the models).
2. Computing the item average with the test ratings included (data leakage).
3. Tuning on the test split. Tune only on validation.
4. Comparing models trained on different splits.
5. Forgetting to exclude already seen items when recommending.
6. Very slow loops over users. Use sparse indexing, sampling for development, and vectorized NumPy.
7. Reporting a single run. Always report mean and spread across 5 seeds.

## 11. Definition of Done

1. FunkSVD and ALS both work, match a library within a small margin, and have convergence plots.
2. Weighted versions pass the three sanity tests.
3. The metrics module is tested.
4. Results tables exist for clean data and for attacked data.
5. The derivation and plots are written up for the report.
6. The three function interface is documented with an example notebook.

## 12. Using LLMs Well

Rules for using an LLM as a helper:
1. Paste this file as context at the start of a chat.
2. Ask for explanations first, then code. Make sure you can explain every line you use.
3. Always test LLM code on a tiny example with a known answer.
4. Ask the LLM to write unit tests along with the code.
5. Never paste real project data that is private. The ratings data is public, but check before sharing anything else.
6. When something fails, give the full error message and the smallest piece of code that causes it.

### Prompts You Can Copy

**Understand the concepts**
```
Explain matrix factorization for recommender systems to a third year CS student. Use a tiny example with 3 users, 4 items and 2 latent factors, and show how the predicted rating is computed.
```

```
Explain the difference between FunkSVD trained with SGD and ALS. When is each one better? Keep it to one page and give a small numeric example.
```

**Setup and data**
```
I have a CSV with columns userId, productId, Rating, Timestamp (about 7.8 million rows). Write pandas code to keep users and items with at least 5 ratings, map IDs to integers, and build a scipy CSR sparse matrix. Do not fill missing values with zero.
```

**Baselines**
```
Write a bias baseline recommender in NumPy: global mean plus user bias plus item bias with shrinkage. Include a function to compute RMSE on a validation set and a small test with made up data.
```

**FunkSVD**
```
Write FunkSVD from scratch in NumPy with biases, SGD, L2 regularization, and per epoch printing of train and validation RMSE. Use arrays of user indices, item indices and ratings. Add a unit test on a tiny matrix where the model should reach near zero training error.
```

**ALS**
```
Derive the ALS update for one user in regularized matrix factorization using observed ratings only. Show the normal equations and explain why the matrix is symmetric positive definite for lambda greater than zero.
```

```
Write ALS from scratch in NumPy using scipy sparse CSR and CSC matrices. Solve each user and item with np.linalg.solve. Track the objective value after each half step and plot it. Use the weighted lambda regularization (lambda times the number of ratings).
```

**Weights**
```
Modify my ALS code so each rating has a weight w between 0 and 1. Show the modified normal equations and code. Add three tests: all weights one equals the unweighted result, weight zero on some rows equals deleting those rows, and random weights run without error.
```

**Spectrum**
```
Show how to compute the top 50 singular values of a mean centered sparse ratings matrix with scipy.sparse.linalg.svds and plot them on a log scale. Explain what the shape of the plot says about a good choice of rank k.
```

**Metrics**
```
Write NumPy functions for Precision at K, Recall at K, NDCG at K, MAP at K, coverage, and novelty for a recommender. Treat ratings of 4 and above as relevant. Include tiny hand checked examples as tests.
```

**Experiments and plots**
```
Write a script that runs a grid over rank k and lambda, trains my model for 5 seeds, and saves mean and standard deviation of validation RMSE to a CSV. Then plot RMSE versus k with error bars using matplotlib.
```

**Debugging**
```
My ALS validation RMSE goes up after a few iterations. Here is my code and the output of the first 10 iterations. List the three most likely causes, in order, and a quick test for each.
```

```
My code is too slow on 500,000 ratings because of a Python loop over users. Show how to speed it up with sparse indexing and vectorization, without changing the results.
```

**Report writing**
```
Here is my derivation and my result tables. Rewrite the explanation in clear academic English for a project report. Keep every equation and number the same, and point out anything that looks incorrect.
```

**Checking your own understanding**
```
Quiz me with 10 questions on matrix factorization, ALS and SVD for recommender systems. Ask one at a time, wait for my answer, and then tell me what is right or wrong.
```

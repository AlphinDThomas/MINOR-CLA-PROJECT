# Recommendation Pipeline - Fake Rating Attack

## Overview

This module prepares Amazon product rating data and simulates a fake-rating
attack against a recommendation pipeline.

The workflow is:

1. Explore the ratings dataset.
2. Filter users with at least 5 ratings.
3. Filter products with at least 5 ratings.
4. Create a 50,000-row sample.
5. Create an 80/10/10 train-validation-test split.
6. Generate fake users and fake ratings.
7. Measure the effect of the attack on a target product.
8. Provide scoring utilities for RMSE and target-score change.

## Dataset Statistics

Original dataset:

- Total ratings: 7,824,482
- Unique users: 4,201,696
- Unique products: 476,002
- Unique timestamps: 5,489

Rating distribution:

- 1 star: 901,765
- 2 stars: 456,322
- 3 stars: 633,073
- 4 stars: 1,485,781
- 5 stars: 4,347,541

## Cleaned Dataset

Filtering criteria:

- Users with at least 5 ratings
- Products with at least 5 ratings

Results:

- Users satisfying minimum rating requirement: 254,064
- Products satisfying minimum rating requirement: 157,783
- Cleaned ratings: 2,109,869

## Train / Validation / Test Split

The cleaned dataset is split using random seed 42:

- Training: 80%
- Validation: 10%
- Test: 10%

Files:

- `ratings_sample.csv`
- `ratings_clean.csv`
- `split.csv`

## Fake Rating Attack

Target product:

`B0045DMA42`

Original target ratings:

- 200

Fake users:

- 50

Each fake user:

- Gives the target product a 5-star rating.
- Rates 20 additional products with ratings near the target product's average.

Total fake ratings:

- 1,050

Fake ratings added to target product:

- 50

Percentage of target-product ratings that are fake after attack:

- 20%

## Attack Results

Target product average before attack:

- 3.390

Target product average after attack:

- 3.712

Change:

- 0.322

The chart showing this effect is stored in:

`attack_impact.png`

## Scoring

`scoring.py` provides:

- RMSE calculation
- Target score change calculation
- Combined prediction evaluation

## Important

The fake-review detector and SVD/ALS recommendation models are not implemented here.
This module produces the prepared data and simulated attack required by those
components.

## Files

- `ratings_sample.csv` - approximately 50,000 cleaned ratings
- `ratings_clean.csv` - filtered ratings dataset
- `split.csv` - train/validation/test assignment
- `attacked_ratings.csv` - cleaned data plus simulated fake ratings
- `attack_impact.png` - before/after attack visualization
- `scoring.py` - evaluation functions
- `requirements.txt` - Python dependencies

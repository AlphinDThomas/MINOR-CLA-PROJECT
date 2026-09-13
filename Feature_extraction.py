import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# 1. Download the VADER lexicon (runs once)
print("Downloading VADER lexicon...")
nltk.download('vader_lexicon', quiet=True)

# 2. Load the dataset
file_path = "Dataset/fake reviews dataset.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
df['target'] = df['label'].map({'OR': 0, 'CG': 1})
df = df.dropna(subset=['text_', 'rating', 'target']).reset_index(drop=True)

# 3. Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 4. Extract Sentiment Scores
print("Calculating sentiment scores...")
# VADER provides a 'compound' score from -1 (extremely negative) to +1 (extremely positive)
df['sentiment_score'] = df['text_'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# 5. Calculate Word Count (Another useful metric for catching bots)
df['word_count'] = df['text_'].apply(lambda x: len(str(x).split()))

# 6. Flag Sentiment-Rating Mismatches
# A mismatch is flagged if the rating is high (>=4) but sentiment is negative (< -0.1),
# OR if the rating is low (<=2) but sentiment is positive (> 0.5)
print("Flagging sentiment-rating mismatches...")
conditions = [
    (df['rating'] >= 4) & (df['sentiment_score'] < -0.1),
    (df['rating'] <= 2) & (df['sentiment_score'] > 0.5)
]
df['sentiment_mismatch'] = np.select(conditions, [1, 1], default=0)

# 7. Save the processed features for the next step
output_file = "reviews_with_features.csv"
df.to_csv(output_file, index=False)

# 8. Verify the extraction
print(f"\nSaved features to {output_file}")
print("\nSample of engineered features:")
print(df[['rating', 'sentiment_score', 'sentiment_mismatch', 'word_count', 'target']].head())
print(f"\nTotal mismatches found: {df['sentiment_mismatch'].sum()}")
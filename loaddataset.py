import pandas as pd

file_path = "Dataset/fake reviews dataset.csv"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(file_path)

# Map labels: 0 = Genuine (OR), 1 = Fake (CG)
df['target'] = df['label'].map({'OR': 0, 'CG': 1})

# Clean missing values if any
df = df.dropna(subset=['text_', 'rating', 'target']).reset_index(drop=True)

print(f"Total reviews loaded: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nColumns available:", df.columns.tolist())
print("\nFirst sample review:")
print(f"Rating: {df.loc[0, 'rating']} | Label: {df.loc[0, 'label']} ({df.loc[0, 'target']})")
print(f"Text: {df.loc[0, 'text_']}")
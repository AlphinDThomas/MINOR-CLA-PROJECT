import os
import pandas as pd

# 1. Download and extract the dataset using the Kaggle CLI
# The dataset identifier is 'mexwell/fake-reviews-dataset'
print("Downloading dataset from Kaggle...")
os.system("kaggle datasets download -d mexwell/fake-reviews-dataset --unzip")

# 2. Load the downloaded CSV into a Pandas DataFrame
# The unzipped file is named 'fake reviews dataset.csv'
file_path = "fake reviews dataset.csv"
df = pd.read_csv(file_path)

# 3. Standardize the labels for binary classification 
# The dataset uses 'OR' (Original Reviews) and 'CG' (Computer-Generated)
# We map these to 0 (Genuine) and 1 (Fake) to establish the ground truth
df['label'] = df['label'].map({'OR': 0, 'CG': 1})

# 4. Verify the import and check the balanced split
print("\nDataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['label'].value_counts())

# Display the text and standard labels for the first few rows
print("\nSample Data:")
print(df[['text_', 'label']].head())
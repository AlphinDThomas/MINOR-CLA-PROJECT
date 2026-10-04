import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import numpy as np

# 1. Load the dataset
file_path = "reviews_with_features.csv"
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

# 2. Setup Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computation device: {device}")

# 3. Load DistilBERT
print("Initializing DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()

# 4. Extract embeddings
batch_size = 32
embeddings_list = []

print("Extracting deep text embeddings (This might take a few minutes)...")
with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):
        # Convert to string to avoid errors on empty text fields
        batch_texts = df['text_'].iloc[i:i+batch_size].astype(str).tolist()
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # Extract the [CLS] token representation for each review
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.append(cls_embeddings)

# 5. Combine and Save
all_embeddings = np.vstack(embeddings_list)
print(f"\nGenerated embeddings shape: {all_embeddings.shape}")

# Convert to DataFrame columns
embedding_cols = [f"emb_{i}" for i in range(all_embeddings.shape[1])]
df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_cols)

# Merge back with the original dataset
final_df = pd.concat([df, df_embeddings], axis=1)

# Save as a pickle file (CSV would be massive and slow to read)
output_file = "final_features_dataset.pkl"
final_df.to_pickle(output_file)
print(f"Saved complete dataset to {output_file}")
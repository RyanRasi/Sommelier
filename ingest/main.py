import zipfile
import pandas as pd
import numpy as np

from load import preview_data
from clean_and_preprocess import drop_nulls, fill_remaining_nulls, build_rich_text
from clean_and_preprocess import validate as preprocess_validate
from generate_embeddings import load_cleaned_data, embedding_model
from generate_embeddings import validation as embeddings_validate
from faiss_index import load_embeddings, build_index, normalise_vectors, build_and_populate_index, save_index, test_sample
from faiss_index import validation as faiss_validate

import os
os.environ['FAISS_OPT_LEVEL'] = ''
data_dir = "wine-data"

# Extract zip
with zipfile.ZipFile("wine-reviews.zip", "r") as zip_ref:
    zip_ref.extractall(data_dir)

# 1. Preview data for any irregularities
df = pd.read_csv(f"{data_dir}/winemag-data-130k-v2.csv", index_col=0)
preview_data(df)

# 2. Clean and preprocess data
df = drop_nulls(df)
df = fill_remaining_nulls(df)
df['text'] = df.apply(build_rich_text, axis=1)

# Preview
print("\nSample text field:")
print(df['text'].iloc[0])

# Save cleaned dataset
filename = "wines_clean"
df.to_csv(f"{data_dir}/{filename}.csv", index=False)
print(f"\n✅ Saved {filename}.csv with {len(df)} rows.")

preprocess_validate(f"{data_dir}/{filename}")

# 3. Generate Embeddings
df = load_cleaned_data(f"{data_dir}/{filename}")
'''
embeddings = embedding_model(df, "all-MiniLM-L6-v2")

# Save embedding model
filename = "wine_embeddings"
np.save(f"{data_dir}/{filename}.npy", embeddings)
print(f"✅ Saved {filename}.npy")

# Verify saved embedding model
embeddings_check = np.load(f"{data_dir}/{filename}.npy")
print(f"Reloaded shape: {embeddings_check.shape}")

embeddings_validate(filename)

'''

# 4. Build FAISS index

df, embeddings = load_embeddings(data_dir)
embeddings, dimensions = build_index(embeddings)
embeddings = normalise_vectors(embeddings)

index = build_and_populate_index(embeddings, dimensions)

filename = f"{data_dir}/wine_faiss"
save_index(filename, index)

test_sample(df, embeddings, filename)
faiss_validate(df, embeddings, filename)

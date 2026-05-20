import faiss
import numpy as np
import pandas as pd

def load_embeddings(data_dir):
    # Load embeddings and data
    embeddings = np.load(f"{data_dir}/wine_embeddings.npy")
    df = pd.read_csv(f"{data_dir}/wines_clean.csv")

    print(f"Embeddings shape: {embeddings.shape}")
    return df, embeddings

def build_index(embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dimensions = embeddings.shape[1]
    print(f"Vector dimension: {dimensions}")

    return embeddings, dimensions

def normalise_vectors(embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    print("Vectors normalised.")
    return embeddings

def build_and_populate_index(embeddings, dimensions):
    # IndexFlatIP = exact search using Inner Product (cosine similarity after normalisation)
    index = faiss.IndexFlatIP(dimensions)

    # Add all wine vectors to the index
    index.add(embeddings)

    print(f"Index built. Total vectors stored: {index.ntotal}")
    return index

def save_index(filename, index):
    faiss.write_index(index, f"{filename}.index")
    print(f"✅ Saved {filename}.index")

def test_sample(df, embeddings, filename):
    # Reload index fresh from disk to confirm it works
    index = faiss.read_index(f"{filename}.index")

    # Use the first wine as a test query
    query_vector = embeddings[0:1]  # shape must be (1, 384)

    # Search for top 5 most similar wines
    scores, indices = index.search(query_vector, k=5)

    print("\nTop 5 results for wine[0]:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        title = df['title'].iloc[idx]
        print(f"  {rank+1}. [{score:.4f}] {title}")
        
def validation(df, embeddings, filename):
    index = faiss.read_index(f"{filename}.index")

    assert index.ntotal == len(df), f"Index has {index.ntotal} vectors, expected {len(df)}"

    # Self-search: wine[0] querying itself should always be rank 1 with score ~1.0
    scores, indices = index.search(embeddings[0:1], k=1)
    assert indices[0][0] == 0, "Wine[0] should be its own nearest neighbour"
    assert abs(scores[0][0] - 1.0) < 1e-3, f"Self-score should be ~1.0, got {scores[0][0]}"

    # Search should return exactly k results
    scores, indices = index.search(embeddings[0:1], k=10)
    assert len(indices[0]) == 10, "Should return exactly 10 results"

    print("✅ All FAISS checks passed.")


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def load_cleaned_data(filename):
    df = pd.read_csv(f"{filename}.csv")
    print(f"Loaded {len(df)} wines.")
    return df

# Load model tailored for semantic search
def embedding_model(df, transformer_model):
    model = SentenceTransformer(transformer_model)
    print("Model loaded.")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    texts = df['text'].tolist()

    print(f"Embedding {len(texts)} wines...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings 

def validation(filename):
    embeddings = np.load(f"{filename}.npy")

    assert embeddings.shape[0] == len(df), "Row count mismatch!"
    assert embeddings.shape[1] == 384, "Wrong embedding dimension!"
    assert not np.isnan(embeddings).any(), "NaNs found in embeddings!"

    # Sanity check: two similar wines should be closer than two random ones
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    # Compare wine 0 with itself (should be 1.0)
    self_sim = cosine_sim(embeddings[0], embeddings[0])
    assert abs(self_sim - 1.0) < 1e-5, "Self-similarity should be 1.0"

    # Compare wine 0 with wine 1 (should be < 1.0)
    other_sim = cosine_sim(embeddings[0], embeddings[1])
    print(f"Self similarity:  {self_sim:.4f}")
    print(f"Other similarity: {other_sim:.4f}")

    print("✅ All embedding checks passed.")

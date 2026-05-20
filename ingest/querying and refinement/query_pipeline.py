import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_artifacts():
    # Load all artifacts from previous steps
    df = pd.read_csv("wines_clean.csv")
    embeddings = np.load("wine_embeddings.npy").astype(np.float32)
    index = faiss.read_index("wine_faiss.index")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"✅ Loaded {len(df)} wines, index has {index.ntotal} vectors.")
    return df, embeddings, index, model

def search_wines(query: str, top_k: int = 10) -> pd.DataFrame:
    """
    Embed a natural language query and retrieve top_k matching wines.
    Returns a DataFrame of results with similarity scores.
    """
    # Embed the query
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    
    # Normalise (must match how we normalised the index vectors)
    faiss.normalize_L2(query_vector)
    
    # Search the index
    scores, indices = index.search(query_vector, k=top_k)
    
    # Build results dataframe
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    results = results.reset_index(drop=True)
    
    return results[['title', 'variety', 'country', 'price', 'points', 
                     'description', 'similarity_score']]

def validation():
    # Test 1: Returns correct shape
    results = search_wines("fruity red wine", top_k=10)
    assert len(results) == 10, "Should return exactly 10 results"
    assert 'similarity_score' in results.columns, "Missing similarity score"
    assert 'description' in results.columns, "Missing description"

    # Test 2: Scores are in valid range
    assert results['similarity_score'].max() <= 1.01, "Scores should be <= 1.0"
    assert results['similarity_score'].min() >= 0.0, "Scores should be >= 0.0"

    # Test 3: Results are sorted best-first
    scores = results['similarity_score'].tolist()
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score"

    # Test 4: Different queries return different results
    r1 = search_wines("sweet dessert wine", top_k=5)
    r2 = search_wines("dry tannic red", top_k=5)
    assert r1['title'].tolist() != r2['title'].tolist(), "Different queries should return different wines"

    print("✅ All query pipeline checks passed.")
df, embeddings, index, model = load_artifacts()

# Test 1: Food pairing
print("=" * 60)
print("QUERY: 'wine for medium rare steak'")
print("=" * 60)
results = search_wines("wine for medium rare steak", top_k=5)
for _, row in results.iterrows():
    print(f"\n🍷 {row['title']}")
    print(f"   {row['variety']} | {row['country']} | ${row['price']}")
    print(f"   Score: {row['similarity_score']:.4f} | Points: {row['points']}")
    print(f"   {row['description'][:120]}...")

# Test 2: Flavour profile
print("\n" + "=" * 60)
print("QUERY: 'fruity wine'")
print("=" * 60)
results = search_wines("fruity wine", top_k=5)
for _, row in results.iterrows():
    print(f"\n🍷 {row['title']}")
    print(f"   {row['variety']} | {row['country']} | ${row['price']}")
    print(f"   Score: {row['similarity_score']:.4f} | Points: {row['points']}")
    print(f"   {row['description'][:120]}...")

# Test 3: Structured query
print("\n" + "=" * 60)
print("QUERY: 'dry wine from France'")
print("=" * 60)
results = search_wines("dry wine from France", top_k=5)
for _, row in results.iterrows():
    print(f"\n🍷 {row['title']}")
    print(f"   {row['variety']} | {row['country']} | ${row['price']}")
    print(f"   Score: {row['similarity_score']:.4f} | Points: {row['points']}")
    print(f"   {row['description'][:120]}...")

validation()
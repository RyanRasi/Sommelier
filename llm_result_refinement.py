import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import re

df = pd.read_csv("wines_clean.csv")
embeddings = np.load("wine_embeddings.npy").astype(np.float32)
index = faiss.read_index("wine_faiss.index")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
print("✅ All assets loaded.")

def search_wines(query: str, df_pool: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Embed query and search within a filtered pool of wines.
    """
    if len(df_pool) == 0:
        return pd.DataFrame()

    # Embed and normalise query
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vector)

    # If pool is the full df, search the full index
    if len(df_pool) == len(df):
        scores, indices = index.search(query_vector, k=top_k)
        results = df.iloc[indices[0]].copy()
        results['similarity_score'] = scores[0]

    else:
        # Build a temporary index for the filtered subset
        pool_indices = df_pool.index.tolist()
        pool_embeddings = embeddings[pool_indices].copy()
        faiss.normalize_L2(pool_embeddings)

        temp_index = faiss.IndexFlatIP(pool_embeddings.shape[1])
        temp_index.add(pool_embeddings)

        k = min(top_k, len(df_pool))
        scores, local_indices = temp_index.search(query_vector, k=k)

        results = df_pool.iloc[local_indices[0]].copy()
        results['similarity_score'] = scores[0]

    return results[['title', 'variety', 'country', 'price', 
                     'points', 'description', 'similarity_score']].reset_index(drop=True)

REFINEMENT_SYSTEM_PROMPT = """
You are a knowledgeable and friendly sommelier.
You will be given a user's wine query and a list of candidate wines.
Your job is to select the best 3 wines from the list and explain each one.

Return ONLY a JSON array with exactly 3 objects. Each object must have:
{
  "rank": number (1 = best match),
  "title": string (exact title from the list),
  "why": string (2-3 sentences explaining why this wine matches the query),
  "food_pairing": string (one sentence suggestion),
  "serving_tip": string (one short tip e.g. temperature, decanting)
}

Rules:
- Select wines that best match the user's intent
- Prefer higher points when quality seems important
- Consider value for money when price is a concern
- Be warm and conversational, not clinical
- Return ONLY the JSON array, no other text
- You MUST copy wine titles EXACTLY as they appear in the candidate list
- Do NOT invent, paraphrase, or modify any wine title under any circumstances, the wine title sometimes has the country afterwards in parentheses, keep that, e.g. Three Pines 2012 Black Granite Red (California) is one title, make sure you don't forget the country when that occurs.
"""

def refine_results(query: str, candidates: pd.DataFrame) -> list:
    """
    Use LLM to select and explain the best 3 wines from candidates.
    """
    if candidates.empty:
        return []

    # Format candidates for the LLM
    wines_text = ""
    for i, row in candidates.iterrows():
        price_str = f"${row['price']:.0f}" if row['price'] > 0 else "Price N/A"
        wines_text += f"""
Wine {i+1}:
  Title: {row['title']}
  Variety: {row['variety']} | Country: {row['country']}
  Price: {price_str} | Points: {row['points']}
  Description: {row['description'][:200]}
"""

    prompt = f"""
User query: "{query}"

Candidate wines:
{wines_text}

Select the best 3 wines and explain why they match the query.
"""

    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        raw = response.choices[0].message.content.strip()
        
        # Strip markdown fences if present
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        
        # Try direct parse
        try:
            recommendations = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON array
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                recommendations = json.loads(match.group())
            else:
                print(f"⚠️ Could not parse refinement response.")
                return []

        # Filter out any hallucinated titles
        valid_titles = candidates['title'].tolist()
        #print("Orginal recommendations")
        #for rec in recommendations:
        #    print(rec.get('title')) 
        recommendations = [
            rec for rec in recommendations 
            if rec.get('title') in valid_titles
        ]

        if len(recommendations) == 0:
            print("⚠️ All recommendations were hallucinated. Falling back to top 3 by score.")
            fallback = candidates.head(3).reset_index(drop=True)
            recommendations = [
                {
                    "rank": i + 1,
                    "title": row['title'],
                    "why": row['description'][:200],
                    "food_pairing": "Pairs well with a variety of dishes.",
                    "serving_tip": "Serve at appropriate temperature for the variety."
                }
                for i, row in fallback.iterrows()
            ]

        return recommendations

    except Exception as e:
        print(f"⚠️ Refinement error: {e}")
        return []
    
def display_recommendations(recommendations: list):
    """
    Pretty print the final recommendations.
    """
    if not recommendations:
        print("No recommendations could be generated.")
        return

    print(f"\n{'🍷 ' * 20}")
    print("YOUR WINE RECOMMENDATIONS")
    print(f"{'🍷 ' * 20}\n")

    for rec in recommendations:
        print(f"#{rec['rank']} — {rec['title']}")
        print(f"   💬 {rec['why']}")
        print(f"   🍽️  Food pairing: {rec['food_pairing']}")
        print(f"   🌡️  Tip: {rec['serving_tip']}")
        print()

def test():
    test_queries = [
        "wine for medium rare steak",
        "fruity wine under $20",
        "dry wine from France"
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"QUERY: '{query}'")
        print(f"{'='*55}")

        # Get candidates from semantic search
        candidates = search_wines(query, df, top_k=15)
        print(f"Retrieved {len(candidates)} candidates from FAISS.")

        # Refine with LLM
        recommendations = refine_results(query, candidates)
        display_recommendations(recommendations)

def validation():
    # Test 1: Returns a list
    candidates = search_wines("bold red wine", df, top_k=15)
    recs = refine_results("bold red wine", candidates)

    titles_in_df = df['title'].tolist()
    for rec in recs:
        assert rec['title'] in titles_in_df, f"Still hallucinating: {rec['title']}"

    print("✅ Hallucination fix confirmed.")

    assert isinstance(recs, list), "Should return a list"
    assert len(recs) > 0, "Should return at least one recommendation"

    # Test 2: Each recommendation has required fields
    required_fields = {'rank', 'title', 'why', 'food_pairing', 'serving_tip'}
    for rec in recs:
        missing = required_fields - set(rec.keys())
        assert not missing, f"Recommendation missing fields: {missing}"

    # Test 3: Empty candidates handled gracefully
    empty_recs = refine_results("test", pd.DataFrame())
    assert empty_recs == [], "Empty candidates should return empty list"

    # Test 4: Titles come from actual wines
    titles_in_df = df['title'].tolist()
    for rec in recs:
        assert rec['title'] in titles_in_df, f"Title not found in dataset: {rec['title']}"

    print("✅ All refinement checks passed.")

if __name__ == "__main__":
    test()
    validation()
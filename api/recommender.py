# recommender.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import re
import time

# ──────────────────────────────────────────────────────────
# ASSETS
# ──────────────────────────────────────────────────────────

print("Loading assets...")
df         = pd.read_csv("../wines_clean.csv")
embeddings = np.load("../wine_embeddings.npy").astype(np.float32)
index      = faiss.read_index("../wine_faiss.index")
model      = SentenceTransformer('all-MiniLM-L6-v2')
client     = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
print(f"✅ Ready. {len(df)} wines loaded.")

# ──────────────────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────────────────

FILTER_PROMPT = """
You are a wine filter extraction assistant.
Extract filters from the user's wine query and return ONLY a JSON object.
Do NOT include any explanation, markdown, or text outside the JSON.

Return exactly this structure (use null if not mentioned):
{
  "country": string or null,
  "variety": string or null,
  "min_price": number or null,
  "max_price": number or null,
  "min_points": number or null,
  "flavour_keywords": [],
  "cleaned_query": string
}

Rules:
- "cleaned_query": rewrite the query as a rich wine description for semantic search
- "flavour_keywords": extract words like fruity, dry, tannic, sweet, oaky, bold, light, floral, earthy, spicy
- For food pairings like "steak", expand to wine characteristics that pair well
- "under $X" means max_price only. "over $X" means min_price only. NEVER set both for a single price constraint.
- Keep country names in English
- Return ONLY the JSON object, nothing else

Example input: dry red wine from France under $30
Example output: {"country": "France", "variety": null, "min_price": null, "max_price": 30, "min_points": null, "flavour_keywords": ["dry", "bold"], "cleaned_query": "dry bold tannic red wine France structured Bordeaux"}
"""

REFINEMENT_PROMPT = """
You are a knowledgeable and friendly sommelier.
You will be given a user's wine query and a list of candidate wines.
Your job is to select the best 3 wines from the list and explain each one.

Return ONLY a JSON array with exactly 3 objects. Each object must have:
{
  "rank": number (1 = best match),
  "title": string (EXACT title from the candidate list, copied character for character),
  "why": string (2-3 sentences explaining why this wine matches the query),
  "food_pairing": string (one sentence suggestion),
  "serving_tip": string (one short tip e.g. temperature, decanting)
}

Rules:
- You MUST copy wine titles EXACTLY as they appear in the candidate list
- Do NOT invent, paraphrase, or modify any wine title under any circumstances
- Select wines that best match the user's intent
- Prefer higher points when quality seems important
- Consider value for money when price is a concern
- Be warm and conversational, not clinical
- Return ONLY the JSON array, no other text
"""

# ──────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ──────────────────────────────────────────────────────────

VAGUE_QUERY_ENRICHMENT = {
    "surprise me": "unique unusual rare interesting complex wine",
    "recommend something": "highly rated complex wine excellent value",
    "good everyday wine": "approachable easy drinking everyday table wine value",
    "something nice": "elegant well-balanced quality wine"
}

MIN_POOL_SIZE = 50


def enrich_query(query: str) -> str:
    lower = query.lower().strip()
    for key, enriched in VAGUE_QUERY_ENRICHMENT.items():
        if key in lower:
            return enriched
    return query


def extract_filters(query: str) -> dict:
    fallback = {
        "country": None, "variety": None,
        "min_price": None, "max_price": None,
        "min_points": None, "flavour_keywords": [],
        "cleaned_query": query
    }
    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": FILTER_PROMPT},
                {"role": "user", "content": f"Extract filters from: {query}"}
            ],
            temperature=0,
            max_tokens=300
        )
        raw = re.sub(r'```(?:json)?', '',
                     response.choices[0].message.content.strip()).strip()
        try:
            filters = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            filters = json.loads(match.group()) if match else None

        if filters is None:
            return fallback

        if filters.get("min_price") and filters.get("max_price"):
            if filters["min_price"] == filters["max_price"]:
                filters["min_price"] = None

        return {**fallback, **filters}

    except Exception as e:
        print(f"⚠️ Filter extraction error: {e}")
        return fallback


def apply_filters(df_in: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df_in.copy()

    if filters.get("country"):
        filtered = filtered[
            filtered['country'].str.lower() == filters['country'].lower()
        ]
    if filters.get("variety"):
        filtered = filtered[
            filtered['variety'].str.lower().str.contains(
                filters['variety'].lower(), na=False
            )
        ]
    if filters.get("max_price"):
        filtered = filtered[
            (filtered['price'] <= filters['max_price']) &
            (filtered['price'] > 0)
        ]
    if filters.get("min_price"):
        filtered = filtered[filtered['price'] >= filters['min_price']]
    if filters.get("min_points"):
        filtered = filtered[filtered['points'] >= filters['min_points']]

    return filtered.reset_index(drop=True)


def search_wines(query: str, df_pool: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    if len(df_pool) == 0:
        return pd.DataFrame()

    query_vector = model.encode(
        [query], convert_to_numpy=True
    ).astype(np.float32)
    faiss.normalize_L2(query_vector)

    if len(df_pool) == len(df):
        scores, indices = index.search(query_vector, k=top_k)
        results = df.iloc[indices[0]].copy()
        results['similarity_score'] = scores[0]
    else:
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
                     'points', 'description',
                     'similarity_score']].reset_index(drop=True)


def refine_results(query: str, candidates: pd.DataFrame) -> list:
    if candidates.empty:
        return []

    wines_text = ""
    for i, row in candidates.iterrows():
        price_str = f"${row['price']:.0f}" if row['price'] > 0 else "N/A"
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

Select the best 3 wines. Copy titles EXACTLY as shown above.
"""

    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": REFINEMENT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        raw = re.sub(r'```(?:json)?', '',
                     response.choices[0].message.content.strip()).strip()
        try:
            recommendations = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            recommendations = json.loads(match.group()) if match else []

        valid_titles = candidates['title'].tolist()
        recommendations = [
            r for r in recommendations
            if r.get('title') in valid_titles
        ]

        if len(recommendations) == 0:
            print("⚠️ Hallucination detected — using fallback.")
            fallback = candidates.head(3).reset_index(drop=True)
            recommendations = [
                {
                    "rank": i + 1,
                    "title": row['title'],
                    "why": row['description'][:200],
                    "food_pairing": "Pairs well with a variety of dishes.",
                    "serving_tip": "Serve at the appropriate temperature."
                }
                for i, row in fallback.iterrows()
            ]

        return recommendations

    except Exception as e:
        print(f"⚠️ Refinement error: {e}")
        return []


def recommend(query: str) -> list:
    """
    Full end-to-end wine recommendation pipeline.
    This is the only function api.py needs to call.
    """
    print(f"\n⏳ Processing: '{query}'")
    start = time.time()

    filters = extract_filters(query)
    print(f"  🔍 Filters: country={filters['country']}, "
          f"max_price={filters['max_price']}, "
          f"variety={filters['variety']}")

    filtered_df = apply_filters(df, filters)
    print(f"  📊 Pool size after filters: {len(filtered_df)} wines")

    if len(filtered_df) < MIN_POOL_SIZE:
        print(f"  ⚠️  Pool too small — relaxing filters.")
        relaxed = {**filters, "country": None, "variety": None}
        filtered_df = apply_filters(df, relaxed)
        print(f"  📊 Relaxed pool size: {len(filtered_df)} wines")

    search_query = enrich_query(filters['cleaned_query'])
    candidates = search_wines(search_query, filtered_df, top_k=15)
    print(f"  🎯 Top {len(candidates)} candidates retrieved.")

    recommendations = refine_results(query, candidates)
    print(f"  ✅ Done in {time.time() - start:.1f}s")

    return recommendations
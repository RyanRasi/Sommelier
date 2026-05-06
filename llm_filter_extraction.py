from openai import OpenAI
import json
import re
import pandas as pd

# Point to local Ollama instead of OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required by library, ignored by Ollama
)

# Quick connection test
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    max_tokens=10
)
print(f"✅ Ollama connected. Model says: {response.choices[0].message.content.strip()}")

def test():

    test_queries = [
        "dry wine from France",
        "wine for medium rare steak",
        "cheap fruity wine under $15",
        "recommend something interesting"
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"QUERY: '{query}'")
        print(f"{'='*55}")
        filters = extract_filters(query)
        print("Extracted filters:")
        for k, v in filters.items():
            if v:
                print(f"  {k}: {v}")
        filtered_df = apply_filters(df, filters)
        print(f"Cleaned query: '{filters['cleaned_query']}'")

def validation():
    # Test 1: France filter
    filters = extract_filters("red wine from France")
    assert filters['country'] == 'France', f"Expected France, got {filters['country']}"

    # Test 2: Price filter
    filters = extract_filters("wine under $20")
    assert filters['max_price'] is not None, "Should have max_price"
    assert filters['max_price'] <= 20, f"Expected <=20, got {filters['max_price']}"

    # Test 3: cleaned_query always present
    filters = extract_filters("recommend something interesting")
    assert filters['cleaned_query'], "cleaned_query should never be empty"

    # Test 4: apply_filters works correctly
    france_filters = {**{k: None for k in ['variety','min_price','max_price',
                    'min_points']}, "country": "France", 
                    "flavour_keywords": [], "cleaned_query": "wine"}
    filtered = apply_filters(df, france_filters)
    assert len(filtered) < len(df)
    assert filtered['country'].unique().tolist() == ['France']

    print("✅ All filter extraction checks passed.")

df = pd.read_csv("wines_clean.csv")

FILTER_SYSTEM_PROMPT = """
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
- Keep country names in English (e.g. "France" not "français")
- Return ONLY the JSON object, nothing else

Example input: dry red wine from France under $30
Example output: {"country": "France", "variety": null, "min_price": null, "max_price": 30, "min_points": null, "flavour_keywords": ["dry", "bold"], "cleaned_query": "dry bold tannic red wine France structured Bordeaux"}
"""

def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract JSON from LLM output even if it contains
    extra text, markdown fences, or minor formatting issues.
    """
    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?', '', text).strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object within the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: return empty filters
    print(f"⚠️ Could not parse JSON from: {text[:200]}")
    return None


def extract_filters(query: str) -> dict:
    """
    Use local Ollama model to extract structured filters from a query.
    """
    fallback = {
        "country": None,
        "variety": None,
        "min_price": None,
        "max_price": None,
        "min_points": None,
        "flavour_keywords": [],
        "cleaned_query": query
    }

    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract filters from: {query}"}
            ],
            temperature=0,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()
        filters = extract_json_from_text(raw)
        
        if filters is None:
            print("⚠️ Using fallback filters.")
            return fallback

        # After parsing filters, add this sanity check:
        if filters.get("min_price") and filters.get("max_price"):
            if filters["min_price"] == filters["max_price"]:
                # "under $15" got parsed as both — clear min_price
                filters["min_price"] = None

        # Merge with fallback to ensure all keys exist
        return {**fallback, **filters}

    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return fallback
    
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply hard filters to dataframe before semantic search.
    """
    filtered = df.copy()

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

    print(f"  Filters applied: {len(df)} → {len(filtered)} wines")
    return filtered.reset_index(drop=True)

test()
validation()
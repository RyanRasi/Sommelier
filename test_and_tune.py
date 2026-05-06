
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import re
import time

from llm_filter_extraction import extract_filters, apply_filters
from llm_result_refinement import refine_results, search_wines, display_recommendations

df = pd.read_csv("wines_clean.csv")


def score_results(results: dict):
    """
    Interactively score each query result 1-3.
    1 = poor, 2 = acceptable, 3 = excellent
    """
    scores = {}
    print("\n🧪 MANUAL SCORING (1=poor, 2=ok, 3=great)")
    print("Press Enter to skip a query.\n")

    for query, recs in results.items():
        if not recs:
            scores[query] = 0
            continue

        print(f"Query: '{query}'")
        print(f"Top result: {recs[0]['title']}")
        print(f"Why: {recs[0]['why'][:100]}...")
        
        score = input("Score (1-3): ").strip()
        scores[query] = int(score) if score in ('1','2','3') else None
        print()

    valid = [s for s in scores.values() if s is not None]
    avg = sum(valid) / len(valid) if valid else 0
    print(f"\n📊 Average score: {avg:.1f} / 3.0")
    return scores

VAGUE_QUERY_ENRICHMENT = {
    "surprise me": "unique unusual rare interesting complex wine",
    "recommend something": "highly rated complex wine excellent value",
    "good everyday wine": "approachable easy drinking everyday table wine value",
    "something nice": "elegant well-balanced quality wine"
}

def enrich_query(query: str) -> str:
    """Enrich vague queries before embedding."""
    lower = query.lower().strip()
    for key, enriched in VAGUE_QUERY_ENRICHMENT.items():
        if key in lower:
            print(f"  💡 Enriched query: '{enriched}'")
            return enriched
    return query

def recommend(query: str) -> list:
    """
    Full end-to-end wine recommendation pipeline.

    Steps:
      1. Extract filters from query using LLM
      2. Apply hard filters to narrow the dataset
      3. Semantic search on filtered pool using FAISS
      4. LLM refines and explains top results

    Returns list of recommendation dicts.
    """
    print(f"\n⏳ Processing: '{query}'")
    start = time.time()

    # Step 1: Extract filters
    filters = extract_filters(query)
    search_query = enrich_query(filters['cleaned_query'])

    print(f"  🔍 Filters: country={filters['country']}, "
          f"max_price={filters['max_price']}, "
          f"variety={filters['variety']}")

    # Step 2: Apply hard filters
    filtered_df = apply_filters(df, filters)
    print(f"  📊 Pool size after filters: {len(filtered_df)} wines")

    if len(filtered_df) == 0:
        print("  ⚠️  No wines matched filters — relaxing to full dataset.")
        filtered_df = df

    # In recommend(), replace the zero-check with:
    MIN_POOL_SIZE = 50

    if len(filtered_df) < MIN_POOL_SIZE:
        print(f"  ⚠️  Pool too small ({len(filtered_df)}) — relaxing filters.")
        # Relax: keep price filter but drop country/variety
        relaxed = {**filters, "country": None, "variety": None}
        filtered_df = apply_filters(df, relaxed)
        print(f"  📊 Relaxed pool size: {len(filtered_df)} wines")

    # Step 3: Semantic search
    candidates = search_wines(search_query, filtered_df, top_k=15)
    print(f"  🎯 Top {len(candidates)} candidates retrieved.")

    # Step 4: LLM refinement
    recommendations = refine_results(query, candidates)
    print(f"  ✅ Done in {time.time() - start:.1f}s")

    return recommendations

TEST_QUERIES = [
    # Category 1: Food pairings
    "wine for medium rare steak",
    "wine to go with salmon",
    "something for a cheese board",

    # Category 2: Flavour profiles
    "fruity and light wine",
    "dry and earthy wine",
    "something sweet for dessert",

    # Category 3: Structured filters
    "dry wine from France",
    "Italian red under $25",
    "highly rated wine over 95 points",

    # Category 4: Vague / open ended
    "recommend something interesting",
    "surprise me",
    "good everyday wine",
]

results = {}

for query in TEST_QUERIES:
    print(f"\n{'='*55}")
    recs = recommend(query)
    display_recommendations(recs)
    results[query] = recs

# Summary
print("\n" + "="*55)
print("TEST SUITE SUMMARY")
print("="*55)
for query, recs in results.items():
    status = "✅" if len(recs) > 0 else "❌"
    print(f"{status} [{len(recs)} recs] '{query}'")

# These are the four original goal queries from the project spec
GOAL_QUERIES = [
    "wine for medium rare steak",
    "fruity wine",
    "dry wine from France",
    "recommend something interesting"
]

print("🎯 FINAL SMOKE TEST — Original Project Goals\n")
all_passed = True

for query in GOAL_QUERIES:
    recs = recommend(query)
    passed = len(recs) > 0 and all(
        all(f in r for f in ['rank','title','why','food_pairing','serving_tip'])
        for r in recs
    )
    status = "✅" if passed else "❌"
    print(f"{status} '{query}' → {len(recs)} recommendations")
    if passed:
        print(f"   Top pick: {recs[0]['title']}")
    else:
        all_passed = False

print(f"\n{'✅ ALL GOAL QUERIES PASSED' if all_passed else '❌ SOME QUERIES FAILED'}")
# 🍷 Sommelier — AI Wine Recommendation System

A natural language wine recommendation system powered by semantic search and a local LLM. Describe what you want in plain English and get back curated, explained recommendations from a dataset of 120,000 wines.

---

## Overview

Type something like *"bold red for a medium-rare steak"* or *"dry French wine under $30"* or *""Give me a wine from Italy that pairs well with seafood* and the system:

1. Uses an LLM to extract structured filters (country, price, variety, flavour profile)
2. Applies those filters to narrow a 120k wine dataset
3. Runs semantic search using sentence embeddings + FAISS vector search
4. Passes the top candidates back to the LLM to select and explain the best 3

Everything runs locally — no OpenAI API key required.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  LLM Filter         │  Extracts: country, max_price, variety,
│  Extraction         │  flavour keywords, cleaned search query
│  (Ollama / llama3.2)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Hard Filter        │  Narrows 120k wines to matching subset
│  (pandas)           │  by country, price, variety, points
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Semantic Search    │  Embeds cleaned query → searches FAISS
│  (FAISS + sentence- │  index → returns top 15 candidates
│   transformers)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Refinement     │  Selects best 3, writes explanations,
│  (Ollama / llama3.2)│  food pairings, and serving tips
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI            │  REST API served on localhost:8000
│  REST API           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  React Frontend     │  Search UI with animated cards
│  (Vite)             │  served on localhost:5173
└─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dimensions) |
| Vector search | `FAISS` — exact cosine similarity search |
| LLM | `Ollama` running `llama3.2` locally |
| API | `FastAPI` + `uvicorn` |
| Frontend | `React` + `Vite` |
| Dataset | Wine Reviews — 120k wines from Kaggle |

---

## Prerequisites

Install these on the new machine before anything else:

| Tool | Version | Download |
|---|---|---|
| Python | 3.9+ | https://python.org |
| Node.js | 18+ | https://nodejs.org (LTS) |
| Ollama | Latest | https://ollama.com/download |

Verify:

```bash
python --version   # 3.9+
node --version     # v18+
ollama --version
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2. Set up Python environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Download the dataset

The CSV is not committed to the repo (~50MB). Run this once:

```bash
python download_data.py
```

This downloads `winemag-data-130k-v2.csv` into your project root.

---

### 4. Generate embeddings and build the FAISS index

These files are too large for GitHub and must be regenerated locally.

> ⚠️ The embedding step takes **10 - 20 minutes** on first run. This is normal.

```bash
python preprocess.py            # → wines_clean.csv
python generate_embeddings.py   # → wine_embeddings.npy  (~180MB, takes 3–8 min)
python build_index.py           # → wine_faiss.index
```

Or if your pipeline is a single file:

```bash
python sommelier.py
```

---

### 5. Pull the Ollama model

```bash
ollama pull llama3.2
```

Downloads `llama3.2` (~2GB) to your local Ollama store. Only needed once.

Verify it's available:

```bash
ollama list
# should show: llama3.2
```

---

### 6. Install React dependencies

```bash
cd sommelier-ui
npm install
cd ..
```

---

## Running the App

You need two terminals running simultaneously.

**Terminal 1 — Python API:**

```bash
uvicorn api:app --reload --port 8000
```

Expected output:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: 🍷 Sommelier API starting up...
INFO: ✅ 119895 wines loaded and ready.
```

**Terminal 2 — React frontend:**

```bash
cd sommelier-ui
npm run dev
```

Expected output:
```
VITE v5.x.x  ready in 300ms
➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

---

## Project Structure

```
.
├── api.py                    # FastAPI REST API (HTTP layer only)
├── recommender.py            # Full recommendation pipeline
├── sommelier.py              # Original pipeline (standalone / CLI)
├── download_data.py          # One-time dataset download script
├── requirements.txt          # Python dependencies
│
├── sommelier-ui/             # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx           # Main UI component
│   │   └── main.jsx          # React entry point
│   ├── package.json
│   └── vite.config.js
│
│   # Generated files — not committed to git:
├── winemag-data-130k-v2.csv  # Raw dataset
├── wines_clean.csv           # Cleaned dataset
├── wine_embeddings.npy       # Sentence embeddings (~180MB)
└── wine_faiss.index          # FAISS vector index
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /`

Returns API status and basic info.

**Response:**
```json
{
  "name": "AI Sommelier API",
  "status": "running",
  "wines_loaded": 119895,
  "usage": "POST /recommend with {query: string}",
  "docs": "/docs"
}
```

---

### `GET /health`

Health check — use this to verify the API is ready before making requests.

**Response `200`:**
```json
{
  "status": "healthy",
  "wines_loaded": 119895,
  "model": "llama3.2",
  "version": "1.0.0"
}
```

**Response `503`** — if the wine database failed to load.

---

### `POST /recommend`

Returns 3 wine recommendations for a natural language query.

**Request body:**
```json
{
  "query": "dry wine from France"
}
```

**Response `200`:**
```json
{
  "query": "dry wine from France",
  "recommendations": [
    {
      "rank": 1,
      "title": "Château Pichon Baron 2012 (Pauillac)",
      "why": "This classic Bordeaux is the definition of dry and structured...",
      "food_pairing": "Perfect with lamb chops or duck confit.",
      "serving_tip": "Decant for 30 minutes before serving."
    }
  ],
  "count": 3,
  "elapsed_seconds": 4.21
}
```

**Error responses:**

| Status | Reason |
|---|---|
| `400` | Empty query or query over 500 characters |
| `404` | No recommendations found — try rephrasing |
| `500` | Pipeline error |

**Error shape:**
```json
{
  "error": "Query cannot be empty.",
  "status_code": 400
}
```

---

### Interactive docs

FastAPI generates a full Swagger UI automatically:

```
http://localhost:8000/docs
```

---

## Example Queries

```
wine for medium rare steak
fruity and light wine
dry wine from France
Italian red under $25
highly rated wine over 95 points
something sweet for dessert
bold earthy Barolo
recommend something interesting
cheap everyday white wine
wine to pair with salmon
```

---

## License

MIT
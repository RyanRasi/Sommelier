from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import logging

# Import everything from the existing sommelier pipeline
from recommender import recommend, df

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# LIFESPAN — runs on startup and shutdown
# ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🍷 Sommelier API starting up...")
    logger.info(f"✅ {len(df)} wines loaded and ready.")
    yield
    # Shutdown
    logger.info("🍷 Sommelier API shutting down.")

# ──────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sommelier API",
    description="Natural language wine recommendations powered by embeddings and LLM.",
    version="1.0.0",
    lifespan=lifespan
)

# ──────────────────────────────────────────────────────────
# CORS — allows React frontend to call this API
# ──────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    "http://localhost:3000",    # React default (Create React App)
    "http://localhost:5173",    # React default (Vite)
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLERS
# ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception anywhere in the app.
    Logs the full error server-side, returns a clean message to client.
    """
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred.",
            "detail": "The server encountered a problem. Please try again."
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Standardises all HTTP errors into a consistent JSON shape.
    """
    logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# ──────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "dry wine from France"
            }
        }


class WineRecommendation(BaseModel):
    rank: int
    title: str
    why: str
    food_pairing: str
    serving_tip: str


class RecommendResponse(BaseModel):
    query: str
    recommendations: list[WineRecommendation]
    count: int
    elapsed_seconds: float

class HealthResponse(BaseModel):
    status: str
    wines_loaded: int
    model: str
    version: str

# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AI Sommelier API",
        "status": "running",
        "wines_loaded": len(df),
        "usage": "POST /recommend with {query: string}",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Used by React frontend to check if API is ready before
    allowing the user to submit queries.

    Returns 200 if healthy, 503 if something is wrong.
    """
    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=503,
            detail="Wine database not loaded."
        )

    return HealthResponse(
        status="healthy",
        wines_loaded=len(df),
        model="llama3.2",
        version="1.0.0"
    )

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):
    """
    Takes a natural language wine query and returns 3 recommendations.

    Example queries:
    - "wine for medium rare steak"
    - "dry wine from France"
    - "fruity wine under $20"
    - "recommend something interesting"
    """

    query = request.query.strip()

    # Validate query
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if len(query) > 500:
        raise HTTPException(
            status_code=400,
            detail="Query too long. Please keep it under 500 characters."
        )

    logger.info(f"Query received: '{query}'")
    start = time.time()

    try:
        recs = recommend(query)
    except Exception as e:
        logger.error(f"Pipeline failed for query '{query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Recommendation pipeline failed. Please try again."
        )

    elapsed = round(time.time() - start, 2)
    logger.info(f"Query '{query}' completed in {elapsed}s — {len(recs)} results")

    # Handle empty results
    if not recs:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this query. Try rephrasing."
        )

    return RecommendResponse(
        query=request.query,
        recommendations=[WineRecommendation(**r) for r in recs],
        count=len(recs),
        elapsed_seconds=elapsed
    )
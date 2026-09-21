from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional, Dict, Any
import os

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core project modules
from data_store import load_dataset, get_indexed_documents
from bm25 import BM25Retriever
from vector_search import VectorSearchEngine
from hybrid_search import HybridSearchEngine
from groq_client import GroqClient
from evaluation import evaluate_system, BENCHMARK_QUERIES

# Global application state holder
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes search and retrieval engines once at server startup."""
    print("Initializing RAG Core Engine...")
    products = load_dataset(".")
    docs = get_indexed_documents(products)
    corpus = [d[1] for d in docs]
    doc_ids = [d[0] for d in docs]

    bm25 = BM25Retriever(corpus, doc_ids)
    vector_engine = VectorSearchEngine(corpus, doc_ids)
    hybrid_engine = HybridSearchEngine(bm25, vector_engine)

    state["products"] = products
    state["prod_map"] = {p["product_id"]: p for p in products}
    state["bm25"] = bm25
    state["vector_engine"] = vector_engine
    state["hybrid_engine"] = hybrid_engine
    state["corpus"] = corpus
    state["doc_ids"] = doc_ids

    print(f"RAG Core Engine ready with {len(products)} catalog items.")
    yield
    state.clear()

app = FastAPI(
    title="Custom Hybrid RAG API",
    description="High-performance REST API for Custom BM25, Dense Vector Search, and Groq-powered RAG.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- PYDANTIC SCHEMAS -----------------

class SearchMode(str, Enum):
    bm25 = "bm25"
    vector = "vector"
    hybrid_rrf = "hybrid_rrf"
    hybrid_score = "hybrid_score"

class SearchRequest(BaseModel):
    query: str = Field(..., example="Best laptop under ₹80k", description="The search query text.")
    mode: SearchMode = Field(default=SearchMode.hybrid_rrf, description="Retrieval strategy to use.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of results to retrieve.")
    k1: Optional[float] = Field(default=1.5, ge=0.1, le=3.0, description="BM25 term saturation parameter.")
    b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0, description="BM25 document length normalization.")
    rrf_k: Optional[int] = Field(default=60, ge=1, le=100, description="RRF smoothing constant.")
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Score fusion weight for BM25.")

class ProductResult(BaseModel):
    product_id: str
    product_name: str
    category: str
    brand: str
    price_inr: int
    rating: float
    reviews_count: int
    specs: Dict[str, Any]
    customer_review: str
    score: float
    score_type: str

class SearchResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    results: List[ProductResult]

class RAGRequest(BaseModel):
    query: str = Field(..., example="Best laptop under ₹80k", description="User question to answer.")
    mode: SearchMode = Field(default=SearchMode.hybrid_rrf, description="Retrieval strategy for context.")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of products to supply as context.")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key (starts with gsk_).")
    model: Optional[str] = Field(default="llama-3.3-70b-versatile", description="Groq LLM model name.")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt.")

class RAGResponse(BaseModel):
    query: str
    mode: str
    answer: str
    retrieved_products: List[ProductResult]

# ----------------- HELPER FUNCTIONS -----------------

def _build_product_results(raw_results: List[dict], score_label: str) -> List[ProductResult]:
    prod_map = state["prod_map"]
    results = []
    for r in raw_results:
        p = prod_map[r["id"]]
        results.append(ProductResult(
            product_id=p["product_id"],
            product_name=p["product_name"],
            category=p["category"],
            brand=p["brand"],
            price_inr=p["price_inr"],
            rating=p["rating"],
            reviews_count=p["reviews_count"],
            specs=p["specs"] if isinstance(p["specs"], dict) else {},
            customer_review=p["customer_review"],
            score=round(float(r["score"]), 5),
            score_type=score_label
        ))
    return results

# ----------------- API ENDPOINTS -----------------

@app.get("/", tags=["General"])
def root():
    """Root status and navigation."""
    return {
        "project": "Custom Hybrid RAG System",
        "status": "healthy",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "total_catalog_products": len(state.get("products", []))
    }

@app.get("/health", tags=["General"])
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "indexed_products": len(state.get("products", []))
    }

@app.get("/api/v1/products", tags=["Catalog"])
def list_products(
    category: Optional[str] = Query(None, description="Filter by category (e.g. Laptop, Smartphone, Headphones)"),
    max_price: Optional[int] = Query(None, description="Maximum price in INR"),
    limit: int = Query(20, ge=1, le=100)
):
    """Browse or filter products in the catalog."""
    products = state.get("products", [])
    filtered = products
    if category:
        filtered = [p for p in filtered if p["category"].lower() == category.lower()]
    if max_price:
        filtered = [p for p in filtered if p["price_inr"] <= max_price]
    return {
        "total_matches": len(filtered),
        "products": filtered[:limit]
    }

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
def search(req: SearchRequest):
    """
    Search product catalog using lexical (BM25), dense vector, or hybrid fusion search.
    """
    bm25: BM25Retriever = state["bm25"]
    vector_engine: VectorSearchEngine = state["vector_engine"]
    hybrid_engine: HybridSearchEngine = state["hybrid_engine"]

    if req.mode == SearchMode.bm25:
        # Dynamically apply custom k1, b if requested
        if req.k1 != bm25.k1 or req.b != bm25.b:
            bm25 = BM25Retriever(state["corpus"], state["doc_ids"], k1=req.k1, b=req.b)
        raw_results = bm25.search(req.query, top_k=req.top_k)
        score_label = "BM25 Score"
    elif req.mode == SearchMode.vector:
        raw_results = vector_engine.search(req.query, top_k=req.top_k)
        score_label = "Cosine Similarity"
    elif req.mode == SearchMode.hybrid_rrf:
        raw_results = hybrid_engine.search(req.query, top_k=req.top_k, method="rrf", k=req.rrf_k)
        score_label = "RRF Score"
    elif req.mode == SearchMode.hybrid_score:
        raw_results = hybrid_engine.search(req.query, top_k=req.top_k, method="score_fusion", alpha=req.alpha)
        score_label = "Fused Score"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported search mode: {req.mode}")

    results = _build_product_results(raw_results, score_label)
    return SearchResponse(
        query=req.query,
        mode=req.mode.value,
        total_results=len(results),
        results=results
    )

@app.post("/api/v1/rag/chat", response_model=RAGResponse, tags=["RAG Generation"])
def rag_chat(req: RAGRequest):
    """
    End-to-End RAG endpoint:
    1. Retrieves relevant product context via selected search mode.
    2. Packages context and user query.
    3. Calls Groq LLM (Llama 3.3) to generate grounded recommendations.
    """
    # 1. Retrieve products
    search_req = SearchRequest(query=req.query, mode=req.mode, top_k=req.top_k)
    search_res = search(search_req)

    # 2. Format context for LLM
    retrieved_docs_formatted = []
    for p in search_res.results:
        p_doc = (
            f"Product Name: {p.product_name} | "
            f"Category: {p.category} | "
            f"Brand: {p.brand} | "
            f"Price: INR {p.price_inr:,} | "
            f"Specs: {p.specs} | "
            f"Customer Review: {p.customer_review}"
        )
        retrieved_docs_formatted.append({
            "text": p_doc,
            "score": p.score
        })

    # 3. Call Groq
    groq_key = req.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return RAGResponse(
            query=req.query,
            mode=req.mode.value,
            answer="⚠️ Error: Groq API Key was not provided. Please provide groq_api_key in the request body or set the GROQ_API_KEY environment variable.",
            retrieved_products=search_res.results
        )

    groq_client = GroqClient(api_key=groq_key, model=req.model or "llama-3.3-70b-versatile")
    answer = groq_client.generate_answer(
        query=req.query,
        retrieved_docs=retrieved_docs_formatted,
        system_prompt=req.system_prompt
    )

    return RAGResponse(
        query=req.query,
        mode=req.mode.value,
        answer=answer,
        retrieved_products=search_res.results
    )

@app.get("/api/v1/evaluate", tags=["Evaluation"])
def evaluate(k: int = Query(3, ge=1, le=10, description="Evaluation top-k cutoff")):
    """
    Executes automated evaluation benchmarks comparing BM25, Vector Search,
    and Hybrid Search on Precision@K, Recall@K, MRR, and NDCG@K.
    """
    bm25 = state["bm25"]
    vector_engine = state["vector_engine"]
    hybrid_engine = state["hybrid_engine"]

    df_summary, df_details = evaluate_system(bm25, vector_engine, hybrid_engine, k=k)

    return {
        "k": k,
        "benchmark_queries_count": len(BENCHMARK_QUERIES),
        "summary": df_summary.to_dict(orient="records"),
        "details": df_details.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

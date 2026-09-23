"""
Benchmarks every retriever on the labelled catalog queries, with and without
cross-encoder reranking, then runs the FAISS ANN recall/latency sweep.

    python benchmark_retrieval.py
"""
import pandas as pd

from data_store import load_dataset, get_indexed_documents
from bm25 import BM25Retriever
from hybrid_search import HybridSearchEngine
from ann_index import Embedder, FaissVectorSearch, ann_recall_benchmark
from reranker import CrossEncoderReranker, RerankedSearch
from evaluation import evaluate_retrievers


def build_catalog_retrievers(corpus, ids, embedder=None, reranker=None):
    embedder = embedder or Embedder()
    reranker = reranker or CrossEncoderReranker()

    bm25 = BM25Retriever(corpus, ids)
    flat = FaissVectorSearch(corpus, ids, kind="flat", embedder=embedder)
    hnsw = FaissVectorSearch(corpus, ids, kind="hnsw", embedder=embedder)
    ivf = FaissVectorSearch(corpus, ids, kind="ivf", embedder=embedder, nlist=8, nprobe=2)
    hybrid = HybridSearchEngine(bm25, flat)

    return {
        "BM25": bm25,
        "FAISS Flat (MiniLM)": flat,
        "FAISS HNSW (MiniLM)": hnsw,
        "FAISS IVF nprobe=2 (MiniLM)": ivf,
        "Hybrid RRF (BM25 + FAISS)": hybrid,
        "BM25 + rerank": RerankedSearch(bm25, reranker, corpus),
        "FAISS Flat + rerank": RerankedSearch(flat, reranker, corpus),
        "Hybrid RRF + rerank": RerankedSearch(hybrid, reranker, corpus, method="rrf"),
    }


if __name__ == "__main__":
    docs = get_indexed_documents(load_dataset("."))
    corpus = [d[1] for d in docs]
    ids = [d[0] for d in docs]

    retrievers = build_catalog_retrievers(corpus, ids)
    for k in (3, 5):
        print(f"\nCatalog benchmark at K={k} (6 labelled queries):")
        print(evaluate_retrievers(retrievers, k=k).to_string(index=False, float_format="%.3f"))

    print("\nANN recall / latency trade-off (100k synthetic 384-d vectors):")
    print(pd.DataFrame(ann_recall_benchmark()).to_string(index=False, float_format="%.3f"))

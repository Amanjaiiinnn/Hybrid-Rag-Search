import math
import pandas as pd
import numpy as np

# Ground truth benchmarks mapping query -> list of relevant product IDs
# Based on the 100-product electronics dataset
BENCHMARK_QUERIES = {
    "Best laptop under ₹80k": [
        "ELEC089",  # TechNova Laptop Model 89 - ₹71,625, Rating 4.8
        "ELEC065",  # TechNova Laptop Model 65 - ₹57,957, Rating 4.9
        "ELEC057",  # TechNova Laptop Model 57 - ₹51,182, Rating 4.2
        "ELEC009",  # TechNova Laptop Model 9 - ₹39,780, Rating 4.8
        "ELEC073",  # TechNova Laptop Model 73 - ₹35,811, Rating 4.6
        "ELEC081"   # TechNova Laptop Model 81 - ₹39,861, Rating 4.1
    ],
    "High speed SSD for gaming with PCIe Gen4": [
        "ELEC023",  # FlashCore SSD Model 23 - PCIe Gen4, Rating 4.8
        "ELEC039",  # FlashCore SSD Model 39 - PCIe Gen4, Rating 4.9
        "ELEC047",  # FlashCore SSD Model 47 - PCIe Gen4, Rating 4.3
        "ELEC071"   # FlashCore SSD Model 71 - PCIe Gen4, Rating 4.8
    ],
    "Noise canceling headphones with good battery life": [
        "ELEC043",  # AudioMax Headphones Model 43 - ANC, Rating 4.9
        "ELEC027",  # AudioMax Headphones Model 27 - ANC, Rating 4.6
        "ELEC011",  # AudioMax Headphones Model 11 - Hi-Res Audio, Rating 4.3 (battery review)
        "ELEC051"   # AudioMax Headphones Model 51 - Bluetooth 5.3, Rating 4.3 (battery review)
    ],
    "Smartphone with Tensor G4 processor": [
        "ELEC002",  # NovaMobile Smartphone Model 2 - Tensor G4
        "ELEC010",  # NovaMobile Smartphone Model 10 - Tensor G4
        "ELEC034",  # NovaMobile Smartphone Model 34 - Tensor G4
        "ELEC042",  # NovaMobile Smartphone Model 42 - Tensor G4
        "ELEC050"   # NovaMobile Smartphone Model 50 - Tensor G4
    ],
    "Smartwatch with ECG support": [
        "ELEC005",  # FitTrack Smartwatch Model 5 - ECG
        "ELEC021",  # FitTrack Smartwatch Model 21 - ECG
        "ELEC029",  # FitTrack Smartwatch Model 29 - ECG
        "ELEC037",  # FitTrack Smartwatch Model 37 - ECG
        "ELEC061",  # FitTrack Smartwatch Model 61 - ECG
        "ELEC093"   # FitTrack Smartwatch Model 93 - ECG
    ],
    "Mesh router for large house": [
        "ELEC008",  # NetWave Router Model 8 - Mesh
        "ELEC040",  # NetWave Router Model 40 - Mesh
        "ELEC048",  # NetWave Router Model 48 - Mesh
        "ELEC072",  # NetWave Router Model 72 - Mesh
        "ELEC096"   # NetWave Router Model 96 - Mesh
    ]
}

def precision_at_k(retrieved_ids, ground_truth_ids, k):
    """
    Computes Precision@K: (number of relevant retrieved items) / k.
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = len(set(retrieved_k) & set(ground_truth_ids))
    return float(relevant_retrieved / k)

def recall_at_k(retrieved_ids, ground_truth_ids, k):
    """
    Computes Recall@K: (number of relevant retrieved items) / (total relevant items).
    """
    if len(ground_truth_ids) == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = len(set(retrieved_k) & set(ground_truth_ids))
    return float(relevant_retrieved / len(ground_truth_ids))

def reciprocal_rank(retrieved_ids, ground_truth_ids):
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant document.
    Returns 0.0 if no relevant documents are retrieved.
    """
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in ground_truth_ids:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    """
    if k <= 0 or len(ground_truth_ids) == 0:
        return 0.0
        
    retrieved_k = retrieved_ids[:k]
    
    # Calculate DCG@K
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_k):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        # Discount rank starts at 1, so math.log2(rank + 2) is used (since log2(1+1) = log2(2) = 1)
        dcg += rel / math.log2(rank + 2)
        
    # Calculate Ideal DCG@K (all relevant documents ranked at the top)
    idcg = 0.0
    num_relevant_in_top_k = min(k, len(ground_truth_ids))
    for rank in range(num_relevant_in_top_k):
        idcg += 1.0 / math.log2(rank + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def evaluate_system(bm25_retriever, vector_engine, hybrid_engine, k=5):
    """
    Runs the benchmark suite on BM25, Vector Search, and Hybrid Search.
    Returns a pandas DataFrame summarizing the metrics.
    """
    methods = ["BM25", "Vector Search", "Hybrid (RRF)", "Hybrid (Score Fusion)"]
    metrics = ["Precision@K", "Recall@K", "MRR", "NDCG@K"]
    
    # Initialize score tables
    scores = {m: {met: [] for met in metrics} for m in methods}
    detailed_results = []
    
    for query, ground_truth in BENCHMARK_QUERIES.items():
        # Retrieve ranked results from all models
        results_bm25 = bm25_retriever.search(query, top_k=20)
        results_vec = vector_engine.search(query, top_k=20)
        results_rrf = hybrid_engine.search(query, top_k=20, method="rrf")
        results_fusion = hybrid_engine.search(query, top_k=20, method="score_fusion")
        
        # Extract lists of IDs
        ids_bm25 = [r["id"] for r in results_bm25]
        ids_vec = [r["id"] for r in results_vec]
        ids_rrf = [r["id"] for r in results_rrf]
        ids_fusion = [r["id"] for r in results_fusion]
        
        candidates = {
            "BM25": ids_bm25,
            "Vector Search": ids_vec,
            "Hybrid (RRF)": ids_rrf,
            "Hybrid (Score Fusion)": ids_fusion
        }
        
        # Calculate metrics for each method
        for m, retrieved in candidates.items():
            p_k = precision_at_k(retrieved, ground_truth, k)
            r_k = recall_at_k(retrieved, ground_truth, k)
            rr = reciprocal_rank(retrieved, ground_truth)
            n_k = ndcg_at_k(retrieved, ground_truth, k)
            
            scores[m]["Precision@K"].append(p_k)
            scores[m]["Recall@K"].append(r_k)
            scores[m]["MRR"].append(rr)
            scores[m]["NDCG@K"].append(n_k)
            
            detailed_results.append({
                "Query": query,
                "Method": m,
                "Precision@K": p_k,
                "Recall@K": r_k,
                "MRR": rr,
                "NDCG@K": n_k,
                "Retrieved IDs": retrieved[:k]
            })
            
    # Calculate average scores
    summary_data = []
    for m in methods:
        summary_data.append({
            "Method": m,
            f"Avg Precision@{k}": np.mean(scores[m]["Precision@K"]),
            f"Avg Recall@{k}": np.mean(scores[m]["Recall@K"]),
            "Avg MRR": np.mean(scores[m]["MRR"]),
            f"Avg NDCG@{k}": np.mean(scores[m]["NDCG@K"])
        })
        
    return pd.DataFrame(summary_data), pd.DataFrame(detailed_results)

if __name__ == "__main__":
    # Test file
    from data_store import load_dataset, get_indexed_documents
    from bm25 import BM25Retriever
    from vector_search import VectorSearchEngine
    from hybrid_search import HybridSearchEngine
    
    products = load_dataset(".")
    docs = get_indexed_documents(products)
    corpus = [d[1] for d in docs]
    ids = [d[0] for d in docs]
    
    bm25 = BM25Retriever(corpus, ids)
    vector = VectorSearchEngine(corpus, ids)
    hybrid = HybridSearchEngine(bm25, vector)
    
    df_summary, df_details = evaluate_system(bm25, vector, hybrid, k=3)
    print("Evaluation Summary at K=3:")
    print(df_summary.to_string(index=False))

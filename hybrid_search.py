class HybridSearchEngine:
    def __init__(self, bm25_retriever, vector_engine):
        """
        Initializes the Hybrid Search Engine.
        :param bm25_retriever: Instance of BM25Retriever.
        :param vector_engine: Instance of VectorSearchEngine.
        """
        self.bm25 = bm25_retriever
        self.vector = vector_engine
        self.corpus = bm25_retriever.corpus
        self.doc_ids = bm25_retriever.doc_ids

    def search_rrf(self, query, top_k=5, k=60):
        """
        Retrieves documents using Reciprocal Rank Fusion (RRF).
        RRF_Score(d) = sum_{m in Models} 1 / (k + rank_m(d))
        """
        # Run both searches and retrieve scores for ALL documents to avoid missing ranks
        bm25_scores = self.bm25.get_scores(query)
        vector_scores = self.vector.get_scores(query)
        
        # Get ranks for all documents in BM25
        # We sort document indices by score descending
        bm25_rankings = sorted(range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True)
        # Create map of doc_id -> rank (1-indexed)
        bm25_ranks = {self.doc_ids[idx]: rank + 1 for rank, idx in enumerate(bm25_rankings)}
        
        # Get ranks for all documents in Vector Search
        vector_rankings = sorted(range(len(vector_scores)), key=lambda idx: vector_scores[idx], reverse=True)
        vector_ranks = {self.doc_ids[idx]: rank + 1 for rank, idx in enumerate(vector_rankings)}
        
        # Calculate RRF scores for all documents
        rrf_scores = {}
        for doc_id in self.doc_ids:
            r_bm25 = bm25_ranks[doc_id]
            r_vec = vector_ranks[doc_id]
            
            # RRF Formula
            score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_vec))
            rrf_scores[doc_id] = score
            
        # Format results
        ranked_docs = []
        for idx, doc_id in enumerate(self.doc_ids):
            ranked_docs.append({
                "index": idx,
                "id": doc_id,
                "score": rrf_scores[doc_id],
                "bm25_rank": bm25_ranks[doc_id],
                "vector_rank": vector_ranks[doc_id]
            })
            
        ranked_docs.sort(key=lambda x: x["score"], reverse=True)
        return ranked_docs[:top_k]

    def search_score_fusion(self, query, top_k=5, alpha=0.5):
        """
        Retrieves documents using Normalized Score Fusion.
        First, normalizes BM25 and Vector Search scores to [0, 1] across the corpus.
        Then computes: score = alpha * norm_bm25 + (1 - alpha) * norm_vector.
        """
        bm25_scores = self.bm25.get_scores(query)
        vector_scores = self.vector.get_scores(query)
        
        # Normalize BM25 scores
        min_bm25, max_bm25 = min(bm25_scores), max(bm25_scores)
        range_bm25 = max_bm25 - min_bm25
        norm_bm25 = []
        for s in bm25_scores:
            if range_bm25 > 0:
                norm_bm25.append((s - min_bm25) / range_bm25)
            else:
                norm_bm25.append(0.0)
                
        # Normalize Vector scores (Cosine similarity is already [0, 1] typically, 
        # but let's min-max normalize it across the current result set to be consistent)
        min_vec, max_vec = min(vector_scores), max(vector_scores)
        range_vec = max_vec - min_vec
        norm_vec = []
        for s in vector_scores:
            if range_vec > 0:
                norm_vec.append((s - min_vec) / range_vec)
            else:
                norm_vec.append(0.0)
                
        # Compute fused score
        fused_docs = []
        for idx in range(len(self.corpus)):
            fused_score = alpha * norm_bm25[idx] + (1.0 - alpha) * norm_vec[idx]
            fused_docs.append({
                "index": idx,
                "id": self.doc_ids[idx],
                "score": fused_score,
                "raw_bm25": bm25_scores[idx],
                "norm_bm25": norm_bm25[idx],
                "raw_vec": vector_scores[idx],
                "norm_vec": norm_vec[idx]
            })
            
        fused_docs.sort(key=lambda x: x["score"], reverse=True)
        return fused_docs[:top_k]

    def search(self, query, top_k=5, method="rrf", **kwargs):
        """Wrapper search method."""
        if method == "rrf":
            k = kwargs.get("k", 60)
            return self.search_rrf(query, top_k=top_k, k=k)
        elif method == "score_fusion":
            alpha = kwargs.get("alpha", 0.5)
            return self.search_score_fusion(query, top_k=top_k, alpha=alpha)
        else:
            raise ValueError(f"Unknown hybrid search method: {method}")

if __name__ == "__main__":
    from bm25 import BM25Retriever
    from vector_search import VectorSearchEngine
    
    corpus = [
        "TechNova Laptop Model 1 Ryzen 7 8GB RAM 256GB SSD price 95247 INR. Customer review: Good overall purchase.",
        "NovaMobile Smartphone Model 2 Tensor G4 32GB RAM price 51769 INR. Review: Excellent value.",
        "TechNova Laptop Model 9 Intel i5 12GB RAM 256GB storage price 39780 INR. Review: Great value for money."
    ]
    ids = ["ELEC001", "ELEC002", "ELEC009"]
    
    bm25 = BM25Retriever(corpus, ids)
    vector = VectorSearchEngine(corpus, ids)
    hybrid = HybridSearchEngine(bm25, vector)
    
    query = "laptop under 40000 INR"
    results_rrf = hybrid.search(query, top_k=2, method="rrf")
    results_fusion = hybrid.search(query, top_k=2, method="score_fusion", alpha=0.4)
    
    print("RRF results:")
    for r in results_rrf:
        print(f"ID: {r['id']}, RRF Score: {r['score']:.5f}, BM25 Rank: {r['bm25_rank']}, Vec Rank: {r['vector_rank']}")
        
    print("\nScore Fusion results:")
    for r in results_fusion:
        print(f"ID: {r['id']}, Fused Score: {r['score']:.4f}, Norm BM25: {r['norm_bm25']:.2f}, Norm Vec: {r['norm_vec']:.2f}")

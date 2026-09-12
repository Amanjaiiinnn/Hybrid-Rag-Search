import numpy as np
from embeddings_client import EmbeddingClient

class VectorSearchEngine:
    def __init__(self, corpus, doc_ids=None, embedding_client=None):
        """
        Initializes the Vector Search Engine.
        :param corpus: List of document strings.
        :param doc_ids: Optional list of document identifiers.
        :param embedding_client: Optional EmbeddingClient instance.
        """
        self.corpus = corpus
        self.doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(corpus))]
        self.embedding_client = embedding_client or EmbeddingClient()
        
        # Set up the TF-IDF fallback in the embedding client just in case
        self.embedding_client.setup_tfidf_fallback(self.corpus)
        
        self.doc_vectors = None
        self.dimension = None
        
        # Build the index (encode all documents)
        self.build_index()

    def build_index(self):
        """Generates embeddings for all documents in the corpus."""
        print(f"Encoding {len(self.corpus)} documents for vector search...")
        # Get embeddings (returns a list of list of floats)
        embeddings_list = self.embedding_client.get_embeddings(self.corpus)
        
        # Convert to numpy array for fast linear algebra
        self.doc_vectors = np.array(embeddings_list)
        self.dimension = self.doc_vectors.shape[1]
        print(f"Index built. Vector dimension: {self.dimension}")

    def cosine_similarity(self, vec_a, vec_b):
        """
        Computes the cosine similarity between two vectors from scratch.
        Formula: (A . B) / (||A|| * ||B||)
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))

    def get_scores(self, query):
        """
        Computes similarity scores between the query and all documents in the corpus.
        """
        # Get query embedding
        query_vector_list = self.embedding_client.get_embeddings([query])
        query_vector = np.array(query_vector_list[0])
        
        scores = []
        # Calculate cosine similarity for all documents
        # We perform this in a loop to show the math clearly, but we can also do it
        # as a vectorized numpy operation:
        # dot = np.dot(self.doc_vectors, query_vector)
        # doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
        # query_norm = np.linalg.norm(query_vector)
        # scores = dot / (doc_norms * query_norm)
        
        for idx in range(len(self.corpus)):
            doc_vec = self.doc_vectors[idx]
            sim = self.cosine_similarity(doc_vec, query_vector)
            scores.append(sim)
            
        return scores

    def search(self, query, top_k=5):
        """
        Searches the vector space for the query.
        Returns a list of dicts with product info, score, and original corpus index.
        """
        scores = self.get_scores(query)
        
        # Pair each document with its index, id, and score
        ranked_docs = []
        for idx, score in enumerate(scores):
            ranked_docs.append({
                "index": idx,
                "id": self.doc_ids[idx],
                "score": score
            })
            
        # Sort by score in descending order (higher cosine similarity is better)
        ranked_docs.sort(key=lambda x: x["score"], reverse=True)
        return ranked_docs[:top_k]

if __name__ == "__main__":
    # Test Vector Search
    corpus = [
        "TechNova Laptop Model 1 Ryzen 7 8GB RAM 256GB SSD price 95247 INR. Customer review: Good overall purchase.",
        "NovaMobile Smartphone Model 2 Tensor G4 32GB RAM price 51769 INR. Review: Excellent value.",
        "TechNova Laptop Model 9 Intel i5 12GB RAM 256GB storage price 39780 INR. Review: Great value for money."
    ]
    ids = ["ELEC001", "ELEC002", "ELEC009"]
    
    # We will use the TF-IDF fallback of embedding client by not providing HF Token and letting it fail to fallback
    engine = VectorSearchEngine(corpus, ids)
    
    query = "laptop with Ryzen processor"
    results = engine.search(query, top_k=2)
    print("Search query:", query)
    for r in results:
        print(f"ID: {r['id']}, Cosine Sim: {r['score']:.4f}, Doc: {corpus[r['index']]}")

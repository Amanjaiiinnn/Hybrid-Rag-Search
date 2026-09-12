import os
import requests
import time
import numpy as np

class EmbeddingClient:
    def __init__(self, api_token=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the Embedding client.
        :param api_token: Optional Hugging Face API token.
        :param model_name: Hugging Face model identifier for embeddings.
        """
        self.api_token = api_token or os.environ.get("HF_API_TOKEN", "")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        
        # Local model placeholder
        self.local_model = None
        self.use_local = False
        
        # Custom TF-IDF fallback setup (in case of offline + no local library)
        self.tfidf_fallback = None

    def _call_hf_api(self, texts):
        """Calls the Hugging Face Inference API to get embeddings."""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        # The inference API can be keyless for low volumes
        response = requests.post(
            self.api_url,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            embeddings = response.json()
            
            # Hugging Face sometimes returns a 3D array if the model isn't pooled.
            # For sentence-transformers, it usually returns [num_texts, embedding_dim].
            # Let's perform mean pooling if it returns token-level embeddings [num_texts, num_tokens, dim].
            embeddings_np = np.array(embeddings)
            if len(embeddings_np.shape) == 3:
                # Shape is (num_texts, num_tokens, dim). Perform mean pooling over tokens.
                embeddings_np = np.mean(embeddings_np, axis=1)
            elif len(embeddings_np.shape) == 1:
                # Single text input returning [dim], reshape to [1, dim]
                embeddings_np = embeddings_np.reshape(1, -1)
                
            return embeddings_np.tolist()
        else:
            raise Exception(f"HF API Error: Status {response.status_code}, Response: {response.text}")

    def load_local_model(self):
        """Attempts to load sentence-transformers locally."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local sentence-transformer model: {self.model_name}...")
            self.local_model = SentenceTransformer(self.model_name)
            self.use_local = True
            return True
        except ImportError:
            print("sentence-transformers library not installed. Cannot load local model.")
            return False
        except Exception as e:
            print(f"Failed to load local model: {e}")
            return False

    def setup_tfidf_fallback(self, corpus):
        """Sets up a custom TF-IDF representation as a zero-dependency fallback."""
        from bm25 import BM25Retriever
        # We reuse the BM25 class tokenization and IDF to construct sparse vectors
        print("Setting up custom TF-IDF Vectorizer fallback...")
        retriever = BM25Retriever(corpus)
        vocab = list(retriever.doc_occurrence.keys())
        vocab_index = {term: i for i, term in enumerate(vocab)}
        
        # Build document vectors
        doc_vectors = []
        for doc_tokens in retriever.tokenized_corpus:
            vec = np.zeros(len(vocab))
            # Calculate Term Frequency
            freqs = {}
            for t in doc_tokens:
                freqs[t] = freqs.get(t, 0) + 1
            # Multiply by IDF to get TF-IDF score
            for term, count in freqs.items():
                if term in vocab_index:
                    tf = count / len(doc_tokens)
                    idf = retriever.get_idf(term)
                    vec[vocab_index[term]] = tf * idf
            # Normalize vector to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            doc_vectors.append(vec)
            
        self.tfidf_fallback = {
            "retriever": retriever,
            "vocab": vocab,
            "vocab_index": vocab_index,
            "doc_vectors": np.array(doc_vectors)
        }

    def get_embeddings(self, texts):
        """
        Retrieves embeddings for a list of texts.
        Attempts HF API -> Local SentenceTransformer -> Custom TF-IDF vector fallback.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 1. Try local SentenceTransformer if already loaded
        if self.use_local and self.local_model:
            try:
                embeddings = self.local_model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                print(f"Local encoding failed: {e}. Trying API...")

        # 2. Try Hugging Face Inference API
        try:
            return self._call_hf_api(texts)
        except Exception as e:
            print(f"HF API embedding failed ({e}). Trying to load local model...")
            
            # 3. Try to load local model on the fly
            if not self.use_local:
                if self.load_local_model():
                    try:
                        embeddings = self.local_model.encode(texts, show_progress_bar=False)
                        return embeddings.tolist()
                    except Exception as e2:
                        print(f"Local encoding after on-the-fly load failed: {e2}")

            # 4. Fallback to TF-IDF if initialized
            if self.tfidf_fallback:
                print("Falling back to custom TF-IDF vector embeddings...")
                query_vectors = []
                retriever = self.tfidf_fallback["retriever"]
                vocab_index = self.tfidf_fallback["vocab_index"]
                vocab_size = len(self.tfidf_fallback["vocab"])
                
                for text in texts:
                    tokens = retriever._tokenize(text)
                    vec = np.zeros(vocab_size)
                    freqs = {}
                    for t in tokens:
                        freqs[t] = freqs.get(t, 0) + 1
                    for term, count in freqs.items():
                        if term in vocab_index:
                            tf = count / max(1, len(tokens))
                            idf = retriever.get_idf(term)
                            vec[vocab_index[term]] = tf * idf
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    query_vectors.append(vec.tolist())
                return query_vectors

            # If everything else fails, raise error
            raise RuntimeError("All embedding options failed (API failed, local model not installed, and TF-IDF fallback not set up).")

if __name__ == "__main__":
    # Test embeddings client
    corpus = [
        "TechNova Laptop Model 1 Ryzen 7 8GB RAM 256GB SSD",
        "NovaMobile Smartphone Model 2 Tensor G4 32GB RAM",
        "TechNova Laptop Model 9 Intel i5 12GB RAM 256GB storage"
    ]
    
    client = EmbeddingClient()
    client.setup_tfidf_fallback(corpus)
    
    # Test TF-IDF embedding fallback
    embeddings = client.get_embeddings(["laptop with 8GB RAM"])
    print("Embedding vector shape (TF-IDF fallback):", np.array(embeddings).shape)

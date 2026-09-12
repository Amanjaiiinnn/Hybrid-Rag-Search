import math
import re

# A basic list of English stop words to filter out noisy terms
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is',
    'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of',
    'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll',
    'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt',
    'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which',
    'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll',
    'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
}

class BM25Retriever:
    def __init__(self, corpus, doc_ids=None, k1=1.5, b=0.75):
        """
        Initializes the BM25 retriever.
        :param corpus: List of document strings.
        :param doc_ids: Optional list of document identifiers (defaults to 0..len-1).
        :param k1: Term frequency saturation tuning parameter (typically 1.2 to 2.0).
        :param b: Document length normalization tuning parameter (typically 0.75).
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(corpus))]
        
        # Preprocess and tokenize all documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.N = len(corpus)
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Calculate term frequencies per document and document frequency per term
        self.doc_freqs = []  # List of dicts: [ {term: count}, ... ]
        self.doc_occurrence = {}  # dict: {term: number of docs containing it}
        
        self._build_index()
        self._calculate_idf()

    def _tokenize(self, text):
        """
        Cleans and tokenizes text.
        Converts to lowercase, removes punctuation, splits, and filters stop words.
        """
        # Convert to lowercase and clean non-alphanumeric chars (preserving currency symbols like ₹ if needed,
        # but let's split clean words and numbers)
        # Note: we want to keep numbers since user might query "80k" or "16GB" or "8GB"
        text = text.lower()
        # Replace punctuation with spaces (keeping letters, numbers, and currency characters like ₹)
        # We can keep '₹' so the model can match price constraints
        cleaned = re.sub(r'[^\w\s₹]', ' ', text)
        tokens = cleaned.split()
        
        # Filter stop words, but keep numbers and specific query-relevant keywords
        filtered_tokens = []
        for t in tokens:
            # We want to keep numbers and tokens like "₹80k", "80k", "16gb"
            if t not in STOP_WORDS or t.isdigit() or '₹' in t:
                filtered_tokens.append(t)
        return filtered_tokens

    def _build_index(self):
        """Builds term frequency and document frequency mappings."""
        for doc_tokens in self.tokenized_corpus:
            frequencies = {}
            for token in doc_tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            
            # Record term occurrences across documents (document frequency)
            for token in frequencies.keys():
                self.doc_occurrence[token] = self.doc_occurrence.get(token, 0) + 1

    def _calculate_idf(self):
        """
        Calculates the Inverse Document Frequency (IDF) for all unique terms.
        Using the standard BM25 IDF formula with +1 to prevent negative IDFs:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        """
        self.idf = {}
        for term, n_q in self.doc_occurrence.items():
            # Standard BM25 IDF:
            numerator = self.N - n_q + 0.5
            denominator = n_q + 0.5
            # We use log(1 + num/den) to guarantee IDF is >= 0
            self.idf[term] = math.log(1.0 + (numerator / denominator))

    def get_idf(self, term):
        """Returns IDF of a term (defaulting to a small positive value if out-of-vocabulary)."""
        return self.idf.get(term, 0.0)

    def get_scores(self, query):
        """
        Calculates the BM25 scores for all documents in the corpus for the given query.
        """
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.N
        
        # If query is empty, return zeros
        if not query_tokens:
            return scores
            
        for i in range(self.N):
            doc_len = self.doc_lengths[i]
            freqs = self.doc_freqs[i]
            score = 0.0
            
            # If doc length is 0, score is 0
            if doc_len == 0 or self.avgdl == 0:
                continue
                
            for token in query_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.get_idf(token)
                    
                    # BM25 TF Normalization factor:
                    # (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    
                    score += idf * (numerator / denominator)
            scores[i] = score
            
        return scores

    def search(self, query, top_k=5):
        """
        Searches the corpus for the query.
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
            
        # Sort by score in descending order
        ranked_docs.sort(key=lambda x: x["score"], reverse=True)
        return ranked_docs[:top_k]

if __name__ == "__main__":
    # Test simple BM25
    corpus = [
        "TechNova Laptop Model 1 Ryzen 7 8GB RAM 256GB SSD price 95247 INR. Customer review: Good overall purchase.",
        "NovaMobile Smartphone Model 2 Tensor G4 32GB RAM price 51769 INR. Review: Excellent value.",
        "TechNova Laptop Model 9 Intel i5 12GB RAM 256GB storage price 39780 INR. Review: Great value for money."
    ]
    ids = ["ELEC001", "ELEC002", "ELEC009"]
    retriever = BM25Retriever(corpus, ids)
    
    # Query test
    query = "laptop under 40000 INR"
    results = retriever.search(query, top_k=2)
    print("Search query:", query)
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}, Doc: {corpus[r['index']]}")

from functools import lru_cache

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder(model_name=DEFAULT_RERANKER):
    """Loads a cross-encoder model once per process."""
    # lru_cache keys f() and f(DEFAULT_RERANKER) separately, so always pass the name through
    return _load_cross_encoder(model_name)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name):
    from sentence_transformers import CrossEncoder
    print(f"Loading cross-encoder reranker: {model_name}...")
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """
    Second-stage reranker. A bi-encoder embeds query and document separately (fast,
    indexable); a cross-encoder reads the (query, document) pair together through
    one transformer (slow, far more precise). So: retrieve a wide candidate set
    cheaply, then rerank only those candidates.
    """
    def __init__(self, model_name=DEFAULT_RERANKER):
        self.model = get_cross_encoder(model_name)

    def score(self, query, texts):
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]

    def rerank(self, query, candidates, corpus, top_k=5):
        """
        :param candidates: result dicts from any retriever (need an 'index' key into corpus).
        Returns the top_k candidates re-sorted by cross-encoder score; the first-stage
        score is kept as 'first_stage_score'.
        """
        scores = self.score(query, [corpus[c["index"]] for c in candidates])
        reranked = [
            {**c, "first_stage_score": c["score"], "score": s}
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]


class RerankedSearch:
    """Wraps any retriever with a search(query, top_k) method: fetch fetch_k, rerank to top_k."""
    def __init__(self, base_retriever, reranker, corpus, fetch_k=20, **search_kwargs):
        self.base = base_retriever
        self.reranker = reranker
        self.corpus = corpus
        self.fetch_k = fetch_k
        self.search_kwargs = search_kwargs

    def search(self, query, top_k=5):
        candidates = self.base.search(query, top_k=max(self.fetch_k, top_k), **self.search_kwargs)
        return self.reranker.rerank(query, candidates, self.corpus, top_k=top_k)


if __name__ == "__main__":
    corpus = [
        "FitTrack Smartwatch with ECG and SpO2 sensor, 7 day battery",
        "AudioMax Headphones with active noise cancelling",
        "FitTrack Smartwatch with GPS, no ECG",
    ]
    reranker = CrossEncoderReranker()
    candidates = [{"index": i, "id": str(i), "score": 0.0} for i in range(len(corpus))]
    for r in reranker.rerank("smartwatch with ECG", candidates, corpus, top_k=3):
        print(f"{r['score']:.3f}  {corpus[r['index']]}")

import time
from functools import lru_cache

import numpy as np
import faiss

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_sentence_model(model_name=DEFAULT_MODEL):
    """Loads a sentence-transformers model once per process."""
    # lru_cache keys f() and f(DEFAULT_MODEL) separately, so always pass the name through
    return _load_sentence_model(model_name)


@lru_cache(maxsize=2)
def _load_sentence_model(model_name):
    from sentence_transformers import SentenceTransformer
    print(f"Loading sentence-transformer model: {model_name}...")
    return SentenceTransformer(model_name)


class Embedder:
    """
    Bi-encoder: embeds queries and documents independently into the same space.
    Vectors are L2-normalised, so inner product == cosine similarity.
    """
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = get_sentence_model(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


def build_faiss_index(vectors, kind="flat", hnsw_m=32, ef_construction=64, ef_search=32, nlist=8, nprobe=2):
    """
    Builds one of three FAISS indexes over L2-normalised vectors (inner product = cosine):
    - flat: IndexFlatIP, exact brute-force search. The ground truth for ANN recall.
    - hnsw: IndexHNSWFlat, a layered proximity graph. M = links per node,
            efConstruction / efSearch = candidate list size at build / query time.
    - ivf:  IndexIVFFlat, k-means partitions the space into nlist cells; a query
            only scans the nprobe nearest cells. Needs a training step.
    """
    dim = vectors.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = nprobe
    else:
        raise ValueError(f"Unknown FAISS index kind: {kind}")
    index.add(vectors)
    return index


class FaissVectorSearch:
    """
    Dense retriever backed by a FAISS index. Same interface as VectorSearchEngine
    (search / get_scores), so it can be dropped into HybridSearchEngine.
    """
    def __init__(self, corpus, doc_ids=None, kind="flat", embedder=None, **index_params):
        self.corpus = corpus
        self.doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(corpus))]
        self.kind = kind
        self.embedder = embedder or Embedder()
        self.doc_vectors = self.embedder.encode(corpus)
        self.dimension = self.doc_vectors.shape[1]
        self.index = build_faiss_index(self.doc_vectors, kind=kind, **index_params)

    def _raw_search(self, query, top_k):
        query_vector = self.embedder.encode([query])
        scores, indices = self.index.search(query_vector, top_k)
        # FAISS pads with -1 when an ANN index returns fewer than top_k hits
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]

    def search(self, query, top_k=5):
        return [
            {"index": idx, "id": self.doc_ids[idx], "score": score}
            for idx, score in self._raw_search(query, top_k)
        ]

    def get_scores(self, query):
        """Scores for every document (documents an ANN index skipped get -1.0)."""
        scores = [-1.0] * len(self.corpus)
        for idx, score in self._raw_search(query, len(self.corpus)):
            scores[idx] = score
        return scores


def ann_recall_benchmark(num_vectors=100_000, dim=384, num_queries=200, k=10, seed=0):
    """
    Recall@k vs exact search and per-query latency for FAISS indexes at a scale
    where ANN actually matters. Uses clustered random unit vectors (real embeddings
    are clustered, uniform random vectors make every index look worse than it is).
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((256, dim)).astype("float32")
    data = centers[rng.integers(0, 256, num_vectors)] + 0.35 * rng.standard_normal((num_vectors, dim)).astype("float32")
    queries = centers[rng.integers(0, 256, num_queries)] + 0.35 * rng.standard_normal((num_queries, dim)).astype("float32")
    faiss.normalize_L2(data)
    faiss.normalize_L2(queries)

    def timed_search(index):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        return ids, (time.perf_counter() - start) * 1000 / num_queries

    def recall(found, truth):
        return float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)]))

    rows = []
    start = time.perf_counter()
    flat = build_faiss_index(data, "flat")
    build_ms = (time.perf_counter() - start) * 1000
    truth, latency = timed_search(flat)
    rows.append({"Index": "Flat (exact)", "Params": "-", "Build (ms)": build_ms,
                 f"Recall@{k}": 1.0, "Latency / query (ms)": latency})

    start = time.perf_counter()
    hnsw = build_faiss_index(data, "hnsw", hnsw_m=32, ef_construction=64)
    build_ms = (time.perf_counter() - start) * 1000
    for ef in (16, 32, 64, 128):
        hnsw.hnsw.efSearch = ef
        found, latency = timed_search(hnsw)
        rows.append({"Index": "HNSW", "Params": f"M=32, efSearch={ef}", "Build (ms)": build_ms,
                     f"Recall@{k}": recall(found, truth), "Latency / query (ms)": latency})

    nlist = 1024
    start = time.perf_counter()
    ivf = build_faiss_index(data, "ivf", nlist=nlist)
    build_ms = (time.perf_counter() - start) * 1000
    for nprobe in (1, 4, 16, 64):
        ivf.nprobe = nprobe
        found, latency = timed_search(ivf)
        rows.append({"Index": "IVF-Flat", "Params": f"nlist={nlist}, nprobe={nprobe}", "Build (ms)": build_ms,
                     f"Recall@{k}": recall(found, truth), "Latency / query (ms)": latency})
    return rows


if __name__ == "__main__":
    import pandas as pd
    from data_store import load_dataset, get_indexed_documents

    docs = get_indexed_documents(load_dataset("."))
    corpus = [d[1] for d in docs]
    ids = [d[0] for d in docs]

    embedder = Embedder()
    for kind in ("flat", "hnsw", "ivf"):
        engine = FaissVectorSearch(corpus, ids, kind=kind, embedder=embedder)
        print(kind, [r["id"] for r in engine.search("smartwatch with ECG", top_k=5)])

    print("\nANN recall / latency trade-off (100k synthetic vectors):")
    print(pd.DataFrame(ann_recall_benchmark()).to_string(index=False, float_format="%.3f"))

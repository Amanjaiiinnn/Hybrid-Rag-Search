# Hybrid RAG Search for an Electronics Catalog

A Retrieval-Augmented Generation (RAG) app built from scratch in Python. It searches a 100-product electronics catalog with a hand-written **BM25** retriever, a **vector search** engine and two **hybrid fusion** methods, answers shopping questions with **Llama 3 on Groq**, and measures every retriever with **Precision@K, Recall@K, MRR and NDCG@K**.

The retrieval maths (tokenisation, BM25, cosine similarity, rank fusion, metrics) is implemented directly with Python and NumPy. It doesn't use a search library or vector database.

---

## Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Evaluation Results](#evaluation-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Using the App](#using-the-app)
- [Dataset](#dataset)
- [Running the Modules on Their Own](#running-the-modules-on-their-own)
- [Customisation](#customisation)

---

## Features

- **Custom BM25** keyword retriever with adjustable `k1` and `b`.
- **Vector search** with cosine similarity and a four-step embedding fallback chain.
- **Hybrid search** using Reciprocal Rank Fusion (RRF) or min-max normalised score fusion.
- **Side-by-side results** from BM25, vector and hybrid search for the same query.
- **Grounded answers** from Groq (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`), using the results of whichever retriever you pick as context.
- **Retrieval benchmark** with six hand-labelled queries and four standard IR metrics.
- **Interactive maths walkthrough** with sliders for BM25 saturation, 2-D cosine similarity, fusion and the metrics.

---

## How It Works

### 1. Documents

`data_store.py` loads `electronics_catalog_100.json` (or the CSV if the JSON is missing) and turns every product into one text document:

```
Product Name: TechNova Laptop Model 1 | Category: Laptop | Brand: TechNova | Price: INR 95,247 |
Specifications: feature_1: Ryzen 7, ram: 8GB, storage: 256GB | Reviews: Good overall purchase with strong feature set.
```

These 100 documents are indexed by both retrievers.

### 2. BM25 retriever (`bm25.py`)

**Tokenisation:** text is lower-cased, punctuation is replaced with spaces (letters, digits and `₹` are kept), and English stop words are removed. Numbers are always kept, so queries like `16GB` or `80k` still match.

**Index:** for every document the retriever stores its term frequencies and length. For every term it stores how many documents contain it.

**IDF**, with a `+1` inside the log so it can never go negative:

```
IDF(t) = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
```

**Score** of document D for query Q:

```
score(D, Q) = Σ over t in Q:  IDF(t) · tf(t, D) · (k1 + 1) / ( tf(t, D) + k1 · (1 − b + b · |D| / avgdl) )
```

- `k1` (default **1.5**) controls how quickly repeated terms stop adding score.
- `b` (default **0.75**) controls how much long documents are penalised.
- Both can be changed from the sidebar.

### 3. Vector search (`vector_search.py`, `embeddings_client.py`)

Every document is embedded once when the app starts. A query is embedded on each search and compared with every document using cosine similarity:

```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

`EmbeddingClient.get_embeddings()` tries these backends in order:

1. A local **sentence-transformers** model, if one is already loaded.
2. The **Hugging Face Inference API** with `sentence-transformers/all-MiniLM-L6-v2`, mean-pooling the output if it comes back per token.
3. Loading **sentence-transformers** locally on the spot, if the package is installed.
4. A **TF-IDF fallback** built from the BM25 index: term frequency × BM25 IDF, normalised to unit length.

**Which backend you get today:**
- **HF API (step 2):** the `api-inference.huggingface.co` host that `embeddings_client.py` calls no longer resolves (checked September 2026).
- **sentence-transformers (steps 1 and 3):** not in `requirements.txt`.

With a plain install, vector search therefore runs on the TF-IDF fallback, with one dimension per vocabulary term (310 for this catalog), and the terminal prints the fallback messages.

To use real 384-dimensional MiniLM embeddings instead, install sentence-transformers and the client will pick it up at step 3:

```bash
pip install sentence-transformers
```

### 4. Hybrid search (`hybrid_search.py`)

Both fusion methods score all 100 documents with both retrievers first.

**Reciprocal Rank Fusion** uses only the rank each retriever gives a document, so the two score scales never need to match:

```
RRF(d) = 1 / (k + rank_BM25(d)) + 1 / (k + rank_vector(d))
```

`k` defaults to **60** (adjustable from 10 to 100).

**Normalised score fusion** min-max scales both score lists to [0, 1] and blends them:

```
score(d) = α · norm_BM25(d) + (1 − α) · norm_vector(d)
```

`α` is the BM25 weight and defaults to **0.5**.

### 5. Answer generation (`groq_client.py`)

The top results from the retriever you choose (hybrid, BM25 or vector) are formatted as numbered context documents with their scores. They are sent to Groq's chat completions API with temperature **0.2** and up to **1,024** tokens.

The system prompt tells the model to:

- answer only from the retrieved products
- compare multiple products on price, specs, rating and reviews
- respect price limits such as "under ₹80k"
- say clearly when the catalog doesn't have enough information
- answer in Markdown, with tables for specifications and prices in rupees

### 6. Evaluation (`evaluation.py`)

Six benchmark queries have hand-labelled relevant product IDs:

| Query | Relevant products |
|---|---|
| Best laptop under ₹80k | 6 |
| High speed SSD for gaming with PCIe Gen4 | 4 |
| Noise canceling headphones with good battery life | 4 |
| Smartphone with Tensor G4 processor | 5 |
| Smartwatch with ECG support | 6 |
| Mesh router for large house | 5 |

Each retriever returns its top 20 results for every query. The metrics are then averaged over the six queries (binary relevance):

| Metric | Meaning |
|---|---|
| **Precision@K** | Relevant items in the top K ÷ K |
| **Recall@K** | Relevant items in the top K ÷ all relevant items |
| **MRR** | Mean of 1 ÷ rank of the first relevant item |
| **NDCG@K** | DCG@K ÷ ideal DCG@K, with DCG = Σ rel_i / log2(i + 1) |

---

## Evaluation Results

Measured with a plain `pip install -r requirements.txt`, so vector search used the **TF-IDF fallback**. Settings were the defaults: `k1 = 1.5`, `b = 0.75`, RRF `k = 60`, `α = 0.5`.

**K = 3**

| Method | Precision@3 | Recall@3 | MRR | NDCG@3 |
|---|---|---|---|---|
| BM25 | 0.667 | 0.403 | **1.000** | 0.735 |
| Vector search (TF-IDF) | 0.667 | 0.417 | 0.792 | 0.667 |
| Hybrid (RRF) | **0.778** | **0.472** | 0.917 | **0.784** |
| Hybrid (score fusion) | 0.722 | 0.444 | 0.806 | 0.706 |

**K = 5**

| Method | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| BM25 | **0.733** | 0.731 | **1.000** | **0.791** |
| Vector search (TF-IDF) | **0.733** | **0.744** | 0.792 | 0.744 |
| Hybrid (RRF) | **0.733** | 0.731 | 0.917 | 0.782 |
| Hybrid (score fusion) | **0.733** | **0.744** | 0.806 | 0.748 |

**What the numbers show:**
- **BM25:** its first result is relevant for every query (MRR 1.0).
- **Hybrid RRF:** best on every metric except MRR at K = 3.
- **K = 5:** BM25 edges ahead on NDCG.

Because both signals are lexical in this setup, running with sentence-transformers installed will give different numbers. The **Retrieval Evaluation** tab recomputes the table live with whatever backend is active.

---

## Tech Stack

| Area | Tools |
|---|---|
| App | Streamlit |
| Retrieval maths | Python, NumPy |
| Data | Pandas, JSON / CSV |
| Embeddings | Hugging Face Inference API, optional sentence-transformers, TF-IDF fallback |
| LLM | Groq chat completions API (Llama 3.3 70B / Llama 3.1 8B) via `requests` |

---

## Project Structure

```text
Hybrid-Rag-Search/
├── app.py                          # Streamlit app: sidebar settings and the three tabs
├── bm25.py                         # BM25Retriever: tokeniser, index, IDF, scoring
├── vector_search.py                # VectorSearchEngine: document embeddings + cosine similarity
├── embeddings_client.py            # EmbeddingClient: HF API → sentence-transformers → TF-IDF
├── hybrid_search.py                # HybridSearchEngine: RRF and normalised score fusion
├── groq_client.py                  # GroqClient: grounded answer generation
├── evaluation.py                   # Benchmark queries + Precision@K, Recall@K, MRR, NDCG@K
├── data_store.py                   # Catalog loading and document construction
├── electronics_catalog_100.json    # Product catalog (primary)
├── electronics_catalog_100.csv     # Same catalog as CSV (fallback)
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A Groq API key for answer generation, free at [console.groq.com](https://console.groq.com). Search and evaluation work without it.

### Install

```bash
git clone https://github.com/Amanjaiiinnn/Hybrid-Rag-Search.git
cd Hybrid-Rag-Search
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Run the command from the project folder, because the catalog is loaded from the current directory. Streamlit opens the app at `http://localhost:8501`.

### API keys

Both keys are entered in the sidebar. The app doesn't read them from environment variables.

| Field | Needed for |
|---|---|
| **Groq API Key** | The **Generate Answer with Groq** button |
| **Hugging Face Token** (optional) | The Hugging Face embedding call (backend step 2) |

---

## Using the App

### Sidebar

| Setting | Range | Default |
|---|---|---|
| BM25 `k1` | 1.0 – 3.0 | 1.5 |
| BM25 `b` | 0.0 – 1.0 | 0.75 |
| Top K retrieved | 1 – 10 | 4 |
| Fusion method | RRF / Normalised score fusion | RRF |
| RRF `k` | 10 – 100 | 60 |
| Score fusion `α` (BM25 weight) | 0.0 – 1.0 | 0.5 |
| LLM model | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | 3.3 70B |

### 🔍 RAG Search Explorer

1. Type a query (for example *Smartwatch with ECG support*) or pick one of the benchmark queries.
2. Compare the BM25, vector and hybrid results in three columns of product cards, each with its score.
3. Choose which results to use as context: **Hybrid**, **BM25** or **Vector**.
4. Click **Generate Answer with Groq** to get a grounded answer.

### 📊 Retrieval Evaluation

Choose K (1 to 5) to see the average metrics for all four methods, with the best value in each column highlighted. You can also open any benchmark query to see its relevant product IDs, what each method retrieved, and which of those results were relevant. This tab uses the sidebar's BM25 settings and the default fusion settings (RRF `k = 60`, `α = 0.5`).

### 📚 Math & Concept Walkthrough

Formulas and small simulations for:

- **TF-IDF vs BM25:** sliders for `k1`, `b` and document length.
- **Vector embeddings and cosine similarity:** move a query vector and a document vector in 2-D.
- **Hybrid search fusion:** RRF and score fusion.
- **Evaluation metrics:** Precision@K, Recall@K, MRR and NDCG@K.

---

## Dataset

`electronics_catalog_100.json` holds 100 products across 8 categories: Laptop, Smartphone, Headphones and Tablet (13 each), and Smartwatch, Monitor, SSD and Router (12 each).

| Field | Example |
|---|---|
| `product_id` | `ELEC001` |
| `category` | `Laptop` |
| `product_name` | `TechNova Laptop Model 1` |
| `brand` | `TechNova` |
| `price_inr` | `95247` |
| `specs` | `{"feature_1": "Ryzen 7", "ram": "8GB", "storage": "256GB"}` |
| `rating` | `4.3` |
| `reviews_count` | `568` |
| `customer_review` | `Good overall purchase with strong feature set.` |

The CSV version stores `specs` as the columns `spec_feature_1`, `spec_ram` and `spec_storage`.

---

## Running the Modules on Their Own

Each module has a small self-test:

```bash
python bm25.py              # BM25 on a 3-document sample
python vector_search.py     # vector search on the same sample
python hybrid_search.py     # RRF and score fusion on the sample
python data_store.py        # loads the catalog and prints one document
python evaluation.py        # full benchmark on the catalog, K = 3
```

---

## Customisation

| What | Where |
|---|---|
| Add or change products | Edit the JSON (or CSV) using the same fields |
| Add benchmark queries | `BENCHMARK_QUERIES` in `evaluation.py` |
| Change the answer style or rules | `DEFAULT_SYSTEM_PROMPT` in `groq_client.py` |
| Change the embedding model | `model_name` in `EmbeddingClient` |
| Change stop words or tokenisation | `STOP_WORDS` and `_tokenize()` in `bm25.py` |

---

## Author

**Aman Jain** · [GitHub](https://github.com/Amanjaiiinnn)

# Hybrid RAG Search for an Electronics Catalog

A Retrieval-Augmented Generation (RAG) app built from scratch in Python. It searches a 100-product electronics catalog with a hand-written **BM25** retriever, a **vector search** engine and two **hybrid fusion** methods, answers shopping questions with **Llama 3 on Groq**, and measures every retriever with **Precision@K, Recall@K, MRR and NDCG@K**.

The retrieval maths (tokenisation, BM25, cosine similarity, rank fusion, metrics) is implemented directly with Python and NumPy. It doesn't use a search library or vector database.

The same engine is also available as a **FastAPI** REST API, and a **Docker Compose** setup runs the API and the app together, ready to deploy on a server such as **AWS EC2**.

---

## Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Evaluation Results](#evaluation-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Using the App](#using-the-app)
- [REST API](#rest-api)
- [Run with Docker](#run-with-docker)
- [Deploy on AWS EC2](#deploy-on-aws-ec2)
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
- **REST API** (FastAPI) for search, grounded answers, catalog filtering and the benchmark, with Swagger docs.
- **Docker Compose** setup that runs the API and the Streamlit app as two containers, with steps for deploying on AWS EC2.

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
| App | Streamlit, Matplotlib |
| Retrieval maths | Python, NumPy |
| Data | Pandas, JSON / CSV |
| Embeddings | Hugging Face Inference API, optional sentence-transformers, TF-IDF fallback |
| LLM | Groq chat completions API (Llama 3.3 70B / Llama 3.1 8B) via `requests` |
| API | FastAPI, Uvicorn, Pydantic |
| Deployment | Docker, Docker Compose, AWS EC2 |

---

## Project Structure

```text
Hybrid-Rag-Search/
├── app.py                          # Streamlit app: sidebar settings and the three tabs
├── api.py                          # FastAPI REST API over the same engine
├── bm25.py                         # BM25Retriever: tokeniser, index, IDF, scoring
├── vector_search.py                # VectorSearchEngine: document embeddings + cosine similarity
├── embeddings_client.py            # EmbeddingClient: HF API → sentence-transformers → TF-IDF
├── hybrid_search.py                # HybridSearchEngine: RRF and normalised score fusion
├── groq_client.py                  # GroqClient: grounded answer generation
├── evaluation.py                   # Benchmark queries + Precision@K, Recall@K, MRR, NDCG@K
├── data_store.py                   # Catalog loading and document construction
├── electronics_catalog_100.json    # Product catalog (primary)
├── electronics_catalog_100.csv     # Same catalog as CSV (fallback)
├── Dockerfile                      # Python 3.11 image shared by both services
├── docker-compose.yml              # backend (FastAPI, port 8000) + frontend (Streamlit, port 8501)
├── .env.example                    # Optional keys for Docker Compose (copy to .env)
├── .dockerignore
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A Groq API key for answer generation, free at [console.groq.com](https://console.groq.com). Search and evaluation work without it.
- Docker with Compose (optional), to run everything in containers

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

### Run the REST API (optional)

In a second terminal, also from the project folder:

```bash
uvicorn api:app --reload
```

The API starts at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`. Once it's up, the Streamlit sidebar shows **🟢 FastAPI Backend: Online**. See [REST API](#rest-api) for the endpoints.

### API keys

Both keys are entered in the sidebar. The app doesn't read them from environment variables.

| Field | Needed for |
|---|---|
| **Groq API Key** | The **Generate Answer with Groq** button |
| **Hugging Face Token** (optional) | The Hugging Face embedding call (backend step 2) |

---

## Using the App

### Sidebar

The top of the sidebar shows the **Architecture Mode**:
- **🟢 FastAPI Backend: Online:** the REST API is reachable at `FASTAPI_URL` (default `http://localhost:8000`). A link to its Swagger docs appears underneath.
- **🟡 Standalone Engine:** the API isn't reachable.

Either way, the app runs searches itself; the API is a separate way into the same engine.

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

## REST API

`api.py` serves the same retrieval engine over HTTP with FastAPI. The indexes are built once, when the server starts.

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Health check, with the number of indexed products |
| `GET` | `/api/v1/products` | Browse the catalog, filtered by `category` and `max_price` (`limit` 1–100, default 20) |
| `POST` | `/api/v1/search` | Search with BM25, vector or hybrid retrieval |
| `POST` | `/api/v1/rag/chat` | Retrieve products and generate a grounded Groq answer from them |
| `GET` | `/api/v1/evaluate` | Run the benchmark and return the metrics at cutoff `k` (1–10, default 3) |

Interactive docs are at `/docs` (Swagger UI, with **Try it out** buttons) and `/redoc`. CORS is open to all origins.

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Smartwatch with ECG support", "mode": "hybrid_rrf", "top_k": 2}'
```

| Field | Default | Allowed values |
|---|---|---|
| `query` | required | Any text |
| `mode` | `hybrid_rrf` | `bm25`, `vector`, `hybrid_rrf`, `hybrid_score` |
| `top_k` | 4 | 1 – 20 |
| `k1`, `b` | 1.5, 0.75 | BM25 parameters, applied in `bm25` mode |
| `rrf_k` | 60 | 1 – 100, for `hybrid_rrf` |
| `alpha` | 0.5 | 0 – 1, the BM25 weight for `hybrid_score` |

Each result has the product's fields plus its `score` and `score_type`. The response below is trimmed to its first result:

```json
{
  "query": "Smartwatch with ECG support",
  "mode": "hybrid_rrf",
  "total_results": 2,
  "results": [
    {
      "product_id": "ELEC029",
      "product_name": "FitTrack Smartwatch Model 29",
      "category": "Smartwatch",
      "brand": "FitTrack",
      "price_inr": 83383,
      "rating": 4.2,
      "reviews_count": 1666,
      "specs": {"feature_1": "ECG", "ram": "12GB", "storage": "256GB"},
      "customer_review": "Great performance for productivity and entertainment.",
      "score": 0.03252,
      "score_type": "RRF Score"
    }
  ]
}
```

### Grounded answers

`POST /api/v1/rag/chat` takes `query`, `mode` and `top_k` (1 – 10) like the search endpoint, plus optional `groq_api_key`, `model` (default `llama-3.3-70b-versatile`) and `system_prompt`. It returns the `answer` and the `retrieved_products` used as context.

The Groq key comes from `groq_api_key` in the request, or from the server's `GROQ_API_KEY` environment variable when the request has none. Without either, the endpoint still returns the retrieved products, with a warning in `answer`.

> **On a public server, leave `GROQ_API_KEY` unset.** Otherwise anyone who can reach the API can generate answers with your key.

---

## Run with Docker

`docker-compose.yml` runs two containers built from the same `Dockerfile`:

| Service | Container | Port | Runs |
|---|---|---|---|
| `backend` | `rag-fastapi-backend` | 8000 | `uvicorn api:app` |
| `frontend` | `rag-streamlit-frontend` | 8501 | `streamlit run app.py`, with `FASTAPI_URL=http://backend:8000` |

From the project folder:

```bash
docker compose up -d --build
```

Then open `http://localhost:8501` for the app and `http://localhost:8000/docs` for the API. Both containers have health checks and restart automatically unless you stop them.

| Command | What it does |
|---|---|
| `docker compose ps` | Shows the containers, marked `(healthy)` once ready |
| `docker compose logs -f` | Follows the logs |
| `docker compose down` | Stops and removes the containers |

**Keys (optional):** Compose reads `GROQ_API_KEY` and `HF_API_TOKEN` from a `.env` file next to `docker-compose.yml`. Copy `.env.example` to `.env` and fill in what you need. `.env` is kept out of both Git and the Docker image. The Streamlit app always takes the Groq key from its sidebar, so `GROQ_API_KEY` only matters for the REST API.

---

## Deploy on AWS EC2

The Docker setup runs unchanged on a small EC2 instance.

1. **Launch an instance** from the EC2 console: **Ubuntu Server 24.04 LTS**, a free-tier-eligible type such as `t3.micro`, a new key pair, and **20 GiB** of storage.
2. **Add inbound rules** to its security group:

   | Port | Source | For |
   |---|---|---|
   | 22 | My IP | SSH |
   | 8501 | Anywhere | Streamlit app |
   | 8000 | Anywhere | REST API and Swagger docs (optional) |

3. **Connect** with the key pair's `.pem` file:

   ```bash
   ssh -i path/to/your-key.pem ubuntu@<public-ip>
   ```

4. **On the server**, add swap (recommended on 1 GB instances), install Docker and start the app:

   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   git clone https://github.com/Amanjaiiinnn/Hybrid-Rag-Search.git
   cd Hybrid-Rag-Search
   sudo docker compose up -d --build
   ```

5. **Open** `http://<public-ip>:8501` for the app and `http://<public-ip>:8000/docs` for the API.

Docker starts on boot and brings the containers back up, so the app survives a reboot.

**Updating:** after pushing changes to GitHub, run this on the server in `Hybrid-Rag-Search`:

```bash
git pull && sudo docker compose up -d --build
```

**Costs:** stop or terminate the instance when you no longer need it. The public IP changes whenever the instance is stopped and started again; attach an Elastic IP if you need a fixed address.

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
| Change ports or container settings | `docker-compose.yml` |

---

## Author

**Aman Jain** · [GitHub](https://github.com/Amanjaiiinnn)

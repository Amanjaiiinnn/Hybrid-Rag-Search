# Custom Hybrid RAG System for Electronics Search

An interactive Retrieval-Augmented Generation (RAG) application for searching and evaluating an electronics product catalog. The project compares keyword search, semantic vector search, and hybrid retrieval, then uses a Groq-hosted Llama model to generate grounded shopping-assistant responses from retrieved product context.

## What This Project Does

- Loads a 100-item electronics catalog from JSON or CSV.
- Converts each product into a searchable document containing product name, category, brand, price, specifications, rating, review count, and customer review.
- Implements a custom BM25 keyword retriever from scratch.
- Implements vector search using embeddings from Hugging Face, a local `sentence-transformers` model when available, or a custom TF-IDF fallback.
- Combines BM25 and vector search with hybrid fusion:
  - Reciprocal Rank Fusion (RRF)
  - Normalized score fusion
- Generates final RAG answers using Groq Llama models.
- Provides retrieval evaluation using Precision@K, Recall@K, MRR, and NDCG@K.
- Includes an interactive Streamlit interface with search comparison, evaluation dashboards, and retrieval math walkthroughs.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Requests
- Groq API
- Hugging Face Inference API

## Project Structure

```text
RAG_PROJECT/
|-- app.py                         # Streamlit application
|-- bm25.py                        # Custom BM25 retriever
|-- vector_search.py               # Vector search engine with cosine similarity
|-- embeddings_client.py           # Hugging Face, local model, and TF-IDF embedding fallback
|-- hybrid_search.py               # RRF and normalized score fusion
|-- groq_client.py                 # Groq LLM client for answer generation
|-- evaluation.py                  # Retrieval benchmark metrics
|-- data_store.py                  # Dataset loading and document construction
|-- electronics_catalog_100.json   # Product dataset
|-- electronics_catalog_100.csv    # Product dataset backup/source format
`-- requirements.txt               # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd RAG_PROJECT
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## API Keys

The app can run retrieval without a Groq key, but answer generation requires one.

- Groq API key: used for LLM response generation.
- Hugging Face token: optional, used to avoid rate limits for embedding requests.

You can enter both keys in the Streamlit sidebar. The code also supports environment variables:

```bash
set GROQ_API_KEY=your_groq_key
set HF_API_TOKEN=your_huggingface_token
```

## Main Features

### RAG Search Explorer

Enter a product query such as:

- `Best laptop under 80k`
- `Smartwatch with ECG support`
- `Noise canceling headphones with good battery life`
- `High speed SSD for gaming with PCIe Gen4`

The app shows BM25, vector search, and hybrid results side by side, then lets you choose which retrieved context should be sent to the LLM.

### Retrieval Evaluation

The evaluation tab compares retrieval methods against benchmark queries with manually labeled relevant product IDs. It reports:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K

### Math Walkthrough

The app includes explanations and visual simulations for:

- TF-IDF and BM25
- Vector embeddings
- Cosine similarity
- Reciprocal Rank Fusion
- Score fusion
- Retrieval evaluation metrics

## Repository Name Suggestion

For GitHub, a clear repository name would be:

```text
hybrid-rag-electronics-search
```

Other good options:

- `custom-rag-search-engine`
- `electronics-rag-assistant`
- `hybrid-search-rag-streamlit`

## LinkedIn Project Description

Built a custom Hybrid RAG system for an electronics product catalog using Python and Streamlit. The project implements BM25 keyword retrieval, vector search with embeddings, hybrid fusion using Reciprocal Rank Fusion and score normalization, retrieval evaluation metrics, and Groq Llama-based answer generation. It includes an interactive dashboard to compare retrieval methods, evaluate Precision@K, Recall@K, MRR, and NDCG@K, and generate grounded shopping-assistant responses from retrieved product data.

## Future Improvements

- Add persistent vector storage with FAISS or ChromaDB.
- Add user authentication for saved searches.
- Add support for larger product catalogs.
- Improve query parsing for price and specification filters.
- Add automated tests for retrieval and evaluation logic.

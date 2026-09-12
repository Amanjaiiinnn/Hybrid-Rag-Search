import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import custom modules
from data_store import load_dataset, get_indexed_documents
from bm25 import BM25Retriever
from vector_search import VectorSearchEngine
from hybrid_search import HybridSearchEngine
from groq_client import GroqClient
from evaluation import evaluate_system, BENCHMARK_QUERIES

# Set page configuration
st.set_page_config(
    page_title="Custom RAG Explorer & Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism Theme and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title banner styling */
    .title-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .title-banner h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .title-banner p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Product Card Glassmorphism */
    .product-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2), 0 8px 10px -6px rgba(99, 102, 241, 0.2);
    }
    
    .product-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 8px;
    }
    
    .product-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 12px;
        display: flex;
        gap: 16px;
    }
    
    .product-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .product-price {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34d399;
    }
    
    .product-specs {
        background: rgba(0, 0, 0, 0.2);
        padding: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 12px;
        border-left: 3px solid #6366f1;
    }
    
    .product-review {
        font-style: italic;
        font-size: 0.9rem;
        color: #d1d5db;
        border-left: 2px solid rgba(255,255,255,0.2);
        padding-left: 8px;
    }

    /* Math formulas container */
    .math-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.image("https://img.icons8.com/nolan/128/search.png", width=70)
st.sidebar.title("Configuration Panel")

# 1. API Keys Section
st.sidebar.subheader("🔑 API Setup")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Paste your Groq API key (starts with gsk_)")
hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password", help="Hugging Face API token to avoid rate limits on embedding service")

# 2. Search Engine Parameters
st.sidebar.subheader("⚙️ Retrieval Parameters")
k1_val = st.sidebar.slider("BM25 k₁ (TF Saturation)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
b_val = st.sidebar.slider("BM25 b (Length Normalization)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
retrieval_k = st.sidebar.slider("Top K Retrieved", min_value=1, max_value=10, value=4)

# 3. Hybrid Search Setup
st.sidebar.subheader("🔀 Hybrid Fusion Parameters")
hybrid_method = st.sidebar.selectbox("Fusion Method", ["RRF (Reciprocal Rank Fusion)", "Normalized Score Fusion"])
rrf_k = st.sidebar.slider("RRF Smoothing (k)", min_value=10, max_value=100, value=60, step=5, disabled=(hybrid_method != "RRF (Reciprocal Rank Fusion)"))
score_alpha = st.sidebar.slider("Score Fusion weight (α = BM25 weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=(hybrid_method != "Normalized Score Fusion"))

# 4. LLM Options
st.sidebar.subheader("🤖 LLM Settings")
llm_model = st.sidebar.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

# Load Dataset & Initialize search engines
@st.cache_resource
def init_retrieval_engines(k1, b):
    try:
        products = load_dataset(".")
        docs = get_indexed_documents(products)
        corpus = [d[1] for d in docs]
        doc_ids = [d[0] for d in docs]
        
        # Initialize custom BM25
        bm25 = BM25Retriever(corpus, doc_ids, k1=k1, b=b)
        
        # Initialize Vector search (we will configure it with the user's HF Token if provided)
        # We can update the client token dynamically later
        vector_search = VectorSearchEngine(corpus, doc_ids)
        
        # Initialize Hybrid
        hybrid_search = HybridSearchEngine(bm25, vector_search)
        
        return products, bm25, vector_search, hybrid_search
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None, None, None

products, bm25, vector_search, hybrid_search = init_retrieval_engines(k1_val, b_val)

# Dynamically update Hugging Face token if provided
if vector_search and vector_search.embedding_client:
    vector_search.embedding_client.api_token = hf_token

# Header Banner
st.markdown("""
<div class="title-banner">
    <h1>Custom Hybrid RAG System</h1>
    <p>Compare custom BM25 (Lexical) vs. custom Vector (Semantic) Search, evaluate metrics, and generate answers with Groq Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs(["🔍 RAG Search Explorer", "📊 Retrieval Evaluation", "📚 Math & Concept Walkthrough"])

# Helper function to render a product dictionary nicely as a custom styled card
def render_product_card(product, score, score_label="Score"):
    specs = product.get("specs", {})
    specs_str = ", ".join([f"{k.upper()}: {v}" for k, v in specs.items()]) if isinstance(specs, dict) else str(specs)
    
    st.markdown(f"""
    <div class="product-card">
        <div class="product-title">{product['product_name']}</div>
        <div class="product-meta">
            <span class="product-badge">ID: {product['product_id']}</span>
            <span class="product-badge">Category: {product['category']}</span>
            <span class="product-badge">Brand: {product['brand']}</span>
            <span class="product-badge">Rating: ⭐ {product['rating']} ({product['reviews_count']})</span>
        </div>
        <div class="product-price">Price: ₹{product['price_inr']:,}</div>
        <div class="product-specs"><b>Specs:</b> {specs_str}</div>
        <div class="product-review"><b>Review summary:</b> "{product['customer_review']}"</div>
        <div style="font-size: 0.8rem; color: #a5b4fc; text-align: right; margin-top: 8px;">{score_label}: {score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- TAB 1: RAG SEARCH EXPLORER -----------------
with tab1:
    st.header("Search & RAG Explorer")
    
    # Query input and suggestions
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("Enter your product or review query:", placeholder="e.g. Best laptop under ₹80k, or smartwatch with ECG, or hi-res audio headphones...")
    with col2:
        suggested_query = st.selectbox("Or choose a benchmark query:", ["None"] + list(BENCHMARK_QUERIES.keys()))
        if suggested_query != "None":
            user_query = suggested_query

    # Run retrieval
    if user_query:
        st.subheader(f"Results for: '{user_query}'")
        
        # 1. Fetch search results from all systems
        bm25_res = bm25.search(user_query, top_k=retrieval_k)
        vector_res = vector_search.search(user_query, top_k=retrieval_k)
        
        if hybrid_method.startswith("RRF"):
            hybrid_res = hybrid_search.search(user_query, top_k=retrieval_k, method="rrf", k=rrf_k)
            hybrid_score_label = "RRF Score"
        else:
            hybrid_res = hybrid_search.search(user_query, top_k=retrieval_k, method="score_fusion", alpha=score_alpha)
            hybrid_score_label = "Fused Score"

        # Display side-by-side search results comparison
        st.write("### 🔍 Search Systems Comparison")
        col_bm25, col_vec, col_hyb = st.columns(3)
        
        # Load helper maps to fetch products by ID quickly
        prod_map = {p["product_id"]: p for p in products}
        
        with col_bm25:
            st.markdown("#### 📝 Custom BM25 (Keyword)")
            for r in bm25_res:
                prod = prod_map[r["id"]]
                render_product_card(prod, r["score"], "BM25 Score")
                
        with col_vec:
            st.markdown("#### 🔮 Custom Dense Vector (Semantic)")
            for r in vector_res:
                prod = prod_map[r["id"]]
                render_product_card(prod, r["score"], "Cosine Similarity")
                
        with col_hyb:
            st.markdown(f"#### 🧬 Hybrid Search ({hybrid_method.split(' ')[0]})")
            for r in hybrid_res:
                prod = prod_map[r["id"]]
                render_product_card(prod, r["score"], hybrid_score_label)

        # 2. LLM RAG Response Section
        st.write("---")
        st.write("### 🤖 Groq LLM Response Generation (RAG)")
        
        # Choose which search results to feed as context
        context_source = st.radio(
            "Select context source for the LLM:", 
            ["Hybrid Search (Recommended)", "BM25 Search", "Vector Search"], 
            horizontal=True
        )
        
        if context_source == "Hybrid Search (Recommended)":
            selected_retrieved = hybrid_res
        elif context_source == "BM25 Search":
            selected_retrieved = bm25_res
        else:
            selected_retrieved = vector_res
            
        # Format the context for the LLM call
        retrieved_docs_formatted = []
        for idx, r in enumerate(selected_retrieved):
            p = prod_map[r["id"]]
            p_doc = (
                f"Product Name: {p['product_name']} | "
                f"Category: {p['category']} | "
                f"Brand: {p['brand']} | "
                f"Price: INR {p['price_inr']:,} | "
                f"Specs: {p['specs']} | "
                f"Customer Review: {p['customer_review']}"
            )
            retrieved_docs_formatted.append({
                "text": p_doc,
                "score": r["score"]
            })
            
        # Generate Answer
        if st.button("Generate Answer with Groq", type="primary"):
            if not groq_api_key:
                st.warning("⚠️ Groq API Key is missing! Please enter your key in the sidebar.")
            else:
                with st.spinner(f"Contacting Groq ({llm_model})..."):
                    client = GroqClient(api_key=groq_api_key, model=llm_model)
                    answer = client.generate_answer(user_query, retrieved_docs_formatted)
                    
                    st.markdown("#### LLM Output:")
                    st.info(answer)

# ----------------- TAB 2: RETRIEVAL EVALUATION -----------------
with tab2:
    st.header("Search Engine Retrieval Evaluation")
    st.markdown("""
    Evaluate how effectively each retrieval engine performs. We run a set of benchmark queries (which have manually labeled "ground-truth" relevant products) and compute standard metrics:
    - **Precision@K:** Ratio of retrieved documents that are relevant.
    - **Recall@K:** Ratio of relevant documents that were successfully retrieved.
    - **MRR (Mean Reciprocal Rank):** Evaluation of where the first correct product was placed.
    - **NDCG@K (Normalized Discounted Cumulative Gain):** Measures ranking quality.
    """)
    
    eval_k = st.slider("Evaluation K", min_value=1, max_value=5, value=3)
    
    if st.button("Run System-wide Evaluation", type="primary"):
        with st.spinner("Evaluating BM25, Vector Search, and Hybrid models..."):
            df_summary, df_details = evaluate_system(bm25, vector_search, hybrid_search, k=eval_k)
            
            # Show overall results
            st.subheader("📊 Overall Averages Comparison")
            st.dataframe(df_summary.style.highlight_max(axis=0, color="#1e3a8a"))
            
            # Plot metrics comparison chart
            fig, ax = plt.subplots(figsize=(10, 5))
            methods = df_summary["Method"].tolist()
            metrics_cols = [c for c in df_summary.columns if c != "Method"]
            
            x = np.arange(len(methods))
            width = 0.18
            
            # Use a nice modern color palette
            colors = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b"]
            
            # Plot each metric group
            for idx, col in enumerate(metrics_cols):
                ax.bar(x + (idx - len(metrics_cols)/2)*width + width/2, df_summary[col], width, label=col, color=colors[idx % len(colors)])
                
            ax.set_ylabel("Score (0.0 to 1.0)")
            ax.set_title(f"Average Retrieval Metrics Comparison (K={eval_k})")
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            # Match streamlit theme background color
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e293b')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('white')
            
            st.pyplot(fig)
            
            # Query-specific breakdown
            st.subheader("🔍 Breakdown by Benchmark Query")
            selected_eval_query = st.selectbox("Select a benchmark query to view detailed rankings:", list(BENCHMARK_QUERIES.keys()))
            
            # Ground truth items for this query
            gt_ids = BENCHMARK_QUERIES[selected_eval_query]
            st.markdown(f"**Ground Truth Relevant Product IDs:** {', '.join([f'`{id}`' for id in gt_ids])}")
            
            # Show what each retriever returned for this query
            res_bm25 = [r["id"] for r in bm25.search(selected_eval_query, top_k=eval_k)]
            res_vec = [r["id"] for r in vector_search.search(selected_eval_query, top_k=eval_k)]
            res_rrf = [r["id"] for r in hybrid_search.search(selected_eval_query, top_k=eval_k, method="rrf")]
            res_fusion = [r["id"] for r in hybrid_search.search(selected_eval_query, top_k=eval_k, method="score_fusion")]
            
            df_q_breakdown = pd.DataFrame({
                "System": ["BM25", "Vector Search", "Hybrid (RRF)", "Hybrid (Score Fusion)"],
                "Retrieved IDs": [res_bm25, res_vec, res_rrf, res_fusion],
                "Matches (Relevant)": [
                    list(set(res_bm25) & set(gt_ids)),
                    list(set(res_vec) & set(gt_ids)),
                    list(set(res_rrf) & set(gt_ids)),
                    list(set(res_fusion) & set(gt_ids))
                ]
            })
            st.table(df_q_breakdown)

# ----------------- TAB 3: MATH & CONCEPT WALKTHROUGH -----------------
with tab3:
    st.header("📚 Theoretical Foundations & Math Walkthrough")
    st.write("Understand the mathematics powering our custom retrieval pipeline.")
    
    math_select = st.radio("Select Concept:", ["TF-IDF vs BM25", "Vector Embeddings & Cosine Similarity", "Hybrid Search Fusion (RRF & Score)", "Evaluation Metrics"], horizontal=True)
    
    if math_select == "TF-IDF vs BM25":
        st.subheader("1. TF-IDF Formulation")
        st.markdown("""
        In classical Term Frequency-Inverse Document Frequency (TF-IDF), term weighting is linear with respect to term frequency:
        $$TF(t, D) = \\frac{f(t, D)}{|D|}$$
        $$IDF(t) = \\ln \\left( \\frac{N}{n(t)} \\right)$$
        $$Score(D, Q) = \\sum_{t \\in Q} TF(t, D) \\cdot IDF(t)$$
        
        **Limitations:** If a term appears 100 times in a document, TF-IDF weights it 100 times more than if it appeared once. In real documents, this leads to spamming keywords.
        """)
        
        st.subheader("2. BM25 (Best Matching 25)")
        st.markdown("""
        BM25 improves TF-IDF by adding **term frequency saturation** ($k_1$) and **document length normalization** ($b$):
        $$IDF(q_i) = \\ln \\left( 1 + \\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \\right)$$
        
        $$Score(D, Q) = \\sum_{q_i \\in Q} IDF(q_i) \\cdot \\frac{f(q_i, D) \\cdot (k_1 + 1)}{f(q_i, D) + k_1 \\cdot \\left( 1 - b + b \\cdot \\frac{|D|}{avgdl} \\right)}$$
        
        **Parameters Explained:**
        - $k_1$ regulates **term frequency saturation**. As term frequency $tf$ increases, the term's contribution asymptotically approaches a limit. Higher $k_1$ makes TF scale slower.
        - $b$ controls **document length normalization**. If $b=1$, we fully divide by document length (penalizing long documents). If $b=0$, document length is ignored.
        """)
        
        # Interactive Plot for BM25 Term Frequency Saturation Curve
        st.write("#### 📈 Interactive BM25 Curve Simulator")
        sim_k1 = st.slider("Simulate k₁", 0.5, 3.0, 1.5, 0.1)
        sim_b = st.slider("Simulate b", 0.0, 1.0, 0.75, 0.05)
        sim_dl_ratio = st.slider("Document Length Ratio ( |D| / avgdl )", 0.1, 3.0, 1.0, 0.1)
        
        tfs = np.arange(0, 15)
        weights_bm25 = (tfs * (sim_k1 + 1)) / (tfs + sim_k1 * (1 - sim_b + sim_b * sim_dl_ratio))
        weights_tfidf = tfs * 0.4  # Linear TF-IDF approximation
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tfs, weights_bm25, label="BM25 Saturation Curve (Non-linear)", color="#4f46e5", linewidth=3)
        ax.plot(tfs, weights_tfidf, label="Standard TF-IDF (Linear)", color="#d1d5db", linestyle="--")
        ax.set_xlabel("Term Frequency (tf)")
        ax.set_ylabel("Term Frequency Weight Contribution")
        ax.set_title("How BM25 Saturation k₁ and Length Normalization b Shape Term Weights")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Stylings
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')
            
        st.pyplot(fig)
        
    elif math_select == "Vector Embeddings & Cosine Similarity":
        st.subheader("1. Vector Embedding Representation")
        st.markdown("""
        Dense retrieval transforms a text document $D$ into a fixed-length high-dimensional vector $\\vec{v}_D \\in \\mathbb{R}^d$.
        Our model `all-MiniLM-L6-v2` encodes documents into a $384$-dimensional space:
        $$\\vec{v}_D = [v_1, v_2, \\dots, v_{384}]$$
        
        Unlike BM25 which matches exact keywords, embeddings capture **semantic concepts** (e.g. "Ryzen" is closer to "processor" than "headphones").
        """)
        
        st.subheader("2. Cosine Similarity Calculation")
        st.markdown("""
        To match a query vector $\\vec{q}$ and document vector $\\vec{d}$, we measure the angle $\\theta$ between them. The cosine similarity is:
        $$\\text{Cosine Similarity}(q, d) = \\cos(\\theta) = \\frac{\\vec{q} \\cdot \\vec{d}}{\\|\\vec{q}\\|_2 \\|\\vec{d}\\|_2}$$
        
        Expanded form:
        $$\\text{Cosine Similarity}(q, d) = \\frac{\\sum_{i=1}^{384} q_i d_i}{\\sqrt{\\sum_{i=1}^{384} q_i^2} \\sqrt{\\sum_{i=1}^{384} d_i^2}}$$
        
        - A score of **1.0** indicates identical directions (maximum semantic overlap).
        - A score of **0.0** indicates orthogonality (no correlation).
        - A score of **-1.0** indicates opposite directions.
        """)
        
        # Interactive vector simulator
        st.write("#### 📐 Simple 2D Cosine Similarity Simulator")
        col_vec1, col_vec2 = st.columns(2)
        with col_vec1:
            q_x = st.slider("Query X", -1.0, 1.0, 0.8, 0.1)
            q_y = st.slider("Query Y", -1.0, 1.0, 0.6, 0.1)
        with col_vec2:
            d_x = st.slider("Doc X", -1.0, 1.0, 0.5, 0.1)
            d_y = st.slider("Doc Y", -1.0, 1.0, 0.8, 0.1)
            
        vec_q = np.array([q_x, q_y])
        vec_d = np.array([d_x, d_y])
        
        dot = np.dot(vec_q, vec_d)
        norm_q = np.linalg.norm(vec_q)
        norm_d = np.linalg.norm(vec_d)
        cos_sim = dot / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0
        
        # Plot vectors
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.quiver(0, 0, q_x, q_y, angles='xy', scale_units='xy', scale=1, color='#4f46e5', label="Query Vector (q)")
        ax.quiver(0, 0, d_x, d_y, angles='xy', scale_units='xy', scale=1, color='#10b981', label="Doc Vector (d)")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_title(f"Cosine Similarity = {cos_sim:.4f}")
        
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')
            
        st.pyplot(fig)
        
    elif math_select == "Hybrid Search Fusion (RRF & Score)":
        st.subheader("1. Reciprocal Rank Fusion (RRF)")
        st.markdown("""
        RRF takes the ranked lists of multiple retrieval models and merges them without score calibration. RRF scores documents based on their position (rank) in both lists:
        $$RRF\\_Score(d \\in D) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}$$
        
        Where:
        - $M$ is the set of retrieval models (BM25 and Dense).
        - $r_m(d)$ is the rank (1-indexed position) of document $d$ in model $m$.
        - $k$ is a constant (smoothing factor), typically set to **60**. It dampens the influence of top ranks, ensuring that a single extremely high rank doesn't dominate if other models rank the document very low.
        """)
        
        st.subheader("2. Score Fusion")
        st.markdown("""
        Score Fusion normalizes the raw outputs of the retrieval engines to a common scale $[0, 1]$ before performing a weighted linear combination:
        $$Score_{norm}(d) = \\frac{Score_{raw}(d) - Score_{min}}{Score_{max} - Score_{min} + \\epsilon}$$
        $$Score_{hybrid}(d) = \\alpha \\cdot Score_{norm, BM25}(d) + (1 - \\alpha) \\cdot Score_{norm, Dense}(d)$$
        
        - $\\alpha$ controls the relative weight. A higher $\\alpha$ shifts importance towards keyword matching (BM25).
        """)
        
    elif math_select == "Evaluation Metrics":
        st.subheader("Information Retrieval Evaluation Metrics")
        st.markdown("""
        Let $R$ be the set of ground-truth relevant document IDs, and let $P_K$ be the ordered set of top $K$ retrieved document IDs.
        
        #### A. Precision@K
        The fraction of retrieved documents in the top $K$ that are relevant:
        $$\\text{Precision@K} = \\frac{|R \\cap P_K|}{K}$$
        
        #### B. Recall@K
        The fraction of all ground-truth relevant documents retrieved in the top $K$:
        $$\\text{Recall@K} = \\frac{|R \\cap P_K|}{|R|}$$
        
        #### C. MRR (Mean Reciprocal Rank)
        Computes the rank of the first relevant retrieved document, reciprocalizes it, and averages across all queries $Q$:
        $$\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}$$
        Where $\\text{rank}_i$ is the position (1-indexed) of the first relevant document for query $i$.
        
        #### D. NDCG@K (Normalized Discounted Cumulative Gain)
        Evaluates ranking position. Documents that are relevant should be ranked high.
        
        **Discounted Cumulative Gain (DCG@K):**
        $$\\text{DCG@K} = \\sum_{i=1}^{K} \\frac{rel_i}{\\log_2(i + 1)}$$
        (where $rel_i = 1$ if the document at rank $i$ is relevant, and $0$ otherwise).
        
        **Ideal DCG (IDCG@K):** The maximum possible DCG if all relevant documents were sorted at the top.
        $$\\text{NDCG@K} = \\frac{\\text{DCG@K}}{\\text{IDCG@K}}$$
        """)

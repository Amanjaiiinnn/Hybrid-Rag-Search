"""
LangChain integration for the hand-written retrievers.

- CatalogRetriever: a custom BaseRetriever around HybridSearchEngine (or any engine
  with get_scores-style search), with optional cross-encoder reranking and metadata
  filters (max price, category). Returns LangChain Documents.
- build_rag_chain: an LCEL chain  retriever -> prompt -> ChatGroq -> parser.
"""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import ConfigDict


class CatalogRetriever(BaseRetriever):
    """
    Wraps the custom hybrid (BM25 + dense) search engine as a LangChain retriever.

    Metadata filters are applied after scoring the whole catalog (post-filtering
    over all 100 products, so no result is lost to the filter). With a large
    corpus you would push the filter into the vector store instead (pre-filtering).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: Any
    products: List[dict]
    k: int = 4
    fetch_k: int = 20
    method: str = "rrf"
    reranker: Optional[Any] = None
    max_price: Optional[int] = None
    category: Optional[str] = None

    def with_filters(self, max_price=None, category=None):
        """Returns a copy of this retriever with metadata filters set."""
        return self.model_copy(update={"max_price": max_price, "category": category})

    def _passes_filters(self, product):
        if self.max_price is not None and product["price_inr"] > self.max_price:
            return False
        if self.category and product["category"].lower() != self.category.lower():
            return False
        return True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        corpus = self.engine.corpus
        # Score the full catalog, then filter, then keep fetch_k candidates for reranking
        ranked = self.engine.search(query, top_k=len(corpus), method=self.method)
        ranked = [r for r in ranked if self._passes_filters(self.products[r["index"]])][:self.fetch_k]

        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked, corpus, top_k=self.k)
        else:
            ranked = ranked[:self.k]

        docs = []
        for r in ranked:
            p = self.products[r["index"]]
            docs.append(Document(
                page_content=corpus[r["index"]],
                metadata={
                    "product_id": p["product_id"],
                    "product_name": p["product_name"],
                    "category": p["category"],
                    "brand": p["brand"],
                    "price_inr": p["price_inr"],
                    "rating": p["rating"],
                    "score": r["score"],
                },
            ))
        return docs


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a shopping assistant for an electronics catalog. Answer ONLY from the "
     "products in the context. If the context does not contain the answer, say so. "
     "Mention product names and prices (INR). Use Markdown.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[{i + 1}] {d.page_content} | Rating: {d.metadata.get('rating')}" for i, d in enumerate(docs)
    )


def make_llm(api_key=None, model="llama-3.3-70b-versatile", temperature=0.2):
    """ChatGroq model. Reads GROQ_API_KEY from the environment when api_key is None."""
    from langchain_groq import ChatGroq
    kwargs = {"model": model, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatGroq(**kwargs)


def build_rag_chain(retriever, llm):
    """
    LCEL chain. Input: question string. Output: {"answer": str, "docs": [Document]}.

        question --+--> retriever ----> docs ---------------------+
                   +--> passthrough --> question --> prompt -> llm -> str --> {"docs", "question", "answer"}
    """
    answer_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
    )


def build_catalog_retriever(products=None, reranker=True, k=4):
    """Convenience factory: loads the catalog, builds BM25 + FAISS hybrid, wraps it."""
    from data_store import load_dataset, get_indexed_documents
    from bm25 import BM25Retriever
    from hybrid_search import HybridSearchEngine
    from ann_index import FaissVectorSearch
    from reranker import CrossEncoderReranker

    products = products or load_dataset(".")
    docs = get_indexed_documents(products)
    corpus = [d[1] for d in docs]
    ids = [d[0] for d in docs]
    engine = HybridSearchEngine(BM25Retriever(corpus, ids), FaissVectorSearch(corpus, ids, kind="flat"))
    return CatalogRetriever(
        engine=engine,
        products=products,
        k=k,
        reranker=CrossEncoderReranker() if reranker else None,
    )


if __name__ == "__main__":
    import os

    retriever = build_catalog_retriever()

    print("Plain retriever:")
    for d in retriever.invoke("Best laptop under 80k"):
        print(f"  {d.metadata['product_id']}  INR {d.metadata['price_inr']:>7,}  {d.metadata['product_name']}")

    print("Same query with a max_price=80000, category=Laptop filter:")
    for d in retriever.with_filters(max_price=80000, category="Laptop").invoke("Best laptop under 80k"):
        print(f"  {d.metadata['product_id']}  INR {d.metadata['price_inr']:>7,}  {d.metadata['product_name']}")

    print("Batch (LCEL .batch over the retriever):")
    for q, docs in zip(["Tensor G4 phone", "mesh router"], retriever.batch(["Tensor G4 phone", "mesh router"])):
        print(f"  {q!r}: {[d.metadata['product_id'] for d in docs]}")

    if os.environ.get("GROQ_API_KEY"):
        chain = build_rag_chain(retriever, make_llm())
        print("\nStreaming answer:")
        for chunk in chain.stream("Smartwatch with ECG support"):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print()
    else:
        print("\nSet GROQ_API_KEY to run the LCEL RAG chain.")

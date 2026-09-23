"""
Plain RAG (LCEL chain) vs corrective RAG (LangGraph) on the labelled benchmark
queries plus deliberately vague / impossible ones. Needs GROQ_API_KEY.

    python eval_crag.py

Reports, per query: context precision (share of context docs that are labelled
relevant, benchmark queries only), rewrites, regenerations, and the groundedness
verdict of the same LLM judge applied to both systems' answers.
"""
import os
import sys
import uuid

import pandas as pd

from crag_graph import GROUNDED_PROMPT, GroundednessGrade, build_crag_graph, format_context
from evaluation import BENCHMARK_QUERIES
from lc_retriever import build_catalog_retriever, build_rag_chain, make_llm

EXTRA_QUERIES = [
    "something to track my heart at the gym",   # vague: needs Smartwatch
    "cheap computer for college",               # vague: needs Laptop + budget
    "SSD under 3000 rupees",                     # impossible: cheapest SSD costs more
    "best gaming console",                       # not in catalog
]


def precision(docs, query):
    relevant = BENCHMARK_QUERIES.get(query)
    if relevant is None or not docs:
        return None
    return sum(d.metadata["product_id"] in relevant for d in docs) / len(docs)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY first.")

    retriever = build_catalog_retriever(k=4)
    generator = make_llm(model="llama-3.3-70b-versatile")
    judge = make_llm(model="llama-3.1-8b-instant", temperature=0.0)
    plain = build_rag_chain(retriever, generator)
    crag = build_crag_graph(retriever, generator, judge)
    grounded_judge = GROUNDED_PROMPT | judge.with_structured_output(GroundednessGrade)

    rows = []
    for query in list(BENCHMARK_QUERIES) + EXTRA_QUERIES:
        base = plain.invoke(query)
        base_grounded = grounded_judge.invoke({"context": format_context(base["docs"], []), "generation": base["answer"]})

        result = crag.invoke({"user_input": query}, {"configurable": {"thread_id": str(uuid.uuid4())}})
        path = [s["node"] for s in result["trace"]]

        rows.append({
            "Query": query,
            "Plain ctx precision": precision(base["docs"], query),
            "CRAG ctx precision": precision(result["documents"], query),
            "Plain grounded": base_grounded.grounded,
            "CRAG grounded": result["grounded"],
            "CRAG route": result.get("route"),
            "Rewrites": path.count("rewrite_query"),
            "Generations": path.count("generate"),
            "Fallback": "fallback" in path,
        })
        print(f"done: {query}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format="%.2f"))
    print(f"\nRewrite rate: {(df['Rewrites'] > 0).mean():.0%} | Regeneration rate: {(df['Generations'] > 1).mean():.0%}"
          f" | Plain grounded: {df['Plain grounded'].mean():.0%} | CRAG grounded: {df['CRAG grounded'].mean():.0%}")

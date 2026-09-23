"""
Corrective RAG (CRAG) over the electronics catalog, built with LangGraph.

    start_turn -> route --off_topic--> off_topic --------------------------------+
                    |                                                            |
                    v                                                            |
                 retrieve -> grade_documents --enough relevant--> generate       |
                    ^              |                                 |           |
                    |              +--too few, retries left--+       v           |
                    +---------- rewrite_query <--------------+  grade_generation |
                    ^              |                             |   |   |       |
                    |              +--too few, no retries--> fallback |   |       |
                    |                                   (web / none)  |   |       |
                    +------- not answering, retries left -------------+   |       |
                             not grounded, retries left -> generate  <----+       |
                                                  pass / out of retries -> finish <+

- route: an LLM router picks a specialist agent (spec_lookup / compare_budget) and
  extracts metadata filters (category, max price) from the question.
- grade_documents / grade_generation: LLM-as-judge with structured output, run on a
  small fast model; the generator runs on a larger model.
- A checkpointer keeps per-thread history, so follow-up questions are rewritten into
  standalone questions before retrieval.
"""
import os
from typing import List, Literal, Optional, TypedDict

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

CATEGORIES = ["Laptop", "Smartphone", "Headphones", "Tablet", "Smartwatch", "Monitor", "SSD", "Router"]


# ----------------- Structured outputs for the router and judges -----------------
class RouteDecision(BaseModel):
    """Which specialist should handle the question, plus metadata filters."""
    route: Literal["spec_lookup", "compare_budget", "off_topic"] = Field(
        description="spec_lookup: questions about specific features/specs. "
                    "compare_budget: recommendations, comparisons or price limits. "
                    "off_topic: not about shopping for electronics.")
    category: Optional[str] = Field(default=None, description=f"One of {CATEGORIES} if the question targets one, else null.")
    max_price_inr: Optional[int] = Field(default=None, description="Upper price limit in INR if stated (80k -> 80000), else null.")


class DocumentGrade(BaseModel):
    """Relevance of one retrieved product to the question."""
    relevant: bool = Field(description="True if the product helps answer the question.")
    reason: str = Field(description="One short sentence.")


class GroundednessGrade(BaseModel):
    """Whether the answer is supported by the context."""
    grounded: bool = Field(description="True if every product, spec and price in the answer appears in the context.")
    reason: str = Field(description="One short sentence.")


class AnswerGrade(BaseModel):
    """Whether the answer resolves the question."""
    answers_question: bool = Field(description="True if the answer addresses what the user asked.")
    reason: str = Field(description="One short sentence.")


class CRAGState(TypedDict, total=False):
    user_input: str            # raw input for this turn
    question: str              # standalone question (after resolving follow-ups)
    search_query: str          # query sent to the retriever (may be rewritten)
    route: str
    category: Optional[str]
    max_price: Optional[int]
    documents: List[Document]  # retrieved, then filtered to judge-approved ones
    num_retrieved: int
    web_results: List[str]
    generation: str
    grounded: bool
    answers_question: bool
    rewrites: int
    generations: int
    trace: List[dict]          # per-turn log of node decisions, for the UI
    history: List[dict]        # persisted across turns by the checkpointer


# ----------------- Prompts -----------------
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as a standalone shopping question, using the "
               "conversation for missing context. Return only the question."),
    ("human", "Conversation:\n{history}\n\nLatest message: {question}"),
])

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You route questions for an electronics store whose catalog has these categories: "
               f"{', '.join(CATEGORIES)}. Pick the specialist and extract filters."),
    ("human", "{question}"),
])

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You grade whether a retrieved catalog product is relevant to a shopper's question. "
               "It is relevant if it matches the product type and the key requirements "
               "(features, budget). Be strict about product type."),
    ("human", "Question: {question}\n\nProduct: {document}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "The search query below returned poor results from a product catalog search engine "
               "(keyword + semantic). Rewrite it as a short search query naming the product type "
               "and key specs. Return only the query."),
    ("human", "Original question: {question}\nPrevious query: {search_query}"),
])

AGENT_PROMPTS = {
    "spec_lookup": "You are a product specialist. Answer precisely which products have the requested "
                   "features, quoting the relevant specs, product names and IDs.",
    "compare_budget": "You are a budget advisor. Respect any price limit, compare the candidates in a "
                      "Markdown table (name, price, key specs, rating) and recommend the best value.",
}

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{agent_instructions}\n\nUse ONLY the context. Prices are in INR. If the context has "
               "no suitable product, say the catalog has none. Label anything taken from web "
               "results as outside the catalog.{retry_note}\n\nContext:\n{context}"),
    ("human", "{question}"),
])

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You check an assistant's answer against its context. Mark it ungrounded if it names "
               "a product, spec, price or rating that is not in the context."),
    ("human", "Context:\n{context}\n\nAnswer:\n{generation}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You check whether an answer addresses the user's question. An honest 'the catalog "
               "has no such product' counts as addressing it."),
    ("human", "Question: {question}\n\nAnswer:\n{generation}"),
])


def normalize_category(category):
    if not category:
        return None
    wanted = category.strip().lower().rstrip("s")
    for c in CATEGORIES:
        if c.lower().rstrip("s") == wanted:
            return c
    return None


def format_context(documents, web_results):
    parts = [f"[{i + 1}] {d.page_content} | Rating: {d.metadata.get('rating')}" for i, d in enumerate(documents)]
    parts += [f"[web {i + 1}] {w}" for i, w in enumerate(web_results)]
    return "\n\n".join(parts) if parts else "(no products found)"


def tavily_search(query, max_results=3):
    """Web search fallback. Returns [] when TAVILY_API_KEY is not set or the call fails."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        response.raise_for_status()
        return [f"{r['title']} ({r['url']}): {r['content']}" for r in response.json().get("results", [])]
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []


def build_crag_graph(retriever, generator_llm, judge_llm, min_relevant=2, max_rewrites=2,
                     max_generations=2, web_search=tavily_search, checkpointer=None):
    """
    :param retriever: CatalogRetriever (needs .with_filters(max_price, category) and .invoke).
    :param generator_llm: chat model that writes answers (e.g. llama-3.3-70b-versatile).
    :param judge_llm: chat model for routing, grading and rewriting (e.g. llama-3.1-8b-instant).
    Returns a compiled graph; invoke with {"user_input": ...} and a thread_id config.
    """
    contextualizer = CONTEXTUALIZE_PROMPT | judge_llm | StrOutputParser()
    router = ROUTER_PROMPT | judge_llm.with_structured_output(RouteDecision)
    grader = GRADER_PROMPT | judge_llm.with_structured_output(DocumentGrade)
    rewriter = REWRITE_PROMPT | judge_llm | StrOutputParser()
    generator = GENERATE_PROMPT | generator_llm | StrOutputParser()
    grounded_judge = GROUNDED_PROMPT | judge_llm.with_structured_output(GroundednessGrade)
    answer_judge = ANSWER_PROMPT | judge_llm.with_structured_output(AnswerGrade)

    def log(state, node, detail):
        return state.get("trace", []) + [{"node": node, "detail": detail}]

    # ----------------- Nodes -----------------
    def start_turn(state):
        user_input = state["user_input"]
        history = state.get("history", [])
        question = user_input
        if history:
            transcript = "\n".join(f"User: {h['question']}\nAssistant: {h['answer'][:400]}" for h in history[-3:])
            question = contextualizer.invoke({"history": transcript, "question": user_input}).strip()
        detail = f"Standalone question: {question}" if question != user_input else "New question"
        return {
            "question": question, "search_query": question, "documents": [], "web_results": [],
            "generation": "", "grounded": False, "answers_question": False,
            "rewrites": 0, "generations": 0,
            "trace": [{"node": "start_turn", "detail": detail}],
        }

    def route(state):
        try:
            decision = router.invoke({"question": state["question"]})
        except Exception as e:
            decision = RouteDecision(route="spec_lookup")
            print(f"Router failed ({e}); defaulting to spec_lookup")
        category = normalize_category(decision.category)
        detail = f"{decision.route} | category={category} | max_price={decision.max_price_inr}"
        return {"route": decision.route, "category": category, "max_price": decision.max_price_inr,
                "trace": log(state, "route", detail)}

    def off_topic(state):
        answer = ("I can only help with shopping questions about this electronics catalog "
                  f"({', '.join(CATEGORIES)}).")
        return {"generation": answer, "grounded": True, "answers_question": True,
                "trace": log(state, "off_topic", "Declined")}

    def retrieve(state):
        filtered = retriever.with_filters(max_price=state.get("max_price"), category=state.get("category"))
        documents = filtered.invoke(state["search_query"])
        ids = [d.metadata["product_id"] for d in documents]
        return {"documents": documents, "trace": log(state, "retrieve", f"'{state['search_query']}' -> {ids}")}

    def grade_documents(state):
        documents = state["documents"]
        try:
            grades = grader.batch([{"question": state["question"], "document": d.page_content} for d in documents])
        except Exception as e:
            print(f"Grader failed ({e}); keeping all documents")
            grades = [DocumentGrade(relevant=True, reason="grader error") for _ in documents]
        kept = [d for d, g in zip(documents, grades) if g.relevant]
        detail = ", ".join(
            f"{d.metadata['product_id']}:{'yes' if g.relevant else 'no'}" for d, g in zip(documents, grades)
        )
        return {"documents": kept, "num_retrieved": len(documents),
                "trace": log(state, "grade_documents", f"{len(kept)}/{len(documents)} relevant ({detail})")}

    def rewrite_query(state):
        new_query = rewriter.invoke({"question": state["question"], "search_query": state["search_query"]}).strip()
        return {"search_query": new_query, "rewrites": state["rewrites"] + 1,
                "trace": log(state, "rewrite_query", f"Attempt {state['rewrites'] + 1}: '{new_query}'")}

    def fallback(state):
        results = web_search(state["question"])
        detail = f"{len(results)} web results" if results else "No web search available; answering from catalog only"
        return {"web_results": results, "trace": log(state, "fallback", detail)}

    def generate(state):
        retry_note = ""
        if state["generations"] > 0:
            retry_note = ("\nYour previous answer included claims not supported by the context. "
                          "Only state facts that appear in the context.")
        generation = generator.invoke({
            "agent_instructions": AGENT_PROMPTS.get(state["route"], AGENT_PROMPTS["spec_lookup"]),
            "retry_note": retry_note,
            "context": format_context(state["documents"], state.get("web_results", [])),
            "question": state["question"],
        })
        return {"generation": generation, "generations": state["generations"] + 1,
                "trace": log(state, "generate", f"{state['route']} agent, attempt {state['generations'] + 1}")}

    def grade_generation(state):
        context = format_context(state["documents"], state.get("web_results", []))
        try:
            grounded = grounded_judge.invoke({"context": context, "generation": state["generation"]})
            answers = answer_judge.invoke({"question": state["question"], "generation": state["generation"]})
        except Exception as e:
            print(f"Generation judge failed ({e}); accepting answer")
            grounded = GroundednessGrade(grounded=True, reason="judge error")
            answers = AnswerGrade(answers_question=True, reason="judge error")
        detail = (f"grounded={grounded.grounded} ({grounded.reason}) | "
                  f"answers={answers.answers_question} ({answers.reason})")
        return {"grounded": grounded.grounded, "answers_question": answers.answers_question,
                "trace": log(state, "grade_generation", detail)}

    def finish(state):
        history = state.get("history", []) + [{"question": state["question"], "answer": state["generation"]}]
        return {"history": history[-10:], "trace": log(state, "finish", "Done")}

    # ----------------- Conditional edges -----------------
    def after_route(state):
        return "off_topic" if state["route"] == "off_topic" else "retrieve"

    def after_grading(state):
        kept = len(state["documents"])
        # Enough relevant docs, or the filters only matched a few and all of them passed
        if kept >= min_relevant or (kept > 0 and kept == state["num_retrieved"]):
            return "generate"
        if state["rewrites"] < max_rewrites:
            return "rewrite_query"
        return "fallback"

    def after_generation(state):
        if state["grounded"] and state["answers_question"]:
            return "finish"
        if not state["grounded"] and state["generations"] < max_generations:
            return "generate"
        if state["grounded"] and not state["answers_question"] and state["rewrites"] < max_rewrites:
            return "rewrite_query"
        return "finish"

    graph = StateGraph(CRAGState)
    for name, fn in [("start_turn", start_turn), ("route", route), ("off_topic", off_topic),
                     ("retrieve", retrieve), ("grade_documents", grade_documents),
                     ("rewrite_query", rewrite_query), ("fallback", fallback), ("generate", generate),
                     ("grade_generation", grade_generation), ("finish", finish)]:
        graph.add_node(name, fn)

    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "route")
    graph.add_conditional_edges("route", after_route, ["off_topic", "retrieve"])
    graph.add_edge("off_topic", "finish")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", after_grading, ["generate", "rewrite_query", "fallback"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("fallback", "generate")
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges("grade_generation", after_generation, ["finish", "generate", "rewrite_query"])
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def build_default_crag(groq_api_key=None, generator_model="llama-3.3-70b-versatile",
                       judge_model="llama-3.1-8b-instant", retriever=None):
    from lc_retriever import build_catalog_retriever, make_llm
    return build_crag_graph(
        retriever or build_catalog_retriever(k=4),
        generator_llm=make_llm(groq_api_key, generator_model, temperature=0.2),
        judge_llm=make_llm(groq_api_key, judge_model, temperature=0.0),
    )


if __name__ == "__main__":
    import sys
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY to run the corrective-RAG graph (python test_crag.py runs offline).")

    app = build_default_crag()
    config = {"configurable": {"thread_id": "cli"}}
    for q in ["Best laptop under 80k", "which of those has the most RAM?", "SSD under 3000 rupees"]:
        result = app.invoke({"user_input": q}, config)
        print(f"\n=== {q}")
        for step in result["trace"]:
            print(f"  [{step['node']}] {step['detail']}")
        print(result["generation"])

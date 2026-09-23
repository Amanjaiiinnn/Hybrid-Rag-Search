"""
Offline tests for the corrective-RAG graph: every branch is driven by a stub LLM
and a fake retriever, so no API key or model download is needed.

    python test_crag.py        (or: pytest test_crag.py)
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from crag_graph import (AnswerGrade, DocumentGrade, GroundednessGrade, RouteDecision, build_crag_graph)


class StubLLM(RunnableLambda):
    """Chat-model stand-in: text replies come from text_fn, structured replies from structured[schema name]."""
    def __init__(self, text_fn, structured):
        super().__init__(lambda prompt: AIMessage(content=text_fn(prompt.to_string())))
        self.structured = structured

    def with_structured_output(self, schema, **kwargs):
        fn = self.structured[schema.__name__]
        return RunnableLambda(lambda prompt: fn(prompt.to_string()))


class FakeRetriever(BaseRetriever):
    """Returns canned documents per query and records the filters it was called with."""
    results: dict
    calls: list
    max_price: Optional[int] = None
    category: Optional[str] = None

    def with_filters(self, max_price=None, category=None):
        return self.model_copy(update={"max_price": max_price, "category": category})

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:
        self.calls.append({"query": query, "max_price": self.max_price, "category": self.category})
        return [Document(page_content=text, metadata={"product_id": pid, "rating": 4.5})
                for pid, text in self.results.get(query, [])]


LAPTOPS = [("ELEC009", "Laptop Model 9 INR 39,780"), ("ELEC073", "Laptop Model 73 INR 35,811")]
PHONES = [("ELEC002", "Smartphone Model 2 Tensor G4"), ("ELEC010", "Smartphone Model 10 Tensor G4")]


def make_judge(route="compare_budget", category="Laptop", max_price=80000,
               relevant=lambda prompt: "Laptop" in prompt, grounded=None, rewrite_to="laptop under 80000"):
    grounded = grounded or (lambda prompt: True)
    return StubLLM(
        text_fn=lambda prompt: rewrite_to if "Rewrite it as a short search query" in prompt else "Standalone: follow-up",
        structured={
            "RouteDecision": lambda p: RouteDecision(route=route, category=category, max_price_inr=max_price),
            "DocumentGrade": lambda p: DocumentGrade(relevant=relevant(p), reason="stub"),
            "GroundednessGrade": lambda p: GroundednessGrade(grounded=grounded(p), reason="stub"),
            "AnswerGrade": lambda p: AnswerGrade(answers_question=True, reason="stub"),
        },
    )


GENERATOR = StubLLM(text_fn=lambda prompt: "Recommend ELEC009.", structured={})


def run(graph, text, thread="t"):
    result = graph.invoke({"user_input": text}, {"configurable": {"thread_id": thread}})
    return result, [s["node"] for s in result["trace"]]


def test_happy_path_uses_router_filters():
    retriever = FakeRetriever(results={"Best laptop under 80k": LAPTOPS}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge())
    result, path = run(graph, "Best laptop under 80k")
    assert path == ["start_turn", "route", "retrieve", "grade_documents", "generate", "grade_generation", "finish"]
    assert retriever.calls[0] == {"query": "Best laptop under 80k", "max_price": 80000, "category": "Laptop"}
    assert result["generation"] == "Recommend ELEC009."


def test_irrelevant_docs_trigger_rewrite_then_recover():
    retriever = FakeRetriever(results={"cheap computer": PHONES, "laptop under 80000": LAPTOPS}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge())
    result, path = run(graph, "cheap computer")
    assert path.count("rewrite_query") == 1
    assert [c["query"] for c in retriever.calls] == ["cheap computer", "laptop under 80000"]
    assert [d.metadata["product_id"] for d in result["documents"]] == ["ELEC009", "ELEC073"]


def test_fallback_after_max_rewrites():
    web_calls = []
    retriever = FakeRetriever(results={}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge(),
                             web_search=lambda q: web_calls.append(q) or ["web: some result"])
    result, path = run(graph, "SSD under 3000 rupees")
    assert path.count("rewrite_query") == 2 and "fallback" in path
    assert web_calls == ["SSD under 3000 rupees"]
    assert result["web_results"] == ["web: some result"]


def test_single_filtered_match_skips_rewrite():
    retriever = FakeRetriever(results={"Best laptop under 36k": LAPTOPS[1:]}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge(max_price=36000))
    _, path = run(graph, "Best laptop under 36k")
    assert "rewrite_query" not in path and "generate" in path


def test_ungrounded_answer_is_regenerated_with_stricter_prompt():
    seen = {"n": 0}

    def grounded(prompt):
        seen["n"] += 1
        return seen["n"] > 1  # first answer fails the hallucination check

    prompts = []
    generator = StubLLM(text_fn=lambda p: prompts.append(p) or "Recommend ELEC009.", structured={})
    retriever = FakeRetriever(results={"Best laptop under 80k": LAPTOPS}, calls=[])
    graph = build_crag_graph(retriever, generator, make_judge(grounded=grounded))
    result, path = run(graph, "Best laptop under 80k")
    assert path.count("generate") == 2 and result["grounded"]
    assert "not supported by the context" in prompts[1] and "not supported by the context" not in prompts[0]


def test_off_topic_short_circuits():
    retriever = FakeRetriever(results={}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge(route="off_topic", category=None, max_price=None))
    result, path = run(graph, "Write me a poem")
    assert path == ["start_turn", "route", "off_topic", "finish"]
    assert retriever.calls == []


def test_checkpointer_keeps_history_per_thread():
    retriever = FakeRetriever(results={"Best laptop under 80k": LAPTOPS, "Standalone: follow-up": LAPTOPS}, calls=[])
    graph = build_crag_graph(retriever, GENERATOR, make_judge())
    run(graph, "Best laptop under 80k", thread="a")
    result, _ = run(graph, "which has more RAM?", thread="a")
    assert result["question"] == "Standalone: follow-up"   # follow-up was contextualised
    assert len(result["history"]) == 2
    other, _ = run(graph, "Best laptop under 80k", thread="b")
    assert len(other["history"]) == 1                       # separate thread, separate memory


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")

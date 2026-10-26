"""Per-call results and progress stay isolated when agents are shared."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest

from librarian import (
    LibrarianAgent,
    RunResult,
    SynthesisAgent,
    SynthesisResult,
    load_runtime_config,
)
from librarian.literature_search import LiteratureSearchError
from librarian.scope_port import ScopeVerdict


def retrieval_agent(barrier=None):
    agent = LibrarianAgent(
        runtime_config=replace(load_runtime_config(), paragraphs_per_judge_batch=1),
        llm_client=object(),
        literature_source=object(),
    )

    def generate(query):
        if barrier:
            barrier.wait(timeout=10)
        return [query]

    def paragraphs(query):
        if query == "empty":
            return []
        return [
            {
                "paper": {"pmid": f"{query}-{i}", "title": query},
                "text": f"Evidence for {query}.",
                "source_paragraph_index": 0,
                "word_start": 0,
            }
            for i in range(1 if query == "alpha" else 2)
        ]

    def evaluate(query, batch, template, state):
        if barrier and batch[0]["id"] == "paragraph_0":
            barrier.wait(timeout=10)
        return [sentence["id"] for item in batch for sentence in item["sentences"]]

    agent._generate_queries = generate
    agent._validate_queries = lambda queries: queries
    agent._paragraphs_for_subquery = paragraphs
    agent._evaluate_batch = evaluate
    return agent


def test_concurrent_retrieval_isolates_callbacks_counters_and_debug():
    agent = retrieval_agent(threading.Barrier(2))
    before = vars(agent).copy()
    messages = {query: [] for query in ("alpha", "beta")}
    steps = {query: [] for query in messages}

    def run(query):
        caller = threading.get_ident()
        return agent.run(
            query,
            on_progress=messages[query].append,
            progress_callback=lambda step: steps[query].append(
                (step, threading.get_ident() == caller)
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(zip(messages, pool.map(run, messages)))
    for query, total in (("alpha", 1), ("beta", 2)):
        result = results[query]
        assert isinstance(result, RunResult)
        assert result.debug["search_queries"] == [query]
        assert result.debug["paragraph_count"] == total
        assert result.debug["relevant_count"] == total
        assert result.debug["final_pmids"] == [f"{query}-{i}" for i in range(total)]
        assert len(result.papers) == total
        assert all(paper["title"] == query for paper in result.papers)
        assert steps[query] == [(0, True), (1, True), (2, True)]
        assert f"Found {total} candidate paragraphs" in messages[query]
        batch_messages = [m for m in messages[query] if m.endswith(" batches")]
        assert sorted(batch_messages) == sorted(
            [
                f"Judging paragraph relevance · {i}/{total} batches"
                for i in range(1, total + 1)
            ]
        )
    assert vars(agent) == before
    assert results["alpha"].debug is not results["beta"].debug


def test_empty_and_failed_runs_do_not_change_previous_result():
    agent = retrieval_agent()
    first = agent.run("alpha")
    empty = agent.run("empty")
    assert empty == RunResult([], {"search_queries": ["empty"]})

    def cancel(step):
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        agent.run("beta", progress_callback=cancel)
    assert first.debug["search_queries"] == ["alpha"]
    assert agent.run("beta").debug["relevant_count"] == 2
    assert not hasattr(agent, "last_run_debug")


def test_failed_search_preserves_partial_debug():
    agent = retrieval_agent()
    error = LiteratureSearchError("search unavailable")
    with (
        patch.object(agent, "_paragraphs_for_subquery", side_effect=error),
        pytest.raises(LiteratureSearchError) as caught,
    ):
        agent.run("alpha")
    assert caught.value is error
    assert error.run_debug == {"search_queries": ["alpha"]}
    agent.run("beta")
    assert error.run_debug == {"search_queries": ["alpha"]}


def test_failed_synthesis_search_preserves_route_and_subqueries():
    librarian = retrieval_agent()
    agent = SynthesisAgent(llm_client=object(), librarian=librarian)
    error = LiteratureSearchError("search unavailable")
    decision = {"action": "search", "query": "rewritten", "language": "English"}
    with (
        patch.object(agent, "_route", return_value=decision),
        patch.object(librarian, "_paragraphs_for_subquery", side_effect=error),
        pytest.raises(LiteratureSearchError) as caught,
    ):
        agent.run("original")
    assert caught.value is error
    assert error.run_debug == {
        "action": "search",
        "query": "rewritten",
        "language": "English",
        "passages": [],
        "search_queries": ["rewritten"],
    }


def test_concurrent_synthesis_with_shared_librarian():
    agent = SynthesisAgent(
        llm_client=object(), librarian=retrieval_agent(threading.Barrier(2))
    )
    barrier = threading.Barrier(2)
    agent._route = lambda query, history: {
        "action": "search",
        "query": query,
        "language": query,
    }

    def summarize(query, passages, **kwargs):
        barrier.wait(timeout=10)
        answer = f"answer {query}"
        kwargs["stream_callback"](answer)
        return answer

    agent._summarize = summarize
    before = vars(agent).copy()
    streams = {query: [] for query in ("alpha", "beta")}
    steps = {query: [] for query in streams}

    def run(query):
        return agent.run(
            query,
            stream_callback=streams[query].append,
            progress_callback=steps[query].append,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(zip(streams, pool.map(run, streams)))
    for query, result in results.items():
        assert isinstance(result, SynthesisResult)
        assert result.answer == f"answer {query}"
        assert result.debug["action"] == "search"
        assert result.debug["query"] == query
        assert result.debug["language"] == query
        assert result.debug["search_queries"] == [query]
        assert all(p["title"] == query for p in result.debug["passages"])
        assert streams[query] == [result.answer]
        assert steps[query] == [0, 1, 2, 3]
    assert vars(agent) == before


def test_synthesis_early_paths_return_their_own_debug():
    agent = SynthesisAgent(llm_client=object())
    with patch.object(agent, "_summarize", return_value="summary"):
        summary = agent.run("supplied", passages=[])
    assert summary.answer == "summary"
    assert summary.debug["action"] == "summarize"
    with patch.object(
        agent,
        "_route",
        return_value={"action": "reply", "query": "rewritten", "answer": "hello"},
    ):
        reply = agent.run("direct")
    assert reply.answer == "hello"
    assert reply.debug["action"] == "reply"
    assert reply.debug["search_queries"] == []
    with patch.object(agent, "_screen_query", return_value="stop"):
        stopped = agent.run("screened")
    assert stopped.answer == "stop"
    assert stopped.debug["action"] == "stopped"
    assert summary.debug["query"] == "supplied"
    assert not hasattr(agent, "last_run_debug")


def test_partial_scope_note_is_returned_and_streamed():
    class Guard:
        def check(self, query, conversation_history):
            return ScopeVerdict(in_scope=True, query="accepted", note="left out")

    agent = SynthesisAgent(llm_client=object(), scope_guard=Guard())
    stream = []
    with patch.object(
        agent, "_route", return_value={"action": "reply", "answer": "answer"}
    ):
        result = agent.run("mixed", stream_callback=stream.append)
    assert result.answer == "answer\n\nleft out"
    assert result.debug["query"] == "accepted"
    assert "".join(stream) == result.answer


def test_orchestrator_preserves_response_shape():
    import orchestrator

    retrieval = RunResult([{"pmid": "1"}], {"search_queries": ["subquery"]})
    with (
        patch.object(orchestrator.LibrarianAgent, "run", return_value=retrieval),
        patch.object(
            orchestrator.SynthesisAgent,
            "run",
            return_value=SynthesisResult("summary", {}),
        ),
    ):
        result = orchestrator._run(
            orchestrator.RunRequest(query="question", agent="literature_synthesis")
        )
    assert result == {
        "query": "question",
        "search_queries": ["subquery"],
        "papers": [{"pmid": "1"}],
        "summary": "summary",
        "action": "search",
    }

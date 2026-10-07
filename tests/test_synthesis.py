"""Offline tests for ``SynthesisAgent``: fake LLM, fake librarian, no network.

Run from the repository root with:

    uv run --with pytest pytest tests/test_synthesis.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from librarian import synthesis
from librarian.synthesis import SynthesisAgent

_ROUTE_SEARCH = json.dumps(
    {"language": "Italian", "action": "search", "query": "metformin lifespan mammals"}
)
_ROUTE_REPLY = json.dumps(
    {"language": "English", "action": "reply", "answer": "It is a protein."}
)


def _passage(pmid: str = "111", author: str = "Chen X", year: str = "2023") -> dict:
    """A record shaped like ``LibrarianAgent.run`` returns."""
    return {
        "title": f"Paper {pmid}",
        "authors": author,
        "journal": "Nature",
        "year": year,
        "pmid": pmid,
        "doi": "",
        "url": f"https://europepmc.org/article/MED/{pmid}",
        "evidence_snippets": [f"Evidence sentence of {pmid}."],
    }


class _FakeLLM:
    """Hands back scripted replies in order and records every call."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat_completion(self, messages, temperature=0.7, max_tokens=None, timeout=180):
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.replies.pop(0)

    def prompt(self, index: int) -> str:
        """The user-role text sent in call ``index``."""
        return self.calls[index]["messages"][-1]["content"]


class _FakeLibrarian:
    """Stands in for ``LibrarianAgent``: fixed passages, recorded calls."""

    def __init__(self, passages=None, error: Exception | None = None):
        self.passages = [_passage()] if passages is None else passages
        self.error = error
        self.calls: list[dict] = []
        self.last_run_debug: dict = {}

    def run(self, query, max_loops=1, on_progress=None, progress_callback=None):
        self.calls.append({"query": query, "max_loops": max_loops})
        if on_progress is not None:
            on_progress("searching")
        if progress_callback is not None:
            progress_callback(1)
        if self.error is not None:
            raise self.error
        self.last_run_debug = {"search_queries": ["sub query one", "sub query two"]}
        return self.passages


class _RecordingTracer:
    """A ``TracingPort`` that remembers span names and errors."""

    def __init__(self):
        self.spans: list[str] = []
        self.errors: list[Exception] = []

    def start_span(self, name, *, attributes=None):
        from contextlib import nullcontext

        self.spans.append(name)
        return nullcontext(name)

    def set_span_attributes(self, span, attributes):
        pass

    def mark_span_error(self, span, exc):
        self.errors.append(exc)

    def bind_current_trace_context(self, fn):
        return fn


class _StoppingAgent(SynthesisAgent):
    """Overrides the extension point: stops any query containing ``STOP``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[tuple] = []

    def _screen_query(self, query, conversation_history):
        self.seen.append((query, conversation_history))
        return "Stopped here." if "STOP" in query else None


# ── run(query, passages): behaves as before ──────────────────────────────────


def test_passages_path_writes_one_grounded_answer() -> None:
    llm = _FakeLLM("  The answer.  ")
    agent = SynthesisAgent(llm_client=llm)

    answer = agent.run("does X do Y?", [_passage()])

    assert answer == "The answer."
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["temperature"] == 0.2 and call["max_tokens"] == 8192
    assert call["messages"][0]["role"] == "system"
    prompt = llm.prompt(0)
    assert "does X do Y?" in prompt
    assert "[Chen 2023](https://europepmc.org/article/MED/111)" in prompt
    assert "(none — first turn)" in prompt
    assert "the language the USER QUERY above is written in" in prompt


def test_empty_passages_give_the_no_papers_line_without_a_model_call() -> None:
    llm = _FakeLLM()
    agent = SynthesisAgent(llm_client=llm)

    assert agent.run("q", []) == synthesis._NO_PAPERS_MESSAGE
    assert llm.calls == []


def test_prompt_is_filled_in_one_pass() -> None:
    """A question that looks like a placeholder is data, not a placeholder."""
    agent = SynthesisAgent(llm_client=_FakeLLM())

    prompt = agent._build_prompt("what is {papers_text}?", [_passage()])

    assert "what is {papers_text}?" in prompt
    for token in ("{today_date}", "{conversation_history}", "{answer_language}"):
        assert token not in prompt


def test_a_search_without_a_librarian_says_what_is_missing() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH)
    agent = SynthesisAgent(llm_client=llm)

    with pytest.raises(ValueError, match="librarian"):
        agent.run("q")


def test_a_direct_reply_needs_no_librarian() -> None:
    agent = SynthesisAgent(llm_client=_FakeLLM(_ROUTE_REPLY))

    assert agent.run("what is a protein?") == "It is a protein."


# ── run(query): route, search, answer ────────────────────────────────────────


def test_full_path_routes_searches_and_answers() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "Final answer.")
    librarian = _FakeLibrarian()
    agent = SynthesisAgent(llm_client=llm, librarian=librarian)
    steps: list[int] = []

    answer = agent.run("and in mice?", progress_callback=steps.append)

    assert answer == "Final answer."
    # The librarian searched the router's standalone query, not the raw question.
    assert librarian.calls == [{"query": "metformin lifespan mammals", "max_loops": 1}]
    assert steps == [1, 3]  # the librarian's step, then the one run() adds
    assert len(llm.calls) == 2
    assert llm.calls[0]["temperature"] == 0.0 and llm.calls[0]["max_tokens"] == 512
    summarizer_prompt = llm.prompt(1)
    assert "metformin lifespan mammals" in summarizer_prompt
    assert "Evidence sentence of 111." in summarizer_prompt
    # The router named the language, so the summarizer is told it outright.
    assert "Italian" in summarizer_prompt
    assert agent.last_run_debug == {
        "action": "search",
        "query": "metformin lifespan mammals",
        "language": "Italian",
        "passages": librarian.passages,
        "search_queries": ["sub query one", "sub query two"],
    }


def test_direct_reply_skips_the_search() -> None:
    llm = _FakeLLM(_ROUTE_REPLY)
    librarian = _FakeLibrarian()
    agent = SynthesisAgent(llm_client=llm, librarian=librarian)

    answer = agent.run("what is a protein?")

    assert answer == "It is a protein."
    assert librarian.calls == []
    assert len(llm.calls) == 1
    assert agent.last_run_debug["action"] == "reply"
    assert agent.last_run_debug["passages"] == []
    assert agent.last_run_debug["search_queries"] == []


def test_unparseable_router_output_searches_the_original_question() -> None:
    llm = _FakeLLM("sorry, no json here", "Final answer.")
    librarian = _FakeLibrarian()
    agent = SynthesisAgent(llm_client=llm, librarian=librarian)

    agent.run("original question")

    assert librarian.calls[0]["query"] == "original question"
    assert agent.last_run_debug["action"] == "search"
    assert agent.last_run_debug["language"] == ""
    assert "the language the USER QUERY above is written in" in llm.prompt(1)


def test_search_with_no_relevant_papers_reports_it_without_summarizing() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH)
    agent = SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian(passages=[]))

    assert agent.run("q") == synthesis._NO_PAPERS_MESSAGE
    assert len(llm.calls) == 1  # the router only


def test_a_failing_search_propagates_and_keeps_the_route_for_the_caller() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH)
    tracer = _RecordingTracer()
    boom = RuntimeError("Europe PMC is down")
    agent = SynthesisAgent(
        llm_client=llm, librarian=_FakeLibrarian(error=boom), tracer=tracer
    )

    with pytest.raises(RuntimeError, match="Europe PMC is down"):
        agent.run("q")

    assert agent.last_run_debug["action"] == "search"
    assert agent.last_run_debug["query"] == "metformin lifespan mammals"
    assert boom in tracer.errors


# ── conversation history ─────────────────────────────────────────────────────

_HISTORY = [
    {"role": "user", "content": "Tell me about metformin and ageing."},
    {"role": "assistant", "content": "Here is what the papers say about metformin."},
]


def test_history_reaches_both_the_router_and_the_summarizer() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "Final answer.")
    agent = SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian())

    agent.run("and in mice?", conversation_history=_HISTORY)

    for index in (0, 1):
        assert "user: Tell me about metformin and ageing." in llm.prompt(index)
        assert "assistant: Here is what the papers say" in llm.prompt(index)
    assert "Conversation so far:" in llm.prompt(0)
    assert "(none — first turn)" not in llm.prompt(1)


def test_history_reaches_the_summarizer_on_the_passages_path() -> None:
    llm = _FakeLLM("Answer.")
    agent = SynthesisAgent(llm_client=llm)

    agent.run("and in mice?", [_passage()], conversation_history=_HISTORY)

    assert "user: Tell me about metformin and ageing." in llm.prompt(0)


def test_no_history_leaves_no_history_block_in_the_router_prompt() -> None:
    llm = _FakeLLM(_ROUTE_REPLY)
    SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian()).run("q")

    assert "Conversation so far" not in llm.prompt(0)
    assert "{history_block}" not in llm.prompt(0)


def test_history_is_trimmed_to_recent_turns_and_capped_per_turn() -> None:
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    block = synthesis._render_history_block(history)
    assert block.splitlines() == [f"user: turn {i}" for i in range(4, 10)]

    long_turn = [{"role": "user", "content": "x" * 5000}]
    assert synthesis._render_history_block(long_turn) == "user: " + "x" * 2000

    assert synthesis._render_history_block(None) == ""
    assert synthesis._render_history_block([]) == ""


# ── the _screen_query extension point ────────────────────────────────────────


def test_screen_query_does_nothing_in_the_base_class() -> None:
    agent = SynthesisAgent(llm_client=_FakeLLM())

    assert agent._screen_query("any question", None) is None
    assert agent._screen_query("any question", _HISTORY) is None


def test_screen_query_text_stops_the_run_before_any_model_call() -> None:
    llm = _FakeLLM()
    librarian = _FakeLibrarian()
    agent = _StoppingAgent(llm_client=llm, librarian=librarian)

    assert (
        agent.run("please STOP now", conversation_history=_HISTORY) == "Stopped here."
    )

    assert llm.calls == []
    assert librarian.calls == []
    assert agent.seen == [("please STOP now", _HISTORY)]
    assert agent.last_run_debug["action"] == "stopped"
    assert agent.last_run_debug["passages"] == []


def test_screen_query_also_covers_the_passages_path() -> None:
    llm = _FakeLLM()
    agent = _StoppingAgent(llm_client=llm)

    assert agent.run("STOP", [_passage()]) == "Stopped here."
    assert llm.calls == []


def test_screen_query_none_lets_the_normal_flow_run() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "Final answer.")
    agent = _StoppingAgent(llm_client=llm, librarian=_FakeLibrarian())

    assert agent.run("a fine question") == "Final answer."
    assert len(llm.calls) == 2


# ── what run() hands to each step (so a subclass overrides a method, not run) ─


def test_progress_callbacks_reach_the_librarian() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "Final answer.")
    agent = SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian())
    messages: list[str] = []
    steps: list[int] = []

    agent.run("q", on_progress=messages.append, progress_callback=steps.append)

    assert messages == ["searching"]
    assert steps == [1, 3]  # the librarian's step, then the one run() adds


def test_output_channel_defaults_to_the_terminal_guidance() -> None:
    llm = _FakeLLM("Answer.", "Answer.")
    agent = SynthesisAgent(llm_client=llm)

    agent.run("q", [_passage()])
    agent.run("q", [_passage()], output_channel="anything-else")

    for index in (0, 1):
        assert "OUTPUT CHANNEL: terminal" in llm.prompt(index)


def test_a_subclass_adds_a_channel_by_overriding_one_method() -> None:
    class _TwoChannels(SynthesisAgent):
        def _output_formatting(self, output_channel):
            if output_channel == "chat":
                return "chat", "Write short chat-style paragraphs."
            return super()._output_formatting(output_channel)

    llm = _FakeLLM("Answer.", "Answer.")
    agent = _TwoChannels(llm_client=llm)

    agent.run("q", [_passage()], output_channel="chat")
    agent.run("q", [_passage()])

    assert "OUTPUT CHANNEL: chat" in llm.prompt(0)
    assert "Write short chat-style paragraphs." in llm.prompt(0)
    assert "OUTPUT CHANNEL: terminal" in llm.prompt(1)


def test_stream_callback_gets_the_answer_once_on_every_path() -> None:
    def chunks_for(agent, *args, **kwargs):
        chunks: list[str] = []
        agent.run(*args, stream_callback=chunks.append, **kwargs)
        return chunks

    # summary of given passages
    agent = SynthesisAgent(llm_client=_FakeLLM("  The answer.  "))
    assert chunks_for(agent, "q", [_passage()]) == ["The answer."]

    # nothing to answer from
    agent = SynthesisAgent(llm_client=_FakeLLM())
    assert chunks_for(agent, "q", []) == [synthesis._NO_PAPERS_MESSAGE]

    # full path: the router's JSON is never streamed, only the answer
    agent = SynthesisAgent(
        llm_client=_FakeLLM(_ROUTE_SEARCH, "Final answer."), librarian=_FakeLibrarian()
    )
    assert chunks_for(agent, "q") == ["Final answer."]

    # direct reply
    agent = SynthesisAgent(llm_client=_FakeLLM(_ROUTE_REPLY))
    assert chunks_for(agent, "q") == ["It is a protein."]

    # a text returned by the extension point
    agent = _StoppingAgent(llm_client=_FakeLLM())
    assert chunks_for(agent, "STOP", [_passage()]) == ["Stopped here."]


def test_a_subclass_replaces_retrieve_and_summarize_without_touching_run() -> None:
    """What a downstream subclass relies on: run() hands each step its arguments."""
    seen: dict = {}

    class _Subclass(SynthesisAgent):
        def _retrieve(
            self, query, max_loops=1, on_progress=None, progress_callback=None
        ):
            seen["retrieve"] = (query, max_loops, on_progress, progress_callback)
            return [_passage("333")]

        def _summarize(
            self,
            query,
            passages,
            conversation_history=None,
            language="",
            output_channel=None,
            stream_callback=None,
        ):
            seen["summarize"] = (
                query,
                [p["pmid"] for p in passages],
                conversation_history,
                language,
                output_channel,
                stream_callback,
            )
            return "subclass answer"

    def on_progress(message):
        pass

    def on_step(step):
        pass

    def on_chunk(text):
        pass

    # No librarian at all: the subclass supplies its own retrieval.
    agent = _Subclass(llm_client=_FakeLLM(_ROUTE_SEARCH))
    answer = agent.run(
        "and in mice?",
        conversation_history=_HISTORY,
        max_loops=1,
        on_progress=on_progress,
        progress_callback=on_step,
        output_channel="chat",
        stream_callback=on_chunk,
    )

    assert answer == "subclass answer"
    assert seen["retrieve"] == (
        "metformin lifespan mammals",
        1,
        on_progress,
        on_step,
    )
    assert seen["summarize"] == (
        "metformin lifespan mammals",
        ["333"],
        _HISTORY,
        "Italian",
        "chat",
        on_chunk,
    )
    assert agent.last_run_debug["action"] == "search"
    assert [p["pmid"] for p in agent.last_run_debug["passages"]] == ["333"]


# ── run bookkeeping ──────────────────────────────────────────────────────────


def test_last_run_debug_is_rebuilt_on_every_run() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "First.", "Second.")
    agent = SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian())

    agent.run("first question")
    assert agent.last_run_debug["action"] == "search"

    agent.run("second question", [_passage("222")])
    assert agent.last_run_debug["action"] == "summarize"
    assert agent.last_run_debug["query"] == "second question"
    assert agent.last_run_debug["search_queries"] == []
    assert [p["pmid"] for p in agent.last_run_debug["passages"]] == ["222"]


def test_a_failed_run_does_not_leave_the_previous_run_in_last_run_debug() -> None:
    llm = _FakeLLM(_ROUTE_SEARCH, "First.")
    agent = SynthesisAgent(llm_client=llm, librarian=_FakeLibrarian())
    agent.run("first question")
    assert agent.last_run_debug["action"] == "search"

    class _BrokenLLM:
        def chat_completion(self, messages, **kwargs):
            raise RuntimeError("model down")

    agent.llm = _BrokenLLM()
    with pytest.raises(RuntimeError, match="model down"):
        agent.run("second question")

    assert agent.last_run_debug == {}


def test_spans_are_traced_in_order() -> None:
    tracer = _RecordingTracer()
    SynthesisAgent(
        llm_client=_FakeLLM(_ROUTE_SEARCH, "Answer."),
        librarian=_FakeLibrarian(),
        tracer=tracer,
    ).run("q")
    assert tracer.spans == ["synthesis.run", "synthesis.route", "synthesis.summarize"]

    tracer = _RecordingTracer()
    SynthesisAgent(llm_client=_FakeLLM("Answer."), tracer=tracer).run("q", [_passage()])
    assert tracer.spans == ["synthesis.run", "synthesis.summarize"]


# ── the router prompt ────────────────────────────────────────────────────────


def test_router_prompt_names_no_language_and_asks_for_one() -> None:
    """The language rule must state "mirror the question" without an example.

    A named example primes the model to answer in that language whatever the
    question is written in.
    """
    text = (ROOT / "librarian" / "prompts" / "router.md").read_text(encoding="utf-8")

    for name in ("Italian", "German", "French", "Spanish", "Portuguese", "Chinese"):
        assert name not in text, f"router.md names {name!r}"
    assert '"language"' in text
    assert "{history_block}" in text and "{query}" in text


# ── the benchmark subclass keeps working ─────────────────────────────────────


def test_numbered_citation_subclass_still_builds_its_prompt() -> None:
    path = ROOT / "evals/Literature/SQA_bench/numbered_citations.py"
    spec = importlib.util.spec_from_file_location("numbered_citations_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    llm = _FakeLLM("Answer [1].", "Answer [1].")
    agent = module.NumberedCitationSynthesisAgent(llm_client=llm)

    assert agent.run("q", [_passage()]) == "Answer [1]."
    assert "[REF_1] Paper #1:" in llm.prompt(0)
    assert "(none — first turn)" in llm.prompt(0)

    agent.run("q", [_passage()], conversation_history=_HISTORY)
    assert "user: Tell me about metformin and ageing." in llm.prompt(1)

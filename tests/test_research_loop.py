"""The research loop's ledger, planning rounds and final selection, offline.

uv run pytest tests/test_research_loop.py
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from librarian.agent import RunResult
from librarian.config import load_research_loop_config, load_runtime_config
from librarian.llm_client import LLMClient
from librarian.research_loop import (
    DEFAULT_LOOP_PROMPT_PATH,
    ResearchLoop,
    merge_passages,
)

# The shipped prompt's placeholders, with a marker between the two evidence blocks.
TEMPLATE = (
    "{research_question}\nPREVIOUS\n{previous_searches}\nLAST\n{last_searches}\n"
    "{eligible_context_pmids} {max_context_queries} {max_specific_queries}"
)


def _paper(pmid: str) -> dict:
    """One librarian paper with a single snippet naming its PMID."""
    return {
        "pmid": pmid,
        "evidence_snippets": [f"evidence {pmid}"],
        "evidence_sections": ["Abstract"],
    }


class FakeRetriever:
    """Answers each question from a fixed table; a missing question raises."""

    def __init__(self, papers_by_question: dict):
        self.papers_by_question = papers_by_question

    def run(self, query: str) -> RunResult:
        """Return the table's papers for *query*."""
        papers = [_paper(pmid) for pmid in self.papers_by_question[query]]
        return RunResult(papers=papers, debug={"search_queries": [query]})


class ScriptedPlanner:
    """Replies in order and keeps every prompt it was sent."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.prompts: list = []

    def chat_completion(self, messages, temperature, max_tokens, timeout) -> str:
        """Record the prompt, return the next scripted reply."""
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)


def _reply(context=(), specific=(), pmids=()) -> str:
    """A planner reply as JSON text."""
    payload = {
        "context_queries": list(context),
        "specific_queries": list(specific),
        "selected_pmids": list(pmids),
    }
    return json.dumps(payload)


def _loop(papers_by_question: dict, replies: list, **overrides) -> tuple:
    """A loop over fake retrievers with two rounds and three papers per specific."""
    config = replace(
        load_research_loop_config(),
        max_rounds=2,
        specific_max_papers=3,
        planner_max_retries=2,
        **overrides,
    )
    retriever = FakeRetriever(papers_by_question)
    retrievers = {"seed": retriever, "context": retriever, "specific": retriever}
    planner = ScriptedPlanner(replies)
    return ResearchLoop(retrievers, planner, TEMPLATE, config), planner


def test_ledger_rounds_and_final_selection():
    papers = {
        "seed": ["1", "2"],
        "C1": ["3", "4"],
        "S1": ["5", "6", "7", "8"],
        "C2": ["9", "2"],
    }
    replies = [
        _reply(context=["C1"], specific=["S1"], pmids=["1"]),
        # 5 is specific-only and 999 invented: both ignored, the round goes on.
        _reply(context=["C2"], pmids=["4", "5", "999"]),
        _reply(pmids=["2"]),
    ]
    loop, planner = _loop(papers, replies)
    result = loop.run("seed")

    # The specific result keeps its first three papers in the judge's ranking.
    specific = result.specific_searches[0]
    assert [paper["pmid"] for paper in specific.passages] == ["5", "6", "7"]
    assert specific.debug["paper_selection"]["omitted_pmids"] == ["8"]
    # Round 2 sees the seed through its own selection only, and S1 as latest.
    previous, latest = planner.prompts[1].split("LAST")
    assert "<pmid_1>" in previous and "<pmid_2>" not in previous
    assert "<pmid_3>" in latest and "<pmid_5>" in latest
    assert '["3", "4"] 16 64' in latest
    # S1 moved to the history, so the final review sees only C2 as latest.
    previous, latest = planner.prompts[2].split("LAST")
    assert "<pmid_5>" in previous and "<pmid_5>" not in latest
    assert '["2", "9"] 0 0' in latest
    # The seed's paper 2 stays discarded although C2's copy was selected later.
    assert [s.selected_pmids for s in result.context_searches] == [["1"], ["4"], ["2"]]
    assert result.selected_pmids == ["1", "4", "2"]
    assert [p["pmid"] for p in result.passages] == ["1", "4", "2", "5", "6", "7"]
    assert result.final_selection["round"] == 3


def test_round_without_questions_counts_and_skips_final_review():
    replies = [_reply(pmids=["1"]), _reply()]
    loop, planner = _loop({"seed": ["1"]}, replies)
    result = loop.run("seed")
    assert len(planner.prompts) == 2
    assert result.final_selection is None
    assert [p["pmid"] for p in result.passages] == ["1"]


def test_invalid_replies_are_retried_twice_then_raise():
    valid = _reply(context=["seed", " SEED "], pmids=["1"])
    loop, _ = _loop(
        {"seed": ["1"]}, ["not json", '{"context_queries": []}', valid, _reply()]
    )
    result = loop.run("seed")
    trace = result.planning[0]
    assert trace["attempt"] == 3 and len(trace["previous_attempts"]) == 2
    # The seed question, however spelled, is never asked again.
    assert trace["context_queries"] == []

    over_budget = _reply(context=["A", "B"])
    loop, _ = _loop({"seed": ["1"]}, [over_budget] * 3, max_context_queries=1)
    with pytest.raises(ValueError, match="budget") as raised:
        loop.run("seed")
    # The partial result survives, with the seed and every failed attempt.
    partial = raised.value.loop_result
    assert partial.context_searches[0].question == "seed"
    assert len(partial.planner_failure["failed_attempts"]) == 3


def test_planner_client_ignores_reasoning_content(monkeypatch):
    choice = {"message": {"content": "", "reasoning_content": '{"draft": 1}'}}
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"choices": [choice]},
    )
    monkeypatch.setattr(
        "librarian.llm_client.requests.post", lambda *args, **kwargs: response
    )
    messages = [{"role": "user", "content": "q"}]
    assert LLMClient(model_name="m").chat_completion(messages) == '{"draft": 1}'
    planner = LLMClient(model_name="m", content_only=True)
    assert planner.chat_completion(messages) == ""


def test_failed_retrieval_is_shown_and_can_be_asked_again():
    replies = [_reply(context=["missing"]), _reply(context=["missing"]), _reply()]
    loop, planner = _loop({"seed": ["1"]}, replies)
    result = loop.run("seed")
    assert result.context_searches[1].error.startswith("KeyError")
    # The planner sees the failure, not an empty result, and may ask again.
    latest = planner.prompts[1].split("LAST")[1]
    assert "<query>\nmissing\n</query>\n<evidences>\nretrieval failed" in latest
    assert result.planning[1]["context_queries"] == ["missing"]

    loop, _ = _loop({}, [])
    with pytest.raises(RuntimeError, match="seed retrieval failed"):
        loop.run("seed")


def test_merge_keeps_each_snippet_with_its_section():
    first = {"pmid": "1", "evidence_snippets": ["a"], "evidence_sections": ["Abstract"]}
    second = {
        "pmid": "1",
        "evidence_snippets": ["a", "b"],
        "evidence_sections": ["Abstract", "Supplementary: S1.pdf"],
    }
    (merged,) = merge_passages([[first], [second]])
    assert merged["evidence_snippets"] == ["a", "b"]
    assert merged["evidence_sections"] == ["Abstract", "Supplementary: S1.pdf"]
    # The per-question record is untouched.
    assert first["evidence_sections"] == ["Abstract"]


def test_prompt_contract_and_config():
    loop_config = load_research_loop_config()
    retrievers = {"seed": None, "context": None, "specific": None}
    with pytest.raises(ValueError, match="last_searches"):
        ResearchLoop(retrievers, None, "{previous_searches}", loop_config)
    shipped = DEFAULT_LOOP_PROMPT_PATH.read_text(encoding="utf-8")
    ResearchLoop(retrievers, None, shipped, loop_config)
    # The [research_loop] table leaves the runtime config loadable.
    assert load_runtime_config().num_subqueries >= 3
    assert loop_config.planner_max_retries == 2

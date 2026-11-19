"""A failed sub-query search fails only that sub-query, not the whole run.

uv run --extra supplementary pytest tests
"""

import json
import re
from typing import Any, Dict, List

import pytest

from librarian import LibrarianAgent, load_runtime_config
from librarian.literature_search import EuropePmcSource, LiteratureSearchError


class PlannerAndJudgeLLM:
    """Plans the given sub-queries, then judges every sentence relevant."""

    def __init__(self, subqueries: List[str]) -> None:
        self.subqueries = subqueries

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 0,
    ) -> str:
        """Return a search plan, or every sentence id when asked to judge."""
        prompt = messages[-1]["content"]
        sentence_ids = re.findall(r"paragraph_\d+_[A-Z]+", prompt)
        if sentence_ids:
            return json.dumps({"relevant_ids": sentence_ids})
        return json.dumps({"queries": self.subqueries, "ranking_terms": []})


class SourceFailingOn(EuropePmcSource):
    """One paper per sub-query; the search of each query in *failing* raises."""

    def __init__(self, failing: Dict[str, LiteratureSearchError]) -> None:
        self.failing = failing

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Raise for a failing query, else one paper named after the query."""
        if query in self.failing:
            raise self.failing[query]
        return [
            {"pmid": f"pmid-{query}", "title": query, "abstract": f"{query} evidence."}
        ]


def _agent(
    subqueries: List[str], failing: Dict[str, LiteratureSearchError]
) -> LibrarianAgent:
    # Abstracts only: full-text fetching is not what these tests are about.
    return LibrarianAgent(
        load_runtime_config(),
        llm_client=PlannerAndJudgeLLM(subqueries),
        literature_source=SourceFailingOn(failing),
        full_text_enrichment=False,
    )


def test_one_failed_search_keeps_the_other_subqueries_papers():
    error = LiteratureSearchError("Europe PMC request failed: read timed out")
    agent = _agent(["kinase", "phosphatase"], {"phosphatase": error})

    result = agent.run("question")

    assert [paper["pmid"] for paper in result.papers] == ["pmid-kinase"]
    assert result.debug["failed_subqueries"] == [
        {"query": "phosphatase", "error": str(error)}
    ]


def test_run_fails_when_every_subquery_search_fails():
    first = LiteratureSearchError("Europe PMC is down")
    second = LiteratureSearchError("Europe PMC is still down")
    agent = _agent(["kinase", "phosphatase"], {"kinase": first, "phosphatase": second})

    with pytest.raises(LiteratureSearchError) as raised:
        agent.run("question")

    assert raised.value is first
    assert raised.value.run_debug["failed_subqueries"] == [
        {"query": "kinase", "error": str(first)},
        {"query": "phosphatase", "error": str(second)},
    ]

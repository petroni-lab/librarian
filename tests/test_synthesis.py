"""Self-check for SynthesisAgent and the shared citation helpers.

Run with: uv run python tests/test_synthesis.py

No LLM backend and no network: the librarian and the LLM client are both stubs,
so this checks the wiring the agent owns — placeholder substitution, the
no-papers short circuit, and the citation keys the answer must agree with.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from librarian.citations import citation_keys, render_papers  # noqa: E402
from librarian.tracing_port import NullTracer  # noqa: E402
from librarian.synthesis import (  # noqa: E402
    _NO_PAPERS_MESSAGE,
    _SUMMARIZER_PROMPT_PATH,
    SynthesisAgent,
)

PASSAGES: List[Dict[str, Any]] = [
    {
        "title": "Prime editing in primary cells",
        "authors": "Chen X, Wang Y",
        "journal": "Nature",
        "year": "2023",
        "pmid": "11111111",
        "doi": "10.1/a",
        "url": "https://europepmc.org/article/MED/11111111",
        "has_fulltext": True,
        "evidence_snippets": ["First span.", "Second span."],
    },
    {
        # Same first author and year as above: the keys must diverge.
        "title": "Prime editing follow-up",
        "authors": "Chen X, Other Z",
        "journal": "Cell",
        "year": "2023",
        "pmid": "22222222",
        "doi": "",
        "url": "https://europepmc.org/article/MED/22222222",
        "has_fulltext": False,
        "evidence_snippets": ["Only span."],
    },
]


class _StubLLM:
    """Records the prompt it was handed and returns a canned answer."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def chat_completion(self, messages, **_kwargs) -> str:
        self.calls.append(messages[-1]["content"])
        return "  Executive Summary\nAn answer [Chen 2023a](url).  "


class _StubLibrarian:
    """Returns ``passages`` verbatim and records the progress messages it saw."""

    def __init__(self, passages: List[Dict[str, Any]], with_debug: bool = True) -> None:
        self._passages = passages
        self.progress: List[str] = []
        if with_debug:
            self.last_run_debug = {"search_queries": ["prime editing"]}

    def run(self, query: str, on_progress=None) -> List[Dict[str, Any]]:
        if on_progress:
            on_progress("Searching Europe PMC")
        return self._passages


def _agent_with(librarian: Any) -> SynthesisAgent:
    """A synthesis agent wired to ``librarian``, with the LLM stubbed out."""
    agent = object.__new__(SynthesisAgent)
    agent.librarian = librarian
    agent.verbose = False
    agent._tracer = NullTracer()
    agent.llm = _StubLLM()
    agent._summarizer_prompt = _SUMMARIZER_PROMPT_PATH.read_text(encoding="utf-8")
    return agent


def test_no_papers_skips_the_llm() -> None:
    """An empty result is answered without paying for a generation call."""
    agent = _agent_with(_StubLibrarian([]))

    result = agent.run("prime editing")

    assert result["summary"] == _NO_PAPERS_MESSAGE
    assert result["passages"] == []
    assert agent.llm.calls == [], "no LLM call may be made when nothing was retrieved"


def test_missing_last_run_debug_is_survivable() -> None:
    """last_run_debug is unset until Stage 3 completes; that must not raise."""
    agent = _agent_with(_StubLibrarian([], with_debug=False))

    assert agent.run("prime editing")["search_queries"] == []


def test_every_placeholder_is_substituted() -> None:
    """No ``{placeholder}`` may survive into the prompt the model receives."""
    agent = _agent_with(_StubLibrarian(PASSAGES))

    result = agent.run("does prime editing work in primary cells?")
    prompt = agent.llm.calls[0]

    for token in (
        "{today_date}",
        "{today_year}",
        "{output_channel}",
        "{formatting_guidance}",
        "{conversation_history}",
        "{user_query}",
        "{answer_language}",
        "{papers_text}",
    ):
        assert token not in prompt, f"{token} left unsubstituted"
    assert "does prime editing work in primary cells?" in prompt
    assert result["summary"] == "Executive Summary\nAn answer [Chen 2023a](url)."
    assert result["search_queries"] == ["prime editing"]


def test_substitution_is_single_pass() -> None:
    """A placeholder inside the user's question is never itself expanded."""
    agent = _agent_with(_StubLibrarian(PASSAGES))

    agent.run("what about {papers_text}?")

    # The literal token survives as typed; it must not have pulled in the papers.
    assert "what about {papers_text}?" in agent.llm.calls[0]


def test_progress_reaches_both_stages() -> None:
    """Retrieval and synthesis report to the same callback."""
    seen: List[str] = []
    agent = _agent_with(_StubLibrarian(PASSAGES))

    agent.run("prime editing", on_progress=seen.append)

    assert seen == ["Searching Europe PMC", "Synthesizing answer"]


def test_papers_text_carries_a_cite_as_line_per_paper() -> None:
    """The summarizer copies these verbatim, so every paper must have one."""
    rendered = render_papers(PASSAGES)

    assert rendered.count("- Cite as: ") == len(PASSAGES)
    assert "- Evidence: First span." in rendered


def test_shared_author_year_keys_diverge() -> None:
    """Two papers sharing first author and year take distinct a/b suffixes."""
    assert citation_keys(PASSAGES) == ["Chen 2023a", "Chen 2023b"]


def test_prompt_names_no_concrete_language() -> None:
    """An example language primes the model to answer in it, whatever was asked.

    The rule must say "mirror the question's language" without ever naming one.
    """
    prompt = _SUMMARIZER_PROMPT_PATH.read_text(encoding="utf-8")

    for name in (
        "Italian",
        "German",
        "French",
        "Spanish",
        "Portuguese",
        "Chinese",
        "Japanese",
    ):
        assert name not in prompt, (
            f"summarizer.md names {name!r}: an example language primes the model "
            "to answer in it. State the mirroring rule without naming one."
        )


if __name__ == "__main__":
    test_no_papers_skips_the_llm()
    test_missing_last_run_debug_is_survivable()
    test_every_placeholder_is_substituted()
    test_substitution_is_single_pass()
    test_progress_reaches_both_stages()
    test_papers_text_carries_a_cite_as_line_per_paper()
    test_shared_author_year_keys_diverge()
    test_prompt_names_no_concrete_language()
    print("Synthesis agent, prompt substitution, and citation keys passed.")

"""Offline replay of a recorded Librarian run: does the deterministic logic still hold?

Each case under ``fixtures/integration/`` records what the agent's external calls
answered during one live run (see ``fixture_replay``). These tests hand those
answers back through the agent's ``llm_client`` and ``literature_source`` and call
the real ``LibrarianAgent.run``, so everything between the external calls runs for
real and has to land where it landed when recorded: sub-query parsing and
validation, Stage 2's thread fan-out, BM25 ranking and paragraph merging, Stage 3's
sentence registry, batching and span regrouping, and evidence extraction.

Only ``test_run_reproduces_recorded_evidence`` states the contract. Every other
test narrows down *where* the pipeline diverged when that one fails, so a
regression names its stage instead of dumping a final diff.

To refresh a case after an intended change, re-record it:

    uv run python tests/record_fixture.py \\
      --case <name> --query "<question>" --overwrite
"""

from __future__ import annotations

import copy
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import fixture_replay
import pytest

from librarian.agent import _QUERY_PROMPT_PATH, LibrarianAgent
from librarian.literature_search import Fulltext, normalize_pmcid

# Collected at import so every case parametrizes the suite; adding a case is a
# recording, not an edit here.
_RECORDED_CASES = fixture_replay.available_cases()

# Prompt slots are ``{lower_snake_case}`` (e.g. ``{query_budget_guidance}``,
# ``{today_date}``); the JSON examples in the prompt files also use braces, and
# this pattern does not match them.
_PROMPT_SLOT = re.compile(r"\{[a-z][a-z0-9_]*\}")


def prompt_slots(*templates: str) -> list[str]:
    """Every ``{slot}`` appearing in the given prompt templates, deduplicated.

    Derived from the templates themselves rather than listed by hand, so a slot
    added to a prompt file is covered without editing this test.

    :param templates: Raw prompt template text (and any config-supplied fragment
        substituted into it, such as ``query_budget_guidance``).
    :returns: The distinct slot names, in sorted order.
    """
    found = {slot for template in templates for slot in _PROMPT_SLOT.findall(template)}
    return sorted(found)


@dataclass
class ReplayedCalls:
    """What the pipeline asked its boundaries for, checked once ``run`` has returned."""

    llm_requests: list[dict[str, Any]] = field(default_factory=list)
    searched_subqueries: list[str] = field(default_factory=list)
    fetched_pmcids: list[str] = field(default_factory=list)
    # Stage 2 and Stage 3 call the boundaries from thread pools, as the recorder's
    # proxies are; the log is shared across those threads.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def requests_for(self, key: str) -> list[dict[str, Any]]:
        """Every LLM request the run made under ``key``."""
        return [request for request in self.llm_requests if request["key"] == key]


class ReplayLLM:
    """An ``llm_client`` that answers each call with its recorded response.

    The test-only half of the LLM boundary: ``RecordingLLM`` (in
    ``record_fixture.py``) recorded what this one answers with.
    """

    def __init__(self, case: fixture_replay.RecordedCase, calls: ReplayedCalls):
        self._case = case
        self._calls = calls
        self._responses = {call["key"]: call["response"] for call in case.llm_calls}

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: Any = None,
        max_tokens: Any = None,
    ) -> str:
        """Return the recorded answer for this call, or say which calls were recorded."""
        key = fixture_replay.llm_call_key(messages)
        with self._calls.lock:
            self._calls.llm_requests.append(
                {
                    "key": key,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
        if key not in self._responses:
            raise fixture_replay.MissingRecording(
                f"no recorded LLM response for {key!r}. Recorded: "
                f"{sorted(self._responses)}. The agent's prompts or Stage 3 "
                f"batching changed — re-record {self._case.directory.name}."
            )
        return self._responses[key]


class ReplaySource:
    """A ``LiteratureSource`` that answers from the recording, logging what was asked.

    The test-only half of the literature boundary: ``RecordingSource`` (in
    ``record_fixture.py``) recorded what this one answers with.

    A replayed value is deep-copied on the way out: a live search returns fresh
    dicts for every sub-query, and Stage 3 writes ``evidence_sentences_full_text``
    onto the paper dicts it keeps, so handing the same object to two calls would
    not reproduce the recorded run.
    """

    def __init__(self, case: fixture_replay.RecordedCase, calls: ReplayedCalls):
        self._case = case
        self._calls = calls

    def search(self, query: str, page_size: int) -> list[dict[str, Any]]:
        """Return the papers this sub-query really returned, as fresh dicts."""
        # page_size only decides how many papers the live API sends back, which the
        # recording has already fixed; accepted to match the real signature.
        del page_size
        with self._calls.lock:
            self._calls.searched_subqueries.append(query)
        if query not in self._case.search_results:
            raise fixture_replay.MissingRecording(
                f"no recorded Europe PMC search for {query!r}. Recorded: "
                f"{sorted(self._case.search_results)}. Stage 1 planned a different "
                f"sub-query — re-record {self._case.directory.name}."
            )
        return copy.deepcopy(self._case.search_results[query])

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> dict[str, Fulltext]:
        """Return this batch's recorded JATS XML, wrapped as the real call returns it.

        Mirrors ``EuropePmcSource.fetch_fulltext_many``'s contract: blank ids
        (papers with no free full text, per ``_fulltext_pmcid``) are dropped, the
        rest are deduplicated and keyed by ``normalize_pmcid`` — so a caller
        relying on that keying behaves the same in replay as live.
        """
        unique_pmcids = dict.fromkeys(
            normalize_pmcid(p) for p in pmcids if str(p or "").strip()
        )
        results: dict[str, Fulltext] = {}
        for pmcid in unique_pmcids:
            with self._calls.lock:
                self._calls.fetched_pmcids.append(pmcid)
            xml_text = self._case.fulltext(pmcid)
            if xml_text:
                results[pmcid] = Fulltext(pmcid=pmcid, xml=xml_text, status=200)
            else:
                results[pmcid] = Fulltext(
                    pmcid=pmcid, status=None, error="recorded as failed"
                )
        return results

    def fetch_supplementary_many(self, pmcids: Iterable[str], options: Any) -> Any:
        """Supplementary files are not recorded yet, so any call is a gap.

        Recordings pin ``supplementary_enrichment = false``; reaching this means a
        case was recorded with it on, or the agent started calling it regardless.
        """
        raise fixture_replay.MissingRecording(
            f"fetch_supplementary_many called for {sorted(pmcids)}, but recordings "
            f"do not capture supplementary files. Record "
            f"{self._case.directory.name} with supplementary_enrichment = false."
        )


@dataclass(frozen=True, repr=False)
class ReplayedRun:
    """One recorded case, replayed once: the agent, what it asked, what it returned."""

    case: fixture_replay.RecordedCase
    agent: LibrarianAgent
    calls: ReplayedCalls
    stages: fixture_replay.StageCapture
    evidence: list[dict[str, Any]]

    def __repr__(self) -> str:
        """Short: pytest prints every fixture value in a failure header."""
        return f"<ReplayedRun {self.case.directory.name}>"


@pytest.fixture(
    params=_RECORDED_CASES or [None],
    ids=lambda path: path.name if path else "no-cases",
)
def case(request: pytest.FixtureRequest) -> fixture_replay.RecordedCase:
    """One recorded case, skipped by name when its recording is absent or partial."""
    case_dir = request.param
    if case_dir is None:
        pytest.skip(f"no recorded cases under {fixture_replay.CASE_ROOT}")
    missing = fixture_replay.missing_files(case_dir)
    if missing:
        pytest.skip(
            f"{case_dir.name}: incomplete recording, missing {missing}. "
            "Re-record with tests/record_fixture.py"
        )
    loaded_case = fixture_replay.load_case(case_dir)
    stemming_now = fixture_replay.bm25_stemming_enabled()
    if loaded_case.bm25_stemming != stemming_now:
        pytest.skip(
            f"{case_dir.name}: recorded with bm25_stemming="
            f"{loaded_case.bm25_stemming}, this process has {stemming_now}. "
            "Check PyStemmer and LITERATURE_BM25_STEMMING."
        )
    return loaded_case


@pytest.fixture
def replayed(
    case: fixture_replay.RecordedCase, monkeypatch: pytest.MonkeyPatch
) -> ReplayedRun:
    """Run the recorded query once against the recorded boundaries.

    Every test below reads this same run, so the pipeline executes once per case
    and each test is an assertion about it rather than a separate replay.
    """
    calls = ReplayedCalls()
    agent = fixture_replay.build_agent(
        case,
        llm_client=ReplayLLM(case, calls),
        literature_source=ReplaySource(case, calls),
    )
    stages = fixture_replay.StageCapture()
    stages.install(agent, monkeypatch.setattr)
    try:
        evidence = agent.run(case.query).papers
    except fixture_replay.MissingRecording as gap:
        # Re-raised as pytest's own outcome: a bare BaseException escaping a
        # parametrized fixture corrupts pytest's fixture cache, and every later
        # case then errors with an unrelated AssertionError.
        pytest.fail(str(gap), pytrace=False)
    return ReplayedRun(case, agent, calls, stages, evidence)


def test_run_reproduces_recorded_evidence(replayed: ReplayedRun) -> None:
    """The whole pipeline still turns the recorded answers into the recorded evidence.

    This is the contract. A failure here means some deterministic step changed; the
    stage tests below say which one.
    """
    actual = replayed.evidence
    expected = replayed.case.expected(fixture_replay.EVIDENCE_FILE)
    assert actual == expected


def test_stage1_planning_matches_recording(replayed: ReplayedRun) -> None:
    """Stage 1: the planner's parsing, dedup and cap, then Europe PMC validation.

    The recorded LLM text is replayed verbatim, so a difference here is in
    ``_generate_queries``'s own handling of it or in ``_validate_queries``.
    """
    case = replayed.case

    actual_generated = replayed.stages.generated_queries
    expected_generated = case.expected(fixture_replay.GENERATED_QUERIES_FILE)
    assert actual_generated == expected_generated

    actual_validated = replayed.stages.validated_queries
    expected_validated = case.expected(fixture_replay.VALIDATED_QUERIES_FILE)
    assert actual_validated == expected_validated


def test_stage1_asks_the_llm_the_way_it_was_recorded(replayed: ReplayedRun) -> None:
    """Stage 1 made as many planning and strict-JSON retry calls as the live run.

    A replayed answer is returned whatever the request looked like, so how the
    agent asked can only be checked against the calls it made. A retry starting
    (or stopping) to fire would otherwise pass unnoticed, since a recording holds
    an answer for whichever branch the live run took.
    """
    for key in (fixture_replay.QUERY_GENERATION_KEY, fixture_replay.QUERY_RETRY_KEY):
        recorded = [call for call in replayed.case.llm_calls if call["key"] == key]
        actual = replayed.calls.requests_for(key)
        assert len(actual) == len(recorded), key


def test_llm_calls_use_the_recorded_sampling(replayed: ReplayedRun) -> None:
    """Every LLM call asked with the temperature and ``max_tokens`` it was recorded with.

    The replayed answer is the same whatever was asked, so a drifted budget would
    pass unnoticed here and only show live: ``_generate_queries`` documents that a
    truncated planner answer collapses retrieval to one raw-question search, and
    ``config.py`` that a too-small judge budget returns no ids at all.
    """
    recorded_sampling = {
        call["key"]: {
            "temperature": call["temperature"],
            "max_tokens": call["max_tokens"],
        }
        for call in replayed.case.llm_calls
    }
    for request in replayed.calls.llm_requests:
        key = request["key"]
        actual_sampling = {
            "temperature": request["temperature"],
            "max_tokens": request["max_tokens"],
        }
        assert actual_sampling == recorded_sampling[key], key


def test_stage1_prompt_is_fully_substituted(replayed: ReplayedRun) -> None:
    """No ``{slot}`` from the planner's prompt survives into the message sent.

    The slot list is read from the prompt file and the configured budget guidance
    rather than listed here, so a slot added to either is covered without touching
    this test. A dropped or misspelled substitution reaches the LLM silently
    otherwise, since the replayed answer is the same either way.
    """
    template = _QUERY_PROMPT_PATH.read_text(encoding="utf-8")
    guidance = replayed.case.runtime_config.query_budget_guidance
    slots = prompt_slots(template, guidance)
    assert slots, "no prompt slots found — the slot pattern stopped matching"

    request = replayed.calls.requests_for(fixture_replay.QUERY_GENERATION_KEY)[0]
    user_prompt = request["messages"][1]["content"]
    assert replayed.case.query in user_prompt, "recorded question missing from prompt"
    unsubstituted = [slot for slot in slots if slot in user_prompt]
    assert unsubstituted == [], f"unsubstituted prompt slots: {unsubstituted}"


def test_stage2_searches_every_validated_subquery(replayed: ReplayedRun) -> None:
    """Stage 2 searched each validated sub-query exactly once, and nothing else.

    Pins the fan-out itself: a sub-query silently dropped, duplicated, or rewritten
    before the search shows up here rather than as a smaller paragraph pool.
    """
    assert sorted(replayed.calls.searched_subqueries) == sorted(
        replayed.stages.validated_queries
    )


def test_stage2_selects_the_recorded_paragraphs(replayed: ReplayedRun) -> None:
    """Stage 2: full-text chunking and the BM25 ranking each sub-query selects from.

    Compared per sub-query, as one line per selection (see ``paragraph_line``), so
    a failure names the paragraph that moved rather than printing whole Europe PMC
    records.
    """
    actual = replayed.stages.subquery_paragraphs_in_order(
        replayed.stages.validated_queries
    )
    expected = replayed.case.expected(fixture_replay.SUBQUERY_PARAGRAPHS_FILE)
    assert [item["query"] for item in actual] == [item["query"] for item in expected]
    for actual_item, expected_item in zip(actual, expected):
        subquery = actual_item["query"]
        assert actual_item["paragraphs"] == expected_item["paragraphs"], subquery


def test_stage2_merges_into_the_recorded_pool(replayed: ReplayedRun) -> None:
    """Stage 2: the pool Stage 3 judges, after overlapping selections are merged.

    ``test_paragraph_merge`` covers the union rules themselves; this pins the pool
    one real run arrives at, including the order the sub-queries contribute in.
    """
    actual = replayed.stages.merged_paragraphs
    expected = replayed.case.expected(fixture_replay.MERGED_PARAGRAPHS_FILE)
    assert actual == expected


def test_stage3_cites_the_recorded_papers(replayed: ReplayedRun) -> None:
    """Stage 3: sentence ids, judge batching, and regrouping citations into spans.

    The judge's raw answers are replayed, so a difference here is in the sentence
    registry, the batch split, or ``_papers_from_cited_sentences`` — not the LLM.
    """
    actual = replayed.stages.relevant_papers
    expected = replayed.case.expected(fixture_replay.RELEVANT_PAPERS_FILE)
    assert actual == expected

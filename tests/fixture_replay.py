"""What a recorded Librarian case is: its files, its keys, and its stage capture.

``LibrarianAgent.run`` reaches the outside world only through the two
collaborators it is constructed with:

- ``llm_client.chat_completion`` — Stage 1 sub-query planning and Stage 3 judging,
- ``literature_source.search`` — Stage 2's Europe PMC search,
- ``literature_source.fetch_fulltext_many`` — Stage 2's JATS full-text fetch.

(``literature_source.fetch_supplementary_many`` is a fourth call, made only when
``supplementary_enrichment`` is on. Recordings pin it off; see ``recording.toml``.)

Everything between those calls is the agent's own deterministic logic. Recording
passes in collaborators that forward to the real ones and keep what they returned,
then runs ``run`` for real; replaying passes in collaborators that answer from the
recording, then runs ``run`` for real again. Neither side re-implements a stage, so
a fixture always describes what the agent actually does rather than what a
recording script believed it did.

This module holds only what both directions must agree on: the case file layout,
:func:`llm_call_key` (so a recorded call and a replayed request name it the same
way), and :class:`StageCapture` (so a stage file always means the same thing on
the way in as on the way out). ``record_fixture.py`` and ``test_recorded_run.py``
each additionally define their own one-way half of the recording/replay pair —
writing boundary answers, or feeding them back in — since that half is never
shared.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from librarian import agent as librarian_module
from librarian.agent import LibrarianAgent
from librarian.config import LibrarianRuntimeConfig
from librarian.literature_search import LiteratureSource, normalize_pmcid

CASE_ROOT = Path(__file__).parent / "fixtures" / "integration"

# Stage files say what the pipeline produced, and exist to localize a failure.
RUN_FILE = "00_run.json"
GENERATED_QUERIES_FILE = "01_generated_queries.json"
VALIDATED_QUERIES_FILE = "02_validated_queries.json"
SUBQUERY_PARAGRAPHS_FILE = "03_subquery_paragraphs.json"
MERGED_PARAGRAPHS_FILE = "04_merged_paragraphs.json"
RELEVANT_PAPERS_FILE = "05_relevant_papers.json"
EVIDENCE_FILE = "06_evidence.json"

# Boundary files say what the outside world answered, and are what makes an
# offline ``run`` possible at all.
LLM_CALLS_FILE = "boundaries/llm_calls.json"
SEARCH_RESULTS_FILE = "boundaries/search_results.json"
FULLTEXT_DIR = "boundaries/fulltext"

# Every file (and the full-text directory) a complete recording has. A partial
# directory is a failed recording, which the tests report as such instead of
# failing somewhere further in.
CASE_FILES = (
    RUN_FILE,
    GENERATED_QUERIES_FILE,
    VALIDATED_QUERIES_FILE,
    SUBQUERY_PARAGRAPHS_FILE,
    MERGED_PARAGRAPHS_FILE,
    RELEVANT_PAPERS_FILE,
    EVIDENCE_FILE,
    LLM_CALLS_FILE,
    SEARCH_RESULTS_FILE,
)

# ``setattr`` while recording, ``monkeypatch.setattr`` while testing — the stage
# capture is identical apart from whether the patch is undone.
SetAttr = Callable[[Any, str, Any], None]

# How the agent's three ``chat_completion`` call sites are told apart using only
# their request. Each marker is a literal from the matching system prompt in
# ``librarian/agent.py``, and each must match positively: a request matching none
# of them raises rather than being answered as one of the known calls.
_JUDGE_SYSTEM_MARKER = "relevant_ids"
# The planner's whole system prompt; the judge's starts with the same sentence, so
# this is compared for equality rather than containment.
_QUERY_GENERATION_SYSTEM_PROMPT = "You are a helpful assistant."
_QUERY_RETRY_SYSTEM_MARKER = "'queries' array"
_BATCH_PARAGRAPH_ID = re.compile(r'"id": "(paragraph_\d+)"')

QUERY_GENERATION_KEY = "query_generation"
QUERY_RETRY_KEY = "query_generation_retry"


class MissingRecording(BaseException):
    """The pipeline asked its boundaries for something the fixture never recorded.

    Deliberately a ``BaseException`` and not an ``Exception``: ``_evaluate_batch``
    catches ``Exception`` around its LLM call and degrades to an empty response, so
    an ordinary error here would be swallowed and a missing recording would surface
    as a quietly smaller result instead of a failure naming the gap.
    """


def llm_call_key(messages: list[dict[str, str]]) -> str:
    """Stable id for one ``chat_completion`` call, derived from its request alone.

    Stage 3's judge calls run concurrently, so the order they arrive in is not
    reproducible; they are keyed instead by the paragraph ids in the batch, which
    also stays correct when ``_evaluate_batch`` splits a batch after a bad
    response. No part of the key depends on the dates the prompts embed, so a
    fixture stays replayable long after the day it was recorded.

    :param messages: The chat messages as passed to ``chat_completion``.
    :returns: The key this call is recorded and replayed under.
    :raises MissingRecording: if the request matches none of the known call sites,
        e.g. a new ``chat_completion`` call or a reworded system prompt.
    """
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    if _JUDGE_SYSTEM_MARKER in system_prompt:
        return "judge:" + ",".join(_BATCH_PARAGRAPH_ID.findall(user_prompt))
    if _QUERY_RETRY_SYSTEM_MARKER in system_prompt:
        return QUERY_RETRY_KEY
    if system_prompt == _QUERY_GENERATION_SYSTEM_PROMPT:
        return QUERY_GENERATION_KEY
    raise MissingRecording(
        f"unrecognised chat_completion call site (system prompt: "
        f"{system_prompt[:120]!r}). Teach llm_call_key to key it, then re-record."
    )


def bm25_stemming_enabled() -> bool:
    """Whether this process's BM25 stems tokens, which decides Stage 2's ranking.

    ``agent._BM25_STEMMER`` is fixed at import from ``LITERATURE_BM25_STEMMING`` and
    from whether ``PyStemmer`` imports, so it is recorded with each case and a
    replay under the other setting is skipped rather than failed.
    """
    return librarian_module._BM25_STEMMER is not None


def read_json(path: Path) -> Any:
    """Load one JSON file belonging to a recorded case."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write one readable, diff-stable JSON file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{rendered}\n", encoding="utf-8")


def fulltext_filename(pmcid: str) -> str:
    """Filename holding one paper's recorded JATS XML, keyed as ``fetch_fulltext_many`` keys it."""
    return f"{normalize_pmcid(pmcid)}.xml"


def available_cases() -> list[Path]:
    """Every recorded case directory, for parametrizing the tests over all of them."""
    if not CASE_ROOT.is_dir():
        return []
    return sorted(path for path in CASE_ROOT.iterdir() if path.is_dir())


def missing_files(case_dir: Path) -> list[str]:
    """Names of the case files a complete recording needs but this directory lacks."""
    missing = [name for name in CASE_FILES if not (case_dir / name).is_file()]
    fulltext_dir_present = (case_dir / FULLTEXT_DIR).is_dir()
    if not fulltext_dir_present:
        missing.append(FULLTEXT_DIR)
    return missing


def paragraph_line(paragraph: dict[str, Any]) -> str:
    """One Stage 2 selection as a line: where it came from and what it selects.

    Stage 2's fixtures record lines rather than whole paragraph records, for two
    reasons. Each record embeds the full Europe PMC paper dict, so a recorded pool
    ran to ~120 KB of mostly duplicated metadata; and pytest diffs a list of
    strings line by line while it truncates a list of nested dicts — the difference
    between a failure naming the paragraph that moved and one printing an elided
    blob. The text these coordinates select stays reproducible from the recorded
    boundaries, so nothing is lost by not storing it.

    :param paragraph: One Stage 2 paragraph record.
    :returns: Its source coordinates and a readable head of the selected text.
    """
    text = str(paragraph.get("text") or "")
    section_type = str(paragraph.get("section_type") or "body")
    section_title = str(paragraph.get("section_title") or "")
    return (
        f"{librarian_module._candidate_id(paragraph['paper'])}"
        f" | p{paragraph['source_paragraph_index']}"
        f" w{paragraph['word_start']}-{paragraph['word_end']}"
        f" | {len(text.split())} words"
        f" | {section_type}:{section_title}"
        f" | {text[:80]}"
    )


def judged_paper_lines(papers: list[dict[str, Any]]) -> list[str]:
    """Stage 3's ranked papers as one line per paper plus one per evidence span.

    Records only what Stage 3 itself decides — which papers the judge cited, the
    order it cited them in, and the spans its citations regrouped into. Every other
    field on a judged paper is Europe PMC metadata that arrived unchanged through
    the recorded search, so repeating it here would only add diff noise.

    :param papers: ``_relevance_filter``'s output, most to least relevant.
    :returns: One line per paper, each followed by a line per evidence span.
    """
    lines: list[str] = []
    for rank, paper in enumerate(papers):
        paper_id = librarian_module._candidate_id(paper)
        spans = paper.get("evidence_sentences_full_text") or []
        lines.append(f"{rank}. {paper_id} | {len(spans)} spans | {paper.get('title')}")
        lines.extend(
            f"{rank}. {paper_id} | span {index} | {span}"
            for index, span in enumerate(spans)
        )
    return lines


@dataclass(frozen=True)
class RecordedCase:
    """One recorded run: the inputs it started from and the answers it was given."""

    directory: Path
    query: str
    full_text_enrichment: bool
    bm25_stemming: bool
    runtime_config: LibrarianRuntimeConfig
    llm_calls: list[dict[str, Any]]
    search_results: dict[str, list[dict[str, Any]]]

    def expected(self, filename: str) -> Any:
        """The ground-truth value recorded for one stage, by its file name.

        Named for what every caller uses it for: the right-hand side of an
        ``actual == expected`` comparison against something the replayed run
        just produced. Never the source of anything replay feeds *into* the
        agent — that's ``query``, ``llm_calls``, and ``search_results`` above.
        """
        return read_json(self.directory / filename)

    def fulltext(self, pmcid: str) -> str:
        """Recorded JATS XML for ``pmcid``.

        A file that exists but is empty is a fetch that failed during recording,
        and replays as the failed fetch it was. A file that is absent means this
        paper was never fetched live, which is a divergence rather than a failure
        to reproduce — so it raises.

        :raises MissingRecording: if the run asked for an unrecorded paper.
        """
        path = self.directory / FULLTEXT_DIR / fulltext_filename(pmcid)
        if not path.is_file():
            raise MissingRecording(
                f"no recorded full text for {pmcid!r}. Stage 2 reached a paper the "
                f"live run never fetched — re-record {self.directory.name}."
            )
        return path.read_text(encoding="utf-8")


def load_case(case_dir: Path) -> RecordedCase:
    """Load one complete case directory.

    :param case_dir: A directory under ``fixtures/integration/``.
    :returns: The case, ready to rebuild an agent from and replay.
    """
    run = read_json(case_dir / RUN_FILE)
    return RecordedCase(
        directory=case_dir,
        query=run["query"],
        full_text_enrichment=run["full_text_enrichment"],
        bm25_stemming=run["bm25_stemming"],
        runtime_config=LibrarianRuntimeConfig.from_dict(run["runtime_config"]),
        llm_calls=read_json(case_dir / LLM_CALLS_FILE),
        search_results=read_json(case_dir / SEARCH_RESULTS_FILE),
    )


def build_agent(
    case: RecordedCase, *, llm_client: Any, literature_source: LiteratureSource
) -> LibrarianAgent:
    """Rebuild the agent that produced ``case``, around the given boundaries.

    Every other constructor argument comes from the recording, so the test cannot
    drift from how the fixture was made. Both collaborators are required: an agent
    built here never falls back to a live LLM or Europe PMC.

    :param case: The recorded case to rebuild.
    :param llm_client: Answers ``chat_completion`` — recording or replaying.
    :param literature_source: Answers search and full-text fetches, likewise.
    :returns: An agent configured exactly as the recorded one was.
    """
    return LibrarianAgent(
        runtime_config=case.runtime_config,
        full_text_enrichment=case.full_text_enrichment,
        llm_client=llm_client,
        literature_source=literature_source,
    )


@dataclass
class StageCapture:
    """What each pipeline stage really produced during one ``run``.

    Installed by both the recorder and the tests: the recorder writes these out,
    the tests compare them against what was written, and because it is one class
    the two can never disagree about which value a stage file holds.
    """

    generated_queries: list[str] = field(default_factory=list)
    validated_queries: list[str] = field(default_factory=list)
    # (sub-query, its selected paragraph lines), in the order Stage 2's threads
    # finished; re-ordered to the sub-query order by subquery_paragraphs_in_order.
    subquery_paragraphs: list[dict[str, Any]] = field(default_factory=list)
    merged_paragraphs: list[str] = field(default_factory=list)
    relevant_papers: list[str] = field(default_factory=list)

    def install(self, agent: LibrarianAgent, set_attribute: SetAttr = setattr) -> None:
        """Wrap each stage of ``agent`` so it reports its output as it runs.

        Every wrapper delegates to the real implementation and returns its value
        untouched — capturing is a side effect of a normal run, never a substitute
        for one.

        :param agent: The agent whose stages to observe.
        :param set_attribute: ``monkeypatch.setattr`` under pytest, so the patches
            are undone; plain ``setattr`` when recording a throwaway agent.
        """
        # Stage 2 fans out over a thread pool, so its capture is the one that needs
        # a lock; the rest are called once, from the calling thread.
        lock = threading.Lock()
        generate_queries = agent._generate_queries
        validate_queries = agent._validate_queries
        paragraphs_for_subquery = agent._paragraphs_for_subquery
        merge_selected = librarian_module._merge_selected_paragraphs
        relevance_filter = agent._relevance_filter

        def capture_generated(query: str, *args: Any, **kwargs: Any) -> Any:
            """Stage 1: the planner's parsed, deduplicated, capped sub-queries."""
            plan = generate_queries(query, *args, **kwargs)
            self.generated_queries = plan.queries
            return plan

        def capture_validated(queries: list[str]) -> list[str]:
            """Stage 1: the sub-queries that survived Europe PMC syntax validation."""
            self.validated_queries = validate_queries(queries)
            return self.validated_queries

        def capture_subquery(
            subquery: str, ranking_terms: Any = ()
        ) -> list[dict[str, Any]]:
            """Stage 2: one sub-query's top-k paragraphs, rendered before Stage 3.

            Rendered here rather than after the run because Stage 3 later writes
            ``evidence_sentences_full_text`` onto the very paper dicts these
            records point at, which would otherwise leak into a Stage 2 fixture.
            """
            paragraphs = paragraphs_for_subquery(subquery, ranking_terms)
            lines = [paragraph_line(paragraph) for paragraph in paragraphs]
            with lock:
                self.subquery_paragraphs.append(
                    {"query": subquery, "paragraphs": lines}
                )
            return paragraphs

        def capture_merged(
            paragraphs: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            """Stage 2: the merged pool Stage 3 judges, rendered for the same reason."""
            merged = merge_selected(paragraphs)
            self.merged_paragraphs = [paragraph_line(p) for p in merged]
            return merged

        def capture_relevant(
            query: str, paragraphs: list[dict[str, Any]], state: Any
        ) -> list[dict[str, Any]]:
            """Stage 3: which papers the judge cited, and the spans it cited for them."""
            papers = relevance_filter(query, paragraphs, state)
            self.relevant_papers = judged_paper_lines(papers)
            return papers

        set_attribute(agent, "_generate_queries", capture_generated)
        set_attribute(agent, "_validate_queries", capture_validated)
        set_attribute(agent, "_paragraphs_for_subquery", capture_subquery)
        set_attribute(librarian_module, "_merge_selected_paragraphs", capture_merged)
        set_attribute(agent, "_relevance_filter", capture_relevant)

    def subquery_paragraphs_in_order(self, queries: list[str]) -> list[dict[str, Any]]:
        """Stage 2's captures re-ordered to ``queries``, since the threads finish out of order."""
        by_query = {item["query"]: item for item in self.subquery_paragraphs}
        return [by_query[query] for query in queries]

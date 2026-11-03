"""Research loop: iterative retrieval over a PMID ledger, without synthesis.

For questions one retrieval cannot cover:

1. **Seed.** The first query gets one librarian run (``[research_loop.seed]``
   shape); all of its papers form the first unreviewed *context* batch.
2. **Rounds 1..max_rounds.** One planner call reads the history and the latest
   batch, selects the useful papers of the latest context searches into the
   ledger, and plans *context* questions (explore the topic) and *specific*
   questions (one precise point each) under independent ceilings. Both run in
   the same round, one librarian run per question, each type with its own
   retrieval shape. A specific result keeps only its first papers in the
   judge's ranking. A round that asks nothing ends the loop: no gap is left
   and no new evidence would come in.
3. **Final review.** If the last context batch was never reviewed and has a
   paper to select, one more planner call selects from it; any question it asks
   is dropped.

Questions over a ceiling never fail a round: each type keeps its first ones and
the planning trace records the rest as ``dropped_context_queries`` and
``dropped_specific_queries``.

History shows a context search only through the papers selected when its batch
was reviewed, so a later selection never revives a discarded paper; specific
searches are kept whole. ``LoopResult.passages`` is that selected context
evidence plus every specific result, merged per paper. What to extract or
write from it is the caller's task.

One librarian run per question rather than one run with many sub-queries is the
point: each question gets its own relevance judge, so a specific question's
evidence is about its own point.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

from librarian.agent import LibrarianAgent, RunResult
from librarian.config import LibrarianRuntimeConfig, ResearchLoopConfig
from librarian.llm_client import parse_json_response
from librarian.tracing_port import NullTracer, TracingPort

logger = logging.getLogger(__name__)

# The generic planner prompt shipped with the package (``--loop-prompt`` default).
DEFAULT_LOOP_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "research_loop.md"
)
# Placeholders a loop prompt must contain: without them the planner sees no evidence.
REQUIRED_PLACEHOLDERS = ("{previous_searches}", "{last_searches}")

# Which retriever runs a question; the seed is recorded as a context search.
QueryKind = Literal["seed", "context", "specific"]
SearchKind = Literal["context", "specific"]


class Retriever(Protocol):
    """One question in, ranked evidence out: ``LibrarianAgent.run``'s contract."""

    def run(self, query: str) -> RunResult:
        """Retrieve ranked evidence for *query*."""
        ...


class PlannerLLM(Protocol):
    """The ``chat_completion`` shape of ``llm_client.LLMClient``."""

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
    ) -> str:
        """Return the assistant's reply to *messages*."""
        ...


@dataclass
class SearchRecord:
    """One question's own librarian run.

    :param round_index: 0 for the seed, else the round that asked it.
    :param kind: ``context`` (the seed included) or ``specific``.
    :param question: The question sent to the librarian.
    :param passages: The papers it returned, in the judge's ranking; a specific
        search keeps only its first ``specific_max_papers``.
    :param debug: ``RunResult.debug``, plus ``paper_selection`` on specific searches.
    :param error: Why the run failed, or None.
    :param selected_pmids: Ledger PMIDs selected from this context search when
        its batch was reviewed; None until then, and always None for specific ones.
    """

    round_index: int
    kind: SearchKind
    question: str
    passages: List[Dict[str, Any]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    selected_pmids: Optional[List[str]] = None


@dataclass
class LoopResult:
    """Everything one research loop asked, retrieved and selected.

    :param question: The first query, i.e. the seed question.
    :param context_searches: The seed, then every context question, in ask order.
    :param specific_searches: Every specific question, in ask order.
    :param selected_pmids: The ledger: selected context PMIDs, in selection order.
    :param reviewed_context_count: Context searches already reviewed; the rest
        are the unreviewed latest batch.
    :param planning: One planner trace per round.
    :param final_selection: The selection-only review's trace, or None when the
        last batch had no numeric PMID to select (no context question asked, or
        every one failed).
    :param planner_failure: The round and every invalid reply when the planner
        gave up, which ends the run; None otherwise.
    :param passages: Selected context evidence and every specific result,
        merged per paper; filled when the run ends, a failed run included
        (context searches never reviewed contribute nothing).
    """

    question: str
    context_searches: List[SearchRecord] = field(default_factory=list)
    specific_searches: List[SearchRecord] = field(default_factory=list)
    selected_pmids: List[str] = field(default_factory=list)
    reviewed_context_count: int = 0
    planning: List[Dict[str, Any]] = field(default_factory=list)
    final_selection: Optional[Dict[str, Any]] = None
    planner_failure: Optional[Dict[str, Any]] = None
    passages: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """This result as plain JSON-serializable data.

        :return: Every field, searches included with their passages.
        :rtype: dict[str, Any]
        """
        return asdict(self)


def build_retrievers(
    runtime_config: LibrarianRuntimeConfig,
    loop_config: ResearchLoopConfig,
    **agent_kwargs: Any,
) -> Dict[QueryKind, LibrarianAgent]:
    """One in-process ``LibrarianAgent`` per query kind, each with its shape.

    :param runtime_config: The base config; only the four sizing knobs change.
    :type runtime_config: LibrarianRuntimeConfig
    :param loop_config: Holds the ``seed``, ``context`` and ``specific`` shapes.
    :type loop_config: ResearchLoopConfig
    :param agent_kwargs: Passed to every ``LibrarianAgent`` (``verbose``,
        ``llm_client``, ``tracer``, ...).
    :return: Agents keyed by query kind.
    :rtype: dict[str, LibrarianAgent]
    """
    retrievers: Dict[QueryKind, LibrarianAgent] = {}
    kinds: tuple[QueryKind, ...] = ("seed", "context", "specific")
    for kind in kinds:
        shape = asdict(getattr(loop_config, kind))
        kind_config = replace(runtime_config, **shape)
        retrievers[kind] = LibrarianAgent(runtime_config=kind_config, **agent_kwargs)
    return retrievers


def passage_key(passage: Dict[str, Any]) -> str:
    """The identity a paper is grouped and merged under.

    :param passage: One paper as ``LibrarianAgent.run`` returns it.
    :type passage: dict[str, Any]
    :return: Its numeric PMID, else its ``paper_id``, else an empty string.
    :rtype: str
    """
    return _numeric_pmid(passage) or str(passage.get("paper_id") or "").strip()


def _numeric_pmid(passage: Dict[str, Any]) -> str:
    # The paper's ``pmid`` field when it is numeric, else an empty string.
    pmid = str(passage.get("pmid") or "").strip()
    return pmid if pmid.isdigit() else ""


def merge_passages(passage_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """One passage list out of several runs, merged per paper in first-seen order.

    :param passage_groups: Passage lists in the order their questions were asked.
    :type passage_groups: list[list[dict[str, Any]]]
    :return: Merged copies; a paper's ``evidence_snippets`` are unioned, each
        new snippet with its ``evidence_sections`` label.
    :rtype: list[dict[str, Any]]
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for passages in passage_groups:
        for passage in passages:
            key = passage_key(passage)
            if not key:
                continue
            existing = merged.get(key)
            snippets = passage["evidence_snippets"]
            # Parallel to the snippets: section labels (supplementary file, page, ...).
            sections = passage["evidence_sections"]
            if existing is None:
                # Copied so merging never mutates the per-question record.
                merged[key] = {
                    **passage,
                    "evidence_snippets": list(snippets),
                    "evidence_sections": list(sections),
                }
                continue
            for snippet, section in zip(snippets, sections, strict=True):
                if snippet not in existing["evidence_snippets"]:
                    existing["evidence_snippets"].append(snippet)
                    existing["evidence_sections"].append(section)
    return list(merged.values())


def _grouped_spans(passages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    # Paper key -> its whitespace-collapsed, deduplicated snippets, first-seen order.
    grouped: Dict[str, List[str]] = {}
    for passage in passages:
        key = passage_key(passage)
        if not key:
            continue
        spans = grouped.setdefault(key, [])
        for raw in passage.get("evidence_snippets") or []:
            span = " ".join(str(raw or "").split())
            if span and span not in spans:
                spans.append(span)
    return grouped


def render_searches(searches: List[SearchRecord]) -> str:
    """Render searches as the ``<search>`` blocks the planner prompt reads.

    Spans are joined with `` || `` so two distant spans never read as one claim.
    A paper without a numeric PMID is shown with ``PMID: null``: visible for
    planning, never selectable. A failed retrieval shows ``retrieval failed``
    instead of evidence, so it never reads as a search that found nothing.

    :param searches: History or latest batch, in prompt order.
    :type searches: list[SearchRecord]
    :return: The blocks, or ``null`` when there are no searches.
    :rtype: str
    """
    if not searches:
        return "null"
    blocks: List[str] = []
    for search in searches:
        evidence_blocks: List[str] = []
        for key, spans in _grouped_spans(search.passages).items():
            text = " || ".join(spans)
            if key.isdigit():
                evidence_blocks.append(f"<pmid_{key}>\n{text}\n</pmid_{key}>")
            else:
                tag = "article_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
                evidence_blocks.append(f"<{tag}>\nPMID: null\n{text}\n</{tag}>")
        evidence = "\n".join(evidence_blocks) or "null"
        if search.error:
            evidence = "retrieval failed"
        lines = [
            "<search>",
            f"<query_type>{search.kind}</query_type>",
            "<query>",
            search.question,
            "</query>",
            "<evidences>",
            evidence,
            "</evidences>",
            "</search>",
        ]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _selected_copy(search: SearchRecord) -> SearchRecord:
    # A reviewed context search reduced to the papers selected from its own batch.
    if search.selected_pmids is None:
        raise ValueError(f"context search {search.question!r} was never reviewed")
    selected = set(search.selected_pmids)
    passages = [paper for paper in search.passages if passage_key(paper) in selected]
    return replace(search, passages=passages)


def _merged_evidence(result: LoopResult) -> List[Dict[str, Any]]:
    # Selections of the reviewed context searches plus every specific result.
    reviewed = result.context_searches[: result.reviewed_context_count]
    selected_context = [_selected_copy(search) for search in reviewed]
    passage_groups = [s.passages for s in selected_context if s.passages]
    passage_groups.extend(s.passages for s in result.specific_searches)
    return merge_passages(passage_groups)


def _numeric_pmids(searches: List[SearchRecord]) -> set[str]:
    # The PMIDs the planner may select. Not passage_key: a paper without a PMID
    # can have an all-digit paper_id (an EPMC id, or the id() last resort).
    pmids = {_numeric_pmid(paper) for search in searches for paper in search.passages}
    return pmids - {""}


def _normalized_new(candidates: List[str], seen: set[str]) -> List[str]:
    # Whitespace-collapsed questions not yet asked (case-insensitive); updates *seen*.
    questions: List[str] = []
    for candidate in candidates:
        question = " ".join(candidate.split())
        key = question.casefold()
        if question and key not in seen:
            seen.add(key)
            questions.append(question)
    return questions


def parse_planner_reply(
    raw_response: str,
    max_context_queries: int,
    max_specific_queries: int,
    already_asked: List[str],
    latest_context: List[SearchRecord],
) -> tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Validate one planner reply.

    :param raw_response: JSON object with ``context_queries``,
        ``specific_queries`` and ``selected_pmids``.
    :type raw_response: str
    :param max_context_queries: Ceiling on new context questions.
    :type max_context_queries: int
    :param max_specific_queries: Ceiling on new specific questions.
    :type max_specific_queries: int
    :param already_asked: Every earlier question of either kind.
    :type already_asked: list[str]
    :param latest_context: The unreviewed context searches, the only ones
        whose papers may be selected.
    :type latest_context: list[SearchRecord]
    :return: New context questions, new specific questions, selected PMIDs,
        then the context and specific questions dropped over their ceiling.
        A selected ID that is not a numeric PMID of the latest context batch
        (history-only, specific-only, invented) is ignored rather than failing
        the round; repeated questions are dropped; each type keeps its first
        questions up to its ceiling.
    :rtype: tuple[list[str], list[str], list[str], list[str], list[str]]
    :raises ValueError: Invalid schema.
    """
    payload = parse_json_response(raw_response)
    if payload is None:
        # Repair only a trailing comma: any wider repair could alter the plan.
        payload = parse_json_response(re.sub(r",(\s*[}\]])", r"\1", raw_response))
    if not isinstance(payload, dict):
        raise ValueError("planner must return a JSON object")
    context = payload.get("context_queries")
    specific = payload.get("specific_queries")
    for questions in (context, specific):
        if not isinstance(questions, list) or any(
            not isinstance(question, str) for question in questions
        ):
            raise ValueError(
                "context_queries and specific_queries must be string arrays"
            )
    pmids = payload.get("selected_pmids")
    if not isinstance(pmids, list):
        raise ValueError("planner must return a selected_pmids list")
    available = _numeric_pmids(latest_context)
    selected: List[str] = []
    for candidate in pmids:
        pmid = str(candidate)
        if isinstance(candidate, bool) or pmid not in available:
            continue
        if pmid not in selected:
            selected.append(pmid)
    seen = {" ".join(question.split()).casefold() for question in already_asked}
    context_questions = _normalized_new(context, seen)
    specific_questions = _normalized_new(specific, seen)
    # Over a ceiling the first questions are kept: a long reply still plans the
    # round instead of failing a run whose retrievals are already paid for.
    kept_context = context_questions[:max_context_queries]
    dropped_context = context_questions[max_context_queries:]
    kept_specific = specific_questions[:max_specific_queries]
    dropped_specific = specific_questions[max_specific_queries:]
    return kept_context, kept_specific, selected, dropped_context, dropped_specific


def limit_search_papers(search: SearchRecord, max_papers: int) -> SearchRecord:
    """Keep a search's first papers in the judge's ranking, without reranking.

    :param search: One completed search.
    :type search: SearchRecord
    :param max_papers: Papers to keep; 0 keeps all.
    :type max_papers: int
    :return: A copy with the kept papers and ``paper_selection`` in its debug.
    :rtype: SearchRecord
    :raises ValueError: if *max_papers* is negative.
    """
    if max_papers < 0:
        raise ValueError("paper limit must be nonnegative")
    retained = search.passages[:max_papers] if max_papers else search.passages
    omitted = search.passages[len(retained) :]
    debug = dict(search.debug)
    debug["paper_selection"] = {
        "max_papers": max_papers,
        "order": "librarian_llm_ranking",
        "n_returned": len(search.passages),
        "n_retained": len(retained),
        "omitted_pmids": [str(paper.get("pmid") or "") for paper in omitted],
    }
    return replace(search, passages=retained, debug=debug)


class ResearchLoop:
    """Seed retrieval, planned context/specific rounds, and a final selection."""

    def __init__(
        self,
        retrievers: Dict[QueryKind, Retriever],
        planner: PlannerLLM,
        prompt_template: str,
        config: ResearchLoopConfig,
        on_progress: Optional[Callable[[str], None]] = None,
        tracer: Optional[TracingPort] = None,
    ) -> None:
        """Build a loop around its retrievers and planner.

        :param retrievers: One retriever per kind (``seed``, ``context``,
            ``specific``); see :func:`build_retrievers`.
        :type retrievers: dict[str, Retriever]
        :param planner: The LLM that selects papers and plans questions.
        :type planner: PlannerLLM
        :param prompt_template: Planner prompt. It must contain
            ``{previous_searches}`` and ``{last_searches}``; the loop also fills
            ``{research_question}``, ``{eligible_context_pmids}``,
            ``{max_context_queries}`` and ``{max_specific_queries}`` where
            present. Fill any placeholder of your own before passing the
            template.
        :type prompt_template: str
        :param config: Round count, ceilings, planner and retrieval budgets.
        :type config: ResearchLoopConfig
        :param on_progress: Called with a short status line per step.
        :type on_progress: callable or None
        :param tracer: Span-tracing adapter for the run, its rounds and planner
            calls; defaults to ``NullTracer``. Pass the same tracer to the
            retrievers (``build_retrievers(..., tracer=...)``) so their runs
            nest under the loop's trace.
        :type tracer: TracingPort or None
        :raises ValueError: if a required placeholder or retriever is missing.
        """
        missing = [key for key in REQUIRED_PLACEHOLDERS if key not in prompt_template]
        if missing:
            raise ValueError(f"loop prompt must contain {', '.join(missing)}")
        absent = {"seed", "context", "specific"} - set(retrievers)
        if absent:
            raise ValueError(f"missing retrievers: {', '.join(sorted(absent))}")
        self._retrievers = retrievers
        self._planner = planner
        self._template = prompt_template
        self._config = config
        self._on_progress = on_progress
        self._tracer: TracingPort = tracer if tracer is not None else NullTracer()

    def run(self, result: LoopResult) -> None:
        """Run the seed, every round and the final review, filling *result*.

        The caller creates *result*, so it holds the partial state whatever is
        raised (Ctrl-C included) and the rounds already paid for can be saved.

        :param result: A fresh ``LoopResult(question=...)``; its question is the
            seed, also ``{research_question}``. Filled in place with searches,
            ledger, planning traces and the selected evidence.
        :type result: LoopResult
        :raises ValueError: if *result* is not fresh, or if the planner's reply
            stays invalid after ``planner_max_retries`` more attempts.
        :raises RuntimeError: if the seed retrieval failed, since an empty
            history would then read as empty literature.
        """
        if result != LoopResult(question=result.question):
            raise ValueError("run needs a fresh LoopResult(question=...)")
        with self._tracer.start_span(
            "research_loop.run",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": json.dumps({"query": result.question}),
                "input.mime_type": "application/json",
            },
        ) as run_span:
            try:
                self._run(result)
            except BaseException as exc:
                # The port records exceptions only; Ctrl-C is not a span error.
                if isinstance(exc, Exception):
                    self._tracer.mark_span_error(run_span, exc)
                raise
            finally:
                # Also on failure, so a partial result carries its evidence.
                result.passages = _merged_evidence(result)
            output = {
                "rounds": len(result.planning),
                "selected_pmid_count": len(result.selected_pmids),
                "evidence_count": len(result.passages),
            }
            self._tracer.set_span_attributes(
                run_span,
                {
                    "output.value": json.dumps(output),
                    "output.mime_type": "application/json",
                },
            )

    def _run(self, result: LoopResult) -> None:
        # The loop itself; fills *result* in place so a failure leaves it partial.
        first_query = result.question
        self._progress("Seed retrieval")
        seed = self._retrieve_one("seed", first_query, 0)
        if seed.error:
            raise RuntimeError(f"seed retrieval failed: {seed.error}")
        result.context_searches.append(seed)
        for round_index in range(1, self._config.max_rounds + 1):
            round_attributes = {
                "openinference.span.kind": "CHAIN",
                "research_loop.round": round_index,
            }
            with self._tracer.start_span(
                "research_loop.round", attributes=round_attributes
            ) as round_span:
                asked_anything = self._run_round(result, round_index, round_span)
            # Every gap is closed and nothing new came in: the next planner call
            # would re-read the same evidence. The batch is already reviewed, so
            # the final review is skipped too.
            if not asked_anything:
                break
        # Review the last context batch without asking anything further. A batch
        # with nothing to select (every retrieval failed) is not worth the call.
        latest_context = result.context_searches[result.reviewed_context_count :]
        if _numeric_pmids(latest_context):
            self._progress("Final selection")
            final_round = self._config.max_rounds + 1
            result.final_selection = self._select_and_plan(result, final_round, 0, 0)

    def _run_round(self, result: LoopResult, round_index: int, span: Any) -> bool:
        # One planner call, then its questions; False when it asked nothing.
        self._progress(f"Round {round_index}: planning")
        trace = self._select_and_plan(
            result,
            round_index,
            self._config.max_context_queries,
            self._config.max_specific_queries,
        )
        result.planning.append(trace)
        context_questions = trace["context_queries"]
        specific_questions = trace["specific_queries"]
        self._tracer.set_span_attributes(
            span,
            {
                "research_loop.context_query_count": len(context_questions),
                "research_loop.specific_query_count": len(specific_questions),
                "research_loop.selected_pmid_count": len(result.selected_pmids),
            },
        )
        if not context_questions and not specific_questions:
            return False
        self._progress(
            f"Round {round_index}: retrieving {len(context_questions)} context, "
            f"{len(specific_questions)} specific question(s)"
        )
        self._retrieve_round(result, context_questions, specific_questions, round_index)
        logger.info(
            "Round %d: %d context, %d specific; %d selected context PMIDs",
            round_index,
            len(context_questions),
            len(specific_questions),
            len(result.selected_pmids),
        )
        return True

    def render_prompt(
        self, result: LoopResult, max_context: int, max_specific: int
    ) -> str:
        """The planner prompt for the current state of *result*.

        History: reviewed context searches reduced to their selections, plus
        specific searches of earlier rounds, by round. Latest batch: the
        unreviewed context searches in full, plus the last round's specific ones.

        :param result: The loop state.
        :type result: LoopResult
        :param max_context: Context ceiling; 0 for the final review.
        :type max_context: int
        :param max_specific: Specific ceiling; 0 for the final review.
        :type max_specific: int
        :return: The rendered prompt.
        :rtype: str
        """
        boundary = result.reviewed_context_count
        last_round = len(result.planning)
        reviewed = [_selected_copy(s) for s in result.context_searches[:boundary]]
        earlier_specific = [
            s for s in result.specific_searches if s.round_index < last_round
        ]
        latest_specific = [
            s for s in result.specific_searches if s.round_index == last_round
        ]
        history = reviewed + earlier_specific
        history.sort(key=lambda search: search.round_index)
        latest_context = result.context_searches[boundary:]
        eligible = sorted(_numeric_pmids(latest_context))
        rendered = self._template.replace("{research_question}", result.question)
        rendered = rendered.replace("{previous_searches}", render_searches(history))
        latest = render_searches(latest_context + latest_specific)
        rendered = rendered.replace("{last_searches}", latest)
        rendered = rendered.replace("{eligible_context_pmids}", json.dumps(eligible))
        rendered = rendered.replace("{max_context_queries}", str(max_context))
        return rendered.replace("{max_specific_queries}", str(max_specific))

    def _progress(self, message: str) -> None:
        # Status line for the caller's spinner, if any.
        if self._on_progress is not None:
            self._on_progress(message)

    def _select_and_plan(
        self,
        result: LoopResult,
        round_index: int,
        max_context: int,
        max_specific: int,
    ) -> Dict[str, Any]:
        # One planner call (plus retries on an invalid reply): updates the
        # ledger and the review boundary, and returns the round's trace.
        prompt = self.render_prompt(result, max_context, max_specific)
        boundary = result.reviewed_context_count
        latest_context = result.context_searches[boundary:]
        # A failed question was never answered, so the planner may ask it again.
        searches = result.context_searches + result.specific_searches
        asked = [s.question for s in searches if not s.error]
        messages = [{"role": "user", "content": prompt}]
        attempts: List[Dict[str, Any]] = []
        max_attempts = self._config.planner_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            raw_response = self._call_planner(messages, round_index, attempt)
            trace: Dict[str, Any] = {
                "round": round_index,
                "attempt": attempt,
                "prompt": prompt,
                "response": raw_response,
                "reviewed_search_start": boundary,
                "reviewed_search_end": len(result.context_searches),
            }
            try:
                parsed = parse_planner_reply(
                    raw_response, max_context, max_specific, asked, latest_context
                )
            except ValueError as exc:
                trace["parse_error"] = str(exc)
                attempts.append(trace)
                if attempt == max_attempts:
                    result.planner_failure = {
                        "round": round_index,
                        "failed_attempts": attempts,
                    }
                    raise
                logger.warning("Invalid planner reply (%s); retrying", exc)
                continue
            (
                context_questions,
                specific_questions,
                pmids,
                dropped_context,
                dropped_specific,
            ) = parsed
            if dropped_context or dropped_specific:
                logger.warning(
                    "Round %d: dropped %d context, %d specific question(s) over "
                    "the ceilings",
                    round_index,
                    len(dropped_context),
                    len(dropped_specific),
                )
            trace["parse_error"] = None
            trace["context_queries"] = context_questions
            trace["specific_queries"] = specific_questions
            trace["dropped_context_queries"] = dropped_context
            trace["dropped_specific_queries"] = dropped_specific
            trace["selected_pmids"] = pmids
            trace["previous_attempts"] = attempts
            for pmid in pmids:
                if pmid not in result.selected_pmids:
                    result.selected_pmids.append(pmid)
            # Each reviewed search keeps the selections found in its own papers.
            for search in latest_context:
                available = {passage_key(paper) for paper in search.passages}
                search.selected_pmids = [pmid for pmid in pmids if pmid in available]
            result.reviewed_context_count = len(result.context_searches)
            return trace
        raise AssertionError("planner attempts exhausted without returning or raising")

    def _call_planner(
        self, messages: List[dict], round_index: int, attempt: int
    ) -> str:
        # One planner LLM call in its own span: the largest calls of a run.
        call = {"round": round_index, "attempt": attempt}
        with self._tracer.start_span(
            "research_loop.plan",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": json.dumps(call),
                "input.mime_type": "application/json",
            },
        ) as span:
            try:
                raw_response = self._planner.chat_completion(
                    messages,
                    temperature=self._config.planner_temperature,
                    max_tokens=self._config.planner_max_tokens,
                    timeout=self._config.planner_timeout_seconds,
                )
            except Exception as exc:
                self._tracer.mark_span_error(span, exc)
                raise
            self._tracer.set_span_attributes(
                span,
                {"output.value": raw_response[:2000], "output.mime_type": "text/plain"},
            )
        return raw_response

    def _retrieve_round(
        self,
        result: LoopResult,
        context: List[str],
        specific: List[str],
        round_index: int,
    ) -> None:
        # Both kinds run in one pool, whose width caps the runs in flight.
        # Records land in *result* in ask order, specific ones capped.
        jobs: List[tuple[QueryKind, str]] = [("context", q) for q in context]
        jobs.extend(("specific", q) for q in specific)
        workers = min(self._config.max_concurrent_retrievals, len(jobs))
        # Pool threads keep the round's span, so each run nests under it.
        bound_retrieve = self._tracer.bind_current_trace_context(self._retrieve_one)
        pool = ThreadPoolExecutor(max_workers=workers)
        futures: List[Future[SearchRecord]] = [
            pool.submit(bound_retrieve, kind, question, round_index)
            for kind, question in jobs
        ]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # In practice Ctrl-C: queued questions never start, runs in flight
            # still finish (threads cannot be killed) and keep their records.
            pool.shutdown(cancel_futures=True)
            self._store_finished(result, futures)
            raise
        pool.shutdown()
        self._store_finished(result, futures)

    def _store_finished(
        self, result: LoopResult, futures: List[Future[SearchRecord]]
    ) -> None:
        # Append the records of the futures that completed, in ask order.
        for future in futures:
            finished = future.done() and not future.cancelled()
            if not finished or future.exception() is not None:
                continue
            record: SearchRecord = future.result()
            if record.kind == "context":
                result.context_searches.append(record)
                continue
            capped = limit_search_papers(record, self._config.specific_max_papers)
            result.specific_searches.append(capped)

    def _retrieve_one(
        self, kind: QueryKind, question: str, round_index: int
    ) -> SearchRecord:
        # One librarian run. A failure becomes the record's error: a question
        # that was never answered must stay distinct from one that found nothing.
        record_kind: SearchKind = "specific" if kind == "specific" else "context"
        try:
            run_result = self._retrievers[kind].run(question)
            papers = list(run_result.papers)
            # merge_passages pairs each snippet with its section label. Checked
            # here so a retriever without them fails this question, not the
            # merge after every retrieval has been paid for.
            unaligned = [
                passage_key(paper)
                for paper in papers
                if len(paper["evidence_sections"]) != len(paper["evidence_snippets"])
            ]
            if unaligned:
                raise ValueError(
                    f"evidence_sections not aligned with evidence_snippets: {unaligned}"
                )
        except Exception as exc:  # any pipeline failure ends only this question
            logger.warning("Retrieval failed for %r: %s", question[:60], exc)
            error = f"{type(exc).__name__}: {exc}"
            return SearchRecord(round_index, record_kind, question, error=error)
        return SearchRecord(
            round_index,
            record_kind,
            question,
            passages=papers,
            debug=run_result.debug,
        )

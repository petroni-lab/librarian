"""LibrarianAgent — pure literature retrieval over Europe PMC.

Single-pass pipeline (max_loops=1, the only supported mode):

  Stage 1 — Subquery generation (_generate_queries + _validate_queries):
      LLM turns the user question into N Europe PMC sub-queries; drop those
      whose Europe PMC field syntax is broken.
  Stage 2 — Paper-level retrieval + paragraph BM25 ranking (_paragraphs_for_subquery,
      one thread per sub-query):
        - search the sub-query against Europe PMC (papers_per_subquery papers),
        - decompose every paper into paragraphs (abstract + full-text body chunks,
          no distinction), BM25-rank the pool against the sub-query, keep the top k
          (paragraphs_per_subquery).
      ``run`` concatenates every sub-query's paragraphs into one flat pool and
      merges duplicate or overlapping selections by source coordinates.
  Stage 3 — Filtering, re-ranking & evidence extraction (_relevance_filter):
      judge the paragraphs (one item each, batched by paragraphs_per_judge_batch
      paragraphs per call — a paper's paragraphs may span batches), cite the
      supporting sentences, then regroup the cited sentences by paper and return
      the cited papers ranked most→least relevant, each with its judge-cited
      evidence snippet.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import bm25s
import pysbd
import requests

from librarian.config import load_runtime_config
from librarian.jats import extract_body_paragraphs
from librarian.literature_search import search_scientific_literature_structured
from librarian.llm_client import create_llm_client, parse_json_response

logging.getLogger("bm25s").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Prompts — co-located so this agent is self-contained.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "europepmc_claude_librarian.md"
_FILTER_PROMPT_PATH = _PROMPTS_DIR / "relevance_filter.md"

# Europe PMC full-text fetch endpoint (PMC id -> JATS XML).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Implementation constants — env-independent and never tuned. Every tunable knob
# (query count, page size, evidence words, pool caps, filter batches, ...) lives
# in config.py instead.
#
# Stage 3's judge threads only WAIT on the LLM endpoint (network I/O, no CPU), so
# their pool is a fixed count independent of cores. Stage 2's threads do CPU work
# (search + BM25) and instead size to the CPUs actually available (see
# _available_cpus), so they match the container's cores without oversubscribing.
_JUDGE_WORKERS = 8
# Output budget for one Stage-3 relevance batch. Generous margin over a plain
# JSON id-list response so a wide batch doesn't get silently truncated.
_FILTER_MAX_TOKENS = 8192
# Quoted tokens in a judge reply; used to salvage ids when the JSON is unparseable.
_QUOTED_TOKEN_RE = re.compile(r'"([A-Za-z0-9_\-]+)"')


def _dedupe_preserving_order(ids: Iterable[str]) -> List[str]:
    """Drop repeats but keep first-seen order, which is the judge's ranking."""
    seen: set = set()
    unique: List[str] = []
    for identifier in ids:
        if identifier not in seen:
            seen.add(identifier)
            unique.append(identifier)
    return unique


def _cgroup_cpu_quota() -> Optional[int]:
    """The container's CPU limit in whole cores from the cgroup CFS quota, or None.

    This is the container runtime's ``limits.cpu`` (e.g. Kubernetes), which is
    enforced as a CFS quota (quota/period), NOT as CPU affinity — so
    ``sched_getaffinity`` still reports every host core and misses the limit;
    reading the quota is the reliable signal. Handles cgroup v2 (``cpu.max`` =
    "<quota> <period>", "max" = unlimited) and v1 (``cpu.cfs_quota_us`` = -1
    when unlimited). Rounds up, floored at 1. Returns None when there is no
    quota file or no limit is set.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as handle:  # cgroup v2
            quota_str, period_str = handle.read().split()
        if quota_str == "max":
            return None
        quota, period = int(quota_str), int(period_str)
    except (FileNotFoundError, ValueError):
        try:  # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as handle:
                quota = int(handle.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
                period = int(handle.read())
        except (FileNotFoundError, ValueError):
            return None
        if quota <= 0:  # -1 == unlimited
            return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, -(-quota // period))  # ceil(quota / period)


def _available_cpus() -> int:
    """CPUs this process may actually use, floored at 1.

    Prefers the container's cgroup CPU limit (``_cgroup_cpu_quota``); falls
    back to the CPU affinity, then the host count. Reading the quota matters
    because a container runtime enforces the limit as a CFS quota, not
    affinity, so ``sched_getaffinity``/``cpu_count`` alone report the whole
    node's core count even when the process is capped well below it.
    """
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return quota
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


# ── Pure helpers (no network) ────────────────────────────────────────────────


def _candidate_id(paper: Dict[str, Any]) -> str:
    """Stable per-paper id: PMID → EPMC id → DOI → title → object identity.

    Shared by paragraph dedup and the Stage-3 judge so a paper keys identically
    everywhere. Two papers with the same pmid share an id even if they are
    distinct dicts (e.g. returned by two different sub-queries).
    """
    pmid = str(paper.get("pmid") or "").strip()
    if pmid:
        return pmid
    epmc_id = str(paper.get("epmcId") or "").strip()
    if epmc_id:
        source = str(paper.get("epmcSource") or paper.get("sourceCode") or "").strip()
        return f"{source}:{epmc_id}" if source else epmc_id
    return str(paper.get("doi") or paper.get("title") or id(paper))


def _merge_selected_paragraphs(
    paragraphs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge overlapping selections from the same source paragraph.

    Stage 2 may select the same or overlapping chunks through several
    sub-queries. Group them by paper, source paragraph, and source text; union
    touching word intervals; and rebuild each selected interval once. Group
    order follows first selection while intervals within a source paragraph
    follow document order.
    """
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for paragraph in paragraphs:
        source_text = str(paragraph.get("source_text") or paragraph.get("text") or "")
        source_index = int(paragraph.get("source_paragraph_index", 0))
        key = (_candidate_id(paragraph["paper"]), source_index, source_text)
        groups.setdefault(key, []).append(paragraph)

    merged: List[Dict[str, Any]] = []
    for (_, source_index, source_text), selected in groups.items():
        source_words = source_text.split()
        intervals = sorted(
            (
                int(paragraph.get("word_start", 0)),
                int(paragraph.get("word_end", len(source_words))),
            )
            for paragraph in selected
        )
        merged_intervals: List[List[int]] = []
        for start, end in intervals:
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
            else:
                merged_intervals.append([start, end])
        base = selected[0]
        for start, end in merged_intervals:
            merged.append(
                {
                    **base,
                    "text": " ".join(source_words[start:end]),
                    "source_text": source_text,
                    "source_paragraph_index": source_index,
                    "word_start": start,
                    "word_end": end,
                }
            )
    return merged


def _strip_epmc_operators(query: str) -> str:
    """Reduce a Europe PMC query to bare keywords for BM25.

    Drops field specifiers (``TITLE_ABS:``), quotes, parentheses and the
    boolean operators AND/OR/NOT. A plain keyword query is returned unchanged.
    """
    stripped = re.sub(r"\b[A-Z_]+:", " ", query)
    stripped = stripped.replace('"', " ").replace("(", " ").replace(")", " ")
    tokens = [t for t in stripped.split() if t not in {"AND", "OR", "NOT"}]
    return " ".join(tokens)


def _bm25_rank(query: str, paragraphs: List[str]) -> List[Tuple[int, float]]:
    """Return ``(index, score)`` for every paragraph, highest score first."""
    if not paragraphs:
        return []
    clean_query = _strip_epmc_operators(query) or query
    corpus_tokens = bm25s.tokenize(paragraphs, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize([clean_query], stopwords="en", show_progress=False)
    idxs, scores = retriever.retrieve(
        query_tokens, k=len(paragraphs), show_progress=False
    )
    return [(int(idxs[0][i]), float(scores[0][i])) for i in range(len(idxs[0]))]


def _bm25_doc(record: Dict[str, Any]) -> str:
    """BM25 document for a paragraph record: text + section title/type."""
    return " ".join(
        [
            str(record.get("text", "")),
            str(record.get("section_title", "")),
            str(record.get("section_type", "")),
        ]
    )


def _rank_paragraphs_for_subquery(
    subquery: str, records: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """Top ``k`` paragraph records BM25-scored against ONE sub-query.

    The Stage-2 primitive: each record is scored against the *sub-query* text
    (not the original question), so the paragraph that triggered THAT
    sub-query's lexical match surfaces — which may be a body chunk, not the
    abstract. Europe PMC field/boolean syntax in the sub-query is stripped by
    ``_bm25_rank`` before scoring, so only the keywords matter.

    Returns the records themselves (with their chunk offsets intact), highest
    score first, so the caller can pool paragraphs from several sub-queries.
    **Dedup is a pool-level concern, not handled here**: this helper may
    return a record that another sub-query's call also returned. The caller
    merging the per-sub-query pools merges duplicate and overlapping source
    intervals, so a paragraph relevant to several sub-queries costs the
    downstream judge budget once. Positive-scoring records are preferred; if
    none score above zero the leading ``k`` records are returned as a recall
    fallback.
    """
    if not records or k <= 0:
        return []
    ranked = _bm25_rank(subquery, [_bm25_doc(r) for r in records])
    positive = [i for i, score in ranked if score > 0]
    order = positive if positive else [i for i, _ in ranked]
    return [records[i] for i in order[:k]]


def _chunk_paragraphs(
    records: List[Dict[str, Any]], max_words: int, overlap: int
) -> List[Dict[str, Any]]:
    """Split over-long paragraph records into overlapping word windows.

    Records at or under ``max_words`` pass through untouched. Longer ones
    become several records sharing the same section metadata, each a
    ``max_words`` window stepped by ``max_words - overlap`` so a match
    spanning a cut isn't lost. Keeps BM25 from favouring a long paragraph on
    length alone.

    :raises ValueError: if ``overlap`` is not smaller than ``max_words``,
        which would make the window step 1 word at a time — a 600-word
        paragraph becomes 501 chunks instead of 3, silently flooding Stage 3
        and its LLM cost.
    """
    if overlap >= max_words:
        raise ValueError(
            f"paragraph_overlap_words ({overlap}) must be smaller than "
            f"max_paragraph_words ({max_words})"
        )
    step = max_words - overlap
    chunked: List[Dict[str, Any]] = []
    for record in records:
        source_text = str(record.get("text", ""))
        words = source_text.split()
        if len(words) <= max_words:
            chunked.append(
                {
                    **record,
                    "source_text": source_text,
                    "word_start": 0,
                    "word_end": len(words),
                }
            )
            continue
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            chunked.append(
                {
                    **record,
                    "text": " ".join(window),
                    "source_text": source_text,
                    "word_start": start,
                    "word_end": start + len(window),
                }
            )
            if start + max_words >= len(words):
                break
    return chunked


# One reusable segmenter; sentence splitting runs single-threaded, so a shared
# instance is safe. clean=False keeps the text verbatim so cited sentences map
# back to the source exactly.
_SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences with pySBD (handles ``Dr.``, ``e.g.``, etc.)."""
    if not (text or "").strip():
        return []
    return [s.strip() for s in _SENTENCE_SEGMENTER.segment(text) if s.strip()]


def _sentence_suffix(idx: int) -> str:
    """Stable per-paper sentence suffix: A..Z then S26, S27, ..."""
    return chr(65 + idx) if idx < 26 else f"S{idx}"


@dataclass
class _Sentence:
    """One sentence shown to the relevance judge, keyed by its sentence id."""

    paper_id: str
    text: str
    sentence_index: int  # reading-order position within the selected interval
    source_paragraph_index: int  # paragraph position within the source paper
    source_word_start: int  # selected interval position within the source paragraph


def _build_sentence_items(
    paragraph_id: str,
    paper_id: str,
    source_paragraph_index: int,
    source_word_start: int,
    source_text: str,
    registry: Dict[str, _Sentence],
) -> List[Dict[str, str]]:
    """Split one paragraph into ``{"id": "paragraph_0_A", "text": ...}`` judge items.

    ``paragraph_id`` makes sentence ids unique within Stage 3. Paper and
    source coordinates stay internal so citations map back to document order.
    No sentence cap — the full excerpt is shown to the judge.
    """
    sentences = _split_sentences(source_text)
    if not sentences and (source_text or "").strip():
        sentences = [source_text.strip()]
    items: List[Dict[str, str]] = []
    for idx, sentence in enumerate(sentences):
        sid = f"{paragraph_id}_{_sentence_suffix(idx)}"
        registry[sid] = _Sentence(
            paper_id=paper_id,
            text=sentence,
            sentence_index=idx,
            source_paragraph_index=source_paragraph_index,
            source_word_start=source_word_start,
        )
        items.append({"id": sid, "text": sentence})
    return items


def _group_contiguous_spans(sentences: List[_Sentence]) -> List[str]:
    """Join sentences into contiguous spans, one per adjacent run within a paragraph.

    Two sentences merge only when they are consecutive within the same
    selected interval. A different source paragraph, word interval, or
    sentence gap starts a new span, preserving real document order without
    joining unrelated evidence.
    """
    spans: List[List[str]] = []
    prev: Optional[_Sentence] = None
    for sent in sentences:
        contiguous = (
            prev is not None
            and sent.source_paragraph_index == prev.source_paragraph_index
            and sent.source_word_start == prev.source_word_start
            and sent.sentence_index == prev.sentence_index + 1
        )
        if not contiguous:
            spans.append([])
        spans[-1].append(sent.text)
        prev = sent
    return [" ".join(span) for span in spans]


def _truncate_spans_to_words(spans: List[str], max_words: int) -> List[str]:
    """Cap the total word count across spans, preserving span boundaries.

    Words are kept greedily in order; the span that crosses the budget is
    word-sliced and later spans are dropped, so ``" ".join`` of the result
    holds the first ``max_words`` words.
    """
    out: List[str] = []
    used = 0
    for span in spans:
        if used >= max_words:
            break
        words = span.split()
        remaining = max_words - used
        if len(words) > remaining:
            out.append(" ".join(words[:remaining]))
            break
        out.append(span)
        used += len(words)
    return out


# ── Agent ────────────────────────────────────────────────────────────────────


class LibrarianAgent:
    """Retrieves and ranks evidence for a research question."""

    def __init__(
        self,
        full_text_enrichment: bool = True,
        verbose: bool = False,
        llm_base_url: Optional[str] = None,
        llm_model_name: Optional[str] = None,
    ):
        # Full text is the default retrieval path; the flag is kept for harness
        # compatibility (turning it off falls back to abstract-only paragraphs).
        self.full_text_enrichment = full_text_enrichment
        self.verbose = verbose

        # Progress reporting: set per-run in run(); _progress() no-ops when unset.
        self._on_progress: Optional[Callable[[str], None]] = None
        # Shared counters so the Stage-3 judge batches report one "N/M
        # batches" figure to the progress callback.
        self._filter_lock = threading.Lock()
        self._filter_done = 0
        self._filter_total = 0

        # Explicit constructor args (llm_model_name) take priority over config.
        runtime = load_runtime_config()
        self._max_queries = runtime.max_query_count
        self._query_budget_guidance = runtime.query_budget_guidance
        self._papers_per_subquery = runtime.papers_per_subquery
        self._paragraphs_per_subquery = runtime.paragraphs_per_subquery
        self._paragraphs_per_judge_batch = runtime.paragraphs_per_judge_batch
        self._max_paragraph_words = runtime.max_paragraph_words
        self._paragraph_overlap_words = runtime.paragraph_overlap_words
        self._evidence_snippet_max_words = runtime.evidence_snippet_max_words
        self._filter_temperature = runtime.filter_temperature

        self.llm = create_llm_client(
            base_url=llm_base_url,
            model_name=llm_model_name or runtime.default_model_name,
        )
        self._query_prompt = _QUERY_PROMPT_PATH.read_text(encoding="utf-8")
        self._filter_prompt = _FILTER_PROMPT_PATH.read_text(encoding="utf-8")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Librarian] {message}")

    def _progress(self, message: str) -> None:
        """Report the current pipeline stage to the run's progress callback (if any)."""
        if self._on_progress is not None:
            self._on_progress(message)

    def _on_filter_batch_done(self, _future) -> None:
        """Bump the shared filter-batch counter and report N/M to the callback."""
        with self._filter_lock:
            self._filter_done += 1
            done, total = self._filter_done, self._filter_total
        self._progress(f"Judging paragraph relevance · {done}/{total} batches")

    # ── Step 1: query generation ─────────────────────────────────────────────

    def _generate_queries(
        self,
        query: str,
        previous_evidences: Optional[List[str]] = None,
    ) -> List[str]:
        """Ask the LLM for diverse Europe PMC sub-queries for ``query``."""
        today = datetime.date.today()
        additional_context = ""
        if previous_evidences:
            additional_context = "Evidence already gathered:\n" + "\n".join(
                f"- {e}" for e in previous_evidences
            )

        prompt = (
            self._query_prompt.replace("{today_date}", today.isoformat())
            .replace("{today_year}", str(today.year))
            .replace("{query_budget_guidance}", self._query_budget_guidance)
            .replace("{conversation}", f"User: {query}")
            .replace("{additional_context}", additional_context)
        )

        # max_tokens=8192 (NOT 1024): the claude prompt's JSON response is long;
        # truncation → parse failure → a single raw-question search → bad retrieval.
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
        parsed = parse_json_response(response)
        queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
        if not queries:
            # Retry once, deterministically, demanding strict JSON before falling
            # back to the raw question.
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON with a 'queries' array of Europe "
                        'PMC query strings, e.g. {"queries": ["q1", "q2"]}. '
                        "No prose, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            retry = self.llm.chat_completion(
                retry_messages, temperature=0.0, max_tokens=8192
            )
            parsed = parse_json_response(retry)
            queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
        if not queries:
            self._log("query-gen failed to parse; falling back to raw question")
            queries = [query]

        # Dedup (preserve order) and cap.
        seen, unique = set(), []
        for q in queries:
            q = str(q).strip()
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
        result = unique[: self._max_queries]
        self._log(f"generated {len(result)} queries")
        return result

    # ── Step 2: query validation ─────────────────────────────────────────────

    def _validate_queries(self, queries: List[str]) -> List[str]:
        """Validate against EPMC syntax with the deterministic BatchQueryValidator.

        If every query is rejected, keep them all rather than search nothing.
        """
        from librarian.query_validation.batch_validator import BatchQueryValidator

        report = BatchQueryValidator().validate_queries(queries)
        valid = [r["query"] for r in report["results"] if r["valid"]]
        if self.verbose and len(valid) != len(queries):
            self._log(f"query validation kept {len(valid)}/{len(queries)}")
        return valid or queries

    # ── Stage 2: per-sub-query retrieval + paragraph ranking ─────────────────

    def _paragraph_records_for_paper(
        self, paper: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """One paper → its paragraph records: abstract + full-text body chunks.

        Abstract and body paragraphs share one pool with no distinction. Each
        record carries a ``paper`` reference so its metadata is reachable later.
        """
        records: List[Dict[str, Any]] = []

        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            records.append(
                {
                    "text": abstract,
                    "section_title": "Abstract",
                    "section_type": "abstract",
                }
            )

        if self.full_text_enrichment:
            for record in self._fetch_body_paragraphs(paper):
                records.append(record)

        for source_index, record in enumerate(records):
            record["source_paragraph_index"] = source_index
        chunked = _chunk_paragraphs(
            records,
            self._max_paragraph_words,
            self._paragraph_overlap_words,
        )
        for record in chunked:
            record["paper"] = paper
        return chunked

    def _paragraphs_for_subquery(self, subquery: str) -> List[Dict[str, Any]]:
        """One sub-query → its top-``k`` paragraphs, end to end in this thread.

        Search Europe PMC for the sub-query, decompose every returned paper
        into paragraphs (abstract + full-text chunks), BM25-rank the whole
        pool against the sub-query, and keep the top
        ``paragraphs_per_subquery``. Runs entirely in the calling thread so
        ``run`` can fan these out, one thread per sub-query, with no shared
        state.
        """
        try:
            papers = search_scientific_literature_structured(
                subquery, page_size=self._papers_per_subquery
            )
        except Exception as exc:  # network/parse failure → skip this sub-query
            self._log(f"search failed for {subquery!r}: {exc}")
            return []

        pool: List[Dict[str, Any]] = []
        for paper in papers:
            pool.extend(self._paragraph_records_for_paper(paper))

        top = _rank_paragraphs_for_subquery(
            subquery, pool, self._paragraphs_per_subquery
        )
        self._log(f"{subquery!r} → {len(papers)} papers, {len(top)} paragraphs")
        return top

    def _passage_from_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a relevant paper into the returned evidence passage.

        The one citable artifact every consumer reads is ``evidence_snippets``:
        the sentences the Stage-3 judge cited, grouped into contiguous spans
        in reading order. A paper whose cited sentences are non-adjacent
        yields more than one span, so a consumer can tell them apart instead
        of reading one string that silently splices distant sentences
        together. When the judge cited nothing (a paper kept only as a
        fallback), the abstract stands in as a single span.
        """
        judge_spans = [
            span.strip()
            for span in paper.get("evidence_sentences_full_text") or []
            if span and span.strip()
        ]
        abstract = str(paper.get("abstract") or "").strip()
        spans = judge_spans or ([abstract] if abstract else [])
        # Hard cap the TOTAL evidence at the word budget (~250, OpenScholar
        # passage size). Sentence accumulation in the filter stops once the
        # budget is reached, but the last sentence can overshoot, so bound it
        # here too.
        evidence_snippets = _truncate_spans_to_words(
            spans, self._evidence_snippet_max_words
        )
        paper_id = _candidate_id(paper)
        return {
            "evidence_snippets": evidence_snippets,
            "paper_id": paper_id,
            "title": str(paper.get("title") or "No title"),
            "authors": str(paper.get("authors") or "Unknown authors"),
            "journal": str(paper.get("journal") or ""),
            "year": str(paper.get("year") or ""),
            "pmid": str(paper.get("pmid") or ""),
            "doi": str(paper.get("doi") or ""),
            # Europe PMC record identity, kept verbatim under the upstream
            # camelCase keys literature_search.py already uses. Preprints
            # (source="PPR") have NO pmid, so without these a citation cannot
            # be linked back to its EPMC article page.
            "epmcId": str(paper.get("epmcId") or ""),
            "epmcSource": str(paper.get("epmcSource") or paper.get("sourceCode") or ""),
            "url": str(paper.get("url") or ""),
            "has_fulltext": bool(paper.get("inEPMC") or paper.get("hasFreeFullText")),
        }

    # ── Full-text fetch + body extraction ────────────────────────────────────

    def _fetch_body_paragraphs(self, paper: Dict[str, Any]) -> List[Dict[str, str]]:
        """Fetch full text and return body paragraph records (or [])."""
        if not (paper.get("inEPMC") or paper.get("hasFreeFullText")):
            return []
        full_text_ids = paper.get("fullTextIds") or []
        pmcid = (full_text_ids[0] if full_text_ids else "") or paper.get("pmcid") or ""
        pmcid = str(pmcid).strip()
        if not pmcid:
            return []
        return extract_body_paragraphs(self._fetch_fulltext_xml(pmcid))

    def _fetch_fulltext_xml(self, pmcid: str) -> str:
        """Fetch JATS full-text XML for a PMC id (returns '' on failure)."""
        pmcid = pmcid.upper()
        if pmcid.isdigit():
            pmcid = f"PMC{pmcid}"
        try:
            response = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self._log(f"full-text fetch failed for {pmcid}: {exc}")
            return ""
        return response.text

    # ── Step 5: single-pass LLM relevance filter ─────────────────────────────

    @staticmethod
    def _salvage_relevant_ids(response: str) -> List[str]:
        """Recover sentence ids from a judge reply that failed to parse as JSON.

        The judge returns ``relevant_ids`` ordered most→least relevant, so a
        reply cut off mid-array still carries usable ranking in the part that
        arrived — dropping the whole batch throws away good evidence. Ids the
        batch never issued are harmless: ``_papers_from_cited_sentences``
        skips anything missing from its sentence registry.

        :param response: The raw judge reply.
        :returns: First-seen-order ids, or ``[]`` if this is not a judge reply.
        """
        # Gate on the key so an unrelated error string cannot be mined for ids.
        if "relevant_ids" not in response:
            return []
        return _dedupe_preserving_order(
            token
            for token in _QUOTED_TOKEN_RE.findall(response)
            if token != "relevant_ids"
        )

    def _evaluate_batch(
        self,
        query: str,
        batch_data: List[Dict[str, Any]],
        prompt_template: str,
        depth: int = 0,
    ) -> List[str]:
        """LLM-judge one batch; return relevant sentence IDs (most→least relevant).

        On a malformed/empty response, split the batch and retry (<=2 levels)
        so a whole batch is never silently dropped.
        """
        prompt = prompt_template.replace("{user_query}", query).replace(
            "{paragraphs_batch}", json.dumps(batch_data, indent=2)
        )
        try:
            response = self.llm.chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Return only valid JSON "
                            "with a 'relevant_ids' array ordered most→least relevant."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self._filter_temperature,
                max_tokens=_FILTER_MAX_TOKENS,
            )
        except Exception as exc:
            self._log(f"filter batch error: {exc}")
            response = ""

        if self.verbose:
            preview = str(response).replace("\n", " ")[:200]
            self._log(f"filter raw response (batch={len(batch_data)}): {preview!r}")

        parsed = parse_json_response(response)
        ids = parsed.get("relevant_ids") if isinstance(parsed, dict) else None
        if isinstance(ids, list):
            # Deduplicated even on the happy path: the judge can repeat ids
            # without truncating, which inflates the caller's ranking walk.
            return _dedupe_preserving_order(str(x) for x in ids)

        # Unparseable (usually truncated mid-array): keep the prefix that
        # arrived rather than re-judging or discarding the batch.
        salvaged = self._salvage_relevant_ids(str(response))
        if salvaged:
            self._log(f"salvaged {len(salvaged)} ids from an unparseable judge reply")
            return salvaged

        if depth < 2 and len(batch_data) > 1:
            mid = len(batch_data) // 2
            return self._evaluate_batch(
                query, batch_data[:mid], prompt_template, depth + 1
            ) + self._evaluate_batch(
                query, batch_data[mid:], prompt_template, depth + 1
            )
        return []

    def _relevance_filter(
        self, query: str, paragraphs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Stage 3: judge the paragraph pool → relevant papers, ranked.

        One judge item per PARAGRAPH; the flat list is batched by
        ``paragraphs_per_judge_batch`` paragraphs per LLM call, so a paper
        with many paragraphs can span several batches. Each paragraph is
        split into cited-able sentences keyed so the cited sentence ids map
        back to their paper (and its position) regardless of which batch
        judged them. The judge returns sentence ids most→least relevant;
        those become ``evidence_sentences_full_text`` on their paper, and
        papers are returned in the order the judge first cited them. Papers
        the judge never cites are dropped.
        """
        if not paragraphs:
            return []
        today = datetime.date.today()
        template = self._filter_prompt.replace(
            "{today_date}", today.isoformat()
        ).replace("{today_year}", str(today.year))

        registry: Dict[str, _Sentence] = {}
        paper_by_id: Dict[str, Dict[str, Any]] = {}
        items: List[Dict[str, Any]] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            paper = paragraph["paper"]
            paper_id = _candidate_id(paper)
            paper_by_id[paper_id] = paper
            paragraph_id = f"paragraph_{paragraph_index}"
            items.append(
                {
                    "id": paragraph_id,
                    "title": paper.get("title", "N/A"),
                    "authors": paper.get("authors", "N/A"),
                    "journal": paper.get("journal", "N/A"),
                    "year": paper.get("year", "N/A"),
                    "sentences": _build_sentence_items(
                        paragraph_id,
                        paper_id,
                        int(paragraph["source_paragraph_index"]),
                        int(paragraph["word_start"]),
                        str(paragraph.get("text") or "").strip(),
                        registry,
                    ),
                }
            )

        ranked_ids = self._judge_items_in_batches(query, template, items)
        return self._papers_from_cited_sentences(ranked_ids, registry, paper_by_id)

    def _judge_items_in_batches(
        self, query: str, template: str, items: List[Dict[str, Any]]
    ) -> List[str]:
        """Send items to the judge in batches of ``paragraphs_per_judge_batch`` → sentence ids.

        Batches run in parallel; results are concatenated in batch order so
        the overall most→least-relevant ranking is preserved. Each batch bumps
        the shared progress counter as it finishes (via a done-callback, so
        the count reflects real completion, not submission order).
        """
        batch_size = self._paragraphs_per_judge_batch
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        with self._filter_lock:
            self._filter_total += len(batches)
        ranked_ids: List[str] = []
        # Judge batches only wait on the LLM (network I/O) — size to
        # _JUDGE_WORKERS, not the CPU count.
        with ThreadPoolExecutor(max_workers=_JUDGE_WORKERS) as pool:
            futures = []
            for batch in batches:
                future = pool.submit(self._evaluate_batch, query, batch, template)
                future.add_done_callback(self._on_filter_batch_done)
                futures.append(future)
            for future in futures:  # submission order == batch order
                ranked_ids.extend(future.result())
        return ranked_ids

    def _papers_from_cited_sentences(
        self,
        ranked_ids: List[str],
        registry: Dict[str, _Sentence],
        paper_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cited sentence ids → relevant papers with their evidence spans.

        Walks the judge's ranked sentence ids: the first time a paper is
        cited it joins the result (so paper order follows the judge's
        ranking), and every sentence the judge cited for it is kept. Each
        paper's sentences are then grouped into contiguous reading-order
        spans on ``evidence_sentences_full_text`` for ``_passage_from_paper``.
        """
        relevant: List[Dict[str, Any]] = []
        seen_papers: set = set()
        seen_sentences: set = set()
        sids_by_paper: Dict[str, List[str]] = {}
        for sid in ranked_ids:
            sentence = registry.get(sid)
            if sentence is None or sentence.paper_id not in paper_by_id:
                continue
            coordinate = (
                sentence.paper_id,
                sentence.source_paragraph_index,
                sentence.source_word_start,
                sentence.sentence_index,
            )
            if coordinate in seen_sentences:
                continue
            seen_sentences.add(coordinate)
            paper_id = sentence.paper_id
            if paper_id not in seen_papers:
                seen_papers.add(paper_id)
                relevant.append(paper_by_id[paper_id])
            sids_by_paper.setdefault(paper_id, []).append(sid)
        for paper_id, sids in sids_by_paper.items():
            ordered = sorted(
                sids,
                key=lambda s: (
                    registry[s].source_paragraph_index,
                    registry[s].source_word_start,
                    registry[s].sentence_index,
                ),
            )
            paper_by_id[paper_id]["evidence_sentences_full_text"] = (
                _group_contiguous_spans([registry[s] for s in ordered])
            )
        self._log(f"Stage 3: {len(relevant)} relevant papers")
        return relevant

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        max_loops: int = 1,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Single-pass retrieval: Stage 1 (generate + validate sub-queries) →
        Stage 2 (per-sub-query paragraph BM25 ranking) → Stage 3 (relevance
        judge over the paragraphs, regrouped to papers). Returns every paper
        the judge keeps — the result is variable-size, with no final top_k cap.

        ``on_progress`` (optional) is called with a short human-readable
        status string at each pipeline stage — wire it to a spinner for live
        feedback.
        """
        if max_loops != 1:
            raise NotImplementedError(
                "LibrarianAgent only supports max_loops=1 (single pass)."
            )
        self._on_progress = on_progress
        return self._run(query)

    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Internal implementation of the single-pass retrieval pipeline."""
        self._filter_done = 0  # reset so a reused agent reports fresh counts
        self._filter_total = 0
        self._progress("Generating search queries")
        generated = self._generate_queries(query)
        self._progress("Validating queries")
        queries = self._validate_queries(generated)

        # Stage 2 — one thread per sub-query: search EPMC, decompose papers
        # into paragraphs, BM25-rank against that sub-query, keep its top-k
        # paragraphs. Concatenate every sub-query's paragraphs into one flat
        # pool, merging duplicate/overlapping selections. This flat paragraph
        # pool is what Stage 3 judges.
        self._progress(f"Searching Europe PMC · {len(queries)} sub-queries")
        # One CPU-bound thread per sub-query, capped at the CPUs actually
        # available so a large max_query_count can't oversubscribe a small
        # container.
        stage2_workers = min(max(len(queries), 1), _available_cpus())
        with ThreadPoolExecutor(max_workers=stage2_workers) as pool:
            per_subquery = pool.map(self._paragraphs_for_subquery, queries)
        paragraphs = _merge_selected_paragraphs(
            [para for subquery_paras in per_subquery for para in subquery_paras]
        )
        self._log(f"{len(paragraphs)} paragraphs after Stage 2")
        self._progress(f"Found {len(paragraphs)} candidate paragraphs")
        if not paragraphs:
            return []

        # Stage 3 — the relevance judge reads the paragraphs (batched by
        # paragraphs_per_judge_batch paragraphs per call; a paper's
        # paragraphs may span batches), cites the supporting sentences, and
        # those regroup by paper. Returns the cited papers, most→least
        # relevant. Stage 3's output IS the final set — variable size, with
        # no final top_k cap applied on top of the judge's decision.
        self._progress("Judging paragraph relevance")
        relevant_papers = self._relevance_filter(query, paragraphs)

        self.last_run_debug = {
            "search_queries": queries,
            "query_count": len(queries),
            "paragraph_count": len(paragraphs),
            "relevant_count": len(relevant_papers),
            "final_pmids": [str(p.get("pmid") or "") for p in relevant_papers],
        }
        logger.debug(
            "[Librarian] queries=%d paragraphs=%d relevant=%d",
            len(queries),
            len(paragraphs),
            len(relevant_papers),
        )

        return [self._passage_from_paper(p) for p in relevant_papers]

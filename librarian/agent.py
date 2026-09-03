"""LibrarianAgent — pure literature retrieval over Europe PMC.

Single-pass pipeline (max_loops=1, the only supported mode):

  1. _generate_queries   LLM turns the user question into N Europe PMC sub-queries
                         using the europepmc_claude_librarian prompt.
  2. _validate_queries   drop sub-queries whose Europe PMC field syntax is broken.
  3. multithread_routine per sub-query, run in parallel:
                           - search Europe PMC (<=50 papers/sub-query),
                           - fetch each paper's full-text body,
                           - BM25-rank the body paragraphs against the ORIGINAL
                             user query and keep the top paragraphs up to a
                             per-paper word budget (~2300 words = 3072 tokens) as
                             ``full_text_excerpt``.
  4. dedup candidate papers across all sub-queries.
  5. Two-pass relevance filter (winning bm25_per_paper strategy), merged
     full-text-first: a full-text filter over papers with an excerpt
     (relevance_filter_fulltext.md) + an abstract triage over all candidates
     (relevance_filter.md). Returns the merged relevant papers as evidence
     passages (each with its filter-cited evidence snippet).

"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
_ABSTRACT_FILTER_PROMPT_PATH = _PROMPTS_DIR / "relevance_filter.md"
_FULLTEXT_FILTER_PROMPT_PATH = _PROMPTS_DIR / "relevance_filter_fulltext.md"

# Europe PMC full-text fetch endpoint (PMC id -> JATS XML).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Implementation constants — env-independent and never tuned. Every tunable knob
# (query count, page size, evidence words, pool caps, filter batches, ...) lives
# in config.py instead.
#
# The relevance-filter and full-text-fetch pools below only WAIT on network I/O,
# so they are sized to fixed constants independent of CPU count. The sub-query
# search fan-out does real CPU work (search + downstream BM25) and instead sizes
# to the CPUs actually available (see _available_cpus), so it matches the
# container's cores without oversubscribing.
_MAX_WORKERS = 8
# Parallel full-text fetches per sample (deduped first, so this is unique papers).
_FETCH_WORKERS = 12
# Overall ceiling on returned papers (safety backstop; the two filters do the real
# selecting). Defaults to the sum of the per-pool caps (30 + 30) so it does not
# truncate the intended split.
_DEFAULT_FINAL_TOP_K = 60


def _per_paper_word_limit(config_tokens: int) -> int:
    """Convert a token budget to a word budget (x 0.75), floored at 200 words.

    3072 x 0.75 ~= 2304 words — the value that drove LitQA2 accuracy from
    ~0.65 to ~0.75.
    """
    return max(200, int(config_tokens * 0.75))


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


def _strip_epmc_operators(query: str) -> str:
    """Reduce a Europe PMC query to bare keywords for BM25.

    Drops field specifiers (``TITLE_ABS:``), quotes, parentheses and the boolean
    operators AND/OR/NOT. A plain keyword query is returned unchanged.
    """
    stripped = re.sub(r"\b[A-Z_]+:", " ", query)
    stripped = stripped.replace('"', " ").replace("(", " ").replace(")", " ")
    tokens = [t for t in stripped.split() if t not in {"AND", "OR", "NOT"}]
    return " ".join(tokens)


def _bm25_rank(query: str, passages: List[str]) -> List[Tuple[int, float]]:
    """Return ``(index, score)`` for every passage, highest score first."""
    if not passages:
        return []
    clean_query = _strip_epmc_operators(query) or query
    corpus_tokens = bm25s.tokenize(passages, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize([clean_query], stopwords="en", show_progress=False)
    idxs, scores = retriever.retrieve(
        query_tokens, k=len(passages), show_progress=False
    )
    return [(int(idxs[0][i]), float(scores[0][i])) for i in range(len(idxs[0]))]


def _select_excerpt_within_budget(
    query: str, records: List[Dict[str, str]], word_limit: int
) -> str:
    """Keep the top-BM25 paragraphs up to ``word_limit`` words, joined in source order.

    Each paragraph's BM25 document is ``text + section_title + section_type`` (so
    section terms contribute), ranked against the ORIGINAL query; accumulate
    positive-score paragraphs until the word budget (counted on the TEXT) is
    reached, then re-join the TEXT in document order. Falls back to leading
    paragraphs if nothing scores > 0.
    """
    if not records:
        return ""
    docs = [
        " ".join(
            [
                str(r.get("text", "")),
                str(r.get("section_title", "")),
                str(r.get("section_type", "")),
            ]
        )
        for r in records
    ]
    ranked = _bm25_rank(query, docs)
    positive = [(i, s) for i, s in ranked if s > 0]
    source = positive if positive else [(i, 0.0) for i in range(len(records))]
    kept: List[int] = []
    words = 0
    for index, _ in source:
        kept.append(index)
        words += len(str(records[index].get("text", "")).split())
        if words >= word_limit:
            break
    kept.sort()  # document order → coherent excerpt
    return " ".join(str(records[i].get("text", "")) for i in kept)


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
    idx: int  # reading-order position within its paper


def _build_sentence_items(
    paper_id: str,
    source_text: str,
    registry: Dict[str, _Sentence],
) -> List[Dict[str, str]]:
    """Split text into ``{"id": "<pid>_A", "text": ...}`` items (filter protocol).

    No sentence cap — the full excerpt is shown to the filter. Each sentence is
    also recorded in ``registry`` so judge-selected evidence can later be traced
    back to its paper and restored to reading order.
    """
    sentences = _split_sentences(source_text)
    if not sentences and (source_text or "").strip():
        sentences = [source_text.strip()]
    items: List[Dict[str, str]] = []
    for idx, sentence in enumerate(sentences):
        sid = f"{paper_id}_{_sentence_suffix(idx)}"
        registry[sid] = _Sentence(paper_id=paper_id, text=sentence, idx=idx)
        items.append({"id": sid, "text": sentence})
    return items


def _group_contiguous_spans(sentences: List[_Sentence]) -> List[str]:
    """Join reading-order sentences into contiguous spans.

    Sentences with consecutive in-paper indices form one span; a gap in the
    indices starts a new span. The returned list has one entry per contiguous
    run, so a consumer can tell non-adjacent evidence apart instead of reading a
    single string that silently splices distant sentences together.
    """
    spans: List[List[str]] = []
    prev_idx: Optional[int] = None
    for sent in sentences:
        if prev_idx is None or sent.idx != prev_idx + 1:
            spans.append([])
        spans[-1].append(sent.text)
        prev_idx = sent.idx
    return [" ".join(span) for span in spans]


def _truncate_spans_to_words(spans: List[str], max_words: int) -> List[str]:
    """Cap the total word count across spans, preserving span boundaries.

    Words are kept greedily in order; the span that crosses the budget is
    word-sliced and later spans are dropped, so ``" ".join`` of the result holds
    the first ``max_words`` words.
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
    """Retrieves and ranks evidence passages for a research question."""

    def __init__(
        self,
        full_text_enrichment: bool = True,
        verbose: bool = False,
        llm_base_url: Optional[str] = None,
        llm_model_name: Optional[str] = None,
    ):
        # Full text is the default retrieval path; the flag is kept for harness
        # compatibility (turning it off falls back to abstract-only passages).
        self.full_text_enrichment = full_text_enrichment
        self.verbose = verbose

        # Progress reporting: set per-run in run(); _progress() no-ops when unset.
        self._on_progress: Optional[Callable[[str], None]] = None
        # Shared counters so the two concurrent filter passes report one combined
        # "N/M batches" figure to the progress callback.
        self._filter_lock = threading.Lock()
        self._filter_done = 0
        self._filter_total = 0

        # Explicit constructor args (llm_model_name) take priority over config.
        runtime = load_runtime_config()
        self._max_queries = runtime.max_query_count
        self._query_budget_guidance = runtime.query_budget_guidance
        self._papers_per_subquery = runtime.papers_per_subquery
        self._full_text_target_tokens_per_paper = (
            runtime.full_text_target_tokens_per_paper
        )
        self._abstract_filter_batch = runtime.abstract_filter_batch
        self._fulltext_filter_batch = runtime.fulltext_filter_batch
        self._max_fulltext_papers = runtime.max_fulltext_papers
        self._max_abstract_only_papers = runtime.max_abstract_only_papers
        self._search_page_size = runtime.search_page_size
        self._evidence_snippet_max_words = runtime.evidence_snippet_max_words
        self._filter_temperature = runtime.filter_temperature

        self.llm = create_llm_client(
            base_url=llm_base_url,
            model_name=llm_model_name or runtime.default_model_name,
        )
        self._query_prompt = _QUERY_PROMPT_PATH.read_text(encoding="utf-8")
        self._abstract_filter_prompt = _ABSTRACT_FILTER_PROMPT_PATH.read_text(
            encoding="utf-8"
        )
        self._fulltext_filter_prompt = _FULLTEXT_FILTER_PROMPT_PATH.read_text(
            encoding="utf-8"
        )

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
        self._progress(f"Judging paper relevance · {done}/{total} batches")

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

    # ── Step 3: per-sub-query retrieval ──────────────────────────────────────

    def multithread_routine(
        self,
        subquery: str,
        papers_per_subquery: Optional[int] = None,
        original_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search one sub-query; return up to ``papers_per_subquery`` raw papers.

        Enrichment (full-text fetch + BM25 excerpt) is deferred to ``run`` so it
        happens once per unique paper, in parallel — not per sub-query.
        ``original_query`` is accepted for signature stability but unused here.
        """
        cap = (
            papers_per_subquery
            if papers_per_subquery is not None
            else self._papers_per_subquery
        )
        try:
            papers = search_scientific_literature_structured(
                subquery, page_size=self._search_page_size
            )
        except Exception as exc:  # network/parse failure → skip this sub-query
            self._log(f"search failed for {subquery!r}: {exc}")
            return []
        self._log(f"{subquery!r} → {len(papers[:cap])} papers")
        return papers[:cap]

    def _excerpt_for_paper(self, paper: Dict[str, Any], rank_query: str) -> str:
        """Top BM25 body paragraphs up to the word budget (~2300 words), or ''.

        This full-text excerpt is context-fitting only: it shrinks a long paper so
        the relevance judge can afford to read it. The judge then selects the
        sentences that become the paper's cited evidence. Papers with no
        open-access full text return '' and fall back to their abstract downstream.
        """
        if not self.full_text_enrichment:
            return ""
        paragraphs = self._fetch_body_paragraphs(paper)
        if not paragraphs:
            return ""
        return _select_excerpt_within_budget(
            rank_query,
            paragraphs,
            _per_paper_word_limit(self._full_text_target_tokens_per_paper),
        )

    def _passage_from_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a surviving paper into the returned evidence passage."""
        excerpt = str(paper.get("full_text_excerpt") or "").strip()
        text = excerpt or str(paper.get("abstract") or "").strip()
        # Single citable evidence artifact for every downstream consumer:
        # ``evidence_snippets``, the sentences the relevance judge cited (Step 5)
        # grouped into contiguous spans and kept in reading order. Falls back to
        # the excerpt/abstract as a single span only when the judge cited nothing.
        judge_spans = list(paper.get("evidence_sentences_full_text") or []) or list(
            paper.get("evidence_sentences_abstract") or []
        )
        spans = [s.strip() for s in judge_spans if s and s.strip()] or (
            [text] if text else []
        )
        # Hard cap the TOTAL evidence at the word budget (~250, OpenScholar
        # passage size). Sentence accumulation in the filter stops once the budget
        # is reached, but the last sentence can overshoot, so bound it here too.
        evidence_snippets = _truncate_spans_to_words(
            spans, self._evidence_snippet_max_words
        )
        paper_id = str(
            paper.get("pmid") or paper.get("doi") or paper.get("title") or ""
        )
        # NB: the full ~3072-token ``full_text_excerpt`` is deliberately NOT
        # returned. It exists only as fuel for the internal relevance filter;
        # downstream consumers cite from ``evidence_snippets`` (+ metadata).
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

    # ── Step 5: two-pass LLM relevance filter ────────────────────────────────

    def _evaluate_batch(
        self,
        query: str,
        batch_data: List[Dict[str, Any]],
        prompt_template: str,
        depth: int = 0,
    ) -> List[str]:
        """LLM-judge one batch; return relevant sentence IDs (most→least relevant).

        On a malformed/empty response, split the batch and retry (<=2 levels) so a
        whole batch is never silently dropped.
        """
        prompt = prompt_template.replace("{user_query}", query).replace(
            "{papers_batch}", json.dumps(batch_data, indent=2)
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
                max_tokens=4096,
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
            return [str(x) for x in ids]
        if depth < 2 and len(batch_data) > 1:
            mid = len(batch_data) // 2
            return self._evaluate_batch(
                query, batch_data[:mid], prompt_template, depth + 1
            ) + self._evaluate_batch(
                query, batch_data[mid:], prompt_template, depth + 1
            )
        return []

    def _relevance_filter(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        prompt_template: str,
        text_field: str,
        evidence_key: str,
    ) -> List[Dict[str, Any]]:
        """Keep the papers the LLM judges relevant, ranked most→least relevant.

        ``text_field`` is the per-paper text shown to the judge (``abstract`` for
        the triage pass, ``full_text_excerpt`` for the full-text pass). Cited
        sentences are stashed on each paper as ``evidence_sentences_<key>``.
        """
        if not papers:
            return []
        today = datetime.date.today()
        template = prompt_template.replace("{today_date}", today.isoformat()).replace(
            "{today_year}", str(today.year)
        )
        batch_size = (
            self._abstract_filter_batch
            if text_field == "abstract"
            else self._fulltext_filter_batch
        )

        by_id: Dict[str, Dict[str, Any]] = {}
        sentences: Dict[str, _Sentence] = {}
        batches: List[List[Dict[str, Any]]] = []
        for i in range(0, len(papers), batch_size):
            data = []
            for j, p in enumerate(papers[i : i + batch_size]):
                pid = str(
                    p.get("pmid") or p.get("doi") or p.get("title") or f"paper_{i + j}"
                )
                by_id.setdefault(pid, p)
                affil = "; ".join(
                    f"{a.get('name', '')}: {str(a.get('affiliation', ''))[:200]}"
                    for a in p.get("authorsWithAffiliations", [])
                    if a.get("affiliation")
                )
                data.append(
                    {
                        "id": pid,
                        "title": p.get("title", "N/A"),
                        "authors": p.get("authors", "N/A"),
                        "affiliations": affil,
                        "journal": p.get("journal", "N/A"),
                        "year": p.get("year", "N/A"),
                        "abstract": _build_sentence_items(
                            pid,
                            str(p.get(text_field) or "").strip(),
                            sentences,
                        ),
                    }
                )
            batches.append(data)

        # Judge every batch in parallel (each _evaluate_batch is one LLM call),
        # preserving batch order so the relevance ranking stays stable. Each batch
        # bumps the shared progress counter as it finishes (via a done-callback, so
        # the count reflects real completion, not submission order).
        with self._filter_lock:
            self._filter_total += len(batches)
        ranked_ids: List[str] = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = []
            for b in batches:
                future = pool.submit(self._evaluate_batch, query, b, template)
                future.add_done_callback(self._on_filter_batch_done)
                futures.append(future)
            for future in futures:  # submission order == batch order
                ranked_ids.extend(future.result())

        relevant: List[Dict[str, Any]] = []
        seen: set = set()
        # The judge cites all sentences that support relevance (no length target
        # in its prompt); trim to the evidence word budget (~250 words). Since
        # ``ranked_ids`` arrives most→least relevant, trimming keeps the
        # highest-scoring sentences; they are re-sorted into reading order below.
        budget = self._evidence_snippet_max_words
        evidence_sids: Dict[str, List[str]] = {}
        evidence_words: Dict[str, int] = {}
        for sid in ranked_ids:
            sent = sentences.get(sid)
            if sent is None or sent.paper_id not in by_id:
                continue
            pid = sent.paper_id
            if pid not in seen:
                seen.add(pid)
                relevant.append(by_id[pid])
            if evidence_words.get(pid, 0) >= budget:
                continue
            evidence_sids.setdefault(pid, []).append(sid)
            evidence_words[pid] = evidence_words.get(pid, 0) + len(sent.text.split())
        for pid, sids in evidence_sids.items():
            ordered = sorted(sids, key=lambda s: sentences[s].idx)
            by_id[pid][f"evidence_sentences_{evidence_key}"] = _group_contiguous_spans(
                [sentences[s] for s in ordered]
            )
        self._log(f"{evidence_key} filter: {len(relevant)}/{len(papers)} relevant")
        return relevant

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        max_loops: int = 1,
        top_k: int = _DEFAULT_FINAL_TOP_K,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Single-pass retrieval reproducing the winning bm25_per_paper strategy:
        generate → search+enrich → full-text filter + abstract triage → merge.

        ``on_progress`` (optional) is called with a short human-readable status
        string at each pipeline stage — wire it to a spinner for live feedback.
        """
        if max_loops != 1:
            raise NotImplementedError(
                "LibrarianAgent only supports max_loops=1 (single pass)."
            )
        self._on_progress = on_progress
        return self._run(query, top_k)

    def _run(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Internal implementation of the single-pass retrieval pipeline."""
        self._filter_done = 0  # reset so a reused agent reports fresh counts
        self._filter_total = 0
        self._progress("Generating search queries")
        generated = self._generate_queries(query)
        self._progress("Validating queries")
        queries = self._validate_queries(generated)

        # 1. Search every sub-query in parallel → raw papers. This fan-out does
        # CPU work (search + downstream BM25), so size it to the CPUs actually
        # available rather than a fixed constant — otherwise a large
        # max_query_count can oversubscribe a small container.
        self._progress(f"Searching Europe PMC · {len(queries)} sub-queries")
        all_papers: List[Dict[str, Any]] = []
        search_workers = min(max(len(queries), 1), _available_cpus())
        with ThreadPoolExecutor(max_workers=search_workers) as pool:
            futures = [
                pool.submit(
                    self.multithread_routine, q, self._papers_per_subquery, query
                )
                for q in queries
            ]
            for future in futures:
                all_papers.extend(future.result())

        # 2. Dedup candidate papers across sub-queries BEFORE fetching, so each
        #    unique paper's full text is fetched at most once.
        candidates: List[Dict[str, Any]] = []
        by_pid: Dict[str, Dict[str, Any]] = {}
        for paper in all_papers:
            pid = str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "")
            if pid and pid not in by_pid:
                by_pid[pid] = paper
                candidates.append(paper)
        self._log(f"{len(candidates)} unique candidate papers")
        self._progress(f"Found {len(candidates)} candidate papers")
        if not candidates:
            return []

        # 3. Enrich each unique paper with a full-text excerpt IN PARALLEL (BM25
        #    vs the ORIGINAL query). Doing this once per unique paper, in
        #    parallel, is what keeps a sample within its time budget.
        if self.full_text_enrichment:
            total = len(candidates)
            done = 0
            self._progress(f"Fetching full text · 0/{total}")
            with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
                future_to_paper = {
                    pool.submit(self._excerpt_for_paper, p, query): p
                    for p in candidates
                }
                for future in as_completed(future_to_paper):
                    try:
                        excerpt = future.result()
                    except Exception:
                        excerpt = ""
                    if excerpt:
                        future_to_paper[future]["full_text_excerpt"] = excerpt
                    done += 1
                    self._progress(f"Fetching full text · {done}/{total}")

        # Two independent filters, merged full-text-first (the winning flow):
        #  - full-text filter over papers that have an excerpt,
        #  - abstract triage over ALL candidates (title+abstract).
        # They are independent (different prompts/inputs; they write different
        # keys), so run them concurrently. Each pass still parallelises its own
        # batches internally.
        ft_candidates = [p for p in candidates if p.get("full_text_excerpt")]
        self._progress("Judging paper relevance")
        with ThreadPoolExecutor(max_workers=2) as pool:
            ft_future = pool.submit(
                self._relevance_filter,
                query,
                ft_candidates,
                self._fulltext_filter_prompt,
                "full_text_excerpt",
                "full_text",
            )
            ab_future = pool.submit(
                self._relevance_filter,
                query,
                candidates,
                self._abstract_filter_prompt,
                "abstract",
                "abstract",
            )
            ft_relevant = ft_future.result()
            abstract_relevant = ab_future.result()

        # Compose the answer set from two separately-capped pools so abstract-only
        # papers are never crowded out by full-text ones. The two pools are
        # disjoint by construction:
        #   - fulltext_papers: top papers that have a full-text excerpt, in
        #     full-text-filter relevance order.
        #   - abstract_only_papers: top papers that have ONLY an abstract, in
        #     abstract-filter relevance order.
        fulltext_papers = ft_relevant[: self._max_fulltext_papers]
        abstract_only_candidates = [
            paper for paper in abstract_relevant if not paper.get("full_text_excerpt")
        ]
        abstract_only_papers = abstract_only_candidates[
            : self._max_abstract_only_papers
        ]
        merged = fulltext_papers + abstract_only_papers

        # Fallback: if both filters rejected everything, don't blank the pipeline.
        if not merged:
            merged = candidates
        # Outer ceiling; defaults to the sum of the two pool caps (normally a no-op).
        merged = merged[:top_k]

        # One-line pipeline summary for eval logs.
        n_excerpt = sum(1 for c in candidates if c.get("full_text_excerpt"))
        self.last_run_debug = {
            "search_queries": queries,
            "query_count": len(queries),
            "candidate_count": len(candidates),
            "candidate_pmids": [str(p.get("pmid") or "") for p in candidates],
            "excerpt_count": n_excerpt,
            "ft_relevant_count": len(ft_relevant),
            "abstract_relevant_count": len(abstract_relevant),
            "final_count": len(merged),
            "final_pmids": [str(p.get("pmid") or "") for p in merged],
        }
        logger.debug(
            "[Librarian] queries=%d candidates=%d excerpts=%d/%d "
            "ft_relevant=%d abstract_relevant=%d final=%d",
            len(queries),
            len(candidates),
            n_excerpt,
            len(candidates),
            len(ft_relevant),
            len(abstract_relevant),
            len(merged),
        )

        return [self._passage_from_paper(p) for p in merged]

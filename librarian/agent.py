"""LibrarianAgent — pure literature retrieval over Europe PMC.

Single-pass pipeline (max_loops=1, the only supported mode):

  Stage 1 — Subquery generation (_generate_queries + _validate_queries):
      LLM turns the user question into N Europe PMC sub-queries; drop those
      whose Europe PMC field syntax is broken. The same reply also names ranking
      terms: the words a table answering the question would use, never searched,
      only used to rank Stage 2's paragraphs.
  Stage 2 — Paper-level retrieval + paragraph BM25 ranking (_paragraphs_for_subquery,
      one thread per sub-query):
        - search the sub-query against Europe PMC (papers_per_subquery papers),
        - fetch the open-access full texts of those papers in one batch,
        - decompose every paper into paragraphs (abstract + full-text body chunks
          + supplementary-document chunks when supplementary_enrichment is on,
          no distinction), BM25-rank the pool against the sub-query (fused with
          the ranking terms, see ``_fused_rank``), keep the top k
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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import bm25s
import pysbd

from librarian.config import LibrarianRuntimeConfig
from librarian.fulltext_cache import default_literature_source
from librarian.jats import extract_body_paragraphs, extract_supplementary_captions
from librarian.literature_search import (
    Fulltext,
    LiteratureSource,
    Supplementary,
    SupplementaryOptions,
    declares_supplementary,
    normalize_pmcid,
)
from librarian.llm_client import create_llm_client, parse_json_response
from librarian.query_validation.epmc_rules import BOOLEAN_FIELDS, DATE_FIELDS
from librarian.supplementary import (
    SUPPLEMENTARY_SECTION_TYPE,
    apply_captions,
    require_extraction_dependencies,
)
from librarian.tracing_port import NullTracer, TracingPort

logging.getLogger("bm25s").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Prompts — co-located so this agent is self-contained.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "stage_1_europe_pmc_query_generation.md"
_FILTER_PROMPT_PATH = _PROMPTS_DIR / "stage_3_paragraph_relevance_judge.md"

# Implementation constants — env-independent and never tuned. Every tunable knob
# (query count, page size, evidence words, pool caps, filter batches, ...) lives
# in config.py instead.
#
# Stage 3's judge threads only WAIT on the LLM endpoint (network I/O, no CPU), so
# their pool is a fixed count independent of cores. Stage 2's threads do CPU work
# (search + BM25) and instead size to the CPUs actually available (see
# _available_cpus), so they match the container's cores without oversubscribing.
_JUDGE_WORKERS = 8
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


def _first_affiliation(paper: Dict[str, Any]) -> str:
    """The first non-empty author affiliation on a Europe PMC record, or ''.

    literature_search.py carries a full per-author ``authorsWithAffiliations``
    list, but every display consumer shows a single institution line. The
    record keeps just this one string and drops the list.
    """
    for entry in paper.get("authorsWithAffiliations") or []:
        if not isinstance(entry, dict):
            continue
        affiliation = str(entry.get("affiliation") or "").strip()
        if affiliation:
            return affiliation
    return ""


def _fulltext_pmcid(paper: Dict[str, Any]) -> str:
    """The PMC id whose full text this paper is worth fetching, or ''.

    Gated on ``isOpenAccess``, then ``fullTextIds[0]`` with ``pmcid`` as the
    fallback. Pure, so the batch prefetch and the per-paper lookup derive the
    same key from the same record.
    """
    # Europe PMC serves fullTextXML only for its open-access subset (licensing):
    # free-to-read records (inEPMC=Y, isOpenAccess=N) answer 500 on every call.
    # A 500 is transient to the cache, so without this gate they are refetched
    # every run.
    if not paper.get("isOpenAccess"):
        return ""
    full_text_ids = paper.get("fullTextIds") or []
    raw_id = (full_text_ids[0] if full_text_ids else "") or paper.get("pmcid") or ""
    return normalize_pmcid(raw_id) if str(raw_id).strip() else ""


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


# One clause value: a (group), "phrase", [range], {range} or bare token.
# ponytail: no nested parentheses inside a negated group or filter value.
_EPMC_VALUE = r'(?:\([^)]*\)|"[^"]*"|\[[^\]]*\]|\{[^}]*\}|[^\s()]+)'
# NOT / leading '-' (at start, after whitespace or '(') plus the negated term.
_EPMC_NEGATED_RE = re.compile(r"(?:\bNOT\s+|(?<![^\s(])-)(?:[A-Z_]+:)?" + _EPMC_VALUE)
# Filter fields whose values are dates, Y/N flags or codes, not content words.
_EPMC_FILTER_FIELDS = DATE_FIELDS | BOOLEAN_FIELDS | {"SRC", "LICENSE", "LANG"}
_EPMC_FILTER_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_EPMC_FILTER_FIELDS)) + r"):" + _EPMC_VALUE
)


def _strip_epmc_operators(query: str) -> str:
    """Reduce a Europe PMC query to bare keywords for BM25.

    Drops negated terms (``NOT x``, ``-x``) and filter clauses
    (``PUB_YEAR:[...]``, ``HAS_FT:y``, ``SRC:MED``, ...) entirely, since they
    say nothing about what a relevant passage contains. Then drops field
    specifiers (``TITLE_ABS:``), quotes, parentheses and the boolean operators
    AND/OR/NOT. A plain keyword query is returned unchanged.
    """
    stripped = _EPMC_NEGATED_RE.sub(" ", query)
    stripped = _EPMC_FILTER_RE.sub(" ", stripped)
    stripped = re.sub(r"\b[A-Z_]+:", " ", stripped)
    stripped = stripped.replace('"', " ").replace("(", " ").replace(")", " ")
    tokens = [t for t in stripped.split() if t not in {"AND", "OR", "NOT"}]
    return " ".join(tokens)


def _load_bm25_stemmer() -> Optional[Any]:
    """Snowball (English) stemmer callable for BM25, or ``None`` when disabled.

    Stemming collapses morphological variants (``signaling``/``signal``,
    ``nanoparticles``/``nanoparticle``) so a sub-query keyword matches
    inflected forms in the paragraphs — a recall win for lexical scoring. On
    by default; set ``LITERATURE_BM25_STEMMING`` to a falsey token
    (0/false/no/off) to disable (e.g. for an ablation, or if Snowball
    over-merges a domain term). Built once at import; ``PyStemmer`` is the
    stemmer ``bm25s`` documents. If it is unavailable we fall back to no
    stemming rather than fail the import.
    """
    if os.environ.get("LITERATURE_BM25_STEMMING", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    try:
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer("english").stemWords
    except ImportError:
        logger.warning("PyStemmer not installed — BM25 stemming disabled.")
        return None


# One shared stemmer, built at import (stateless, thread-safe for read-only stem).
_BM25_STEMMER = _load_bm25_stemmer()


def _bm25_rank(query: str, paragraphs: List[str]) -> List[Tuple[int, float]]:
    """Return ``(index, score)`` for every paragraph, highest score first.

    Query and corpus are tokenized with the SAME stemmer (see
    ``_BM25_STEMMER``) so stemmed query terms match stemmed corpus terms;
    stemming both is required or the two token spaces never line up.

    Equal scores are in document order. ``bm25s`` returns ties in reverse
    index order, so without this an all-zero ranking (a sub-query sharing no
    term with the paper) would keep a paper's *last* records, not its first.
    """
    if not paragraphs:
        return []
    clean_query = _strip_epmc_operators(query) or query
    corpus_tokens = bm25s.tokenize(
        paragraphs, stopwords="en", stemmer=_BM25_STEMMER, show_progress=False
    )
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(
        [clean_query], stopwords="en", stemmer=_BM25_STEMMER, show_progress=False
    )
    idxs, scores = retriever.retrieve(
        query_tokens, k=len(paragraphs), show_progress=False
    )
    ranked = [(int(idxs[0][i]), float(scores[0][i])) for i in range(len(idxs[0]))]
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))


# Reciprocal-rank-fusion constant (Cormack et al., 2009): damps the head of each
# ranking so one channel's top hit cannot outvote agreement between channels.
_RRF_K = 60


def _fused_rank(
    subquery: str, ranking_terms: Iterable[str], documents: List[str]
) -> List[Tuple[int, float]]:
    """``_bm25_rank`` for *subquery*, fused with a ranking for the Stage-1 terms.

    A sub-query names what to *search* for, in a paper's words (``never
    smokers``); a data table that answers it is often worded differently
    (``Smoke | No``), so BM25 against the sub-query alone scores it 0. The
    ranking terms are the planner's guess at the table's words. Each channel is
    BM25-ranked on its own and the two rankings are combined by reciprocal-rank
    fusion, ``1 / (_RRF_K + rank)`` per channel, so neither channel's raw BM25
    scale dominates. A channel only votes for documents it scores above zero:
    a document matching neither still scores 0, so callers can still tell
    "matched" from "fallback". Ties stay in document order.

    Without ranking terms this is exactly ``_bm25_rank(subquery, documents)``.
    """
    terms = " ".join(t for t in ranking_terms if t.strip())
    if not terms:
        return _bm25_rank(subquery, documents)
    fused = [0.0] * len(documents)
    for query in (subquery, terms):
        for rank, (index, score) in enumerate(_bm25_rank(query, documents)):
            if score > 0:
                fused[index] += 1.0 / (_RRF_K + rank + 1)
    return sorted(enumerate(fused), key=lambda pair: (-pair[1], pair[0]))


# Paper metadata mixed into a paragraph's BM25 document (scoring only — never
# the returned evidence text), so a sub-query naming it matches at retrieval
# too. The abstract carries the whole citation (also the fields the Stage-3
# judge sees); body chunks carry ONLY the year. Author/journal/title would
# boost every body chunk of a paper equally, hiding which paragraph actually
# matched — so they stay on the abstract; year is a harmless filter-like
# token every chunk can carry.
_ABSTRACT_BM25_FIELDS = ("title", "authors", "journal", "year")
_BODY_BM25_FIELDS = ("year",)


def _bm25_metadata(paper: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    """Space-joined values of ``fields`` from ``paper``, for the BM25 document."""
    return " ".join(str(paper.get(field) or "").strip() for field in fields).strip()


# Most supplementary records extracted (and cached) per paper. Only a safety
# limit: Stage 2 keeps each paper's top max_supplementary_records_per_paper of
# them by BM25 for the sub-query, so which records survive depends on the
# question, not on their position in the files.
_SUPPLEMENTARY_RECORD_SAFETY_LIMIT = 300


# Room for the pieces one record can be re-cut into (``_split_table_records``):
# a record's paragraph coordinate is its stable position times this, plus the
# piece number.
_PIECES_PER_RECORD = 10_000


def _top_supplementary(
    subquery: str,
    supplementary: Supplementary,
    k: int,
    ranking_terms: Iterable[str] = (),
) -> Supplementary:
    """*supplementary* with only its *k* best records for *subquery*, by BM25.

    Ranked with ``_fused_rank``, so a record matching only the Stage-1
    *ranking_terms* can still be kept.

    The kept records stay in document order, so adjacent cited pages remain
    adjacent. Each is a copy tagged with ``supplementary_index``, its position
    among all the paper's records, so its paragraph coordinates are the same
    whichever sub-query selected it; the outcome itself (shared, possibly
    cached) is not modified. When no record matches the sub-query, the first
    *k* are kept (``_bm25_rank`` orders ties by position).
    """
    records = supplementary.records
    keep: Iterable[int] = range(len(records))
    if len(records) > k:
        ranked = _fused_rank(
            subquery, ranking_terms, [_bm25_doc(r) for r in records]
        )
        keep = sorted(index for index, _ in ranked[:k])
    return replace(
        supplementary,
        records=[{**records[i], "supplementary_index": i} for i in keep],
    )


def _bm25_doc(record: Dict[str, Any]) -> str:
    """BM25 document for a paragraph record: text + section title/type + metadata."""
    return " ".join(
        [
            str(record.get("text", "")),
            str(record.get("section_title", "")),
            str(record.get("section_type", "")),
            str(record.get("bm25_metadata", "")),
        ]
    )


def _rank_paragraphs_for_subquery(
    subquery: str,
    records: List[Dict[str, Any]],
    k: int,
    ranking_terms: Iterable[str] = (),
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

    The Stage-1 *ranking_terms* are fused in with ``_fused_rank``, the same
    ranking ``_top_supplementary`` uses, so a supplementary record that matched
    only the terms there is not dropped here.
    """
    if not records or k <= 0:
        return []
    ranked = _fused_rank(subquery, ranking_terms, [_bm25_doc(r) for r in records])
    positive = [i for i, score in ranked if score > 0]
    order = positive if positive else [i for i, _ in ranked]
    return [records[i] for i in order[:k]]


def _words(text: str) -> int:
    return len(text.split())


def _column_groups(
    prefix: str, rows: List[str], max_words: int
) -> List[Tuple[str, List[str]]]:
    """Split a wide table into column groups whose header fits in half the budget.

    Each group keeps the first column (the row's identity) and gets its own
    header, so every piece names exactly the columns it holds. Rows with no
    value in a group are left out of that group.
    """
    *caption, header = prefix.split(" ; ")
    if _words(" ; ".join(caption)) > max_words // 4:  # a long caption would crowd out
        caption = []  # the columns; the header matters more
    names = header.split(" | ")
    limit = max_words // 2 - _words(" ; ".join(caption)) - 1
    groups: List[List[int]] = []
    current: List[int] = []
    size = _words(names[0])
    for column in range(1, len(names)):
        cost = _words(names[column]) + 1  # the cell and its "|"
        if current and size + cost > limit:
            groups.append(current)
            current, size = [], _words(names[0])
        current.append(column)
        size += cost
    if current:
        groups.append(current)
    cells = [row.split(" | ") for row in rows]
    sliced = []
    for group in groups:
        columns = [0] + group
        group_rows = []
        for row in cells:
            values = [row[c] if c < len(row) else "" for c in columns]
            while values and not values[-1]:
                values.pop()
            if any(values[1:]):
                group_rows.append(" | ".join(values))
        if group_rows:
            group_header = " | ".join(names[c] for c in columns)
            sliced.append((" ; ".join(caption + [group_header]), group_rows))
    return sliced


def _row_parts(row: str, room: int) -> List[str]:
    """A row as parts of at most ``room - 1`` words, every column kept in place.

    Each part has all of the row's cells: the first (the row's identity) is
    repeated, and the others are filled in column order with as many of their
    remaining words as fit, empty once used up. So a long cell continues in its
    own column (``SOX2 |  | ...description``), never under another header.
    The ``" ; "`` before a part and every ``" | "`` count as words.
    """
    if _words(row) + 1 <= room:
        return [row]
    identity, *cells = row.split(" | ")
    remaining = [cell.split() for cell in cells]
    budget = room - 1 - _words(identity) - len(cells)  # for the cells' own words
    if budget <= 0:  # the identity and separators alone fill the room
        words = row.split()
        return [
            " ".join(words[i : i + room - 1]) for i in range(0, len(words), room - 1)
        ]
    parts: List[str] = []
    while any(remaining):
        left = budget
        values = [identity]
        for column, words in enumerate(remaining):
            taken = words[:left]
            remaining[column] = words[len(taken) :]
            left -= len(taken)
            values.append(" ".join(taken))
        parts.append(" | ".join(values))
    return parts


def _row_pieces(prefix: str, rows: List[str], max_words: int) -> List[str]:
    """``prefix ; row ; row`` texts of at most ``max_words`` words, whole rows first.

    Every ``" ; "`` separator is a word too, so each row costs its words plus
    one. A row too long on its own is cut by ``_row_parts``, keeping its columns
    aligned under the header and its first cell in every part.
    """
    room = max_words - _words(prefix)  # for "; row ; row ..."
    parts = [part for row in rows for part in _row_parts(row, room)]
    texts: List[str] = []
    chunk: List[str] = []
    used = 0
    for part in parts:
        cost = _words(part) + 1
        if chunk and used + cost > room:
            texts.append(" ; ".join([prefix] + chunk))
            chunk, used = [], 0
        chunk.append(part)
        used += cost
    if chunk:
        texts.append(" ; ".join([prefix] + chunk))
    return texts


def _split_table_records(
    records: List[Dict[str, Any]], max_words: int
) -> List[Dict[str, Any]]:
    """Re-cut table records longer than ``max_words`` so every piece keeps its header.

    ``_chunk_paragraphs`` would split a table like prose, so only its first
    window would keep the column names. Here each piece is the header plus as
    many whole rows as fit, at most ``max_words`` words, so the chunker passes
    it through untouched; a row too long on its own is cut into parts that
    repeat the header and the row's first cell. A table whose header takes more
    than half the budget is first split into column groups, each with its own
    header. Every piece then becomes its own paragraph.
    """
    out: List[Dict[str, Any]] = []
    for record in records:
        prefix = record.get("table_header")
        rows = record.get("table_rows") or []
        base = {
            k: v for k, v in record.items() if k not in ("table_header", "table_rows")
        }
        if not prefix or _words(record["text"]) <= max_words:
            out.append(base)
            continue
        index = int(record.get("source_paragraph_index", 0))  # piece 0, 1, 2, ...
        tables = [(prefix, rows)]
        if _words(prefix) > max_words // 2:
            tables = _column_groups(prefix, rows, max_words)
        for table_prefix, table_rows in tables:
            if _words(table_prefix) >= max_words:  # one enormous header cell
                out.append({**base, "text": " ; ".join([table_prefix] + table_rows)})
                continue
            for text in _row_pieces(table_prefix, table_rows, max_words):
                out.append({**base, "text": text, "source_paragraph_index": index})
                index += 1
    return out


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
    section_title: str = ""  # e.g. "Results", "Abstract", "Supplementary: Table S1"
    section_type: str = ""  # JATS sec-type, "abstract" or "supplementary"


def _build_sentence_items(
    paragraph_id: str,
    paper_id: str,
    source_paragraph_index: int,
    source_word_start: int,
    source_text: str,
    registry: Dict[str, _Sentence],
    section_title: str = "",
    section_type: str = "",
) -> List[Dict[str, str]]:
    """Split one paragraph into ``{"id": "paragraph_0_A", "text": ...}`` judge items.

    ``paragraph_id`` makes sentence ids unique within Stage 3. Paper and
    source coordinates stay internal so citations map back to document order;
    the section is carried along so a cited span can say where it came from.
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
            section_title=section_title,
            section_type=section_type,
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
    return [" ".join(s.text for s in run) for run in _contiguous_runs(sentences)]


def _contiguous_runs(sentences: List[_Sentence]) -> List[List[_Sentence]]:
    """The sentence runs ``_group_contiguous_spans`` joins, one list per span."""
    spans: List[List[_Sentence]] = []
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
        spans[-1].append(sent)
        prev = sent
    return spans


# ── Agent ────────────────────────────────────────────────────────────────────


# Most ranking terms kept from the planner's reply. Each one is a BM25 query
# token for every paragraph of every sub-query, so a long list dilutes the
# words that actually name the answer.
_MAX_RANKING_TERMS = 12


@dataclass(frozen=True)
class SearchPlan:
    """Stage 1's output: the sub-queries to search, and the terms to rank by.

    ``ranking_terms`` are never searched. They are the words a table or
    passage that answers the question would use (``smoking status``,
    ``non-smoker``), fused into both Stage-2 cuts by ``_fused_rank``.
    """

    queries: List[str]
    ranking_terms: List[str] = field(default_factory=list)


def _parse_ranking_terms(parsed: Any) -> List[str]:
    """The reply's ``ranking_terms`` as stripped, case-deduplicated strings, capped.

    A missing or malformed key is no terms, so ranking falls back to the
    sub-query alone rather than failing Stage 1.
    """
    raw = parsed.get("ranking_terms") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []
    seen, terms = set(), []
    for term in raw:
        term = str(term).strip() if isinstance(term, (str, int, float)) else ""
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms[:_MAX_RANKING_TERMS]


@dataclass
class RunResult:
    """Ranked evidence and debug details belonging to one retrieval call."""

    papers: List[Dict[str, Any]]
    debug: Dict[str, Any]


@dataclass
class _RunState:
    """Mutable state owned by one retrieval call, including its judge workers."""

    debug: Dict[str, Any] = field(default_factory=dict)
    on_progress: Optional[Callable[[str], None]] = None
    filter_done: int = 0
    filter_total: int = 0
    filter_lock: threading.Lock = field(default_factory=threading.Lock)
    # One entry per judge LLM call (size, duration, outcome), split retries included.
    judge_requests: List[Dict[str, Any]] = field(default_factory=list)

    def progress(self, message: str) -> None:
        """Send a status message to this call's progress callback."""
        if self.on_progress is not None:
            self.on_progress(message)

    def on_filter_batch_done(self, _future) -> None:
        """Count judge completions under a lock: callbacks run on pool threads."""
        with self.filter_lock:
            self.filter_done += 1
            done, total = self.filter_done, self.filter_total
        self.progress(f"Judging paragraph relevance · {done}/{total} batches")


class LibrarianAgent:
    """Retrieves and ranks evidence for a research question.

    The librarian skill drives this class with a CLI-session LLM client; see
    ``skills/librarian/scripts/search.py`` for what it relies on before changing
    prompts, reply keys or ``RunResult``.

    Per-call state is local, so an instance can serve concurrent runs. Injected
    clients, literature sources and tracers must also support concurrent calls.
    """

    def __init__(
        self,
        runtime_config: LibrarianRuntimeConfig,
        full_text_enrichment: bool = True,
        verbose: bool = False,
        llm_base_url: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        tracer: Optional[TracingPort] = None,
        llm_client: Optional[Any] = None,
        literature_source: Optional[LiteratureSource] = None,
    ):
        """Build a librarian ready to retrieve evidence for a query.

        :param runtime_config: Tuning knobs (sub-query count, page sizes, filter
            temperature, ...). Build one with ``config.load_runtime_config()``
            or a ``dataclasses.replace()`` of it for a one-off override.
        :type runtime_config: LibrarianRuntimeConfig
        :param full_text_enrichment: If ``False``, retrieve abstract-only
            paragraphs instead of fetching open-access full text.
        :type full_text_enrichment: bool
        :param verbose: If ``True``, print per-stage progress and debug info.
        :type verbose: bool
        :param llm_base_url: Override for the LLM endpoint; falls back to the
            client's own default resolution when omitted.
        :type llm_base_url: str or None
        :param llm_model_name: Override for the LLM model name; falls back to
            ``runtime_config.default_model_name`` when omitted.
        :type llm_model_name: str or None
        :param tracer: Span-tracing adapter (see ``tracing_port.py``). Defaults
            to ``NullTracer`` (no-op) — pass your own backend's adapter to
            trace a real run.
        :type tracer: TracingPort or None
        :param llm_client: A ready-made LLM client to use instead of building
            one. It needs ``chat_completion(messages, temperature=...,
            max_tokens=...) -> str``, like ``llm_client.LLMClient``. When
            given, ``llm_base_url`` and ``llm_model_name`` are ignored.
        :type llm_client: object or None
        :param literature_source: Where papers and full texts come from (see
            ``literature_search.LiteratureSource``). Defaults to
            ``default_literature_source()``: Europe PMC, with full texts
            cached on disk when ``LIBRARIAN_CACHE_DIR`` is set.
        :type literature_source: LiteratureSource or None
        """
        # Full text is the default retrieval path; the flag is kept for harness
        # compatibility (turning it off falls back to abstract-only paragraphs).
        self.full_text_enrichment = full_text_enrichment
        self.verbose = verbose
        self._tracer: TracingPort = tracer if tracer is not None else NullTracer()

        # Tuning knobs come from the caller-supplied config (see config.py —
        # load_runtime_config() for the shipped config.toml, or a
        # dataclasses.replace() of it for a one-off override).
        # Explicit constructor args (llm_model_name) take priority over config.
        self._num_subqueries = runtime_config.num_subqueries
        self._query_budget_guidance = runtime_config.query_budget_guidance
        self._papers_per_subquery = runtime_config.papers_per_subquery
        self._paragraphs_per_subquery = runtime_config.paragraphs_per_subquery
        self._paragraphs_per_judge_batch = runtime_config.paragraphs_per_judge_batch
        self._max_paragraph_words = runtime_config.max_paragraph_words
        self._paragraph_overlap_words = runtime_config.paragraph_overlap_words
        self._filter_temperature = runtime_config.filter_temperature
        self._filter_max_tokens = runtime_config.filter_max_tokens
        self._supplementary_enrichment = runtime_config.supplementary_enrichment
        self._max_supplementary_records = (
            runtime_config.max_supplementary_records_per_paper
        )
        self._supplementary_options = SupplementaryOptions(
            max_bytes=runtime_config.max_supplementary_bytes,
            max_records=_SUPPLEMENTARY_RECORD_SAFETY_LIMIT,
            pdf_extractor=runtime_config.supplementary_pdf_extractor,
        )

        self.llm = (
            llm_client
            if llm_client is not None
            else create_llm_client(
                base_url=llm_base_url,
                model_name=llm_model_name or runtime_config.default_model_name,
            )
        )
        self._source: LiteratureSource = (
            literature_source
            if literature_source is not None
            else default_literature_source()
        )
        if self._supplementary_enrichment:
            require_extraction_dependencies(self._supplementary_options.pdf_extractor)
            if getattr(self._source, "fetch_supplementary_many", None) is None:
                logger.warning(
                    "supplementary_enrichment is on but %s has no "
                    "fetch_supplementary_many; no supplementary text will be retrieved",
                    type(self._source).__name__,
                )
        self._query_prompt = _QUERY_PROMPT_PATH.read_text(encoding="utf-8")
        self._filter_prompt = _FILTER_PROMPT_PATH.read_text(encoding="utf-8")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Librarian] {message}")

    def _trace_json(self, payload: Any) -> str:
        """Serialize compact trace payloads safely for span attributes."""
        return json.dumps(payload, default=str, ensure_ascii=True)

    # ── Step 1: query generation ─────────────────────────────────────────────

    def _generate_queries(
        self,
        query: str,
        previous_evidences: Optional[List[str]] = None,
    ) -> SearchPlan:
        """Ask the LLM for diverse Europe PMC sub-queries for ``query``.

        The same reply carries the ranking terms (see ``SearchPlan``), so they
        cost no extra LLM call.
        """
        today = datetime.date.today()
        additional_context = ""
        if previous_evidences:
            additional_context = "Evidence already gathered:\n" + "\n".join(
                f"- {e}" for e in previous_evidences
            )

        # Fill {max_queries} in the budget guidance from the configured cap so the
        # planner is instructed with the same N we later truncate to — no hardcoded
        # count in the prompt.
        budget_guidance = self._query_budget_guidance.replace(
            "{max_queries}", str(self._num_subqueries)
        )
        prompt = (
            self._query_prompt.replace("{today_date}", today.isoformat())
            .replace("{today_year}", str(today.year))
            .replace("{query_budget_guidance}", budget_guidance)
            .replace("{conversation}", f"User: {query}")
            .replace("{additional_context}", additional_context)
        )

        with self._tracer.start_span(
            "librarian.generate_queries",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": self._trace_json({"query": query}),
                "input.mime_type": "application/json",
            },
        ) as span:
            # max_tokens=8192 (NOT 1024): the claude prompt's JSON response is
            # long; truncation → parse failure → a single raw-question search
            # → bad retrieval.
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
                # Retry once, deterministically, demanding strict JSON before
                # falling back to the raw question.
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return only valid JSON with a 'queries' array of Europe "
                            "PMC query strings and a 'ranking_terms' array of words "
                            'a table answering the question would use, e.g. {"queries": '
                            '["q1", "q2"], "ranking_terms": ["t1", "t2"]}. '
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
            result = unique[: self._num_subqueries]
            ranking_terms = _parse_ranking_terms(parsed)
            self._tracer.set_span_attributes(
                span,
                {
                    "search.query_count": len(result),
                    "output.value": self._trace_json(
                        {"queries": result, "ranking_terms": ranking_terms}
                    ),
                    "output.mime_type": "application/json",
                },
            )
        self._log(f"generated {len(result)} queries, ranking terms {ranking_terms}")
        return SearchPlan(result, ranking_terms)

    # ── Step 2: query validation ─────────────────────────────────────────────

    def _validate_queries(self, queries: List[str]) -> List[str]:
        """Validate against EPMC syntax with the deterministic BatchQueryValidator.

        If every query is rejected, keep them all rather than search nothing.
        """
        from librarian.query_validation.batch_validator import BatchQueryValidator

        with self._tracer.start_span(
            "librarian.validate_queries",
            attributes={
                "openinference.span.kind": "CHAIN",
                "search.query_count.input": len(queries),
                "input.value": self._trace_json({"queries": queries}),
                "input.mime_type": "application/json",
            },
        ) as span:
            report = BatchQueryValidator().validate_queries(queries)
            valid = [r["query"] for r in report["results"] if r["valid"]]
            if self.verbose and len(valid) != len(queries):
                self._log(f"query validation kept {len(valid)}/{len(queries)}")
            result = valid or queries
            self._tracer.set_span_attributes(
                span,
                {
                    "search.query_count.valid": len(result),
                    "output.value": self._trace_json({"queries": result}),
                    "output.mime_type": "application/json",
                },
            )
        return result

    # ── Stage 2: per-sub-query retrieval + paragraph ranking ─────────────────

    def _paragraph_records_for_paper(
        self,
        paper: Dict[str, Any],
        fulltexts: Dict[str, Fulltext],
        supplementaries: Optional[Dict[str, Supplementary]] = None,
    ) -> List[Dict[str, Any]]:
        """One paper → its paragraph records: abstract + full-text body chunks
        (+ supplementary-document chunks, see ``_body_paragraphs``).

        Abstract and body paragraphs share one pool with no distinction. Each
        record carries a ``paper`` reference so its metadata is reachable later.

        ``fulltexts`` (and ``supplementaries``, when supplementary enrichment
        is on) are the batches this sub-query already fetched, keyed by
        normalised PMC id; this method only looks its paper up, never fetches.
        """
        records: List[Dict[str, Any]] = []

        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            records.append(
                {
                    "text": abstract,
                    "section_title": "Abstract",
                    "section_type": "abstract",
                    "bm25_metadata": _bm25_metadata(paper, _ABSTRACT_BM25_FIELDS),
                }
            )

        if self.full_text_enrichment:
            body_metadata = _bm25_metadata(paper, _BODY_BM25_FIELDS)
            for record in self._body_paragraphs(paper, fulltexts, supplementaries):
                record["bm25_metadata"] = body_metadata
                records.append(record)

        # Stable coordinates, assigned before re-cutting: the abstract and body
        # by their order (the same for every sub-query), supplementary records
        # after them by their position among all the paper's records, whichever
        # subset this sub-query kept. Equal passages from different sub-queries
        # then merge, and distinct ones never share a coordinate.
        main = [
            r for r in records if r.get("section_type") != SUPPLEMENTARY_SECTION_TYPE
        ]
        position = 0
        for record in records:
            if record.get("section_type") == SUPPLEMENTARY_SECTION_TYPE:
                stable = len(main) + int(record["supplementary_index"])
            else:
                stable, position = position, position + 1
            record["source_paragraph_index"] = stable * _PIECES_PER_RECORD
        records = _split_table_records(records, self._max_paragraph_words)
        chunked = _chunk_paragraphs(
            records,
            self._max_paragraph_words,
            self._paragraph_overlap_words,
        )
        for record in chunked:
            record["paper"] = paper
        return chunked

    def _paragraphs_for_subquery(
        self, subquery: str, ranking_terms: Iterable[str] = ()
    ) -> List[Dict[str, Any]]:
        """One sub-query → its top-``k`` paragraphs, end to end in this thread.

        Search the literature source for the sub-query, fetch the returned
        papers' open-access full texts in one batch, decompose every paper
        into paragraphs (abstract + full-text chunks), BM25-rank the whole
        pool against the sub-query, and keep the top
        ``paragraphs_per_subquery``. Runs entirely in the calling thread so
        ``run`` can fan these out, one thread per sub-query, with no shared
        state. *ranking_terms*, from Stage 1, are fused into both the
        per-paper supplementary cut and the top-k cut.
        """
        # One parent span per sub-query so a trace groups this thread's network
        # and CPU children together instead of interleaving them under stage2.
        with self._tracer.start_span(
            "librarian.subquery",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": subquery,
            },
        ) as subquery_span:
            # Sub-spans split this thread's wall clock into network (search,
            # full text) vs local CPU (paragraph decomposition + BM25).
            # Sub-queries run one per thread, so these durations overlap across
            # siblings — read each on its own, not summed.
            with self._tracer.start_span(
                "librarian.epmc_search",
                attributes={
                    "openinference.span.kind": "RETRIEVER",
                    "input.value": subquery,
                },
            ) as search_span:
                # A search failure propagates out of the thread pool and fails
                # the run, so the caller can tell "the search broke" apart from
                # "no literature found" — silently returning [] here would
                # surface a Europe PMC outage as an empty result set.
                papers = self._source.search(
                    subquery, page_size=self._papers_per_subquery
                )
                self._tracer.set_span_attributes(
                    search_span, {"paper.count": len(papers)}
                )

            # Every open-access full text this sub-query needs, fetched in one
            # call before any paper is decomposed, so a source can fetch the
            # batch concurrently.
            fulltexts: Dict[str, Fulltext] = {}
            if self.full_text_enrichment:
                with self._tracer.start_span(
                    "librarian.epmc_fulltext",
                    attributes={
                        "openinference.span.kind": "RETRIEVER",
                        "paper.count": len(papers),
                    },
                ) as fulltext_span:
                    fulltexts = self._source.fetch_fulltext_many(
                        [_fulltext_pmcid(p) for p in papers]
                    )
                    self._tracer.set_span_attributes(
                        fulltext_span,
                        {"fulltext.count": sum(1 for f in fulltexts.values() if f.ok)},
                    )

            # Supplementary files, only when enrichment is on, only for papers
            # whose JATS declares any, and only if the source can fetch them.
            supplementaries: Dict[str, Supplementary] = {}
            fetch_supplementary = getattr(
                self._source, "fetch_supplementary_many", None
            )
            if self._supplementary_enrichment and fetch_supplementary is not None:
                wanted = [
                    pmcid
                    for pmcid, f in fulltexts.items()
                    if f.ok and declares_supplementary(f.xml)
                ]
                if wanted:
                    with self._tracer.start_span(
                        "librarian.epmc_supplementary",
                        attributes={
                            "openinference.span.kind": "RETRIEVER",
                            "paper.count": len(wanted),
                        },
                    ) as supplementary_span:
                        supplementaries = {
                            pmcid: _top_supplementary(
                                subquery,
                                outcome,
                                self._max_supplementary_records,
                                ranking_terms,
                            )
                            for pmcid, outcome in fetch_supplementary(
                                wanted, self._supplementary_options
                            ).items()
                        }
                        self._tracer.set_span_attributes(
                            supplementary_span,
                            {
                                "supplementary.count": sum(
                                    1 for r in supplementaries.values() if r.records
                                )
                            },
                        )

            # Paragraph decomposition is counted here with BM25: both are local
            # CPU on already-fetched bytes.
            with self._tracer.start_span(
                "librarian.bm25_rank",
                attributes={
                    "openinference.span.kind": "RERANKER",
                    "paper.count": len(papers),
                    "input.value": subquery,
                },
            ) as rank_span:
                pool: List[Dict[str, Any]] = []
                for paper in papers:
                    pool.extend(
                        self._paragraph_records_for_paper(
                            paper, fulltexts, supplementaries
                        )
                    )

                top = _rank_paragraphs_for_subquery(
                    subquery, pool, self._paragraphs_per_subquery, ranking_terms
                )
                self._tracer.set_span_attributes(
                    rank_span,
                    {
                        "paragraph.count.pool": len(pool),
                        "paragraph.count.kept": len(top),
                    },
                )
            self._tracer.set_span_attributes(
                subquery_span,
                {
                    "paper.count": len(papers),
                    "paragraph.count.kept": len(top),
                },
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
        spans = paper.get("evidence_sentences_full_text") or []
        sections = paper.get("evidence_sections") or [""] * len(spans)
        cited = [
            (span.strip(), section)
            for span, section in zip(spans, sections)
            if span and span.strip()
        ]
        abstract = str(paper.get("abstract") or "").strip()
        # Every span the judge cited for this paper — no word cap. Falls back
        # to the abstract as a single span when the judge cited nothing.
        if not cited and abstract:
            cited = [(abstract, "Abstract")]
        paper_id = _candidate_id(paper)
        return {
            "evidence_snippets": [span for span, _ in cited],
            # Parallel to evidence_snippets: the section each span was cited
            # from ("Results", "Abstract", "Supplementary: <file> (p. 3)", ...).
            "evidence_sections": [section for _, section in cited],
            "paper_id": paper_id,
            "title": str(paper.get("title") or "No title"),
            "authors": str(paper.get("authors") or "Unknown authors"),
            "journal": str(paper.get("journal") or ""),
            "year": str(paper.get("year") or ""),
            "pmid": str(paper.get("pmid") or ""),
            # PMC id of the open-access full text (and supplementary files)
            # the evidence was read from; "" for an abstract-only paper.
            "pmcid": _fulltext_pmcid(paper),
            "doi": str(paper.get("doi") or ""),
            # Europe PMC record identity, kept verbatim under the upstream
            # camelCase keys literature_search.py already uses. Preprints
            # (source="PPR") have NO pmid, so without these a citation cannot
            # be linked back to its EPMC article page.
            "epmcId": str(paper.get("epmcId") or ""),
            "epmcSource": str(paper.get("epmcSource") or paper.get("sourceCode") or ""),
            "url": str(paper.get("url") or ""),
            "affiliation": _first_affiliation(paper),
            "has_fulltext": bool(paper.get("inEPMC") or paper.get("hasFreeFullText")),
        }

    # ── Body extraction from the prefetched full text ────────────────────────

    def _body_paragraphs(
        self,
        paper: Dict[str, Any],
        fulltexts: Dict[str, Fulltext],
        supplementaries: Optional[Dict[str, Supplementary]] = None,
    ) -> List[Dict[str, str]]:
        """Body paragraph records for one paper from the prefetched batch (or []).

        Three ways to legitimately get nothing back, all of which leave the
        paper abstract-only: it is not open access, its full text could not be
        fetched, or the JATS had no usable body. Only the fetch failure is
        worth logging — the other two are ordinary.

        With ``supplementary_enrichment`` on, the text of the paper's
        supplementary PDF / Word files follows the body records, each tagged
        ``section_type="supplementary"``.
        """
        pmcid = _fulltext_pmcid(paper)
        if not pmcid:
            return []
        fulltext = fulltexts.get(pmcid)
        if fulltext is None:
            return []
        if not fulltext.ok:
            self._log(f"full-text unavailable for {pmcid}: {fulltext.error}")
            return []
        records = extract_body_paragraphs(fulltext.xml)
        if self._supplementary_enrichment:
            supplementary = (supplementaries or {}).get(pmcid)
            records.extend(
                self._supplementary_paragraphs(pmcid, fulltext, supplementary)
            )
        return records

    def _supplementary_paragraphs(
        self,
        pmcid: str,
        fulltext: Fulltext,
        supplementary: Optional[Supplementary],
    ) -> List[Dict[str, str]]:
        """Paragraph records from one paper's prefetched supplementary outcome (or []).

        The records come extracted (and possibly cached) from the source; only
        the captions from this paper's JATS are applied here.
        """
        if supplementary is None:
            return []
        if not supplementary.ok:
            self._log(f"supplementary unavailable for {pmcid}: {supplementary.error}")
            return []
        if supplementary.failed:
            self._log(f"{pmcid}: could not read supplementary {supplementary.failed}")
        if not supplementary.records:
            return []
        # Copies: the caller annotates them, and the records may be shared by
        # other sub-queries or held in the cache.
        records = [
            {"supplementary_index": index, **record}
            for index, record in enumerate(
                apply_captions(
                    supplementary.records, extract_supplementary_captions(fulltext.xml)
                )
            )
        ]
        files = sorted({r["supplementary_file"] for r in records})
        self._log(
            f"{pmcid}: {len(records)} supplementary paragraphs from "
            f"{len(files)} file(s) {files}"
        )
        return records

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
        state: _RunState,
        depth: int = 0,
    ) -> List[str]:
        """LLM-judge one batch; return relevant sentence IDs (most→least relevant).

        On a malformed/empty response, split the batch and retry (<=2 levels)
        so a whole batch is never silently dropped. Every call is logged on
        ``state.judge_requests`` with its outcome: ``ok``, ``salvaged`` (ids
        recovered from a truncated reply), ``unparseable`` or ``exception``.
        """
        prompt = prompt_template.replace("{user_query}", query).replace(
            "{paragraphs_batch}", json.dumps(batch_data, indent=2)
        )
        request = {
            "depth": depth,
            "paragraphs": len(batch_data),
            "sentences": sum(len(item["sentences"]) for item in batch_data),
            "prompt_chars": len(prompt),
            "outcome": "",
            "error": "",
        }
        started = time.monotonic()
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
                max_tokens=self._filter_max_tokens,
            )
        except Exception as exc:
            self._log(f"filter batch error: {exc}")
            request["outcome"] = "exception"
            request["error"] = f"{type(exc).__name__}: {exc}"[:300]
            response = ""
        request["seconds"] = round(time.monotonic() - started, 1)

        if self.verbose:
            preview = str(response).replace("\n", " ")[:200]
            self._log(f"filter raw response (batch={len(batch_data)}): {preview!r}")

        parsed = parse_json_response(response)
        ids = parsed.get("relevant_ids") if isinstance(parsed, dict) else None
        result: Optional[List[str]] = None
        if isinstance(ids, list):
            # Deduplicated even on the happy path: the judge can repeat ids
            # without truncating, which inflates the caller's ranking walk.
            request["outcome"] = "ok"
            result = _dedupe_preserving_order(str(x) for x in ids)
        else:
            # Unparseable (usually truncated mid-array): keep the prefix that
            # arrived rather than re-judging or discarding the batch.
            salvaged = self._salvage_relevant_ids(str(response))
            if salvaged:
                self._log(
                    f"salvaged {len(salvaged)} ids from an unparseable judge reply"
                )
                request["outcome"] = "salvaged"
                result = salvaged
            else:
                request["outcome"] = request["outcome"] or "unparseable"
        with state.filter_lock:
            state.judge_requests.append(request)
        if result is not None:
            return result

        if depth < 2 and len(batch_data) > 1:
            mid = len(batch_data) // 2
            return self._evaluate_batch(
                query, batch_data[:mid], prompt_template, state, depth + 1
            ) + self._evaluate_batch(
                query, batch_data[mid:], prompt_template, state, depth + 1
            )
        return []

    def _relevance_filter(
        self, query: str, paragraphs: List[Dict[str, Any]], state: _RunState
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
            # Metadata shown to the judge — the same fields folded into the
            # abstract's BM25 doc at Stage 2 (_ABSTRACT_BM25_FIELDS).
            item = {field: paper.get(field, "N/A") for field in _ABSTRACT_BM25_FIELDS}
            items.append(
                {
                    "id": paragraph_id,
                    **item,
                    "sentences": _build_sentence_items(
                        paragraph_id,
                        paper_id,
                        int(paragraph["source_paragraph_index"]),
                        int(paragraph["word_start"]),
                        str(paragraph.get("text") or "").strip(),
                        registry,
                        section_title=str(paragraph.get("section_title") or ""),
                        section_type=str(paragraph.get("section_type") or ""),
                    ),
                }
            )

        ranked_ids = self._judge_items_in_batches(query, template, items, state)
        return self._papers_from_cited_sentences(ranked_ids, registry, paper_by_id)

    def _judge_items_in_batches(
        self, query: str, template: str, items: List[Dict[str, Any]], state: _RunState
    ) -> List[str]:
        """Send items to the judge in batches of ``paragraphs_per_judge_batch`` → sentence ids.

        Batches run in parallel; results are concatenated in batch order so
        the overall most→least-relevant ranking is preserved. Each batch bumps
        the shared progress counter as it finishes (via a done-callback, so
        the count reflects real completion, not submission order).
        """
        batch_size = self._paragraphs_per_judge_batch
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        with state.filter_lock:
            state.filter_total += len(batches)
        ranked_ids: List[str] = []
        bound_evaluate = self._tracer.bind_current_trace_context(self._evaluate_batch)
        # Judge batches only wait on the LLM (network I/O) — size to
        # _JUDGE_WORKERS, not the CPU count.
        with ThreadPoolExecutor(max_workers=_JUDGE_WORKERS) as pool:
            futures = []
            for batch in batches:
                future = pool.submit(bound_evaluate, query, batch, template, state)
                future.add_done_callback(state.on_filter_batch_done)
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
            runs = _contiguous_runs([registry[s] for s in ordered])
            paper_by_id[paper_id]["evidence_sentences_full_text"] = [
                " ".join(s.text for s in run) for run in runs
            ]
            paper_by_id[paper_id]["evidence_sections"] = [
                run[0].section_title for run in runs
            ]
            paper_by_id[paper_id]["evidence_section_types"] = [
                run[0].section_type for run in runs
            ]
        self._log(f"Stage 3: {len(relevant)} relevant papers")
        return relevant

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        max_loops: int = 1,
        on_progress: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> RunResult:
        """Single-pass retrieval: Stage 1 (generate + validate sub-queries) →
        Stage 2 (per-sub-query paragraph BM25 ranking) → Stage 3 (relevance
        judge over the paragraphs, regrouped to papers). Returns every paper
        the judge keeps in ``RunResult.papers``, with no final top_k cap.

        ``on_progress`` (optional) is called with a short human-readable
        status string at each pipeline stage — wire it to a spinner for live
        feedback. It may also be called from worker threads.

        ``progress_callback`` (optional) is called with the step number, 0
        (starting), 1 (sub-queries ready, searching) and 2 (paragraphs
        pooled, judging), always from the thread that called ``run``. An
        exception it raises propagates out of ``run``, so a caller can use it
        as a cancellation checkpoint.

        ``RunResult.debug`` describes this run: at least
        ``search_queries`` once Stage 1 is done, plus ``query_count``,
        ``paragraph_count``, ``supplementary_paragraph_count``,
        ``relevant_count``, ``supplementary_evidence_count`` and
        ``final_pmids`` when Stage 3 ran. Pipeline failures are re-raised with
        partial debug details on the exception's ``run_debug`` attribute.
        """
        if max_loops != 1:
            raise NotImplementedError(
                "LibrarianAgent only supports max_loops=1 (single pass)."
            )
        state = _RunState(on_progress=on_progress)

        with self._tracer.start_span(
            "librarian.run",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": self._trace_json({"query": query}),
                "input.mime_type": "application/json",
            },
        ) as run_span:
            try:
                result = self._run(query, state, progress_callback)
                evidence_items = result.papers
            except Exception as exc:
                exc.run_debug = state.debug
                self._tracer.mark_span_error(run_span, exc)
                raise
            self._tracer.set_span_attributes(
                run_span,
                {
                    "retrieval.evidence_count": len(evidence_items),
                    "output.value": self._trace_json(
                        {
                            "evidence_count": len(evidence_items),
                            "evidence": [
                                {
                                    "pmid": p.get("pmid"),
                                    "title": p.get("title"),
                                    "year": p.get("year"),
                                }
                                for p in evidence_items[:25]
                            ],
                        }
                    ),
                    "output.mime_type": "application/json",
                },
            )
        return result

    def _run(
        self,
        query: str,
        state: _RunState,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> RunResult:
        """Internal implementation of the single-pass retrieval pipeline."""
        debug = state.debug
        if progress_callback:
            progress_callback(0)
        state.progress("Generating search queries")
        plan = self._generate_queries(query)
        state.progress("Validating queries")
        queries = self._validate_queries(plan.queries)
        ranking_terms = list(plan.ranking_terms)
        debug["search_queries"] = queries
        debug["ranking_terms"] = ranking_terms
        if progress_callback:
            progress_callback(1)

        # Stage 2 — one thread per sub-query: search EPMC, decompose papers
        # into paragraphs, BM25-rank against that sub-query, keep its top-k
        # paragraphs. Concatenate every sub-query's paragraphs into one flat
        # pool, merging duplicate/overlapping selections. This flat paragraph
        # pool is what Stage 3 judges.
        state.progress(f"Searching Europe PMC · {len(queries)} sub-queries")
        with self._tracer.start_span(
            "librarian.stage2_paragraphs",
            attributes={
                "openinference.span.kind": "CHAIN",
                "search.query_count": len(queries),
                "input.value": self._trace_json({"queries": queries}),
                "input.mime_type": "application/json",
            },
        ) as stage2_span:
            bound_stage2 = self._tracer.bind_current_trace_context(
                self._paragraphs_for_subquery
            )
            # One CPU-bound thread per sub-query, capped at the CPUs
            # actually available so a large num_subqueries can't
            # oversubscribe a small container.
            stage2_workers = min(max(len(queries), 1), _available_cpus())
            with ThreadPoolExecutor(max_workers=stage2_workers) as pool:
                per_subquery = pool.map(
                    lambda subquery: bound_stage2(subquery, ranking_terms), queries
                )
            paragraphs = _merge_selected_paragraphs(
                [para for subquery_paras in per_subquery for para in subquery_paras]
            )
            self._tracer.set_span_attributes(
                stage2_span,
                {
                    "paragraph.count": len(paragraphs),
                    "output.value": self._trace_json(
                        {"paragraph_count": len(paragraphs)}
                    ),
                    "output.mime_type": "application/json",
                },
            )
        self._log(f"{len(paragraphs)} paragraphs after Stage 2")
        state.progress(f"Found {len(paragraphs)} candidate paragraphs")
        if progress_callback:
            progress_callback(2)
        if not paragraphs:
            return RunResult([], debug)

        # Stage 3 — the relevance judge reads the paragraphs (batched by
        # paragraphs_per_judge_batch paragraphs per call; a paper's
        # paragraphs may span batches), cites the supporting sentences, and
        # those regroup by paper. Returns the cited papers, most→least
        # relevant. Stage 3's output IS the final set — variable size, with
        # no final top_k cap applied on top of the judge's decision.
        state.progress("Judging paragraph relevance")
        with self._tracer.start_span(
            "librarian.filter_relevance",
            attributes={
                "openinference.span.kind": "CHAIN",
                "paragraph.count": len(paragraphs),
                "input.value": self._trace_json({"paragraph_count": len(paragraphs)}),
                "input.mime_type": "application/json",
            },
        ) as filter_span:
            relevant_papers = self._relevance_filter(query, paragraphs, state)
            self._tracer.set_span_attributes(
                filter_span,
                {
                    "paper.count.relevant": len(relevant_papers),
                    "output.value": self._trace_json(
                        {"relevant_count": len(relevant_papers)}
                    ),
                    "output.mime_type": "application/json",
                },
            )

        debug.update(
            {
                "query_count": len(queries),
                "paragraph_count": len(paragraphs),
                # How much of the Stage-2 pool, and of the judge-cited evidence,
                # came from supplementary files rather than abstract/body.
                "supplementary_paragraph_count": sum(
                    1
                    for p in paragraphs
                    if p.get("section_type") == SUPPLEMENTARY_SECTION_TYPE
                ),
                "supplementary_evidence_count": sum(
                    1
                    for p in relevant_papers
                    for t in p.get("evidence_section_types") or []
                    if t == SUPPLEMENTARY_SECTION_TYPE
                ),
                "relevant_count": len(relevant_papers),
                "final_pmids": [str(p.get("pmid") or "") for p in relevant_papers],
                # What Stage 3 was actually sent: merged passages can run past
                # max_paragraph_words, so the longest is measured, not assumed.
                "longest_passage_chars": max(len(p["text"]) for p in paragraphs),
                "judge_requests": state.judge_requests,
            }
        )
        logger.debug(
            "[Librarian] queries=%d paragraphs=%d relevant=%d",
            len(queries),
            len(paragraphs),
            len(relevant_papers),
        )

        return RunResult([self._passage_from_paper(p) for p in relevant_papers], debug)

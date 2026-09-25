"""
Librarian-backed retrieval for the evidence-verification pipeline.

This module is the librarian counterpart of the PubMed / Semantic Scholar
retrieval functions in ``evidence_api.py``.  It builds an in-process
``LibrarianAgent`` and adds the papers it returns to an ``EvidenceState``:

    librarian_search            — run a natural-language query
    librarian_search_for_claim  — initial claim/subclaim search
    librarian_search_for_gaps   — combined gap-targeted search
    librarian_search_for_gap    — single-gap targeted search
    librarian_refine_search     — refined search after failed extraction

Design notes
------------
* The librarian is a ``LibrarianAgent`` from the ``librarian`` package,
  created in-process.  It implements the winning BM25-per-paper retrieval
  strategy: multi-query search → full-text BM25 excerpt → two-pass LLM filter.
  Its LLM (query planner, relevance filter) connects to the vLLM model service
  via an OpenAI-compatible base URL — configured separately from the model that
  runs the ProClaim evidence programmer.
* The librarian LLM endpoint is read from the environment:
      LIBRARIAN_LLM_BASE_URL — OpenAI-compatible base URL of the vLLM service
                               (e.g. http://localhost:8000/v1)
      LIBRARIAN_LLM_MODEL    — model id served by that endpoint
* ``LibrarianAgent.run()`` returns a list of passage dicts, each carrying the
  compact ``evidence_snippets`` (~250-word cited passage as contiguous spans) for a
  paper together with pmid, title, authors, and filter-cited sentences.  ProClaim
  still runs its own fact extraction over ``PaperRecord.full_text``.
* Reaching ``LibrarianAgent`` requires the ``librarian`` package to be
  importable — `uv sync` at the repository root installs it.
"""

import logging
import os
import threading
from typing import Callable

from proclaim.verification.data_models import PaperRecord
from proclaim.verification.evidence_state import EvidenceState
from evals.Literature.ProClaim.backend.librarian_query_generator import (
    generate_librarian_combined_gaps_query,
    generate_librarian_gap_query,
    generate_librarian_refine_query,
    generate_librarian_subclaim_query,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Librarian agent construction
# ---------------------------------------------------------------------------

# Cached per-thread librarian agent. LibrarianAgent keeps per-run mutable
# state (last_run_debug), so parallel retrieval calls should not share one
# instance.
_LIBRARIAN_AGENT_LOCAL = threading.local()


def _get_librarian_agent():
    """Build (once per thread) and return the in-process LibrarianAgent.

    The agent's LLM is pointed at the model service via
    ``LIBRARIAN_LLM_BASE_URL`` / ``LIBRARIAN_LLM_MODEL``.
    """
    cached_agent = getattr(_LIBRARIAN_AGENT_LOCAL, "agent", None)
    if cached_agent is not None:
        return cached_agent

    try:
        from librarian import LibrarianAgent, load_runtime_config
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import the librarian package. Install it into this "
            "environment (`uv sync` at the repository root), or put the "
            "repository root on PYTHONPATH."
        ) from exc

    base_url = os.environ.get("LIBRARIAN_LLM_BASE_URL") or None
    model = os.environ.get("LIBRARIAN_LLM_MODEL") or None
    if not model:
        logger.warning(
            "LIBRARIAN_LLM_MODEL is not set; the librarian agent will fall back "
            "to its default model resolution (LLM_MODEL / runtime default)."
        )
    verbose = os.environ.get("EVIDENCE_DEBUG", "0") == "1"

    # The agent used to take reasoning_effort= as a constructor argument. The
    # shipped LibrarianAgent has no such parameter: its LLMClient reads
    # LLM_REASONING_EFFORT once, at construction. Setting it here keeps the
    # "low" this benchmark was measured with, without overriding a caller who
    # asked for something else.
    os.environ.setdefault("LLM_REASONING_EFFORT", "low")

    agent = LibrarianAgent(
        runtime_config=load_runtime_config(),
        llm_base_url=base_url,
        llm_model_name=model,
        full_text_enrichment=True,
        verbose=verbose,
    )
    _LIBRARIAN_AGENT_LOCAL.agent = agent
    logger.info(
        "LibrarianAgent ready (base_url=%s, model=%s)",
        base_url or "<default>",
        model or "<default>",
    )
    return agent



# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


def _resolve_passage_id(passage: dict) -> str:
    """Return a stable identifier for a LibrarianAgent passage dict.

    Prefers the real PMID; falls back to ``paper_id`` (which LibrarianAgent
    derives from pmid → doi → title) so passages without a PubMed id can
    still be stored and cited.
    """
    pmid = str(passage.get("pmid") or "").strip()
    if pmid:
        return pmid
    paper_id = str(passage.get("paper_id") or "").strip()
    if paper_id:
        return paper_id
    return ""


def _passage_authors(passage: dict) -> list[str]:
    """Extract a list[str] of author names from a LibrarianAgent passage dict.

    ``LibrarianAgent`` stores authors as a comma-separated string.
    """
    authors = passage.get("authors")
    if isinstance(authors, list):
        return [str(a) for a in authors if a]
    if isinstance(authors, str) and authors.strip():
        return [part.strip() for part in authors.split(",") if part.strip()]
    return []


def _add_librarian_papers(
    passages: list,
    state: EvidenceState,
) -> list[str]:
    """Convert ``LibrarianAgent.run()`` passages into PaperRecords in state.

    ``LibrarianAgent`` returns a list of passage dicts, each containing:
    - ``text``: the most relevant text for this paper (full-text BM25 excerpt
      or abstract fallback) — used directly as the extraction text for ProClaim
      fact extraction.
    - ``evidence_sentences_full_text`` / ``evidence_sentences_abstract``: filter-
      cited sentences appended after ``text`` for extra extraction signal.
    - ``pmid``, ``paper_id``, ``title``, ``authors`` — paper metadata.

    Returns the list of newly added identifiers. Papers already present in
    ``state.papers`` are skipped.
    """
    added_ids: list[str] = []
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        pid = _resolve_passage_id(passage)
        if not pid or pid in state.papers:
            continue

        # ``evidence_snippets`` is the librarian's compact (~250-word) cited
        # passage as contiguous spans — the content it returns now that the heavy
        # full-text excerpt is kept internal to its relevance filter.
        extraction_text = " ".join(passage.get("evidence_snippets") or []).strip() or None

        record = PaperRecord(
            pmid=pid,
            title=str(passage.get("title") or ""),
            abstract=extraction_text or "",
            full_text=extraction_text,
            authors=_passage_authors(passage),
            doi=None,
            source="librarian",
        )
        state.add_paper(record)
        added_ids.append(pid)

    state.token_estimate = state.token_count()
    return added_ids


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def _fetch_librarian_passages(query: str) -> list:
    """Run one librarian search for *query* and return passage dicts.

    ``LibrarianAgent.run()`` returns a list of passage dicts (one per relevant
    paper).  Import/runtime failures raise so evaluation runs do not confuse
    infrastructure errors with successful searches that found no papers.
    """
    try:
        agent = _get_librarian_agent()
    except RuntimeError as exc:
        message = f"Librarian unavailable: {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc

    try:
        passages = agent.run(query=query)
    except Exception as exc:
        message = f"Librarian search failed for query {query!r}: {exc}"
        logger.warning(message)
        raise RuntimeError(message) from exc

    return passages or []


def _run_librarian(
    query: str,
    state: EvidenceState,
) -> list[str]:
    """Run one librarian search for *query* and add the papers to *state*.

    Returns the list of newly added identifiers.
    """
    passages = _fetch_librarian_passages(query)
    added = _add_librarian_papers(passages, state)
    logger.info(
        "Librarian search: query=%r retrieved=%d added=%d",
        query,
        len(passages),
        len(added),
    )
    return added


# ---------------------------------------------------------------------------
# Public retrieval functions (the librarian-backed evidence API)
# ---------------------------------------------------------------------------


def librarian_search(
    query: str,
    state: EvidenceState,
) -> list[str]:
    """Run a single natural-language librarian search. Returns added PMIDs.

    Use this when you already have a well-formed natural-language information
    need. For claim/gap/refine searches prefer the dedicated functions below,
    which generate the query for you.
    """
    added = _run_librarian(query, state)
    print(
        f"librarian_search: added {len(added)} new paper(s). "
        f"PMIDs: {', '.join(added) if added else 'none'}"
    )
    return added


def librarian_search_for_claim(
    claim: str,
    state: EvidenceState,
    llm: Callable,
) -> list[str]:
    """Initial librarian search for a claim, using one subclaim-aware request.

    Runs one deterministic natural-language request. When subclaims are present,
    they are included in the same request so the librarian searches for papers
    about the claim while focusing on the decomposed evidence needs. If no
    subclaims are present, the request falls back to the claim-only form.

    Returns the list of added PMIDs.
    """
    query = generate_librarian_subclaim_query(
        claim,
        llm,
        subclaims=state.subclaims,
        extraction_context=state.extraction_context or None,
    )
    passages = _fetch_librarian_passages(query)
    added = _add_librarian_papers(passages, state)
    logger.info(
        "Librarian search: query=%r retrieved=%d added=%d",
        query,
        len(passages),
        len(added),
    )

    print(
        f"librarian_search_for_claim: added {len(added)} new paper(s) with 1 query. "
        f"PMIDs: {', '.join(added) if added else 'none'}"
    )
    return added


def librarian_search_for_gap(
    gap_description: str,
    state: EvidenceState,
    llm: Callable,
) -> list[str]:
    """Gap-targeted librarian search.

    Generates a natural-language information need focused on a specific
    evidence gap and runs the librarian. This is the librarian-backed
    replacement for ``search_for_gap`` and
    ``search_semantic_scholar_recommendations``.

    Returns the list of added PMIDs.
    """
    query = generate_librarian_gap_query(state.claim, gap_description, llm)
    added = _run_librarian(query, state)
    print(
        f"librarian_search_for_gap: added {len(added)} new paper(s). "
        f"PMIDs: {', '.join(added) if added else 'none'}"
    )
    return added


def librarian_search_for_gaps(
    gap_descriptions: list[str],
    state: EvidenceState,
    llm: Callable,
) -> list[str]:
    """Run one librarian search that targets all current evidence gaps.

    Prefer this over looping over ``librarian_search_for_gap``. It asks the
    subagent to compress all gap descriptions into one concise natural-language
    information need, then performs a single librarian search.

    Returns the list of added PMIDs.
    """
    query = generate_librarian_combined_gaps_query(
        state.claim,
        gap_descriptions,
        llm,
    )
    added = _run_librarian(query, state)
    print(
        f"librarian_search_for_gaps: searched {len(gap_descriptions)} gap(s), "
        f"added {len(added)} new paper(s). "
        f"PMIDs: {', '.join(added) if added else 'none'}"
    )
    return added


def librarian_refine_search(
    failed_pmids: list[str],
    state: EvidenceState,
    llm: Callable,
) -> list[str]:
    """Refined librarian search after extraction found no decisive facts.

    Inspects papers that yielded zero facts or only neutral/ambiguous facts,
    generates a more precise natural-language information need, and runs the
    librarian. This is the librarian-backed replacement for
    ``refine_search_for_failed_papers``.

    Returns the list of newly added PMIDs.
    """
    if not failed_pmids:
        return []

    failed_papers: list[dict] = []
    for pmid in failed_pmids[:5]:
        paper = state.papers.get(pmid)
        if paper:
            failed_papers.append(
                {"pmid": pmid, "title": paper.title, "abstract": paper.abstract}
            )
    if not failed_papers:
        return []

    query = generate_librarian_refine_query(
        state.claim,
        failed_papers,
        llm,
    )
    added = _run_librarian(query, state)
    print(
        f"librarian_refine_search: added {len(added)} new paper(s). "
        f"PMIDs: {', '.join(added) if added else 'none'}"
    )
    return added

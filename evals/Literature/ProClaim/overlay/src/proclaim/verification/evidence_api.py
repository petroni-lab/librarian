"""
Evidence API — pure Python library for evidence manipulation.

All functions operate on EvidenceState objects in-memory. No disk I/O,
no MCP dependency. 
"""

import json
import logging
import os
import re
import warnings as _warnings
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root (relative to this file)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
load_dotenv(_project_root / ".env")

from proclaim.verification.compressor import SufficiencyPreservingCompressor
from proclaim.verification.data_models import (
    Conflict,
    Fact,
    PaperRecord,
    Stance,
    SufficiencyResult,
    VerificationVerdict,
)
from proclaim.verification.evidence_state import EvidenceState

# Librarian-backed retrieval functions — re-exported so the orchestrator can
# import them from evidence_api alongside the PubMed/S2 functions.  Selected
# at runtime via the `retrieval_backend` config (RETRIEVAL_BACKEND env var).
from proclaim.verification.librarian_search import (  # noqa: E402, F401
    librarian_refine_search,
    librarian_search,
    librarian_search_for_claim,
    librarian_search_for_gap,
    librarian_search_for_gaps,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug mode — set EVIDENCE_DEBUG=1 to enable verbose per-call output.
# Default (off) prints only essential summaries; debug mode prints
# per-PMID progress, full-text fetch details, and intermediate diagnostics.
# ---------------------------------------------------------------------------

_EVIDENCE_DEBUG = os.environ.get("EVIDENCE_DEBUG", "0") == "1"


def _debug_print(*args, **kwargs) -> None:
    """Print only when EVIDENCE_DEBUG=1."""
    if _EVIDENCE_DEBUG:
        print(*args, **kwargs)

# Shared singleton instances
_compressor = SufficiencyPreservingCompressor()

# Maximum sufficiency checks before forced verdict
MAX_ITERATIONS = 8

# ---------------------------------------------------------------------------
# Biological verb → noun mapping (for PubMed query formulation)
# ---------------------------------------------------------------------------

_BIO_VERB_TO_NOUN: dict[str, str] = {
    "activates": "activation",
    "inhibits": "inhibition",
    "phosphorylates": "phosphorylation",
    "binds": "binding",
    "regulates": "regulation",
    "suppresses": "suppression",
    "promotes": "promotion",
    "blocks": "blocking",
    "induces": "induction",
    "represses": "repression",
    "ubiquitinates": "ubiquitination",
    "methylates": "methylation",
    "acetylates": "acetylation",
    "deactivates": "deactivation",
    "up-regulates": "up-regulation",
    "down-regulates": "down-regulation",
    "upregulates": "up-regulation",
    "downregulates": "down-regulation",
}

_STOP_WORDS: set[str] = {
    "does", "is", "are", "the", "a", "an", "of", "in", "to", "and",
    "or", "that", "this", "it", "by", "with", "from", "for", "on",
    "at", "be", "was", "were", "been", "being", "have", "has", "had",
    "do", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "not", "no", "its", "their", "our", "your",
    "directly", "indirectly", "via", "through",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MaxIterationsExceeded(Exception):
    """Raised when the iteration limit is reached."""

    def __init__(self, *args, state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state


# ---------------------------------------------------------------------------
# Schema docs for system prompts (auto-generated from Pydantic models)
# ---------------------------------------------------------------------------


def schema_docs(backend: str | None = None) -> str:
    """Return a prompt-ready description of data schemas.

    *backend* selects the retrieval helper named in the "Common pitfalls"
    recipe; defaults to the ``RETRIEVAL_BACKEND`` env var.

    Auto-generated from the Pydantic models so the prompt stays in sync
    with the code.  Called at system-prompt construction time by both
    orchestrators.
    """
    if backend is None:
        backend = os.environ.get("RETRIEVAL_BACKEND", "pubmed")
    _refine_fn = (
        "librarian_refine_search"
        if backend == "librarian"
        else "refine_search_for_failed_papers"
    )
    from proclaim.verification.data_models import (
        Fact as _Fact,
        Gap as _Gap,
        PaperRecord as _PR,
        SufficiencyResult as _SR,
        VerificationVerdict as _VV,
    )
    from proclaim.verification.config import get_label_config
    _label_cfg = get_label_config()

    def _fields(model):
        from pydantic_core import PydanticUndefined as _PU
        parts = []
        for name, info in model.model_fields.items():
            ann = info.annotation
            type_name = getattr(ann, "__name__", str(ann))
            default = info.default
            if info.default_factory is not None:
                parts.append(f"  {name}: {type_name}  (default: {info.default_factory()!r})")
            elif default is _PU:
                parts.append(f"  {name}: {type_name}  (required)")
            else:
                parts.append(f"  {name}: {type_name}  (default: {default!r})")
        return "\n".join(parts)

    return (
        "## Data Schemas\n\n"
        "### PaperRecord\n"
        f"{_fields(_PR)}\n\n"
        "### EvidenceState (the `state` object)\n"
        "  state.claim: str\n"
        "  state.subclaims: list[str]  — set this early\n"
        "  state.papers: dict[str, PaperRecord]  — keys are paper IDs (PMIDs or S2 IDs)\n"
        "  state.facts: list[Fact]  — iterate with `for f in state.facts`\n"
        "  state.conflicts: list[Conflict]\n"
        "  state.coverage: dict[str, float]  — keys are subclaim strings\n"
        "  state.synthesis: dict[str, str]\n"
        "  state.sufficiency_history: list[SufficiencyResult]\n"
        "  state.iteration: int  — current iteration counter\n"
        "  state.token_estimate: int\n\n"
        "### SufficiencyResult\n"
        f"{_fields(_SR)}\n\n"
        "### Gap\n"
        f"{_fields(_Gap)}\n\n"
        "### Fact\n"
        f"{_fields(_Fact)}\n\n"
        "### VerificationVerdict\n"
        f"{_fields(_VV)}\n\n"
        "### add_facts_from_dicts — expected dict keys\n"
        "  text (or aliases: statement, fact_text, evidence, description)\n"
        f"  stance: {_label_cfg.stance_options_str()}\n"
        "  source_pmid (or alias: pmid)  — MUST be a paper ID already present in state.papers\n"
        "  relevant_subclaims: list of subclaim strings  (defaults to all subclaims)\n"
        "  subclaim_index: int  (resolved to the subclaim string at that index)\n"
        "  confidence: float 0.0-1.0\n\n"
        "### Common pitfalls — AVOID THESE\n"
        "  - state.facts is a list, NOT a dict. Use `for f in state.facts:` (not .values())\n"
        "  - state.iteration is a top-level int, NOT `state.metadata.iteration`\n"
        "  - source_pmid must reference a paper in state.papers. Facts with unknown paper IDs are rejected.\n"
        "  - Do NOT write fact dicts by hand. Use extract_and_add_facts(llm, pmids, state) instead.\n"
        "  - When creating a PaperRecord manually, `authors` must be a list[str], e.g. [\"Author Name\"].\n"
        "  - extract_and_add_facts(llm, pmids, state) accepts a list of paper IDs and returns dict[paper_id, count].\n"
        f"  - If extract_and_add_facts returns 0 for multiple paper IDs, use {_refine_fn}:\n"
        "      failed_ids = [paper_id for paper_id, count in results.items() if count == 0]\n"
        f"      new_ids = {_refine_fn}(failed_ids, state, llm, max_new_papers=5)\n"
        "      results2 = extract_and_add_facts(llm, new_ids, state)\n"
        "    This generates more precise queries instead of retrying the same irrelevant papers.\n"
    )


def function_docs(backend: str | None = None) -> str:
    """Auto-generate function signature docs from evidence_api and subagents.

    Inspects all public functions that the agent can call via nb_execute,
    producing a prompt-ready signature block.  Called at system-prompt
    construction time so docs stay in sync with code.

    *backend* selects the retrieval functions to document: ``"pubmed"`` lists
    the PubMed/Semantic Scholar functions, ``"librarian"`` lists the
    librarian-backed functions.  Defaults to the ``RETRIEVAL_BACKEND`` env var.
    """
    import inspect
    import re as _re

    if backend is None:
        backend = os.environ.get("RETRIEVAL_BACKEND", "pubmed")

    def _short_sig(fn) -> str:
        """Produce a signature with short type names (no module paths)."""
        sig = str(inspect.signature(fn))
        # proclaim.verification.data_models.Fact -> Fact, etc.
        sig = _re.sub(r"[a-z_]+(?:\.[a-z_]+)*\.([A-Z]\w*)", r"\1", sig)
        # typing qualifiers: Optional[Path] stays, Callable[[str], str] stays
        return sig

    # Retrieval functions depend on the configured backend.
    if backend == "librarian":
        _retrieval_funcs = [
            librarian_search_for_claim,
            librarian_search_for_gaps,
            librarian_search_for_gap,
            librarian_refine_search,
            librarian_search,
        ]
    else:
        _retrieval_funcs = [
            search_pubmed, search_pubmed_llm, find_related_articles,
            search_semantic_scholar, search_semantic_scholar_dual,
            search_semantic_scholar_recommendations,
            formulate_pubmed_query, search_for_gap, refine_search_for_failed_papers,
        ]

    # Backend-independent functions (extraction, features, sufficiency, verdict).
    _common_funcs = [
        get_full_text_article, get_paper_text, extract_and_add_facts,
        add_facts_from_dicts, update_synthesis, add_conflict,
        get_evidence_summary, check_sufficiency, get_sufficiency_history, compress_evidence,
        emit_verdict, populate_paper_features, filter_papers_by_stance,
        add_extraction_context_note,
    ]

    # Functions from evidence_api
    _api_funcs = _retrieval_funcs + _common_funcs

    # Import model registry function for documentation
    from proclaim.verification.model_registry import prewarm_all_models as _prewarm

    # Functions from subagents
    from proclaim.verification.subagents import (
        extract_facts, synthesize_subclaim, detect_conflicts,
        formulate_gap_queries, refine_search_query,
    )
    _sub_funcs = [extract_facts, synthesize_subclaim, detect_conflicts,
                  formulate_gap_queries, refine_search_query]

    lines = ["## Available functions (after setup)\n"]
    lines.append("### evidence_api\n")
    for fn in _api_funcs:
        sig = _short_sig(fn)
        # Get first line of docstring as description
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"    {fn.__name__}{sig}")
        if doc:
            lines.append(f"        {doc}")

    lines.append("\n### subagents\n")
    for fn in _sub_funcs:
        sig = _short_sig(fn)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"    {fn.__name__}{sig}")
        if doc:
            lines.append(f"        {doc}")

    lines.append("\n### model_registry\n")
    lines.append(f"    prewarm_all_models() -> dict[str, float]")
    lines.append(f"        Pre-load all ML models (NLP, MLP classifier) to avoid first-call latency")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query Formulation
# ---------------------------------------------------------------------------


def formulate_pubmed_query(claim: str) -> str:
    """Convert a natural-language claim into a structured PubMed query.

    Extracts gene/protein symbols and maps biological action verbs to
    noun forms. Returns the query string.
    """
    symbols = re.findall(r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b", claim)

    claim_lower = claim.lower().rstrip(".")
    bio_terms: list[str] = []
    for verb, noun in _BIO_VERB_TO_NOUN.items():
        if verb in claim_lower:
            bio_terms.append(noun)

    if not symbols:
        words = re.findall(r"\b\w+\b", claim)
        symbols = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    parts: list[str] = []
    if symbols:
        parts.append(" AND ".join(symbols))
    if bio_terms:
        parts.append("(" + " OR ".join(bio_terms) + ")")

    return " AND ".join(parts) if parts else claim


def _extract_symbol_subtokens(symbol: str) -> list[str]:
    """Extract family-level sub-tokens from a gene symbol."""
    subtokens: list[str] = []
    parts = symbol.split("-")
    if len(parts) > 1 and len(parts[0]) >= 2:
        subtokens.append(parts[0])
    prefix = re.sub(r"\d+$", "", symbol)
    if prefix and prefix != symbol and len(prefix) >= 2:
        subtokens.append(prefix)
    return list(set(subtokens))


def _generate_tiered_queries(
    symbols: list[str], bio_terms: list[str],
) -> list[tuple[str, str]]:
    """Generate queries from most specific to broadest.

    Returns list of (tier_name, query_string) tuples.
    """
    queries: list[tuple[str, str]] = []

    if symbols and bio_terms:
        q = " AND ".join(symbols) + " AND (" + " OR ".join(bio_terms) + ")"
        queries.append(("strict_pair_mechanism", q))

    if len(symbols) >= 2:
        q = " AND ".join(symbols)
        queries.append(("strict_pair", q))

    if bio_terms:
        for sym in symbols:
            q = f"{sym} AND ({' OR '.join(bio_terms)})"
            queries.append(("bridge_mechanism", q))

    if len(symbols) >= 2:
        for i, sym in enumerate(symbols):
            other_syms = [s for j, s in enumerate(symbols) if j != i]
            for other in other_syms:
                for subtoken in _extract_symbol_subtokens(other):
                    if subtoken != sym:
                        bridge_q = f"{sym} {subtoken}"
                        if bio_terms:
                            bridge_q += f" {bio_terms[0]}"
                        queries.append(("bridge_cross", bridge_q))

    return queries


# ---------------------------------------------------------------------------
# Retrieval Functions
# ---------------------------------------------------------------------------


def _search_and_add(
    query: str,
    state: EvidenceState,
    max_results: int = 5,
    _retries: int = 2,
    _delay: float = 1.0,
) -> tuple[int, int, list[str]]:
    """Search PubMed and add papers to state in-memory.

    Returns (found_count, added_count, added_pmids).
    """
    from proclaim.search.custom_pubmed import RelevancePubMedSearcher

    searcher = RelevancePubMedSearcher()

    papers = []
    for attempt in range(_retries + 1):
        try:
            papers = searcher.search(query, max_results=max_results)
            break
        except Exception:
            if attempt < _retries:
                time.sleep(_delay * (attempt + 1))
            else:
                return 0, 0, []

    added_pmids: list[str] = []
    for paper in papers:
        pmid = paper.paper_id
        if pmid in state.papers:
            continue
        record = PaperRecord(
            pmid=pmid,
            title=paper.title or "",
            abstract=paper.abstract or "",
            authors=[a for a in getattr(paper, "authors", [])],
            doi=getattr(paper, "doi", None) or None,
            source="pubmed",
        )
        state.add_paper(record)
        added_pmids.append(pmid)

    state.token_estimate = state.token_count()
    return len(papers), len(added_pmids), added_pmids


def search_pubmed(
    query: str, state: EvidenceState, max_results: int = 5,
) -> list[str]:
    """Search PubMed and add papers to state. Returns list of added PMIDs."""
    found, added, added_pmids = _search_and_add(query, state, max_results)
    print(
        f"PubMed: found {found}, added {added} new. "
        f"PMIDs: {', '.join(added_pmids) if added_pmids else 'none'}"
    )
    return added_pmids


def search_pubmed_llm(
    claim: str,
    state: EvidenceState,
    llm,
    max_results: int = 10,
) -> list[str]:
    """PubMed search using two independent LLM-generated queries, deduplicated.

    Runs a claim-only query and a subclaim-enriched query in sequence so that
    aliases and alternative names introduced during decomposition improve recall.
    Deduplication is handled by ``_search_and_add`` (keyed on PMID).

    Args:
        claim: The scientific claim to verify
        state: EvidenceState to add papers to
        llm: LLM callable for query generation
        max_results: Maximum papers to retrieve per query

    Returns:
        List of added PMIDs (union of both queries, deduplicated)
    """
    from proclaim.search.llm_query_generator import generate_search_query

    all_added: list[str] = []

    ctx = state.extraction_context or None

    # Query A: claim-only (original behaviour)
    query_a = generate_search_query(claim, llm, extraction_context=ctx)
    _debug_print(f"[LLM Query A] {query_a}")
    _, _, pmids_a = _search_and_add(query_a, state, max_results)
    all_added.extend(pmids_a)

    # Query B: subclaim-enriched (uses aliases from decomposition step)
    if state.subclaims:
        query_b = generate_search_query(claim, llm, subclaims=state.subclaims, extraction_context=ctx)
        _debug_print(f"[LLM Query B] {query_b}")
        _, _, pmids_b = _search_and_add(query_b, state, max_results)
        all_added.extend(pmids_b)
    else:
        query_b = None

    if not all_added:
        print("⚠️ No papers found in initial PubMed search.")

    print(
        f"search_pubmed_llm: added {len(all_added)} new paper(s) across "
        f"{'2 queries' if query_b else '1 query'}. "
        f"PMIDs: {', '.join(all_added) if all_added else 'none'}"
    )
    return all_added




def search_for_gap(
    gap_description: str,
    state: EvidenceState,
    max_results: int = 3,
) -> list[str]:
    """Search for papers targeting a specific evidence gap."""
    return search_pubmed(gap_description, state, max_results)


def refine_search_for_failed_papers(
    failed_pmids: list[str],
    state: EvidenceState,
    llm,
    max_new_papers: int = 5,
) -> list[str]:
    """Generate refined search queries and search for better papers when extraction fails.

    When extract_and_add_facts returns 0 facts for certain papers, this function
    analyzes why those papers were irrelevant and generates more precise search
    queries to find better papers. This is a smarter fallback than retrying the
    same papers with full text.

    Args:
        failed_pmids: List of PMIDs that yielded 0 facts.
        state: EvidenceState containing the papers.
        llm: LLM callable for query generation.
        max_new_papers: Maximum number of new papers to retrieve per query (default: 5).

    Returns:
        List of newly added PMIDs from the refined search.

    Example:
        >>> results = extract_and_add_facts(llm, pmids, state)
        >>> failed = [pmid for pmid, count in results.items() if count == 0]
        >>> if failed:
        >>>     new_pmids = refine_search_for_failed_papers(failed, state, llm)
        >>>     results2 = extract_and_add_facts(llm, new_pmids, state)
    """
    from proclaim.verification.subagents import refine_search_query

    if not failed_pmids:
        return []

    # Collect metadata from failed papers
    failed_papers = []
    for pmid in failed_pmids[:5]:  # Limit to 5 to avoid token overflow
        paper = state.papers.get(pmid)
        if paper:
            failed_papers.append({
                "pmid": pmid,
                "title": paper.title,
                "abstract": paper.abstract,
            })

    if not failed_papers:
        return []

    # Generate refined queries using LLM subagent
    refined_queries = refine_search_query(
        llm=llm,
        claim=state.claim,
        subclaims=state.subclaims,
        failed_papers=failed_papers,
    )

    if not refined_queries:
        logger.info("refine_search_for_failed_papers: no refined queries generated")
        return []

    _debug_print(f"Refined queries: {', '.join(refined_queries)}")

    # Execute searches with refined queries
    all_new_pmids: list[str] = []
    for query in refined_queries:
        try:
            new_pmids = search_pubmed(query, state, max_results=max_new_papers)
            all_new_pmids.extend(new_pmids)
        except Exception as e:
            logger.warning(f"Search failed for refined query '{query}': {e}")

    # Deduplicate
    unique_new_pmids = list(dict.fromkeys(all_new_pmids))

    print(f"refine_search_for_failed_papers: queries={len(refined_queries)} new_pmids={len(unique_new_pmids)}")
    return unique_new_pmids


def find_related_articles(
    pmid: str,
    state: EvidenceState,
    max_results: int = 5,
) -> list[str]:
    """Find related PubMed articles via citation co-occurrence.

    Returns list of added PMIDs.
    """
    if pmid.startswith("S2:"):
        _debug_print(
            f"find_related_articles only supports PubMed papers; skipping Semantic Scholar paper {pmid}."
        )
        return []

    import requests
    from xml.etree import ElementTree as ET

    _EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    _ELINK_URL = f"{_EUTILS_BASE}/elink.fcgi"
    _EFETCH_URL = f"{_EUTILS_BASE}/efetch.fcgi"

    # Step 1: elink to find related PMIDs
    link_params = {
        "dbfrom": "pubmed", "db": "pubmed", "id": pmid,
        "cmd": "neighbor_score", "retmode": "xml",
    }
    try:
        link_resp = requests.get(_ELINK_URL, params=link_params, timeout=30)
        link_resp.raise_for_status()
    except requests.RequestException as e:
        _debug_print(f"Error querying elink for PMID {pmid}: {e}")
        return []

    link_root = ET.fromstring(link_resp.content)
    related_pmids: list[str] = []
    for link in link_root.findall(".//LinkSetDb/Link"):
        rid = link.findtext("Id", "")
        if rid and rid != pmid:
            related_pmids.append(rid)
        if len(related_pmids) >= max_results:
            break

    if not related_pmids:
        _debug_print(f"No related articles found for PMID {pmid}.")
        return []

    # Step 2: Fetch metadata
    fetch_params = {
        "db": "pubmed", "id": ",".join(related_pmids), "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
    except requests.RequestException as e:
        _debug_print(f"Error fetching related article metadata: {e}")
        return []

    fetch_root = ET.fromstring(fetch_resp.content)
    added_pmids: list[str] = []

    for article in fetch_root.findall(".//PubmedArticle"):
        try:
            art_pmid = article.findtext(".//PMID", "")
            if not art_pmid or art_pmid in state.papers:
                continue
            title = article.findtext(".//ArticleTitle", "") or ""
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            authors: list[str] = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())

            record = PaperRecord(
                pmid=art_pmid, title=title, abstract=abstract or "",
                authors=authors, source="pubmed",
                doi=(
                    article.findtext(".//ELocationID[@EIdType='doi']", "")
                    or article.findtext(".//ArticleIdList/ArticleId[@IdType='doi']", "")
                    or None
                ),
            )
            state.add_paper(record)
            added_pmids.append(art_pmid)
        except Exception as e:
            logger.warning("Error parsing related article: %s", e)

    state.token_estimate = state.token_count()
    _debug_print(
        f"Related articles for PMID {pmid}: found {len(related_pmids)}, "
        f"added {len(added_pmids)} new."
    )
    return added_pmids


# ---------------------------------------------------------------------------
# Semantic Scholar discovery helpers
# ---------------------------------------------------------------------------

# DOI pattern used for citation chaining (matches all standard DOI prefixes).
_DOI_RE = re.compile(r"\b10\.\d{4,}/[^\s,;)\]\"\'<>]+")


def _s2_paper_to_record(data: dict, state: EvidenceState) -> "PaperRecord | None":
    """Convert a Semantic Scholar API response dict to a PaperRecord.

    Returns ``None`` if the paper is already present in *state* (dedup by
    PMID or DOI).

    ID resolution:
    - If ``externalIds.PubMed`` is set → use that as the PMID string.
    - Otherwise → ``S2:{paperId}``.

    The open-access PDF URL is logged but not stored on PaperRecord
    (a ``pdf_url`` field would be needed).  ``get_full_text_article`` can
    still retrieve the PDF via its Unpaywall fallback.

    Semantic Scholar-only records with no PubMed ID, no DOI, no abstract,
    and no OA PDF are dropped up front because none of the downstream text
    retrieval layers can recover usable evidence from them.
    """
    s2_id = data.get("paperId", "")
    external = data.get("externalIds") or {}
    pmid = external.get("PubMed") or f"S2:{s2_id}"

    # Check dedup by PMID
    if pmid in state.papers:
        return None

    # Check dedup by DOI (cross-source: same paper may have a PubMed PMID in
    # the state already but arrive here with an S2: id).
    doi_raw = (external.get("DOI") or "").strip().lower() or None
    if doi_raw:
        for existing in state.papers.values():
            if existing.doi and existing.doi.lower() == doi_raw:
                return None

    authors: list[str] = []
    for a in data.get("authors") or []:
        name = a.get("name", "")
        if name:
            authors.append(name)

    oa = data.get("openAccessPdf") or {}
    oa_url = oa.get("url") or None
    if oa_url:
        logger.debug("S2 OA PDF available for %s: %s", pmid, oa_url)

    abstract = data.get("abstract") or ""

    if pmid.startswith("S2:") and not (doi_raw or abstract or oa_url):
        logger.info(
            "Skipping Semantic Scholar-only paper %s: no PMID, DOI, abstract, or OA PDF available",
            pmid,
        )
        return None

    return PaperRecord(
        pmid=pmid,
        title=data.get("title") or "",
        abstract=abstract,
        authors=authors,
        doi=doi_raw,
        source="semantic_scholar",
    )


def _add_s2_records(
    results: list[dict], state: EvidenceState,
) -> list[str]:
    """Convert S2 result dicts, dedup, add to state, return added IDs."""
    added: list[str] = []
    for item in results:
        record = _s2_paper_to_record(item, state)
        if record is None:
            continue
        state.add_paper(record)
        added.append(record.pmid)
    if added:
        state.token_estimate = state.token_count()
    return added


def search_semantic_scholar(
    query: str,
    state: EvidenceState,
    max_results: int = 10,
) -> list[str]:
    """Search Semantic Scholar and add papers to state.

    Complements ``search_pubmed_llm`` by covering preprints (bioRxiv) and
    non-MEDLINE sources.  Use the same query string as the PubMed search.

    Args:
        query: Free-text query (PubMed-style Boolean queries also work).
        state: EvidenceState to add papers to.
        max_results: Maximum papers to retrieve (S2 cap: 100).

    Returns:
        List of added paper IDs (PMID strings or ``S2:<id>``).
    """
    from proclaim.search.semantic_scholar import S2Client

    client = S2Client()
    results = client.search(query, limit=max_results)
    added = _add_s2_records(results, state)
    logger.info(
        "S2 Search: found %d, added %d new. IDs: %s",
        len(results), len(added), ', '.join(added) if added else 'none',
    )
    return added


def search_semantic_scholar_dual(
    claim: str,
    state: EvidenceState,
    llm,
    max_results: int = 10,
) -> list[str]:
    """Search Semantic Scholar with two independent LLM-generated queries, deduplicated.

    Runs a claim-only query and a subclaim-enriched query so that aliases
    introduced during decomposition improve recall beyond what the plain claim
    provides.  Deduplication is handled by ``_search_and_add``.

    Args:
        claim: The scientific claim to verify
        state: EvidenceState to add papers to
        llm: LLM callable for query generation
        max_results: Maximum papers to retrieve per query

    Returns:
        List of added paper IDs (union of both queries, deduplicated)
    """
    from proclaim.search.llm_query_generator import generate_search_query_s2
    from proclaim.search.semantic_scholar import S2Client

    client = S2Client()
    all_added: list[str] = []

    ctx = state.extraction_context or None

    # Query C: claim-only
    query_c = generate_search_query_s2(claim, llm, extraction_context=ctx)
    _debug_print(f"[S2 Query C] {query_c}")
    results_c = client.search(query_c, limit=max_results)
    added_c = _add_s2_records(results_c, state)
    all_added.extend(added_c)

    # Query D: subclaim-enriched
    if state.subclaims:
        query_d = generate_search_query_s2(claim, llm, subclaims=state.subclaims, extraction_context=ctx)
        _debug_print(f"[S2 Query D] {query_d}")
        results_d = client.search(query_d, limit=max_results)
        added_d = _add_s2_records(results_d, state)
        all_added.extend(added_d)
    else:
        query_d = None

    logger.info(
        "S2 dual search: added %d new. IDs: %s",
        len(all_added), ', '.join(all_added) if all_added else 'none',
    )
    return all_added


def search_semantic_scholar_recommendations(
    state: EvidenceState,
    seed_pmids: list[str] | None = None,
    max_results: int = 20,
) -> list[str]:
    """Expand paper pool via S2 Recommendations using confirmed-relevant seeds.

    Seeds are auto-derived from papers that produced at least one SUPPORT fact
    when ``seed_pmids`` is ``None``.  Papers with only REFUTE facts are used as
    negative seeds to steer results away from irrelevant directions.

    Requires at least one positive seed; returns ``[]`` with a warning if none
    are available.  Call this at iteration ≥ 1, after facts have been extracted
    from initial PubMed papers.

    Args:
        state: EvidenceState with papers and facts populated.
        seed_pmids: Explicit positive seeds (PMID strings).  Overrides
            auto-selection when provided.
        max_results: Maximum papers to add (S2 cap: 500).

    Returns:
        List of added paper IDs.
    """
    from proclaim.search.semantic_scholar import S2Client
    from proclaim.verification.config import get_label_config as _glc_s2
    _lc_s2 = _glc_s2()
    _stance_names = _lc_s2.stance_names()
    # First configured stance is the "positive" stance (default: SUPPORT)
    _positive_stance = _stance_names[0] if _stance_names else "SUPPORT"
    # Second configured stance is the "negative" stance (default: REFUTE)
    _negative_stance = _stance_names[1] if len(_stance_names) > 1 else "REFUTE"

    if seed_pmids is not None:
        positive_pmids = list(seed_pmids)
        negative_pmids: list[str] = []
    else:
        # Auto-derive: positive = papers with ≥1 positive-stance fact
        positive_pmids = [
            pmid for pmid in state.papers
            if any(
                f.source_pmid == pmid and f.stance == _positive_stance
                for f in state.facts
            )
        ]
        # Negative = papers whose facts are exclusively negative-stance
        positive_set = set(positive_pmids)
        negative_pmids = [
            pmid for pmid in state.papers
            if pmid not in positive_set
            and state.facts  # only when there are facts at all
            and all(
                f.stance == _negative_stance
                for f in state.facts
                if f.source_pmid == pmid
            )
            and any(f.source_pmid == pmid for f in state.facts)
        ]

    if not positive_pmids:
        _debug_print(
            "S2 Recommendations: no positive seeds available yet "
            "(call after extracting facts from initial papers)."
        )
        return []

    # S2 accepts "PMID:<id>" for PubMed papers; S2-origin papers use the
    # bare 40-char S2 paper ID (strip our internal "S2:" prefix).
    def _fmt(pmid: str) -> str:
        if pmid.startswith("S2:"):
            return pmid[3:]  # bare S2 paper ID
        return f"PMID:{pmid}"

    pos_ids = [_fmt(p) for p in positive_pmids]
    neg_ids = [_fmt(p) for p in negative_pmids]

    client = S2Client()
    results = client.recommendations(pos_ids, neg_ids or None, limit=max_results)
    added = _add_s2_records(results, state)
    logger.info(
        "S2 Recommendations: %d positive seeds, %d negative seeds → added %d papers.",
        len(positive_pmids), len(negative_pmids), len(added),
    )
    return added


def expand_via_citations(
    state: EvidenceState,
    max_per_paper: int = 5,
) -> list[str]:
    """Expand evidence pool via reference DOIs stored on PaperRecords.

    For each paper in *state* that has ``reference_dois`` populated (set by
    ``get_full_text_article`` when the JATS XML ``<back><ref-list>`` is
    available), looks up DOIs via the Semantic Scholar API and adds new
    papers to state.  No LLM call is required.

    This implements backward citation chaining: it finds seminal works *cited*
    by the papers already retrieved, which keyword searches systematically miss.

    Call this after ``get_full_text_article`` has been called for the initial
    papers so that ``reference_dois`` fields are populated.

    Args:
        state: EvidenceState; only papers with ``reference_dois`` are processed.
        max_per_paper: Maximum new DOIs to resolve per source paper.

    Returns:
        List of added paper IDs.
    """
    from proclaim.search.semantic_scholar import S2Client

    # Collect existing DOIs so we can skip already-known papers
    existing_dois: set[str] = {
        p.doi.lower()
        for p in state.papers.values()
        if p.doi
    }

    papers_with_refs = [
        p for p in state.papers.values() if p.reference_dois
    ]
    if not papers_with_refs:
        _debug_print("Citation chaining: no papers have reference DOIs yet.")
        return []

    client = S2Client()
    all_added: list[str] = []

    for paper in papers_with_refs:
        candidate_dois: list[str] = []
        seen: set[str] = set()
        for doi in paper.reference_dois:
            doi_lower = doi.lower().rstrip(".,;:)")
            if doi_lower not in seen and doi_lower not in existing_dois:
                seen.add(doi_lower)
                candidate_dois.append(doi_lower)

        candidate_dois = candidate_dois[:max_per_paper]

        for doi in candidate_dois:
            data = client.lookup_doi(doi)
            if data is None:
                continue
            record = _s2_paper_to_record(data, state)
            if record is None:
                # Already in state (dedup inside _s2_paper_to_record)
                continue
            state.add_paper(record)
            all_added.append(record.pmid)
            # Track the newly added DOI so subsequent papers don't re-fetch
            if record.doi:
                existing_dois.add(record.doi.lower())

    if all_added:
        state.token_estimate = state.token_count()

    logger.info(
        "Citation chaining: %d papers scanned, added %d new papers.",
        len(papers_with_refs), len(all_added),
    )
    return all_added


def get_full_text_article(pmid: str, state: EvidenceState) -> str:
    """Retrieve full text through a layered fallback chain.

    Delegates to ``full_text.fetch_full_text`` which tries in order:
        1 → PMC Open Access XML
        1b → Europe PMC REST API
        1.5 → Semantic Scholar OA PDF
        2 → INDRA literature
        3 → Unpaywall + PDF
        4 → PubMed structured abstract (last resort)

    Layer 4 ensures ``fetch_full_text`` returns the PubMed structured
    abstract (with section labels) rather than ``None``, so this function
    almost never falls through to ``paper.abstract``.  Text from Layer 4
    is prefixed with ``[Abstract only — full text unavailable]``.

    Updates ``paper.full_text`` in state on success and returns the text.
    """
    from proclaim.verification.full_text import fetch_full_text

    paper = state.papers.get(pmid)
    if not paper:
        _debug_print(f"Paper {pmid} not found in evidence state.")
        return ""

    if paper.full_text:
        return paper.full_text

    full_text, ref_dois = fetch_full_text(
        pmid,
        doi=getattr(paper, "doi", None),
        title=paper.title,
        existing_abstract=paper.abstract,
    )

    if full_text:
        paper.full_text = full_text
        if ref_dois:
            paper.reference_dois = ref_dois
        state.token_estimate = state.token_count()
        _debug_print(f"Full text retrieved for paper ID {pmid}: {len(full_text)} chars")
        return full_text

    # Final fallback: plain abstract already stored on the paper record
    _debug_print(f"No text available for paper ID {pmid} (all layers failed). Using stored abstract.")
    return paper.abstract


def get_paper_text(pmid: str, state: EvidenceState) -> str:
    """Get the abstract/text for a paper by paper ID."""
    paper = state.papers.get(pmid)
    if not paper:
        _debug_print(f"Paper {pmid} not found in evidence state.")
        return ""
    return (
        f"Paper ID: {paper.pmid}\nTitle: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:5])}\n\n{paper.abstract}"
    )


# ---------------------------------------------------------------------------
# Evidence Mutation Functions
# ---------------------------------------------------------------------------

# Canonical field aliases.  LLMs commonly use synonyms for fact dict keys;
# resolving them here avoids silent data loss (e.g. "statement" -> "text").
# Extend this dict to accept additional field names in the future.
_FACT_FIELD_ALIASES: dict[str, str] = {
    "statement": "text",
    "claim_text": "text",
    "fact_text": "text",
    "evidence": "text",
    "description": "text",
    "pmid": "source_pmid",
    "paper_pmid": "source_pmid",
    "evidence_type": None,       # accepted but ignored
    "quotable_text": None,       # accepted but ignored
    "subclaim_index": None,      # handled specially below
}


def _normalise_fact_dict(
    item: dict, subclaims: list[str],
) -> dict:
    """Normalise a fact dict by resolving field aliases and subclaim indices.

    Returns a new dict with canonical keys ready for ``Fact()`` construction.
    """
    out: dict = {}
    for key, value in item.items():
        canonical = _FACT_FIELD_ALIASES.get(key, key)
        if canonical is None:
            # Explicitly ignored field
            continue
        # First-write wins — don't overwrite a value already set by a
        # higher-priority key (e.g. "text" takes precedence over "statement").
        if canonical not in out:
            out[canonical] = value

    # --- Resolve subclaim_index (int) → actual subclaim string ---
    idx = item.get("subclaim_index")
    if idx is not None and "relevant_subclaims" not in out:
        if isinstance(idx, int) and 0 <= idx < len(subclaims):
            out["relevant_subclaims"] = [subclaims[idx]]
        else:
            out["relevant_subclaims"] = list(subclaims)

    # --- Default relevant_subclaims to all subclaims when missing ---
    if not out.get("relevant_subclaims"):
        out["relevant_subclaims"] = list(subclaims)

    return out


def add_facts_from_dicts(
    facts_data: list[dict], state: EvidenceState,
) -> int:
    """Add facts to state from a list of dicts.

    Accepted dict keys (canonical + aliases)::

        text | statement | fact_text | evidence | description
        stance            "SUPPORT" | "REFUTE" | "NEUTRAL"
        source_pmid | pmid | paper_pmid
        relevant_subclaims   list[str]   (defaults to state.subclaims)
        subclaim_index       int          resolved to subclaim string
        confidence           float        0.0-1.0

    Validation:
    - Skips facts with empty ``text`` (prints a warning).
    - Deduplicates against existing facts by (text, source_pmid).

    Returns number of facts actually added.
    """
    # Build dedup index of existing facts
    existing_keys: set[tuple[str, str]] = {
        (f.text.strip().lower(), f.source_pmid)
        for f in state.facts
        if f.text.strip()
    }

    added = 0
    skipped_empty = 0
    skipped_dup = 0

    for item in facts_data:
        norm = _normalise_fact_dict(item, state.subclaims)

        text = (norm.get("text") or "").strip()
        if not text:
            skipped_empty += 1
            continue

        source_pmid = norm.get("source_pmid", "")

        # Validate source PMID: reject facts citing unknown papers
        if source_pmid and source_pmid not in state.papers:
            _warnings.warn(
                f"Fact cites PMID {source_pmid} which is not in state.papers. "
                f"Rejecting to prevent fabricated evidence."
            )
            _debug_print(
                f"REJECTED: source_pmid '{source_pmid}' not found in "
                f"state.papers ({list(state.papers.keys())[:5]}…). "
                f"Fact text: {text[:80]}…"
            )
            skipped_empty += 1  # reuse counter for rejected facts
            continue

        dedup_key = (text.lower(), source_pmid)
        if dedup_key in existing_keys:
            skipped_dup += 1
            continue
        existing_keys.add(dedup_key)

        from proclaim.verification.config import get_label_config as _glc
        _lc = _glc()
        raw_stance = norm.get("stance") or _lc.default_stance
        stance_str = _lc.validate_stance(raw_stance)

        fact = Fact(
            id=f"fact_{len(state.facts) + added}",
            text=text,
            stance=stance_str,
            source_pmid=source_pmid,
            relevant_subclaims=norm.get("relevant_subclaims", []),
            confidence=norm.get("confidence", 0.5),
        )
        state.add_fact(fact)
        added += 1

    # Recompute coverage
    _recompute_coverage(state)
    state.token_estimate = state.token_count()
    state._auto_save()  # Persist coverage and token count updates

    parts = [f"Added {added} facts."]
    if skipped_empty:
        parts.append(f"Skipped {skipped_empty} with empty text.")
    if skipped_dup:
        parts.append(f"Skipped {skipped_dup} duplicates.")
    parts.append("Coverage updated.")
    logger.info(" ".join(parts))
    return added


def _recompute_coverage(state: EvidenceState) -> None:
    """Recompute per-subclaim coverage scores."""
    from proclaim.verification.config import get_label_config as _glc_cov
    _default_stance = _glc_cov().default_stance
    for sc in state.subclaims:
        relevant = [
            f for f in state.facts
            if sc in f.relevant_subclaims and f.stance != _default_stance
        ]
        state.coverage[sc] = min(1.0, len(relevant) / 3.0)


def update_synthesis(
    subclaim: str, synthesis_text: str, state: EvidenceState,
) -> None:
    """Update the evidence synthesis for a subclaim."""
    state.synthesis[subclaim] = synthesis_text
    state._auto_save()  # Persist synthesis update to disk
    _debug_print(f"Synthesis updated for: {subclaim}")


def add_conflict(
    fact_a_id: str,
    fact_b_id: str,
    description: str,
    severity: float,
    state: EvidenceState,
) -> str:
    """Record a conflict between two facts. Returns conflict id."""
    conflict = Conflict(
        id=f"conflict_{len(state.conflicts)}",
        fact_a_id=fact_a_id,
        fact_b_id=fact_b_id,
        description=description,
        severity=severity,
    )
    state.add_conflict(conflict)
    _debug_print(f"Conflict recorded: {conflict.id}")
    return conflict.id


# ---------------------------------------------------------------------------
# High-level extraction helper
# ---------------------------------------------------------------------------


def _extract_and_add_facts_single(
    llm,
    pmid: str,
    state: EvidenceState,
) -> int:
    """Read a single paper, extract facts via the LLM subagent, and add to state.

    Private helper — call extract_and_add_facts(llm, pmids, state) instead,
    which processes a list of paper IDs in parallel for better GPU utilization.

    Tries full text via PMC first (``get_full_text_article``).  Falls
    back to abstract-level metadata (``get_paper_text``) only when full
    text is unavailable.

    Args:
        llm: ``Callable[[str], str]`` — takes a prompt, returns text.
             Wire this up in the kernel prelude (see ``inject_prelude``).
        pmid: Paper identifier of the paper already in ``state.papers``.
        state: The live EvidenceState object.

    Returns:
        Number of facts added.
    """
    from proclaim.verification.subagents import extract_facts

    # 1. Try full text (PMC lookup, cached in paper.full_text)
    paper_text = get_full_text_article(pmid, state)

    # 2. Fall back to abstract-level metadata
    if not paper_text:
        paper_text = get_paper_text(pmid, state)

    if not paper_text:
        _debug_print(f"_extract_and_add_facts_single: no text available for paper ID {pmid}.")
        return 0

    text_kind = "full text" if len(paper_text) > 2000 else "abstract"
    _debug_print(f"_extract_and_add_facts_single: using {text_kind} ({len(paper_text)} chars) for paper ID {pmid}.")

    facts = extract_facts(
        llm=llm,
        paper_text=paper_text,
        claim=state.claim,
        subclaims=state.subclaims,
        source_pmid=pmid,
        extraction_context=state.extraction_context or None,
    )

    # Always track as processed — prevents re-extraction unless add_extraction_context_note
    # clears this list explicitly.  Must happen before the early return so 0-fact papers
    # are not re-processed on every subsequent call.
    if pmid not in state.extracted_pmids:
        state.extracted_pmids.append(pmid)
        state._auto_save()

    if not facts:
        _debug_print(f"_extract_and_add_facts_single: subagent returned 0 facts for paper ID {pmid}.")
        return 0

    # Convert Fact objects to dicts and add through the validated path
    facts_dicts = [
        {
            "text": f.text,
            "stance": f.stance,
            "source_pmid": f.source_pmid,
            "relevant_subclaims": f.relevant_subclaims,
            "confidence": f.confidence,
        }
        for f in facts
    ]
    added = add_facts_from_dicts(facts_dicts, state)
    return added


def extract_and_add_facts(
    llm,
    pmids: list[str],
    state: EvidenceState,
    max_workers: int = 8,
) -> dict[str, int]:
    """Extract facts from multiple papers in parallel using ThreadPoolExecutor.

    This is the primary fact extraction function. It processes a list of paper IDs
    concurrently to maximize vLLM throughput via continuous batching. Instead
    of processing papers sequentially (which leaves the GPU idle between
    requests), this sends multiple extraction requests concurrently.

    Performance impact:
    - Sequential: 10 papers × 10s = 100s
    - Parallel (max_workers=8): ~12-15s (limited by longest paper)

    Args:
        llm: ``Callable[[str], str]`` — LLM callable (must be thread-safe).
        pmids: List of paper IDs to process in parallel.
        state: The live EvidenceState object (thread-safe for additions).
        max_workers: Maximum number of parallel workers (default: 8).
                     Should match vLLM's --max-num-seqs parameter.

    Returns:
        dict mapping pmid -> number of facts extracted.

    Example:
        >>> pmids = list(state.papers.keys())
        >>> results = extract_and_add_facts(llm, pmids, state, max_workers=8)
        >>> print(f"Total facts: {sum(results.values())}")

    Note:
        The function uses ThreadPoolExecutor (not asyncio) because the LLM
        callable is typically synchronous and releases the GIL during HTTP calls.
        EvidenceState additions are protected by internal locks, making this safe.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_one(pmid: str) -> tuple[str, int]:
        """Process a single paper, catching exceptions gracefully."""
        try:
            count = _extract_and_add_facts_single(llm, pmid, state)
            return pmid, count
        except Exception as e:
            logger.error(f"Error extracting facts from PMID {pmid}: {e}")
            _debug_print(f"⚠ extract_and_add_facts: error processing PMID {pmid}: {e}")
            return pmid, 0

    if not pmids:
        return {}

    # Filter out already-extracted PMIDs to avoid duplicate work
    pmids_to_process = [p for p in pmids if p not in state.extracted_pmids]

    if not pmids_to_process:
        logger.info("extract_and_add_facts: all %d papers already extracted", len(pmids))
        return {p: 0 for p in pmids}

    logger.info(
        "extract_and_add_facts: processing %d papers with %d workers",
        len(pmids_to_process), max_workers,
    )

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pmid = {executor.submit(_process_one, pmid): pmid
                          for pmid in pmids_to_process}

        # Process as they complete (provides progress feedback)
        for future in as_completed(future_to_pmid):
            pmid, count = future.result()
            results[pmid] = count
            if count > 0:
                _debug_print(f"  ✓ PMID {pmid}: extracted {count} facts")
            else:
                _debug_print(f"  ✗ PMID {pmid}: no facts extracted")

    # Add already-processed PMIDs with 0 count
    for pmid in pmids:
        if pmid not in results:
            results[pmid] = 0

    total_facts = sum(results.values())
    print(f"extract_and_add_facts: done. facts={total_facts} papers={len(pmids_to_process)}")

    return results


def add_extraction_context_note(state: EvidenceState, note: str) -> None:
    """Append a supplementary note for fact extraction and clear the extraction cache.

    Use this when new information (synonym mappings, disambiguation, scope
    clarifications) is discovered during the workflow and should influence how
    papers are re-extracted.  All previously extracted PMIDs are cleared so
    that the next call to extract_and_add_facts re-runs with the updated prompt.

    The note must describe TERMINOLOGY ONLY (aliases, synonyms, scope boundaries).
    Do NOT write interpretive conclusions about what the evidence shows — those
    belong in the verdict reasoning, not here.  Injecting conclusions biases the
    extraction LLM and will produce incorrect stance labels.

    Args:
        state: The live EvidenceState object.
        note: A terminology/disambiguation note to inject into the EXTRACT_FACTS prompt.

    Good example (synonym mapping — note contains ONLY terminology, zero conclusions):
        >>> add_extraction_context_note(
        ...     state,
        ...     "CRTC2 (also called TORC2) is a CREB transcription coactivator. "
        ...     "mTOR Complex 2 (mTORC2) is a distinct kinase complex that also appears "
        ...     "in literature as 'TORC2'. Papers discussing mTORC2 phosphorylating AKT "
        ...     "are NOT about CRTC2 unless they explicitly name CRTC2.",
        ... )

    Bad example (do NOT do this — injects a conclusion as if it were a fact):
        >>> add_extraction_context_note(
        ...     state,
        ...     "PMID 12345678 is highly relevant: kinase X phosphorylates protein Y "
        ...     "at serine 9, activating it.",  # pre-conclusion, not a synonym
        ... )
    """
    n_cleared = len(state.extracted_pmids)
    n_papers = len(state.papers)
    state.add_extraction_context(note)
    state.append_trace("add_extraction_context", {
        "note": note,
        "extracted_pmids_cleared": n_cleared,
        "total_context_notes": len(state.extraction_context),
    })
    print(
        f"Added extraction context note. Cleared all extracted_pmids ({n_cleared} → 0).\n"
        f"NEXT STEP: re-extract ALL {n_papers} papers with the updated prompt:\n"
        f"  extract_and_add_facts(llm, list(state.papers.keys()), state)"
    )


# ---------------------------------------------------------------------------
# Feature Population (for MLP Classifier)
# ---------------------------------------------------------------------------


def populate_paper_features(
    state: EvidenceState,
    compute_nli: bool = True,
    max_text_length: int = 10000,
    force_recompute: bool = False,
) -> str:
    """Populate NLP and metadata features for papers.

    MUST be called after extract_and_add_facts(llm, pmids, state) and before check_sufficiency()
    to ensure the MLP classifier has access to all required features.

    This function extracts:
    - NLP features: entity coverage, semantic similarity, NLI entailment
    - Metadata features: publication year, impact factor, citations, h-index

    By default, skips papers that already have both nlp and metadata features
    populated (efficient for incremental processing). Use force_recompute=True
    to recompute all features.

    **Performance note**: Heavy ML models (SBERT, NLI cross-encoder) are cached
    globally via model_registry after first use. To avoid first-call latency,
    use prewarm_all_models() during kernel setup.

    **Execution model**: Feature computation currently runs sequentially.
    Large paper sets can take a while, but the function is idempotent and skips
    papers that already have both NLP and metadata features unless
    ``force_recompute=True``.

    Args:
        state: The evidence state containing papers to process.
        compute_nli: Whether to compute NLI features (requires GPU for best performance).
        max_text_length: Maximum characters to use from each paper's full text.
        force_recompute: If True, recompute features even if already present.

    Returns:
        Status message indicating how many papers were processed.
    """
    # Sequential feature processing
    from proclaim.verification.feature_tools import compute_entity_coverage
    from proclaim.verification.model_registry import (
        get_semantic_similarity_computer,
        get_nli_entailment_computer,
        get_metadata_extractor,
    )

    logger.info("Populating paper features for %d papers", len(state.papers))

    # Process each paper
    processed_count = 0
    skipped_count = 0

    for paper in state.papers.values():
        # Check if features already exist (skip if both nlp and metadata are present)
        if not force_recompute and paper.nlp is not None and paper.metadata is not None:
            logger.debug("Skipping PMID %s: features already populated", paper.pmid)
            skipped_count += 1
            continue

        # Use full text if available, otherwise attempt to fetch fallback text
        text = paper.full_text or paper.abstract
        if not text:
            text = get_full_text_article(paper.pmid, state) or paper.abstract
        if not text:
            logger.warning(
                "Skipping PMID %s: no full text or abstract available after remote retrieval failure; see preceding warning for layer summary",
                paper.pmid,
            )
            # Still attempt metadata extraction even without text
            if force_recompute or paper.metadata is None:
                try:
                    meta_extractor = get_metadata_extractor()
                    paper.metadata = meta_extractor.extract_metadata(paper.pmid)
                except Exception as exc:
                    logger.error("Failed to extract metadata for PMID %s: %s", paper.pmid, exc)
            processed_count += 1
            continue

        text_source = "full_text" if paper.full_text else "abstract"
        if not paper.full_text:
            logger.info(
                "PMID %s: no full text, using abstract (%d chars) for NLP features",
                paper.pmid, len(text),
            )

        # Truncate text if needed
        text = text[:max_text_length]

        # --- NLP features ---
        if force_recompute or paper.nlp is None:
            try:
                # Entity coverage
                nlp = compute_entity_coverage(state.claim, text)

                # Semantic similarity (model loaded from registry)
                sim_computer = get_semantic_similarity_computer()
                nlp.semantic_similarity = sim_computer.compute(state.claim, text)

                # NLI entailment (optional, GPU-intensive, model loaded from registry)
                if compute_nli:
                    nli_computer = get_nli_entailment_computer()
                    nli_result = nli_computer.compute(state.claim, text)
                    nlp.nli_entailment = nli_result["nli_entailment"]
                    nlp.nli_contradiction = nli_result["nli_contradiction"]
                    nlp.nli_neutral = nli_result["nli_neutral"]
                    nlp.nli_best_chunk_text = nli_result["nli_best_chunk_text"]

                paper.nlp = nlp
                logger.debug(
                    "PMID %s: coverage=%.3f, similarity=%.3f",
                    paper.pmid,
                    nlp.claim_entity_coverage or 0.0,
                    nlp.semantic_similarity or 0.0,
                )
            except Exception as exc:
                logger.error("Failed to compute NLP features for PMID %s: %s", paper.pmid, exc)

        # --- Metadata features ---
        if force_recompute or paper.metadata is None:
            try:
                meta_extractor = get_metadata_extractor()
                paper.metadata = meta_extractor.extract_metadata(paper.pmid)
                logger.debug(
                    "PMID %s: year=%s, log_IF=%.3f",
                    paper.pmid,
                    paper.metadata.publication_year,
                    paper.metadata.log_impact_factor or 0.0,
                )
            except Exception as exc:
                logger.error("Failed to extract metadata for PMID %s: %s", paper.pmid, exc)

        processed_count += 1

    # Save state
    state.save()

    msg = f"Populated features for {processed_count} papers, skipped {skipped_count} (already processed)"
    logger.info(msg)
    return msg


# def populate_paper_features_parallel(
#     state: EvidenceState,
#     compute_nli: bool = True,
#     max_text_length: int = 10000,
#     force_recompute: bool = False,
#     max_workers: int = 4,
# ) -> str:
#     """Parallel version of populate_paper_features using ThreadPoolExecutor.

#     This function parallelizes feature extraction for multiple papers to reduce
#     wall-clock time. Due to PyTorch models not being fully thread-safe when using
#     GPU, this implementation uses locks to serialize access to each model while
#     still allowing concurrent execution of different stages (entity extraction,
#     metadata fetching, etc.).

#     Performance impact:
#     - Sequential: 10 papers × 7s = 70s
#     - Parallel (max_workers=4): ~20-25s (with lock contention)

#     Note: For CPU-only models or if you have multiple GPUs, you could use
#     process-based parallelism (multiprocessing) for better scaling, but that
#     requires picklable models and state.

#     Args:
#         state: The evidence state containing papers to process.
#         compute_nli: Whether to compute NLI features (requires GPU for best performance).
#         max_text_length: Maximum characters to use from each paper's full text.
#         force_recompute: If True, recompute features even if already present.
#         max_workers: Maximum number of parallel workers (default: 4).
#                      Higher values increase concurrency but also lock contention.

#     Returns:
#         Status message indicating how many papers were processed.
#     """
#     import threading
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     from proclaim.verification.feature_tools import compute_entity_coverage
#     from proclaim.verification.model_registry import (
#         get_semantic_similarity_computer,
#         get_nli_entailment_computer,
#         get_metadata_extractor,
#     )

#     # Thread locks for model access (PyTorch models are not thread-safe on GPU)
#     _sim_lock = threading.Lock()
#     _nli_lock = threading.Lock()
#     _meta_lock = threading.Lock()

#     logger.info("Populating paper features in parallel for %d papers (max_workers=%d)",
#                 len(state.papers), max_workers)

#     def _process_one_paper(paper) -> tuple[str, bool]:
#         """Process features for a single paper."""
#         pmid = paper.pmid

#         # Check if features already exist
#         if not force_recompute and paper.nlp is not None and paper.metadata is not None:
#             logger.debug("Skipping PMID %s: features already populated", pmid)
#             return pmid, False

#         # Get paper text, attempting fallback retrieval when the record is empty
#         text = paper.full_text or paper.abstract
#         if not text:
#             text = get_full_text_article(pmid, state) or paper.abstract
#         if not text:
#             logger.warning(
#                 "Skipping PMID %s: no full text or abstract available after remote retrieval failure; see preceding warning for layer summary",
#                 pmid,
#             )
#             # Still attempt metadata extraction
#             if force_recompute or paper.metadata is None:
#                 try:
#                     with _meta_lock:
#                         meta_extractor = get_metadata_extractor()
#                         paper.metadata = meta_extractor.extract_metadata(pmid)
#                 except Exception as exc:
#                     logger.error("Failed to extract metadata for PMID %s: %s", pmid, exc)
#             return pmid, True

#         text_source = "full_text" if paper.full_text else "abstract"
#         if not paper.full_text:
#             logger.info(
#                 "PMID %s: no full text, using abstract (%d chars) for NLP features",
#                 pmid, len(text),
#             )

#         text = text[:max_text_length]

#         # --- NLP features (with locks for model access) ---
#         if force_recompute or paper.nlp is None:
#             try:
#                 # Entity coverage (spaCy is reasonably thread-safe)
#                 nlp = compute_entity_coverage(state.claim, text)

#                 # Semantic similarity (SBERT with GPU - needs lock)
#                 with _sim_lock:
#                     sim_computer = get_semantic_similarity_computer()
#                     nlp.semantic_similarity = sim_computer.compute(state.claim, text)

#                 # NLI entailment (CrossEncoder with GPU - needs lock)
#                 if compute_nli:
#                     with _nli_lock:
#                         nli_computer = get_nli_entailment_computer()
#                         nli_result = nli_computer.compute(state.claim, text)
#                         nlp.nli_entailment = nli_result["nli_entailment"]
#                         nlp.nli_contradiction = nli_result["nli_contradiction"]
#                         nlp.nli_neutral = nli_result["nli_neutral"]
#                         nlp.nli_best_chunk_text = nli_result["nli_best_chunk_text"]

#                 paper.nlp = nlp
#                 logger.debug(
#                     "PMID %s: coverage=%.3f, similarity=%.3f",
#                     pmid,
#                     nlp.claim_entity_coverage or 0.0,
#                     nlp.semantic_similarity or 0.0,
#                 )
#             except Exception as exc:
#                 logger.error("Failed to compute NLP features for PMID %s: %s", pmid, exc)

#         # --- Metadata features (API calls can be parallel, but serialize access) ---
#         if force_recompute or paper.metadata is None:
#             try:
#                 with _meta_lock:
#                     meta_extractor = get_metadata_extractor()
#                     paper.metadata = meta_extractor.extract_metadata(pmid)
#                 logger.debug(
#                     "PMID %s: year=%s, log_IF=%.3f",
#                     pmid,
#                     paper.metadata.publication_year,
#                     paper.metadata.log_impact_factor or 0.0,
#                 )
#             except Exception as exc:
#                 logger.error("Failed to extract metadata for PMID %s: %s", pmid, exc)

#         return pmid, True

#     # Filter papers that need processing
#     papers_to_process = [
#         paper for paper in state.papers.values()
#         if force_recompute or paper.nlp is None or paper.metadata is None
#     ]

#     if not papers_to_process:
#         msg = f"All {len(state.papers)} papers already have features populated"
#         logger.info(msg)
#         return msg

#     processed_count = 0
#     skipped_count = len(state.papers) - len(papers_to_process)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_pmid = {
#             executor.submit(_process_one_paper, paper): paper.pmid
#             for paper in papers_to_process
#         }

#         for future in as_completed(future_to_pmid):
#             pmid, was_processed = future.result()
#             if was_processed:
#                 processed_count += 1
#             else:
#                 skipped_count += 1

#     # Save state
#     state.save()

#     msg = f"Populated features for {processed_count} papers in parallel, skipped {skipped_count} (already processed)"
#     logger.info(msg)
#     return msg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_evidence_summary(state: EvidenceState) -> str:
    """Get a human-readable summary of the current evidence state."""
    from proclaim.verification.config import get_label_config as _glc2
    _lc2 = _glc2()
    stance_counts = ", ".join(
        f"{name.lower()}: {sum(1 for f in state.facts if f.stance == name)}"
        for name in _lc2.stance_names()
    )
    lines = [
        f"Claim: {state.claim}",
        f"Iteration: {state.iteration}",
        f"Papers: {len(state.papers)}",
        f"Facts: {len(state.facts)} ({stance_counts})",
        f"Conflicts: {len(state.conflicts)}",
        f"Token estimate: ~{state.token_estimate}",
        "",
        "Coverage per subclaim:",
    ]
    for sc in state.subclaims:
        cov = state.coverage.get(sc, 0.0)
        synth = "yes" if sc in state.synthesis else "no"
        lines.append(f"  [{cov:.1%}] [synth:{synth}] {sc}")

    if state.sufficiency_history:
        latest = state.sufficiency_history[-1]
        lines.extend([
            "",
            f"Latest sufficiency: {latest.label} (confidence: {latest.confidence:.2f})",
            f"Gaps: {len(latest.gaps)}",
        ])
        for gap in latest.gaps[:5]:
            lines.append(f"  - [{gap.gap_type.value}] {gap.description}")

    summary = "\n".join(lines)
    logger.debug("get_evidence_summary:\n%s", summary)
    return summary


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


# Model registry for singleton management of heavy ML models
# (MLP classifier, NLP models, etc. are now managed by model_registry.py)


def check_sufficiency(
    state: EvidenceState,
    llm,
    threshold: float = 0.5,
    min_total_papers: int = 3,
) -> SufficiencyResult:
    """Run the trained MLP sufficiency classifier on the current evidence state.

    Uses ``SufficiencyMLP`` from ``scripts/sufficiency_classifier/test_mlp_classifier.py``
    and ``FeatureAggregator`` from ``scripts/sufficiency_classifier/feature_aggregation.py``.

    Requires per-paper features (``paper.metadata``, ``paper.nlp``) to have
    been populated — otherwise the MLP will see zeros for missing features.

    When the classifier predicts INSUFFICIENT, gap identification is delegated to
    the ``identify_gaps`` LLM subagent.

    **Performance note**: The MLP classifier and feature aggregator are cached
    globally via model_registry after first use. Use prewarm_all_models() during
    kernel setup to avoid first-call latency.

    Args:
        state: Evidence state to check
        llm: LLM instance for gap identification
        threshold: MLP probability threshold for sufficiency (default: 0.5)
        min_total_papers: Minimum total number of papers required in the state
                          before allowing SUFFICIENT result. If fewer papers
                          are present in total, force INSUFFICIENT to
                          encourage more retrieval. Set to 0 to disable.
                          (default: 3)
    The backend is selected via the SUFFICIENCY_BACKEND environment variable
    (propagated by VerificationSettings.build_sdk_env):
      'mlp' (default): trained MLP classifier — original behaviour.
      'llm': Qwen subagent reads paper titles, abstracts, and NLP features.
      'haiku': Claude Haiku via Anthropic API — Qwen still used for gap identification.

    Appends result to state.sufficiency_history and increments iteration.
    If the iteration limit is reached, prints a message and returns the last
    sufficiency result without mutating state so emit_verdict() can still run.
    """
    import os
    backend = os.environ.get("SUFFICIENCY_BACKEND", "mlp").lower()
    if backend == "haiku":
        return _check_sufficiency_haiku(state, llm, threshold, min_total_papers)
    if backend == "llm":
        return _check_sufficiency_llm(state, llm, threshold, min_total_papers)
    return _check_sufficiency_mlp(state, llm, threshold, min_total_papers)


def _check_sufficiency_llm(
    state: EvidenceState,
    llm,
    threshold: float,
    min_total_papers: int,
) -> SufficiencyResult:
    """LLM-based sufficiency check (Qwen subagent backend)."""
    from proclaim.verification.llm_sufficiency import check_sufficiency_llm
    from proclaim.verification.subagents import identify_gaps

    if state.iteration >= state.MAX_ITERATIONS:
        print(
            f"Iteration limit ({state.MAX_ITERATIONS}) reached. "
            "Proceed to emit_verdict() without calling check_sufficiency() again."
        )
        if state.sufficiency_history:
            return state.sufficiency_history[-1]
        return SufficiencyResult(label="insufficient", confidence=0.0, gaps=[])

    current_paper_count = len(state.papers)
    previous_paper_count = (
        state.papers_per_iteration[-1] if state.papers_per_iteration else 0
    )
    papers_added_this_iteration = current_paper_count - previous_paper_count
    state.papers_per_iteration.append(current_paper_count)

    label, score, raw_response = check_sufficiency_llm(state, llm, threshold)

    override_reason = None
    if (
        label == "sufficient"
        and min_total_papers > 0
        and current_paper_count < min_total_papers
        and state.iteration < state.MAX_ITERATIONS
    ):
        override_reason = (
            f"Minimum paper requirement not met: only {current_paper_count} papers "
            f"(need at least {min_total_papers}). Continue searching."
        )
        label = "insufficient"

    gaps: list = []
    if label == "insufficient":
        if override_reason:
            from proclaim.verification.data_models import Gap, GapType, GapPriority
            gaps = [Gap(
                subclaim=state.claim,
                gap_type=GapType.LOW_DIVERSITY,
                description=override_reason,
                priority=GapPriority.HIGH,
            )]
        else:
            gaps = identify_gaps(
                llm=llm, claim=state.claim,
                subclaims=state.subclaims, facts=state.facts,
            )

    result = SufficiencyResult(label=label, confidence=score, gaps=gaps)
    state.sufficiency_history.append(result)
    state.iteration += 1
    state._auto_save()

    logger.info("check_sufficiency: iter=%d papers=%d score=%.4f label=%s",
                state.iteration, current_paper_count, score, label)
    print(f"check_sufficiency: iter={state.iteration} label={label} confidence={score:.4f} papers={current_paper_count}(+{papers_added_this_iteration}) backend=llm")
    if override_reason:
        print(f"  override: sufficient → insufficient (need >={min_total_papers} papers, have {current_paper_count})")
    if gaps:
        print(f"  gaps: {len(gaps)} (top: {gaps[0].gap_type.value} — {gaps[0].description[:80]})")

    return result


def _check_sufficiency_haiku(
    state: EvidenceState,
    llm,
    threshold: float,
    min_total_papers: int,
) -> SufficiencyResult:
    """Haiku-based sufficiency check (Claude Haiku via native Anthropic API).

    Gap identification still uses the general-purpose ``llm`` (Qwen).
    """
    from proclaim.verification.llm_sufficiency import check_sufficiency_llm
    from proclaim.verification.subagents import identify_gaps
    from proclaim.verification.model_registry import get_haiku_llm

    if state.iteration >= state.MAX_ITERATIONS:
        print(
            f"Iteration limit ({state.MAX_ITERATIONS}) reached. "
            "Proceed to emit_verdict() without calling check_sufficiency() again."
        )
        if state.sufficiency_history:
            return state.sufficiency_history[-1]
        return SufficiencyResult(label="insufficient", confidence=0.0, gaps=[])

    current_paper_count = len(state.papers)
    previous_paper_count = (
        state.papers_per_iteration[-1] if state.papers_per_iteration else 0
    )
    papers_added_this_iteration = current_paper_count - previous_paper_count
    state.papers_per_iteration.append(current_paper_count)

    haiku_llm = get_haiku_llm()
    label, score, raw_response = check_sufficiency_llm(state, haiku_llm, threshold)

    override_reason = None
    if (
        label == "sufficient"
        and min_total_papers > 0
        and current_paper_count < min_total_papers
        and state.iteration < state.MAX_ITERATIONS
    ):
        override_reason = (
            f"Minimum paper requirement not met: only {current_paper_count} papers "
            f"(need at least {min_total_papers}). Continue searching."
        )
        label = "insufficient"

    gaps: list = []
    if label == "insufficient":
        if override_reason:
            from proclaim.verification.data_models import Gap, GapType, GapPriority
            gaps = [Gap(
                subclaim=state.claim,
                gap_type=GapType.LOW_DIVERSITY,
                description=override_reason,
                priority=GapPriority.HIGH,
            )]
        else:
            gaps = identify_gaps(
                llm=llm,  # Qwen, not Haiku
                claim=state.claim,
                subclaims=state.subclaims,
                facts=state.facts,
            )

    result = SufficiencyResult(label=label, confidence=score, gaps=gaps)
    state.sufficiency_history.append(result)
    state.iteration += 1
    state._auto_save()

    logger.info("check_sufficiency: iter=%d papers=%d score=%.4f label=%s",
                state.iteration, current_paper_count, score, label)
    print(f"check_sufficiency: iter={state.iteration} label={label} confidence={score:.4f} papers={current_paper_count}(+{papers_added_this_iteration}) backend=haiku")
    if override_reason:
        print(f"  override: sufficient → insufficient (need >={min_total_papers} papers, have {current_paper_count})")
    if gaps:
        print(f"  gaps: {len(gaps)} (top: {gaps[0].gap_type.value} — {gaps[0].description[:80]})")

    return result


def _check_sufficiency_mlp(
    state: EvidenceState,
    llm,
    threshold: float,
    min_total_papers: int,
) -> SufficiencyResult:
    """MLP-based sufficiency check (original implementation)."""

    if state.iteration >= state.MAX_ITERATIONS:
        print(
            f"Iteration limit ({state.MAX_ITERATIONS}) reached. "
            "Proceed to emit_verdict() without calling check_sufficiency() again."
        )
        if state.sufficiency_history:
            return state.sufficiency_history[-1]
        return SufficiencyResult(label="insufficient", confidence=0.0, gaps=[])

    # Track paper count for this iteration
    current_paper_count = len(state.papers)
    previous_paper_count = (
        state.papers_per_iteration[-1] if state.papers_per_iteration else 0
    )
    papers_added_this_iteration = current_paper_count - previous_paper_count

    # Record current count for next iteration
    state.papers_per_iteration.append(current_paper_count)

    # Get MLP classifier and feature aggregator from model registry
    from proclaim.verification.model_registry import get_mlp_classifier, get_feature_aggregator

    mlp = get_mlp_classifier()
    aggregator = get_feature_aggregator()

    # Convert EvidenceState papers → FeatureAggregator input format
    papers_dicts = []
    num_full_text = 0
    for paper in state.papers.values():
        d: dict = {}
        if paper.metadata:
            d["metadata_features"] = paper.metadata.model_dump()
        if paper.nlp:
            d["nlp_features"] = paper.nlp.model_dump()
        papers_dicts.append(d)
        if paper.full_text:
            num_full_text += 1

    # Aggregate features via FeatureAggregator
    nested = aggregator.aggregate_all(papers_dicts)

    # num_full_text is not computed by FeatureAggregator — inject from state

    import importlib

    inference_module = importlib.import_module("sufficiency_classifier.inference")
    flatten_features = inference_module.flatten_features
    mlp_predict = inference_module.mlp_predict

    flat_feats = flatten_features(nested)
    flat_feats["num_full_text"] = float(num_full_text)
    prob, _ = mlp_predict(
        mlp["model"], flat_feats, mlp["expected_features"],
        mlp["mean"], mlp["scale"],
    )

    # Log feature values for transparency
    vec = [flat_feats.get(k, 0.0) for k in mlp["expected_features"]]
    feat_summary = ", ".join(
        f"{k}={v:.3f}" for k, v in zip(mlp["expected_features"], vec)
    )
    logger.info("MLP features: %s → prob=%.4f (threshold=%.2f)", feat_summary, prob, threshold)

    # Determine label based on MLP prediction
    # New classifier returns "sufficient" or "insufficient" (lowercase)
    mlp_label = "sufficient" if prob >= threshold else "insufficient"

    # Override to INSUFFICIENT if minimum paper requirement not met
    # (but only if MLP would have said SUFFICIENT - don't override INSUFFICIENT)
    override_reason = None
    if (
        mlp_label == "sufficient"
        and min_total_papers > 0
        and current_paper_count < min_total_papers
        and state.iteration < state.MAX_ITERATIONS  # Don't force on last iteration
    ):
        override_reason = (
            f"Minimum paper requirement not met: only {current_paper_count} "
            f"total papers gathered (need at least {min_total_papers}). "
            f"Continue searching to gather more evidence."
        )
        label = "insufficient"
        logger.info(
            "Overriding sufficient → insufficient: %d total papers gathered (need at least %d)",
            current_paper_count, min_total_papers
        )
    else:
        label = mlp_label

    # LLM-driven gap identification for insufficient results
    gaps: list = []
    if label == "insufficient":
        from proclaim.verification.subagents import identify_gaps

        # If we overrode due to min papers, add a synthetic gap
        if override_reason:
            from proclaim.verification.data_models import Gap, GapType, GapPriority
            gaps = [Gap(
                subclaim=state.claim,
                gap_type=GapType.LOW_DIVERSITY,
                description=override_reason,
                priority=GapPriority.HIGH,
            )]
        else:
            # Normal gap identification
            gaps = identify_gaps(
                llm=llm,
                claim=state.claim,
                subclaims=state.subclaims,
                facts=state.facts,
            )

    result = SufficiencyResult(label=label, confidence=prob, gaps=gaps)

    state.sufficiency_history.append(result)
    state.iteration += 1
    state._auto_save()  # Persist sufficiency check result to disk

    # Print compact feedback for the agent
    override_note = " (overridden: min papers)" if override_reason else ""
    print(
        f"Sufficiency: {label} (confidence={prob:.4f}, "
        f"papers={current_paper_count}+{papers_added_this_iteration}, "
        f"gaps={len(gaps)}{override_note})"
    )
    # Verbose details only in debug mode
    _debug_print(f"=== SUFFICIENCY CHECK (iteration {state.iteration}) ===")
    _debug_print(f"MLP Prediction: {mlp_label} (confidence: {prob:.6f})")
    _debug_print(f"Threshold: {threshold}")
    if override_reason:
        _debug_print(f"Override reason: {override_reason}")
    if gaps:
        _debug_print(f"Gaps ({len(gaps)}):")
        for i, gap in enumerate(gaps, 1):
            _debug_print(f"  {i}. [{gap.priority.value}] {gap.gap_type.value}")
            _debug_print(f"     Subclaim: {gap.subclaim}")
            _debug_print(f"     Action: {gap.description}")

    return result


# ---------------------------------------------------------------------------
# Sufficiency History / Trend Analysis
# ---------------------------------------------------------------------------


def get_sufficiency_history(
    state: EvidenceState,
    window: int = 3,
    min_delta: float = 0.02,
    decline_delta: float = 0.05,
) -> dict:
    """Report the sufficiency confidence trend over recent iterations.

    A pure-Python diagnostic tool — no LLM calls.  Call this after
    ``check_sufficiency`` to understand whether the evidence search is
    improving, stagnating, or declining.  The LLM is responsible for
    deciding what action to take in response.

    Args:
        state: The current EvidenceState.
        window: Number of recent iterations to analyse. Default 3.
        min_delta: If ``max - min`` of the last ``window`` confidence
            scores is below this, the trend is **flat** (stagnated).
            Default 0.02.
        decline_delta: If the last confidence minus the first confidence
            in the window is below ``-decline_delta``, the trend is
            **declining**.  Default 0.05.

    Returns:
        dict: {"trend": str} where trend is one of:
            - "improving": evidence quality is increasing
            - "flat": evidence quality has stagnated
            - "declining": evidence quality is decreasing
            - "insufficient_history": not enough iterations yet

    The full history table is printed to stdout for human inspection.
    Use state.sufficiency_history to access raw iteration data.
    """
    history = [
        {
            "iteration": i + 1,
            "label": r.label,
            "confidence": r.confidence,
        }
        for i, r in enumerate(state.sufficiency_history)
    ]

    # --- Print history table (debug only) --------------------------------
    _debug_print("=== SUFFICIENCY HISTORY ===")
    if not history:
        _debug_print("  (no sufficiency checks yet)")
    else:
        _debug_print(f"  {'Iter':>4}  {'Label':<12}  {'Confidence':>10}")
        for row in history:
            _debug_print(
                f"  {row['iteration']:>4}  {row['label']:<12}  "
                f"{row['confidence']:>10.6f}"
            )

    # --- Trend analysis ------------------------------------------------------
    if len(history) < window:
        trend = "insufficient_history"
    else:
        recent_scores = [row["confidence"] for row in history[-window:]]
        first, last = recent_scores[0], recent_scores[-1]
        score_range = max(recent_scores) - min(recent_scores)

        if last - first < -decline_delta:
            trend = "declining"
        elif score_range < min_delta:
            trend = "flat"
        else:
            trend = "improving"

    print(f"get_sufficiency_history: trend={trend} iterations={len(history)}")

    # Return only trend to prevent LLM from over-analyzing the data
    return {
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# Paper Filtering
# ---------------------------------------------------------------------------


def filter_papers_by_stance(
    state: EvidenceState,
    keep_stances: Optional[list[str]] = None,
) -> None:
    """Filter papers in-place, keeping only those with facts matching specified stances.

    Args:
        state: EvidenceState to filter
        keep_stances: List of stances to keep. Defaults to ["SUPPORT", "REFUTE"]
                     (excludes papers with only NEUTRAL facts)

    This function removes papers from state.papers that don't have at least one
    fact with a stance in keep_stances. Papers with no facts are also removed.

    The filtering is recorded in state.trace for auditability.

    Example:
        # Remove papers with only NEUTRAL facts
        filter_papers_by_stance(state)  # keeps SUPPORT and REFUTE only

        # Keep all papers with any facts
        filter_papers_by_stance(state, keep_stances=["SUPPORT", "REFUTE", "NEUTRAL"])
    """
    if keep_stances is None:
        from proclaim.verification.config import get_label_config as _glc3
        _lc3 = _glc3()
        # Default: keep all stances except the default_stance (typically NEUTRAL)
        keep_stances = [s for s in _lc3.stance_names() if s != _lc3.default_stance]

    keep_stances_set = set(keep_stances)

    # Build a map of pmid -> facts
    facts_by_paper: dict[str, list[Fact]] = {}
    for fact in state.facts:
        pmid = fact.source_pmid
        if pmid not in facts_by_paper:
            facts_by_paper[pmid] = []
        facts_by_paper[pmid].append(fact)

    # Identify papers to keep
    papers_to_keep = set()
    papers_to_remove = set()

    for pmid in state.papers.keys():
        paper_facts = facts_by_paper.get(pmid, [])

        # Check if paper has at least one fact with a stance in keep_stances
        has_relevant_fact = any(
            f.stance.upper() in keep_stances_set for f in paper_facts
        )

        if has_relevant_fact:
            papers_to_keep.add(pmid)
        else:
            papers_to_remove.add(pmid)

    # Remove papers
    for pmid in papers_to_remove:
        del state.papers[pmid]

    # Log the filtering operation
    state.append_trace("filter_papers_by_stance", {
        "keep_stances": keep_stances,
        "papers_before": len(papers_to_keep) + len(papers_to_remove),
        "papers_after": len(papers_to_keep),
        "papers_removed": len(papers_to_remove),
        "removed_pmids": list(papers_to_remove),
    })

    # Auto-save
    state._auto_save()

    # Print summary
    print(
        f"Paper filter: {len(papers_to_keep) + len(papers_to_remove)} → {len(papers_to_keep)} "
        f"(removed {len(papers_to_remove)}, keep={', '.join(keep_stances)})"
    )
    if papers_to_remove:
        _debug_print(f"Removed PMIDs: {', '.join(sorted(papers_to_remove))}")


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def compress_evidence(
    state: EvidenceState, target_tokens: int = 40000,
) -> EvidenceState:
    """Compress evidence state while preserving sufficiency.

    Returns a new (compressed) EvidenceState. The caller should reassign::

        state = compress_evidence(state)
    """
    before = state.token_count()
    compressed = _compressor.compress(state, state.claim)
    after = compressed.token_count()
    compressed.token_estimate = after

    within = "Within budget." if after <= target_tokens else "Still over budget."
    print(f"Compression: {before} → {after} tokens. {within}")
    return compressed


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def emit_verdict(
    verdict: str,
    confidence: float,
    reasoning: str,
    key_evidence: list[str],
    gaps_remaining: list[str],
    state: EvidenceState,
    workspace: Optional[Path] = None,
) -> VerificationVerdict:
    """Emit the final verification verdict.

    Runs structural quality checks before accepting the verdict.
    If quality is low, confidence is capped and warnings are printed,
    but the verdict is still emitted (to avoid blocking the loop).

    Optionally writes verdict.json to workspace.
    Returns the VerificationVerdict object.
    """
    # --- Quality gate: structural checks ----------------------------------
    quality_warnings: list[str] = []
    nonempty_facts = [f for f in state.facts if f.text.strip()]

    if not nonempty_facts:
        quality_warnings.append(
            "WARNING: No facts with content — verdict is ungrounded."
        )
        confidence = min(confidence, 0.10)
    if not state.papers:
        quality_warnings.append(
            "WARNING: No papers retrieved — verdict has no evidence base."
        )
        confidence = min(confidence, 0.10)
    if state.coverage and all(c == 0.0 for c in state.coverage.values()):
        quality_warnings.append(
            "WARNING: Zero subclaim coverage — evidence not linked to claim."
        )
        confidence = min(confidence, 0.30)

    for w in quality_warnings:
        logger.warning(w)

    v = VerificationVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        key_evidence=key_evidence,
        gaps_remaining=gaps_remaining,
    )

    # Also checkpoint the final evidence state
    if workspace is not None:
        ws = Path(workspace)
        verdict_path = ws / "verdict.json"
        verdict_path.write_text(v.model_dump_json(indent=2), encoding="utf-8")
        state.checkpoint_save(ws)
        print(f"Verdict emitted: {verdict} (confidence: {confidence:.2f}). Saved to {verdict_path}.")
    else:
        print(f"Verdict emitted: {verdict} (confidence: {confidence:.2f}).")

    return v


# ---------------------------------------------------------------------------
# Kernel bootstrap helper
# ---------------------------------------------------------------------------


def setup_kernel(
    claim: str,
    workspace_path: str,
) -> tuple:
    """One-call kernel bootstrap — returns ``(state, llm, workspace)``.

    LLM connection parameters are read from environment variables set by
    the orchestrator before kernel launch:

    - ``LLM_BASE_URL``  — OpenAI-compatible base URL
    - ``LLM_API_KEY``   — API key (falls back to ``GLM_API_KEY`` / ``EMPTY``)
    - ``LLM_MODEL``     — model identifier
    - ``MLP_MODEL_DIR`` — path to MLP classifier weights

    The agent never needs to know or pass these values.

    Example (inside nb_execute)::

        from proclaim.verification.evidence_api import setup_kernel
        state, llm, workspace = setup_kernel(
            claim="MAPK1 directly activates H3-3A.",
            workspace_path="/path/to/workspace",
        )
    """
    # ---------------------------------------------------------------
    # Silence INFO/DEBUG logging in the kernel so it does not pollute
    # nb_execute cell outputs.
    #
    # Problem: ML libraries (transformers, spaCy, tqdm, mlx-lm …) reset
    # the root logger's *level* to NOTSET/INFO when their modules are first
    # imported (often in a later cell), undoing any setLevel(WARNING) done
    # here.  Setting only the level is therefore not robust.
    #
    # Robust fix: replace any existing StreamHandlers on the root logger
    # with a single WARNING-gated one.  Handler-level filtering survives
    # root-level resets because the handler checks its OWN level *after*
    # the root check, not instead of it.  And by ensuring root already has
    # at least one handler, logging.basicConfig() (called by many libs on
    # import) becomes a no-op and cannot inject a promiscuous handler.
    # ---------------------------------------------------------------
    import logging as _logging
    import os as _os
    import warnings as _warnings

    # Set ML library verbosity env vars before any lazy imports trigger them.
    # HF_HUB_VERBOSITY / TRANSFORMERS_VERBOSITY are read by each library's
    # _configure_library_root_logger() on first import, so setting them here
    # ensures their logger level is ERROR even when they install their own
    # StreamHandler.  TQDM_DISABLE suppresses all progress bars (model-load
    # "Loading weights" bars etc.) since tqdm reads this env var on import.
    _os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _os.environ.setdefault("TQDM_DISABLE", "1")

    _root = _logging.getLogger()
    # Remove any StreamHandler that might have been added by transitive
    # imports before this call (e.g. from requests, httpx, urllib3).
    for _h in list(_root.handlers):
        if isinstance(_h, _logging.StreamHandler) and not isinstance(_h, _logging.FileHandler):
            _root.removeHandler(_h)
    # Add a single WARNING-filtered StreamHandler.  Even if a library
    # subsequently calls logging.root.setLevel(DEBUG), this handler won't
    # let INFO/DEBUG through.
    _sh = _logging.StreamHandler()
    _sh.setLevel(_logging.WARNING)
    _sh.setFormatter(_logging.Formatter("%(levelname)s: %(message)s"))
    _root.addHandler(_sh)
    _root.setLevel(_logging.WARNING)

    # Silence noisy named loggers regardless of root level.
    # huggingface_hub and transformers add their own StreamHandlers; setting
    # level to ERROR here is belt-and-suspenders alongside the env vars above.
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)
    _logging.getLogger("urllib3").setLevel(_logging.WARNING)
    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
    _logging.getLogger("transformers").setLevel(_logging.ERROR)

    # Suppress irrelevant third-party Python warnings.
    # Note: TqdmWarning extends Warning (not UserWarning), so no category arg.
    _warnings.filterwarnings("ignore", message=r".*\[W095\].*", category=UserWarning)
    _warnings.filterwarnings("ignore", message=r".*IProgress.*")
    _warnings.filterwarnings("ignore", category=FutureWarning, module=r"spacy|transformers|tqdm|mlx")

    import os

    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)

    state = EvidenceState.init_new(
        claim=claim,
        subclaims=[claim],
        workspace=ws,
    )

    # Override MAX_ITERATIONS from environment (set by build_sdk_env from config)
    max_iter_env = os.environ.get("MAX_ITERATIONS")
    if max_iter_env is not None:
        try:
            state.MAX_ITERATIONS = int(max_iter_env)
        except ValueError:
            pass  # keep default if env var is malformed

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1/")
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or os.environ.get("ZAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    model = os.environ.get("LLM_MODEL", "glm-5")

    from proclaim.verification.llm_factory import make_llm

    # Read disable_thinking config from environment (set by build_sdk_env)
    disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "0") == "1"
    extra_body = None
    if disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    timeout = int(os.environ.get("LLM_TIMEOUT", "300"))
    stream = os.environ.get("LLM_STREAM", "1") == "1"

    llm = make_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        extra_body=extra_body,
        timeout=timeout,
        stream=stream,
    )

    # Initialize label config from environment (JSON-encoded, set by build_sdk_env)
    label_json = os.environ.get("LABEL_CONFIG_JSON")
    if label_json:
        import json as _json
        try:
            from proclaim.verification.config import LabelConfig, set_label_config
            label_data = _json.loads(label_json)
            set_label_config(LabelConfig(**label_data))
        except Exception as exc:
            logger.warning("Failed to parse LABEL_CONFIG_JSON: %s", exc)
    else:
        # Ensure default label config is available
        from proclaim.verification.config import set_label_config, LabelConfig
        set_label_config(LabelConfig())

    print(f"Kernel ready. state=<{len(state.papers)} papers>, llm={model!r}, max_iterations={state.MAX_ITERATIONS}")
    return state, llm, ws


def setup_workspace(
    claim: str,
    workspace_path: str,
) -> tuple:
    """Stateless workspace bootstrap — returns ``(state, llm, workspace)``.

    Like ``setup_kernel()`` but designed for bash-mode execution where
    there is no persistent Jupyter kernel.  Each call:

    - **Loads** existing ``evidence_state.json`` if present (idempotent),
      or creates a new ``EvidenceState`` if not.
    - Builds an ``llm`` callable from environment variables.
    - Initialises label config.

    Safe to call at the top of every bash invocation.

    LLM and config parameters are read from environment variables set by
    the orchestrator:

    - ``LLM_BASE_URL``  — OpenAI-compatible base URL (subagent endpoint)
    - ``LLM_API_KEY``   — API key
    - ``LLM_MODEL``     — model identifier
    - ``MLP_MODEL_DIR`` — path to MLP classifier weights

    Example::

        from proclaim.verification.evidence_api import setup_workspace
        state, llm, workspace = setup_workspace(
            claim="MAPK1 directly activates H3-3A.",
            workspace_path="/path/to/workspace",
        )
    """
    import os

    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)

    state_path = ws / "evidence_state.json"
    if state_path.exists():
        state = EvidenceState.load(state_path)
    else:
        state = EvidenceState.init_new(
            claim=claim,
            subclaims=[claim],
            workspace=ws,
        )

    # Override MAX_ITERATIONS from environment
    max_iter_env = os.environ.get("MAX_ITERATIONS")
    if max_iter_env is not None:
        try:
            state.MAX_ITERATIONS = int(max_iter_env)
        except ValueError:
            pass

    if state.iteration > state.MAX_ITERATIONS:
        raise MaxIterationsExceeded(
            f"Workspace state iteration ({state.iteration}) exceeds "
            f"MAX_ITERATIONS ({state.MAX_ITERATIONS}).",
            state=state,
        )

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1/")
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or os.environ.get("ZAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    model = os.environ.get("LLM_MODEL", "glm-5")

    disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "0") == "1"
    extra_body = None
    if disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    timeout = int(os.environ.get("LLM_TIMEOUT", "300"))
    stream = os.environ.get("LLM_STREAM", "1") == "1"

    from proclaim.verification.llm_factory import make_llm
    llm = make_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        extra_body=extra_body,
        timeout=timeout,
        stream=stream,
    )

    # Initialize label config from environment
    label_json = os.environ.get("LABEL_CONFIG_JSON")
    if label_json:
        import json as _json
        try:
            from proclaim.verification.config import LabelConfig, set_label_config
            label_data = _json.loads(label_json)
            set_label_config(LabelConfig(**label_data))
        except Exception as exc:
            logger.warning("Failed to parse LABEL_CONFIG_JSON: %s", exc)
    else:
        from proclaim.verification.config import set_label_config, LabelConfig
        set_label_config(LabelConfig())

    print(f"Workspace ready. state=<{len(state.papers)} papers, {len(state.facts)} facts>, llm={model!r}")
    return state, llm, ws

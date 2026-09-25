"""
Natural-language query generation for the librarian retrieval backend.

The librarian (Bio-Agent literature-search agent) expects a natural-language
*information need*, not a PubMed Boolean query.  It runs its own internal
query planner over that text.  This module turns ProClaim claims, gaps, and
failed-extraction context into the rich natural-language requests the
librarian works best with.

This is the librarian-specific counterpart of
``proclaim.search.llm_query_generator`` (which targets PubMed/S2 syntax).
The PubMed query generator is intentionally NOT reused: its prompts are built
around Boolean operators and field tags that the librarian does not want.
"""

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts — ask for a single natural-language information need.
# ---------------------------------------------------------------------------

_GAP_QUERY_PROMPT = """\
You are preparing a natural-language search request for a biomedical literature \
search agent. The agent reads plain English and finds relevant papers itself, \
so do NOT use Boolean operators, field tags, or database syntax.

The verification of the claim below is missing specific evidence. Write ONE \
natural-language information need targeting exactly that gap, naming the key \
entities and asking for direct evidence that would resolve it — whether that \
evidence supports OR contradicts the claim. Do not bias the search toward only \
confirming the claim.

Claim: {claim}

Missing evidence (the gap to fill): {gap_description}

ProClaim domain guidance:
{domain_guidance}

Respond with the information need only — one or two sentences, no preamble."""

_COMBINED_GAPS_QUERY_PROMPT = """\
You are preparing a natural-language search request for a biomedical literature \
search agent. The agent reads plain English and finds relevant papers itself, \
so do NOT use Boolean operators, field tags, or database syntax.

The verification of the claim below has several remaining evidence gaps. Write \
ONE concise natural-language information need that asks the search agent to find \
papers that can fill all of these gaps together. Name the key entities and the \
missing mechanisms, and ask for direct evidence that would resolve the gaps — \
whether that evidence supports OR contradicts the claim. Do not bias the search \
toward only confirming the claim. Keep the request compact enough to remain \
searchable.

Claim: {claim}

Evidence gaps to fill:
{gaps_section}

ProClaim domain guidance:
{domain_guidance}

Respond with the information need only — one or two sentences, no preamble."""

_REFINE_QUERY_PROMPT = """\
You are preparing a natural-language search request for a biomedical literature \
search agent. The agent reads plain English and finds relevant papers itself, \
so do NOT use Boolean operators, field tags, or database syntax.

A previous search returned papers from which no decisive evidence could be \
extracted for the claim below. Some papers may have produced only neutral or \
ambiguous facts. The titles/abstracts of those papers are listed so you can see \
what went wrong (wrong entity sense, tangential topic, unclear direction, etc.).

A common reason for finding nothing is that the literature names an entity \
differently than the claim does. Before writing the request, consider each \
entity's common aliases, gene/protein name variants, and complex or subunit \
names, and fold the most likely alternatives into the request.

Write ONE more precise natural-language information need that steers the search \
toward papers with direct evidence that tests the claim — supporting OR \
contradicting it — avoiding the irrelevant directions seen below.

Claim: {claim}

Papers that yielded no decisive evidence:
{failed_papers_section}

ProClaim domain guidance:
{domain_guidance}

Respond with the information need only — one or two sentences, no preamble."""

_BROAD_QUERY_PROMPT = """\
You are preparing a natural-language search request for a biomedical literature \
search agent. The agent reads plain English and finds relevant papers itself, \
so do NOT use Boolean operators, field tags, or database syntax.

Write ONE deliberately recall-oriented natural-language information need for \
the claim below. Name the key biological entities (genes, proteins, molecules) \
and include common aliases, protein names, complex names, and isoform names \
when they are likely to appear in papers. Preserve the claimed relationship and \
direction, but broaden it with mechanistic equivalents: binding/interaction, \
phosphorylation, activation, inhibition, degradation, stabilization, \
translocation, signaling activity, or pathway readouts \
when relevant. Ask for evidence that supports OR contradicts the claimed \
direction. Do not make narrow wording such as "directly", "extracellularly", \
or "ligand-receptor" the only search route; use those words plus equivalent \
experimental descriptions. The goal is to surface a wide pool of on-topic \
papers; a downstream extractor decides which papers contain usable evidence.

Claim: {claim}{subclaims_section}

ProClaim domain guidance:
{domain_guidance}

Respond with the information need only — one or two sentences, no preamble."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_response(text: str) -> str:
    """Strip reasoning tags, code fences, and prefixes from an LLM response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = text.strip().strip("`").strip()
    # Drop a leading label such as "Information need:" or "Query:".
    text = re.sub(
        r"^(information need|query|request)\s*:\s*", "", text, flags=re.IGNORECASE
    )
    # Collapse whitespace so the result is a single clean line.
    return " ".join(text.split())


def _claim_fallback(claim: str) -> str:
    """Deterministic natural-language request for initial librarian search."""
    domain_guidance = _proclaim_domain_guidance(claim)
    return (
        "Find papers that provide direct experimental "
        f'evidence for or against this claim: "{claim.strip()}". Focus on the '
        "named entities, their aliases or protein names, their interaction or "
        "regulatory relationship, directionality, and biological context. Include "
        "mechanistic equivalents such as binding, phosphorylation, activation, "
        "inhibition, degradation, stabilization, signaling activity, and pathway "
        "readouts when relevant. Include contradictory as well as supporting "
        f"evidence. {domain_guidance}"
    )


def _subclaim_fallback(claim: str, subclaims: list[str]) -> str:
    """Deterministic natural-language request enriched with subclaims."""
    domain_guidance = _proclaim_domain_guidance(claim)
    cleaned_subclaims = [
        " ".join(str(subclaim).split())
        for subclaim in subclaims
        if str(subclaim).strip()
    ]
    if not cleaned_subclaims:
        return _claim_fallback(claim)

    subclaim_lines = "\n".join(f"- {subclaim}" for subclaim in cleaned_subclaims)
    return (
        "Find papers that provide direct experimental evidence relevant to the "
        f'following claim:\n"{claim.strip()}"\n'
        "and these subclaims:\n"
        f"{subclaim_lines}\n"
        "Focus on experiments that test the named entities, their aliases or "
        "protein names, the direction of the relationship, and mechanistic "
        "equivalents such as binding, phosphorylation, activation, inhibition, "
        "degradation, stabilization, signaling activity, and pathway readouts. "
        f"{domain_guidance}"
    )


def _format_section(title: str, items: list[str] | None) -> str:
    if not items:
        return ""
    joined = "\n".join(f"- {item}" for item in items)
    return f"\n\n{title}:\n{joined}"


def _proclaim_domain_guidance(claim: str) -> str:
    """Return benchmark-specific retrieval guidance for ProClaim claim shapes."""
    claim_lc = claim.lower()
    if " as ligand " in claim_lc and " as receptor" in claim_lc:
        return (
            "For ligand-receptor claims, search for the named pair without relying "
            "only on the words ligand and receptor. Include direct binding, "
            "cell-surface or extracellular engagement, receptor-complex evidence, "
            "and papers that assign the opposite ligand/receptor roles. If the "
            "receptor belongs to a multi-subunit, heterodimeric, immune-recognition, "
            "or bidirectional membrane-signaling system, include evidence for "
            "complexes or heterodimers containing the named receptor subunit."
        )
    if " directly activates " in claim_lc or " directly inhibits " in claim_lc:
        return (
            "For signed protein-regulation claims, search for direct mechanism "
            "papers and aliases for both proteins. Include phosphorylation, "
            "cleavage, ubiquitination, degradation, stabilization, complex "
            "formation, direct expression regulation, enzymatic activity, and "
            "pathway readouts that establish activation or inhibition. Also look "
            "for opposite-polarity evidence that would refute the claim."
        )
    return (
        "Prioritize primary experimental papers over network, enrichment, docking, "
        "co-expression, or pathway-summary papers unless those summaries cite a "
        "direct experiment involving the named entities."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_librarian_query(
    claim: str,
    llm: Callable,
    subclaims: list[str] | None = None,
    extraction_context: list[str] | None = None,
) -> str:
    """Generate a natural-language information need for the initial claim search.

    Args:
        claim: The scientific claim to verify.
        llm: Unused; kept for API compatibility with the PubMed/S2 query
            generators and existing callers.
        subclaims: Unused for the claim-only request. Use
            ``generate_librarian_subclaim_query`` when subclaims should focus
            the initial librarian request.
        extraction_context: Unused for initial librarian search.

    Returns:
        A deterministic plain-English information need.
    """
    query = _claim_fallback(claim)
    logger.info("[Librarian Query] %s", query)
    return query


def generate_librarian_subclaim_query(
    claim: str,
    llm: Callable,
    subclaims: list[str] | None = None,
    extraction_context: list[str] | None = None,
) -> str:
    """Generate the broad, LLM-written librarian request for the initial search.

    Asks the LLM for a deliberately broad information need — key entities and the
    general relationship, with narrow qualifiers dropped — so the librarian
    returns a wide pool of on-topic papers and the fact extractor decides
    relevance. Falls back to the deterministic subclaim request if the LLM call
    fails or returns nothing.
    """
    del extraction_context
    cleaned_subclaims = [
        " ".join(str(subclaim).split())
        for subclaim in (subclaims or [])
        if str(subclaim).strip()
    ]
    subclaims_section = _format_section(
        "Subclaims (use these to surface the key entities and aliases)",
        cleaned_subclaims,
    )
    prompt = _BROAD_QUERY_PROMPT.format(
        claim=claim,
        subclaims_section=subclaims_section,
        domain_guidance=_proclaim_domain_guidance(claim),
    )
    try:
        query = _clean_response(llm(prompt))
        if query:
            logger.info("[Librarian Broad Query] %s", query)
            return query
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Librarian broad query generation failed: %s", exc)

    fallback = _subclaim_fallback(claim, subclaims or [])
    logger.info("[Librarian Broad Query — fallback] %s", fallback)
    return fallback


def generate_librarian_gap_query(
    claim: str,
    gap_description: str,
    llm: Callable,
) -> str:
    """Generate a natural-language information need targeting an evidence gap."""
    prompt = _GAP_QUERY_PROMPT.format(
        claim=claim,
        gap_description=gap_description,
        domain_guidance=_proclaim_domain_guidance(claim),
    )
    try:
        query = _clean_response(llm(prompt))
        if query:
            logger.info("[Librarian Gap Query] %s", query)
            return query
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Librarian gap query generation failed: %s", exc)

    # Fallback: the raw gap description is already natural language.
    logger.info("[Librarian Gap Query — fallback] %s", gap_description)
    return gap_description


def generate_librarian_combined_gaps_query(
    claim: str,
    gap_descriptions: list[str],
    llm: Callable,
) -> str:
    """Generate one concise information need covering all evidence gaps."""
    cleaned_gaps = [
        " ".join(str(gap).split()) for gap in gap_descriptions if str(gap).strip()
    ]
    if not cleaned_gaps:
        return _claim_fallback(claim)

    gaps_section = "\n".join(f"- {gap}" for gap in cleaned_gaps)
    prompt = _COMBINED_GAPS_QUERY_PROMPT.format(
        claim=claim,
        gaps_section=gaps_section,
        domain_guidance=_proclaim_domain_guidance(claim),
    )
    try:
        query = _clean_response(llm(prompt))
        if query:
            logger.info("[Librarian Combined Gaps Query] %s", query)
            return query
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Librarian combined gap query generation failed: %s", exc)

    fallback = (
        f'Find papers that help verify the claim "{claim.strip()}" and address '
        f"these evidence gaps: {'; '.join(cleaned_gaps)}"
    )
    logger.info("[Librarian Combined Gaps Query — fallback] %s", fallback)
    return fallback


def generate_librarian_refine_query(
    claim: str,
    failed_papers: list[dict],
    llm: Callable,
) -> str:
    """Generate a refined natural-language information need after weak extraction.

    Args:
        claim: The scientific claim to verify.
        failed_papers: List of ``{"pmid", "title", "abstract"}`` dicts for
            papers that yielded no decisive facts.
        llm: LLM callable.
    """
    lines: list[str] = []
    for paper in failed_papers[:5]:
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        lines.append(f"- {title}\n  {abstract[:300]}")
    failed_papers_section = "\n".join(lines) if lines else "- (no metadata available)"

    prompt = _REFINE_QUERY_PROMPT.format(
        claim=claim,
        failed_papers_section=failed_papers_section,
        domain_guidance=_proclaim_domain_guidance(claim),
    )
    try:
        query = _clean_response(llm(prompt))
        if query:
            logger.info("[Librarian Refine Query] %s", query)
            return query
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Librarian refine query generation failed: %s", exc)

    fallback = _claim_fallback(claim)
    logger.info("[Librarian Refine Query — fallback] %s", fallback)
    return fallback

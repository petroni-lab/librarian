"""Librarian runtime configuration — a single flat set of tuning knobs.

The full agent split these between ``config_dev``/``config_prod`` selected by
``OMNIA_ENV``. This minimal version keeps only the defaults that gave the best
LitQA2 accuracy, plus the handful of env-var overrides an operator (or an
ablation sweep) actually needs. Each knob's comment explains what it controls.
"""

import os
from dataclasses import dataclass
from typing import Optional

# ── Defaults (the winning LitQA2 values) ─────────────────────────────────────

_DEFAULT_MODEL_NAME: Optional[str] = None  # falls back to LLM_MODEL env var
_DEFAULT_MAX_QUERY_COUNT = 7
# Europe PMC results fetched per sub-query before the per-query prefilter cap.
_DEFAULT_SEARCH_PAGE_SIZE = 50
# Per-query prefilter cap: how many of each sub-query's top papers are kept for
# full-text enrichment and relevance filtering.
_DEFAULT_PAPERS_PER_SUBQUERY = 50
# Per-paper BM25 excerpt budget in tokens (x 0.75 = words).
# 3072 -> ~2300 words: the value that drove LitQA2 accuracy from 0.65 -> 0.75.
_DEFAULT_FULL_TEXT_TARGET_TOKENS_PER_PAPER = 3072
# Words of evidence per paper handed to the summarizer: a contiguous slice of
# the larger BM25 excerpt. 250 matches OpenScholar's passage size.
_DEFAULT_EVIDENCE_SNIPPET_MAX_WORDS = 250
# LLM relevance-filter batch sizes.
_DEFAULT_ABSTRACT_FILTER_BATCH = 50
_DEFAULT_FULLTEXT_FILTER_BATCH = 10
# Final-merge pool caps. The returned passage set is composed from two separate
# pools so abstract-only papers are never crowded out by full-text papers.
_DEFAULT_MAX_FULLTEXT_PAPERS = 30
_DEFAULT_MAX_ABSTRACT_ONLY_PAPERS = 30
# Relevance-filter LLM temperature. 0.1 is the winning config; set to 0 for
# deterministic, reproducible filtering.
_DEFAULT_FILTER_TEMPERATURE = 0.1

_DEFAULT_QUERY_BUDGET_GUIDANCE = (
    "- Generate AT MOST 7 complementary queries.\n"
    "- Balance recall and precision across the set.\n"
    "- Include one broader high-recall safety-net query unless the request is "
    "already extremely specific.\n"
    "- Use the remaining queries for focused sub-questions, synonyms, or fielded "
    "variants that improve coverage without over-constraining the search."
)


@dataclass(frozen=True)
class LibrarianRuntimeConfig:
    """Tuning knobs for the librarian agent (fields map 1:1 to agent constants)."""

    default_model_name: Optional[str]
    query_budget_guidance: str
    max_query_count: int
    search_page_size: int
    evidence_snippet_max_words: int
    papers_per_subquery: int
    full_text_target_tokens_per_paper: int
    abstract_filter_batch: int
    fulltext_filter_batch: int
    max_fulltext_papers: int
    max_abstract_only_papers: int
    filter_temperature: float


def _env_override(name: str, default, cast):
    """Return ``cast(env[name])`` if the env var is set and valid, else ``default``."""
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        return default


def load_runtime_config() -> LibrarianRuntimeConfig:
    """Build the config from the defaults above, applying any ``LITERATURE_*``
    env-var overrides for the operator-tunable knobs."""
    return LibrarianRuntimeConfig(
        default_model_name=_DEFAULT_MODEL_NAME,
        query_budget_guidance=_DEFAULT_QUERY_BUDGET_GUIDANCE,
        max_query_count=_env_override(
            "LITERATURE_MAX_QUERY_COUNT", _DEFAULT_MAX_QUERY_COUNT, int
        ),
        search_page_size=_DEFAULT_SEARCH_PAGE_SIZE,
        evidence_snippet_max_words=_env_override(
            "LITERATURE_EVIDENCE_SNIPPET_MAX_WORDS",
            _DEFAULT_EVIDENCE_SNIPPET_MAX_WORDS,
            int,
        ),
        papers_per_subquery=_DEFAULT_PAPERS_PER_SUBQUERY,
        full_text_target_tokens_per_paper=_env_override(
            "LITERATURE_FULL_TEXT_TARGET_TOKENS_PER_PAPER",
            _DEFAULT_FULL_TEXT_TARGET_TOKENS_PER_PAPER,
            int,
        ),
        abstract_filter_batch=_DEFAULT_ABSTRACT_FILTER_BATCH,
        fulltext_filter_batch=_DEFAULT_FULLTEXT_FILTER_BATCH,
        max_fulltext_papers=_env_override(
            "LITERATURE_MAX_FULLTEXT_PAPERS", _DEFAULT_MAX_FULLTEXT_PAPERS, int
        ),
        max_abstract_only_papers=_env_override(
            "LITERATURE_MAX_ABSTRACT_ONLY_PAPERS",
            _DEFAULT_MAX_ABSTRACT_ONLY_PAPERS,
            int,
        ),
        filter_temperature=_env_override(
            "LITERATURE_FILTER_TEMPERATURE", _DEFAULT_FILTER_TEMPERATURE, float
        ),
    )

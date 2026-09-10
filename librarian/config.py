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
# 8 pairs with paragraphs_per_judge_batch below: 16 x 8 = 128, so the whole
# paragraph pool is judged in one call (see that knob's comment).
_DEFAULT_MAX_QUERY_COUNT = 8
# Europe PMC results fetched per sub-query — every returned paper is decomposed
# into paragraphs and BM25-ranked, so this is also the per-sub-query recall lever.
_DEFAULT_PAPERS_PER_SUBQUERY = 50
# Stage-2 knob k: top-BM25 paragraphs each sub-query passes to Stage 3.
_DEFAULT_PARAGRAPHS_PER_SUBQUERY = 16
# Stage-3 batch size: paragraphs per relevance-judge LLM call. Should be
# >= paragraphs_per_subquery * max_query_count so the whole pool is judged in
# ONE call: the judge ranks globally, but multi-batch results are merely
# concatenated in batch order, so a smaller value silently degrades the
# ranking to per-batch. Raise all three knobs together.
_DEFAULT_PARAGRAPHS_PER_JUDGE_BATCH = 128
# Body paragraphs longer than this are split into overlapping windows before
# BM25 so a long paragraph can't outrank on length alone. Overlap keeps a
# match that straddles a cut intact.
_DEFAULT_MAX_PARAGRAPH_WORDS = 250
_DEFAULT_PARAGRAPH_OVERLAP_WORDS = 50
# Relevance-filter LLM temperature. 0.1 is the winning config; set to 0 for
# deterministic, reproducible filtering.
_DEFAULT_FILTER_TEMPERATURE = 0.1

_DEFAULT_QUERY_BUDGET_GUIDANCE = (
    "- Generate AT MOST {max_queries} complementary queries.\n"
    "- Balance recall and precision across the set.\n"
    "- Include one broader high-recall safety-net query unless the request is "
    "already extremely specific.\n"
    "- Use the remaining queries for focused sub-questions, synonyms, or fielded "
    "variants that improve coverage without over-constraining the search."
)


# One query per role in query_budget_guidance's recall/coverage/precision split.
MIN_QUERY_COUNT = 3


@dataclass(frozen=True)
class LibrarianRuntimeConfig:
    """Tuning knobs for the librarian agent (fields map 1:1 to agent constants)."""

    default_model_name: Optional[str]
    query_budget_guidance: str
    max_query_count: int
    papers_per_subquery: int
    paragraphs_per_subquery: int
    paragraphs_per_judge_batch: int
    max_paragraph_words: int
    paragraph_overlap_words: int
    filter_temperature: float

    def __post_init__(self) -> None:
        """Reject a sub-query budget too small to cover all three query roles.

        ``query_budget_guidance`` splits the budget across recall, coverage and
        precision queries with at least one each, so a budget below three cannot
        express the allocation it is handed. Runs on construction and on
        ``dataclasses.replace()`` alike.

        :raises ValueError: if ``max_query_count`` is below ``MIN_QUERY_COUNT``.
        """
        if self.max_query_count < MIN_QUERY_COUNT:
            raise ValueError(
                f"max_query_count must be >= {MIN_QUERY_COUNT} "
                f"(got {self.max_query_count}): the query budget is allocated "
                "across recall, coverage and precision roles."
            )


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
        papers_per_subquery=_env_override(
            "LITERATURE_PAPERS_PER_SUBQUERY", _DEFAULT_PAPERS_PER_SUBQUERY, int
        ),
        paragraphs_per_subquery=_env_override(
            "LITERATURE_PARAGRAPHS_PER_SUBQUERY",
            _DEFAULT_PARAGRAPHS_PER_SUBQUERY,
            int,
        ),
        paragraphs_per_judge_batch=_env_override(
            "LITERATURE_PARAGRAPHS_PER_JUDGE_BATCH",
            _DEFAULT_PARAGRAPHS_PER_JUDGE_BATCH,
            int,
        ),
        max_paragraph_words=_env_override(
            "LITERATURE_MAX_PARAGRAPH_WORDS", _DEFAULT_MAX_PARAGRAPH_WORDS, int
        ),
        paragraph_overlap_words=_env_override(
            "LITERATURE_PARAGRAPH_OVERLAP_WORDS",
            _DEFAULT_PARAGRAPH_OVERLAP_WORDS,
            int,
        ),
        filter_temperature=_env_override(
            "LITERATURE_FILTER_TEMPERATURE", _DEFAULT_FILTER_TEMPERATURE, float
        ),
    )

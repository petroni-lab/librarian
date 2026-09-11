"""Librarian agent runtime configuration.

Loaded once via ``load_runtime_config()`` and passed explicitly to
``LibrarianAgent(runtime_config=...)`` — the constructor has no implicit
default, so every caller decides its own config.

Values live in ``config.toml`` (this directory), which is self-contained:
every knob must be set there, there is no fallback.

There is no env-var override mechanism: a one-off change (e.g. an ablation
sweep varying ``num_subqueries``) is a plain ``dataclasses.replace()`` on the
loaded config, done explicitly by the caller:

    cfg = replace(load_runtime_config(), num_subqueries=3)
    LibrarianAgent(runtime_config=cfg, ...)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mashumaro.mixins.toml import DataClassTOMLMixin


# One query per role in query_budget_guidance: the high-recall safety net plus
# the focused sub-question/synonym/fielded variants it is balanced against.
MIN_SUBQUERIES = 3


@dataclass(frozen=True)
class LibrarianRuntimeConfig(DataClassTOMLMixin):
    """Tuning knobs for the librarian agent.

    Fields map directly to the constants they replace in ``agent.py``. See
    each field's comment here for what the knob *is*; see ``config.toml`` for
    why a given value is what it is.

    No field has a default: build an instance via ``from_toml()`` (see
    ``load_runtime_config()``) or by passing every argument explicitly.
    """

    # Model used when llm_model_name isn't passed explicitly to LibrarianAgent.
    # Empty means the LLM client falls back to the LLM_MODEL env var.
    default_model_name: Optional[str]
    # {max_queries} is filled at prompt-build time from num_subqueries, so the cap
    # and the instruction to the planner always agree (one knob drives both).
    query_budget_guidance: str
    # ── The four primary sizing knobs ────────────────────────────────────────
    # How many sub-queries the planner generates and runs. Must be >= MIN_SUBQUERIES
    # (see __post_init__).
    num_subqueries: int
    # Papers Europe PMC returns per sub-query (recall lever, sent as the page size).
    papers_per_subquery: int
    # Stage 2 knob k: top-BM25 paragraphs each sub-query passes to Stage 3.
    paragraphs_per_subquery: int
    # Stage 3 batch size: paragraphs per relevance-judge LLM call. Should be
    # >= paragraphs_per_subquery * num_subqueries so the whole pool is judged
    # in ONE call: the judge ranks globally, but multi-batch results are
    # merely concatenated in batch order, so a smaller value silently
    # degrades the ranking to per-batch. Raise all three knobs together.
    paragraphs_per_judge_batch: int
    # Body paragraphs longer than this are split into overlapping windows before BM25
    # so a long paragraph can't outrank on length alone. Overlap keeps a match that
    # straddles a cut intact.
    max_paragraph_words: int
    paragraph_overlap_words: int
    # Relevance-filter LLM sampling temperature.
    filter_temperature: float

    def __post_init__(self) -> None:
        """Reject a sub-query budget too small to cover all the query roles.

        ``query_budget_guidance`` spends the budget on a broad high-recall
        safety-net query plus the focused variants that balance it, so a budget
        below three cannot express the allocation it is handed. Runs on
        ``from_toml()`` and on ``dataclasses.replace()`` alike, so an ablation
        sweep is checked the same way the config file is.

        :raises ValueError: if ``num_subqueries`` is below ``MIN_SUBQUERIES``.
        """
        if self.num_subqueries < MIN_SUBQUERIES:
            raise ValueError(
                f"num_subqueries must be >= {MIN_SUBQUERIES} "
                f"(one query per role: recall safety net, focused variants), "
                f"got {self.num_subqueries}."
            )


_CONFIG_DIR = Path(__file__).parent
_CONFIG_NAME = "config.toml"


def load_runtime_config() -> LibrarianRuntimeConfig:
    """Return the ``LibrarianRuntimeConfig`` read from ``config.toml``.

    Fails loudly rather than silently substituting a default — a knob missing
    from the file is an error, not an invitation to guess.

    :return: The runtime config.
    :rtype: LibrarianRuntimeConfig
    :raises FileNotFoundError: if ``config.toml`` is missing.
    :raises mashumaro.exceptions.MissingField: if the TOML file is missing a
        required knob.
    :raises ValueError: if the TOML file sets ``num_subqueries`` below
        ``MIN_SUBQUERIES``.
    """
    toml_path = _CONFIG_DIR / _CONFIG_NAME
    return LibrarianRuntimeConfig.from_toml(toml_path.read_text(encoding="utf-8"))

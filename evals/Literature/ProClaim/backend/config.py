"""Configuration for a librarian-backed ProClaim run.

GPL-3.0; see ../NOTICE.

ProClaim's ``VerificationSettings`` knows nothing about the librarian, and its
``model_config`` is ``extra="ignore"``. That combination is the sharpest edge in
this whole integration: a YAML config saying ``retrieval_backend: librarian``
against the plain upstream class is **silently dropped**, the run retrieves with
PubMed instead, and it produces perfectly plausible numbers for the wrong
pipeline.

Subclassing is what closes that. The three fields below are declared, so they
are parsed rather than ignored, and ``load_from_cli`` then asserts that the
backend really did resolve to "librarian" — so a typo in a key, or the wrong
config file, fails at startup instead of forty minutes later in a results table.

``from_yaml`` and ``from_cli`` are inherited unchanged: both end in
``cls(**raw)``, so they build this class without upstream needing to know it
exists.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from proclaim.verification.config import VerificationSettings


class LibrarianSettings(VerificationSettings):
    """``VerificationSettings`` plus the librarian retrieval backend's fields."""

    retrieval_backend: Literal["pubmed", "librarian"] = Field(
        default="librarian",
        description=(
            "Retrieval layer backend. This entry point only runs 'librarian': "
            "an in-process LibrarianAgent provides papers and evidence "
            "snippets, and PubMed/S2 search and full-text fetching are bypassed "
            "entirely. The field exists so that a config naming it is parsed "
            "rather than ignored, and so 'pubmed' can be rejected loudly."
        ),
    )
    librarian_llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description=(
            "OpenAI-compatible base URL of the model service backing the "
            "librarian agent's internal LLM (query planner / relevance filter). "
            "Independent of the evidence-programmer model."
        ),
    )
    librarian_llm_model: str = Field(
        default="",
        description=(
            "Model id served by librarian_llm_base_url. If empty the agent "
            "falls back to its own default model resolution."
        ),
    )


def load_from_cli(argv: list[str] | None = None) -> LibrarianSettings:
    """Parse ProClaim's own CLI into a LibrarianSettings, and check it.

    :raises SystemExit: if the resolved config would not actually run the
        librarian backend. Continuing would silently measure PubMed retrieval
        and report it as the librarian's number.
    """
    cfg = LibrarianSettings.from_cli(argv)
    if cfg.retrieval_backend != "librarian":
        raise SystemExit(
            "ERROR: this entry point runs the librarian retrieval backend, but "
            f"the resolved config says retrieval_backend={cfg.retrieval_backend!r}.\n"
            "       Use one of evals/Literature/ProClaim/configs/*.yaml, or run\n"
            "       ProClaim's own module for a PubMed run."
        )
    return cfg


def _self_check() -> None:
    """Assert the one thing that would otherwise fail silently."""
    fields = LibrarianSettings.model_fields
    for name in ("retrieval_backend", "librarian_llm_base_url", "librarian_llm_model"):
        assert name in fields, f"{name} is not a declared field; it would be ignored"
    # Upstream's extra="ignore" is what makes the subclass necessary; if that
    # ever becomes "forbid" or "allow" upstream, this comment is the trail.
    assert VerificationSettings.model_config.get("extra") == "ignore", (
        "upstream no longer ignores extra keys — revisit why this subclass exists"
    )
    assert "retrieval_backend" not in VerificationSettings.model_fields, (
        "upstream has grown a retrieval_backend field; this subclass may now be "
        "redundant, or worse, conflicting"
    )
    print("ProClaim librarian config self-check ok")


if __name__ == "__main__":
    _self_check()

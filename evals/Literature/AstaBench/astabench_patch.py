"""Run-time resilience patches for the AstaBench clone.

The clone under `vendor/` is pinned and pristine — `../setup.sh --check`
asserts it is byte-for-byte its upstream commit, and nothing of ours is written
inside it. The two adjustments AstaBench needs for long unattended runs are
applied here instead, by replacing two functions after import.

WHAT AND WHY

Both concern the Asta MCP tool endpoint, and both are about a run of hundreds of
questions surviving a transient network fault rather than about what LitQA2
measures:

  * `make_asta_toolsource()` connects with inspect's default 5 s connect
    timeout. Over a long run that is short enough to fail on ordinary jitter.
    ASTA_MCP_CONNECT_TIMEOUT and ASTA_MCP_SSE_READ_TIMEOUT make it settable,
    defaulting to 20 s and 5 min.
  * `_is_retryable_error()` treats only HTTP 429/529/504 and
    `anyio.BrokenResourceError` as retryable. httpx and httpcore timeouts are
    equally transient and equally worth retrying.

WHEN IT APPLIES

Only to the `standard_tooling` config, which is the AstaBench-native baseline.
The paper's +Librarian row runs `custom_tooling` and never touches the Asta MCP
endpoint at all, so none of this is on the path that produces it. run_evals.py
calls `apply()` only when a standard_tooling run is actually selected.

WHY IT FAILS LOUDLY

A patch that silently does not apply is worse than no patch: the run would
proceed with the defaults and the flakiness this exists to prevent. `apply()`
therefore asserts that each target is there and is what it expects before
replacing it, and raises otherwise. A pin bump that renames either function
stops the run rather than quietly reverting its behaviour.
"""

from __future__ import annotations

import os

_APPLIED = False


def apply() -> None:
    """Replace the two functions. Idempotent; raises if the clone has moved."""
    global _APPLIED
    if _APPLIED:
        return

    from astabench import tools as tools_pkg
    from astabench.tools import asta_tools

    for name in (
        "make_asta_toolsource",
        "_is_retryable_error",
        "_unravel_exception_group",
        "create_server_streamable_http",
    ):
        if not callable(getattr(asta_tools, name, None)):
            raise RuntimeError(
                f"astabench.tools.asta_tools.{name} is missing, so the resilience "
                "patch cannot be applied. The clone is pinned, so this means the "
                "pin moved; see evals/Literature/AstaBench/astabench_patch.py."
            )

    _patch_toolsource_timeouts(tools_pkg, asta_tools)
    _patch_retryable_errors(asta_tools)
    _APPLIED = True


def _patch_toolsource_timeouts(tools_pkg, asta_tools) -> None:
    """Make the MCP connect and SSE read timeouts configurable."""
    original = asta_tools.make_asta_toolsource

    def make_asta_toolsource(api_key: str | None = None):
        api_key = api_key if api_key is not None else os.getenv("ASTA_TOOL_KEY")
        if not api_key:
            raise ValueError("api_key not given and ASTA_TOOL_KEY is not set")
        return asta_tools.create_server_streamable_http(
            "https://asta-tools.allen.ai/mcp/v1",
            headers={"x-api-key": api_key},
            timeout=float(os.getenv("ASTA_MCP_CONNECT_TIMEOUT", "20")),
            sse_read_timeout=float(os.getenv("ASTA_MCP_SSE_READ_TIMEOUT", str(60 * 5))),
        )

    make_asta_toolsource.__doc__ = original.__doc__
    asta_tools.make_asta_toolsource = make_asta_toolsource
    # astabench.tools re-exports by value at import time, so the package-level
    # name has to be rebound too or callers that imported it there keep the old
    # function. This is the failure mode that makes a patch look applied when it
    # is not.
    if hasattr(tools_pkg, "make_asta_toolsource"):
        tools_pkg.make_asta_toolsource = make_asta_toolsource


def _patch_retryable_errors(asta_tools) -> None:
    """Treat httpx and httpcore timeouts as retryable, as 429/529/504 already are."""
    import anyio
    import httpx

    try:
        import httpcore
    except ImportError:  # pragma: no cover - httpcore ships with httpx
        httpcore = None

    flatten = asta_tools._unravel_exception_group
    retryable_codes = {429, 529, 504}
    httpcore_timeouts = tuple(
        t
        for t in (
            getattr(httpcore, "ConnectTimeout", None),
            getattr(httpcore, "ReadTimeout", None),
            getattr(httpcore, "WriteTimeout", None),
            getattr(httpcore, "PoolTimeout", None),
        )
        if isinstance(t, type)
    )

    def _is_retryable_error(error: Exception) -> bool:
        for base_error in flatten(error):
            if isinstance(base_error, httpx.HTTPStatusError) and hasattr(
                base_error, "response"
            ):
                if base_error.response.status_code in retryable_codes:
                    return True
            elif isinstance(base_error, httpx.TimeoutException):
                return True
            elif httpcore_timeouts and isinstance(base_error, httpcore_timeouts):
                return True
            elif isinstance(base_error, anyio.BrokenResourceError):
                # A 429 sometimes manifests as BrokenResourceError, because the
                # MCP client's internal post_writer task dies from it.
                return True
        return False

    asta_tools._is_retryable_error = _is_retryable_error

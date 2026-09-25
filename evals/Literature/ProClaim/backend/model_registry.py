"""The outer-agent LLM a librarian run emits its verdict on.

GPL-3.0; see ../NOTICE.

ProClaim's own model registry serves the Qwen evidence subagent. The librarian
path emits its *final verdict* on the outer agent model instead (Sonnet 4.6 in
the paper), which upstream has no accessor for. Rather than add one to
upstream's registry, this keeps its own singleton: the two caches are
independent, and upstream's is left exactly as it is.

The AGENT_LLM_* variables this reads are exported by
``one_shot.build_subprocess_env``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_AGENT_LLM_CACHE: dict[str, object] = {}
_AGENT_LLM = "agent_llm"


def get_agent_llm():
    """Get or create the outer-agent LLM callable for librarian-run verdicts.

    Built from the AGENT_LLM_* environment variables through ProClaim's own
    ``make_llm``, so the callable behaves exactly like one of upstream's.
    Cached as a singleton for the life of the process.
    """
    if _AGENT_LLM in _AGENT_LLM_CACHE:
        logger.debug("Using cached agent LLM callable")
        return _AGENT_LLM_CACHE[_AGENT_LLM]

    model = os.environ.get("AGENT_LLM_MODEL") or os.environ.get("LLM_MODEL", "glm-5")
    base_url = os.environ.get("AGENT_LLM_BASE_URL") or None
    api_key = (
        os.environ.get("AGENT_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "EMPTY"
    )
    temperature = float(os.environ.get("AGENT_LLM_TEMPERATURE", "0.0"))
    timeout = int(os.environ.get("AGENT_LLM_TIMEOUT", "300"))
    stream = os.environ.get("AGENT_LLM_STREAM", "1") == "1"

    from proclaim.verification.llm_factory import make_llm

    logger.info("Creating agent-model LLM callable (model=%s)...", model)
    agent_llm = make_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        stream=stream,
    )
    _AGENT_LLM_CACHE[_AGENT_LLM] = agent_llm
    return agent_llm

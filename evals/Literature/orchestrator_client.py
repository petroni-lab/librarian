"""Run a literature agent on the orchestrator instead of importing it.

Every benchmark here has two ways to reach the agent. In-process is the default:
build ``LibrarianAgent`` / ``SynthesisAgent`` in the eval process and let their
LLM calls go out one at a time. This module is the other way -- ``POST
/run-agent/stream`` against a running ``orchestrator.py``, which builds the same
agents inside the server process.

It is worth it when the server is closer to the LLM endpoint than the eval box
is: the librarian's Stage-2 BM25 is CPU-bound and runs on the server's CPUs, and
a question costs one round trip instead of one per LLM call. Pointed at a
``orchestrator.py`` on localhost it buys nothing -- that is why ``--in-process``
is the default and ``--via-api`` is opt-in.

The trade is that the server runs the configuration it was started with. No
per-run knob -- sub-query count, the full-text toggle, a different model -- can
be honoured here, so any arm that varies one has to stay in-process. Nor can
synthesis and retrieval use different models: both are the server's ``LLM_MODEL``.

Concurrency is the caller's to manage. 16 was the measured sweet spot against
the deployment the paper ran on, where each agent run fanned out to 8 concurrent
judge calls over one shared vLLM; re-measure against your own.

Start the server this talks to from the repository root:

    uv run uvicorn orchestrator:app --port 8080

and point the benchmarks at it with ``--via-api``, or set ``LIBRARIAN_API_URL``.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Iterable

import requests

#: Orchestrator API base -- where ``orchestrator.py`` is listening.
#: ``/run-agent/stream`` is appended directly to it.
DEFAULT_BASE_URL = os.environ.get("LIBRARIAN_API_URL", "http://localhost:8080")

#: An agent run is minutes of retrieval plus synthesis; reaching the Ingress is
#: not. Connect fast, read slow.
DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_READ_TIMEOUT = 1800.0

_local = threading.local()


def _session() -> requests.Session:
    """One pooled session per calling thread; Session is not thread-safe."""
    if getattr(_local, "session", None) is None:
        _local.session = requests.Session()
    return _local.session


def result_from_sse(lines: Iterable[str]) -> dict[str, Any]:
    """Fold an SSE line stream from ``/run-agent/stream`` into its result payload.

    Kept separate from the HTTP call so the parsing is exercisable without a
    server. Frames are ``event: <type>`` followed by ``data: <json>``; only the
    terminal ``result`` and ``error`` events are decoded, because ``done``
    carries the bare string ``[DONE]`` and everything else is progress noise.

    :param lines: Iterable of decoded SSE lines.
    :returns: The run's result object -- ``query``, ``search_queries``,
        ``papers``, ``summary`` and ``action``. ``summary`` is always ``""`` for
        the ``librarian`` agent, which does not write prose.
    :raises RuntimeError: The stream ended on an ``error`` event, or ended
        without a ``result`` one.
    """
    event_type = ""
    for line in lines:
        if not line:
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue
        if not line.startswith("data:"):
            continue
        if event_type == "result":
            payload = json.loads(line[len("data:") :].strip())
            # A deployment that wraps its payload is still understood; ours does
            # not, and returns the result object directly.
            if isinstance(payload.get("results"), dict):
                return payload["results"]
            return payload
        if event_type == "error":
            payload = json.loads(line[len("data:") :].strip())
            raise RuntimeError(payload.get("error", "stream reported an error"))
    raise RuntimeError("stream ended with no result event")


def run_agent(
    query: str,
    *,
    agent: str = "literature_synthesis",
    base_url: str | None = None,
    source: str = "eval",
    session_prefix: str = "eval",
    user_label: str | None = None,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> dict[str, Any]:
    """Run one question on the orchestrator and return its result payload.

    :param query: The question to answer.
    :param agent: ``literature_synthesis`` for retrieval plus a written summary,
        or ``librarian`` for ranked evidence only (its ``summary`` is always
        empty). Pick by what the benchmark scores.
    :param base_url: Orchestrator API base; defaults to :data:`DEFAULT_BASE_URL`.
    :param source: Names the benchmark, so a sweep is separable from other
        traffic in a server that logs it. ``orchestrator.py`` ignores it.
    :param session_prefix: Prefix for the generated per-question ``session_id``.
    :param user_label: Run label for a server that traces per-question tokens,
        cost and latency. Defaults to *session_prefix*.
    :returns: The result object described in :func:`result_from_sse`.
    :raises RuntimeError: The run failed server-side or the stream was truncated.
    :raises requests.HTTPError: The request itself was rejected.
    """
    label = user_label or session_prefix
    body = {
        "query": query,
        "session_id": f"{session_prefix}-{uuid.uuid4().hex[:12]}",
        "source": source,
        "agent": agent,
        "user": {"id": label, "display_name": label},
    }
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/run-agent/stream"
    # Dropping the connection cancels the job server-side, so a read timeout
    # here does not leave a run holding a slot.
    with _session().post(
        url, json=body, stream=True, timeout=(DEFAULT_CONNECT_TIMEOUT, read_timeout)
    ) as response:
        response.raise_for_status()
        return result_from_sse(response.iter_lines(decode_unicode=True))


def librarian_evidence(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Retrieve ranked evidence for *query*, shaped like ``LibrarianAgent.run``.

    The retrieval-only benchmarks (LabBench, LitQA2, ProClaim) answer with their
    own model and only need the papers, so this returns the same flat list of
    per-paper records the in-process agent returns -- ``pmid``, ``title``,
    ``year``, ``evidence_snippets`` and the rest, shipped verbatim by the API.
    Callers can therefore swap between the two paths without reshaping anything.

    :param query: The search query.
    :param kwargs: Forwarded to :func:`run_agent` (``base_url``, ``source``, ...).
    :returns: Per-paper records, best first; empty when the literature is empty.
    """
    results = run_agent(query, agent="librarian", **kwargs)
    return results.get("papers") or []

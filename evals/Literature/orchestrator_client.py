"""Run a literature agent on the orchestrator instead of importing it.

Every benchmark here has two ways to reach the agent. In-process is the default:
build ``LibrarianAgent`` / ``SynthesisAgent`` in the eval process. This module is
the other way -- ``POST /run-agent/stream`` against a running
``orchestrator.py``, which builds the same agents inside the server process. The
librarian's CPU-bound Stage-2 BM25 then runs on the server, and a question costs
one round trip rather than one per LLM call.

The server runs the configuration it was started with, so no per-run knob --
sub-query count, the full-text toggle, a different model -- applies on this path,
and an arm that varies one has to stay in-process. Synthesis and retrieval also
share one model, the server's ``LLM_MODEL``.

Concurrency is the caller's to manage; see ``[concurrency]`` in
``literature_eval.toml``.

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

#: Connect fast, read slow: an agent run is minutes of retrieval and synthesis.
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

    Frames are ``event: <type>`` followed by ``data: <json>``. Only the terminal
    ``result`` and ``error`` events are decoded: ``done`` carries the bare string
    ``[DONE]``, and the rest is progress reporting.

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
            # Some deployments wrap the payload; ours returns it directly.
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
        or ``librarian`` for ranked evidence only, whose ``summary`` is always
        empty.
    :param base_url: Orchestrator API base; defaults to :data:`DEFAULT_BASE_URL`.
    :param source: Names the benchmark, for a server that logs its traffic.
        ``orchestrator.py`` ignores it.
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
    # Dropping the connection cancels the job server-side.
    with _session().post(
        url, json=body, stream=True, timeout=(DEFAULT_CONNECT_TIMEOUT, read_timeout)
    ) as response:
        response.raise_for_status()
        return result_from_sse(response.iter_lines(decode_unicode=True))


def librarian_evidence(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Retrieve ranked evidence for *query*, shaped like ``LibrarianAgent.run``.

    The retrieval-only benchmarks (LabBench, LitQA2, ProClaim) answer with their
    own model and need only the papers, so this returns the flat list of
    per-paper records the in-process agent returns -- ``pmid``, ``title``,
    ``year``, ``evidence_snippets`` and the rest, shipped verbatim by the API.

    :param query: The search query.
    :param kwargs: Forwarded to :func:`run_agent` (``base_url``, ``source``, ...).
    :returns: Per-paper records, best first; empty when the literature is empty.
    """
    results = run_agent(query, agent="librarian", **kwargs)
    return results.get("papers") or []

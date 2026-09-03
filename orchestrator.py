"""Minimal HTTP orchestrator for the librarian agent.

    GET  /health            liveness probe
    POST /run-agent         blocking JSON run
    POST /run-agent/stream  the same run as Server-Sent Events (live progress)

Start it with (LLM config comes from `.env`, same as `main.py`):

    uv run --extra api uvicorn orchestrator:app --port 8080

Then:

    curl -N localhost:8080/run-agent/stream \
        -H 'content-type: application/json' \
        -d '{"query": "does metformin extend lifespan in mammals?"}'
"""

import json
import queue
import threading
from typing import Any, Callable, Dict, Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from librarian import LibrarianAgent

# Same as main.py: pick up LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from a .env file.
load_dotenv()

app = FastAPI(title="Librarian Orchestrator")


class RunRequest(BaseModel):
    """Request body shared by both run endpoints."""

    query: str = Field(..., min_length=1, description="Natural-language question.")
    full_text_enrichment: bool = Field(
        default=True,
        description="Rank over full-text paragraphs; false falls back to abstracts only.",
    )


def _run(
    request: RunRequest, on_progress: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """Run one retrieval pass and shape the response payload.

    :param request: The validated request body.
    :param on_progress: Called with a short status string at each pipeline stage.
    :returns: The query, the sub-queries the planner produced, and the ranked
        evidence passages (``LibrarianAgent.run``'s output, verbatim).
    """
    # ponytail: a fresh agent per request. Construction just reads two prompt
    # files and builds an HTTP client; cache it in a module global if profiling
    # ever says otherwise.
    agent = LibrarianAgent(full_text_enrichment=request.full_text_enrichment)
    papers = agent.run(request.query, on_progress=on_progress)
    return {
        "query": request.query,
        "search_queries": agent.last_run_debug.get("search_queries", []),
        "papers": papers,
    }


def _sse(event: str, payload: Any) -> str:
    """Render one Server-Sent Event frame; every payload is JSON."""
    data = json.dumps(payload, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _stream(
    request: RunRequest, run: Callable[..., Dict[str, Any]] = _run
) -> Iterator[str]:
    """Run the agent on a worker thread and yield its progress as SSE frames.

    The agent is blocking and reports progress through a callback, so the thread
    pushes both the callback's messages and the final result onto a queue that
    this generator drains in order. Events: ``progress`` (repeated), then exactly
    one of ``result`` / ``error``, then ``done``.

    :param request: The validated request body.
    :param run: Injection point for the self-check below; defaults to ``_run``.
    """
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = run(
                request,
                on_progress=lambda message: events.put(
                    ("progress", {"message": message})
                ),
            )
            events.put(("result", result))
        except Exception as exc:
            events.put(("error", {"error": repr(exc)}))
        finally:
            events.put(("done", {}))

    # ponytail: no cancellation. A client that hangs up leaves the run going to
    # completion; thread a stop flag into LibrarianAgent.run if wasted LLM calls
    # start costing real money.
    threading.Thread(target=worker, daemon=True).start()

    while True:
        event, payload = events.get()
        yield _sse(event, payload)
        if event == "done":
            return


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/run-agent")
def run_agent(request: RunRequest) -> Dict[str, Any]:
    """Run the librarian and return the ranked evidence passages inline."""
    return _run(request)


@app.post("/run-agent/stream")
def run_agent_stream(request: RunRequest) -> StreamingResponse:
    """Run the librarian, streaming each pipeline stage as it happens."""
    return StreamingResponse(
        _stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _self_check() -> None:
    """Drive the SSE generator with a stub run — no LLM, no network."""

    def ok_run(request: RunRequest, on_progress=None) -> Dict[str, Any]:
        on_progress("Searching Europe PMC")
        return {"query": request.query, "search_queries": ["q"], "papers": []}

    def boom_run(request: RunRequest, on_progress=None) -> Dict[str, Any]:
        raise RuntimeError("nope")

    frames = list(_stream(RunRequest(query="test"), run=ok_run))
    assert [f.split("\n", 1)[0] for f in frames] == [
        "event: progress",
        "event: result",
        "event: done",
    ], frames
    assert json.loads(frames[0].split("data: ", 1)[1]) == {
        "message": "Searching Europe PMC"
    }
    assert json.loads(frames[1].split("data: ", 1)[1])["search_queries"] == ["q"]

    failed = list(_stream(RunRequest(query="test"), run=boom_run))
    assert [f.split("\n", 1)[0] for f in failed] == ["event: error", "event: done"]
    assert "nope" in failed[0]

    print("orchestrator self-check ok")


if __name__ == "__main__":
    _self_check()

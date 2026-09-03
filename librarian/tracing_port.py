"""Tracing port the librarian depends on — no tracing-backend imports.

Defines the tracing contract :class:`~librarian.agent.LibrarianAgent` calls
into, plus :class:`NullTracer`, the default adapter used when no tracer is
injected. To trace real runs, implement ``TracingPort`` against your own
backend (Phoenix, OpenTelemetry, ...) and pass it as ``LibrarianAgent(tracer=...)``
— nothing in this module knows that backend exists.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)

F = TypeVar("F", bound=Callable[..., Any])


class TracingPort(Protocol):
    """Contract the librarian depends on for span tracing. No tracing backend implied."""

    def start_span(
        self, name: str, *, attributes: Optional[Mapping[str, Any]] = None
    ) -> ContextManager[Any]:
        """Start a span named ``name``, yielding it (or ``None``) as a context manager."""
        ...

    def set_span_attributes(
        self, span: Any, attributes: Optional[Mapping[str, Any]]
    ) -> None:
        """Attach ``attributes`` to ``span``."""
        ...

    def mark_span_error(self, span: Any, exc: Exception) -> None:
        """Record ``exc`` on ``span`` and mark it as failed."""
        ...

    def bind_current_trace_context(self, fn: F) -> F:
        """Wrap ``fn`` so it keeps the calling span when run on another thread."""
        ...


class NullTracer:
    """Default adapter: every operation is a no-op. Ships with the librarian."""

    @contextmanager
    def start_span(
        self, name: str, *, attributes: Optional[Mapping[str, Any]] = None
    ) -> Iterator[None]:
        yield None

    def set_span_attributes(
        self, span: Any, attributes: Optional[Mapping[str, Any]]
    ) -> None:
        pass

    def mark_span_error(self, span: Any, exc: Exception) -> None:
        pass

    def bind_current_trace_context(self, fn: F) -> F:
        return fn

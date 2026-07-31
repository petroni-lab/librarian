"""Librarian — a minimal, self-contained literature retrieval agent.

``LibrarianAgent`` turns a natural-language biology question into ranked
evidence passages from Europe PMC. This is the open-source, infra-free version
of the agent: no tracing, no database, no caching, no Slack — just the flow.
"""

from librarian.agent import LibrarianAgent

__all__ = ["LibrarianAgent"]

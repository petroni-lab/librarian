"""Librarian — a minimal, self-contained literature retrieval agent.

``LibrarianAgent`` turns a natural-language biology question into ranked
evidence passages from Europe PMC, and ``SynthesisAgent`` writes a cited answer
over those passages. This is the open-source, infra-free version of the agent:
no tracing, no database, no caching, no Slack — just the flow.
"""

from librarian.agent import LibrarianAgent
from librarian.config import LibrarianRuntimeConfig, load_runtime_config
from librarian.synthesis import SynthesisAgent

__all__ = [
    "LibrarianAgent",
    "LibrarianRuntimeConfig",
    "SynthesisAgent",
    "load_runtime_config",
]

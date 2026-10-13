"""Librarian — a minimal, self-contained literature retrieval agent.

``LibrarianAgent`` turns a natural-language biology question into ranked
evidence passages from Europe PMC, and ``SynthesisAgent`` writes a cited answer
over those passages.
"""

from librarian.agent import LibrarianAgent
from librarian.config import LibrarianRuntimeConfig, load_runtime_config
from librarian.scope_port import ZeroShotScopeGuard
from librarian.synthesis import SynthesisAgent

__all__ = [
    "LibrarianAgent",
    "LibrarianRuntimeConfig",
    "SynthesisAgent",
    "ZeroShotScopeGuard",
    "load_runtime_config",
]

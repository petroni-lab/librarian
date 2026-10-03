#!/usr/bin/env python3
"""Run the repo's own ``LibrarianAgent`` with a CLI session as its LLM.

    run.sh search --provider claude "<research question>"

The agent does everything (queries, Europe PMC, BM25, judge, evidence); the only
thing swapped is the LLM client, which shells out to a fresh CLI session.
Prints the report the root agent answers from.

What this relies on from ``librarian`` (run.sh puts the repo root on
``PYTHONPATH``):

- ``LibrarianAgent`` and ``load_runtime_config``;
- the client contract ``chat_completion(messages, temperature, max_tokens) -> str``;
- the reply keys the agent parses, ``queries`` and ``relevant_ids``, which the
  Codex/AGY schema in ``_direct_session.py`` hard-codes;
- ``last_run_debug["search_queries"]`` and ``["paragraph_count"]``;
- ``citations.render_report`` and ``prompts/summarizer.md``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import librarian
from _direct_session import PROVIDERS, CliSessionError, run_direct_session
from librarian import LibrarianAgent, load_runtime_config
from librarian.citations import render_report

SUMMARIZER = Path(librarian.__file__).parent / "prompts" / "summarizer.md"


class CliClient:
    """The ``chat_completion`` shape ``LibrarianAgent`` needs, backed by a CLI session."""

    def __init__(self, provider: str):
        self.provider = provider

    # temperature/max_tokens are ignored: a one-shot CLI session exposes neither.
    def chat_completion(self, messages, temperature=0.0, max_tokens=0) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text(
                "\n\n".join(m["content"] for m in messages) + "\n", encoding="utf-8"
            )
            return run_direct_session(self.provider, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("query", help="the user's research question, verbatim")
    args = parser.parse_args()
    query = args.query.strip()
    if not query:
        parser.error("query must not be empty")

    agent = LibrarianAgent(
        runtime_config=load_runtime_config(), llm_client=CliClient(args.provider)
    )
    try:
        evidence = agent.run(
            query, on_progress=lambda m: print(f"[Librarian] {m}", file=sys.stderr)
        )
    except CliSessionError as exc:
        # Query planning has no fallback, so its CLI failure ends the run.
        raise SystemExit(f"[Librarian] {exc}") from None
    debug = agent.last_run_debug
    # paragraph_count is absent when the agent returns early on no paragraphs.
    queries, paragraphs = len(debug["search_queries"]), debug.get("paragraph_count", 0)
    print(f"[Librarian] queries={queries} paragraphs={paragraphs} relevant={len(evidence)}")
    if not paragraphs:
        return 0  # SKILL.md Step 1 says what to do on paragraphs=0.
    summary = (
        f"{queries} sub-queries → {paragraphs} paragraphs "
        f"→ {len(evidence)} papers cited by the judge"
    )
    print()
    print(render_report(query, evidence, summary))
    # Printed here so the root agent needs no extra Read turn to get them.
    print("\n---\n## Synthesis instructions (follow these to write the final answer)\n")
    print(SUMMARIZER.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import time
from pathlib import Path

import librarian
from _direct_session import PROVIDERS, CliSessionError, run_direct_session
from librarian import LibrarianAgent, load_runtime_config
from librarian.citations import render_report

SUMMARIZER = Path(librarian.__file__).parent / "prompts" / "summarizer.md"


# Whole-run wall clock, under the 600 s host timeout SKILL.md asks for. The
# judge retries a failed batch in halves, one after the other, so a hung
# provider would otherwise cost 7 x 180 s for a single batch.
RUN_BUDGET_SECONDS = 540
SESSION_TIMEOUT_SECONDS = 180


class CliClient:
    """The ``chat_completion`` shape ``LibrarianAgent`` needs, backed by a CLI session.

    Every failed call is kept in ``errors``: the agent's judge swallows them
    (it logs only when verbose), so ``main`` reports them itself.
    """

    def __init__(self, provider: str):
        self.provider = provider
        self.deadline = time.monotonic() + RUN_BUDGET_SECONDS
        self.errors: list[str] = []

    # temperature/max_tokens are ignored: a one-shot CLI session exposes neither.
    def chat_completion(self, messages, temperature=0.0, max_tokens=0) -> str:
        try:
            remaining = self.deadline - time.monotonic()
            if remaining < 10:
                raise CliSessionError(
                    f"run budget of {RUN_BUDGET_SECONDS} s used up; model call skipped."
                )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "prompt.md"
                path.write_text(
                    "\n\n".join(m["content"] for m in messages) + "\n", encoding="utf-8"
                )
                return run_direct_session(
                    self.provider, path, min(SESSION_TIMEOUT_SECONDS, remaining)
                )
        except CliSessionError as exc:
            self.errors.append(str(exc))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("query", help="the user's research question, verbatim")
    args = parser.parse_args()
    query = args.query.strip()
    if not query:
        parser.error("query must not be empty")

    client = CliClient(args.provider)
    agent = LibrarianAgent(runtime_config=load_runtime_config(), llm_client=client)
    try:
        evidence = agent.run(
            query, on_progress=lambda m: print(f"[Librarian] {m}", file=sys.stderr)
        )
    except CliSessionError as exc:
        # Query planning has no fallback, so its CLI failure ends the run.
        raise SystemExit(f"[Librarian] {exc}") from None
    if client.errors:
        failed = f"{len(client.errors)} judge call(s) failed: " + " | ".join(
            dict.fromkeys(client.errors)
        )
        if not evidence:
            # Every judge call failed: an empty report would read as "no papers".
            raise SystemExit(f"[Librarian] {failed}")
        print(f"[Librarian] WARNING: {failed} Evidence may be incomplete.", file=sys.stderr)
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

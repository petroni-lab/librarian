#!/usr/bin/env python3
"""Run the repo's own ``LibrarianAgent`` with a CLI session as its LLM.

    search.py --provider claude "<research question>"

The agent does everything (queries, Europe PMC, BM25, judge, evidence); the only
thing swapped is the LLM client, which shells out to a fresh CLI session.
Prints the report the root agent answers from.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _direct_session import PROVIDERS, run_direct_session  # noqa: E402
from librarian import LibrarianAgent, load_runtime_config  # noqa: E402
from librarian.citations import render_report  # noqa: E402

SUMMARIZER = Path(__file__).resolve().parents[3] / "librarian" / "prompts" / "summarizer.md"


class CliClient:
    """The ``chat_completion`` shape ``LibrarianAgent`` needs, backed by a CLI session."""

    def __init__(self, provider: str):
        self.provider = provider

    # ponytail: temperature/max_tokens ignored, a one-shot CLI session exposes neither.
    def chat_completion(self, messages, temperature=0.0, max_tokens=0) -> str:
        system, prompt = messages[0]["content"], messages[-1]["content"]
        key = "relevant_ids" if "relevant_ids" in system else "queries"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text(
                f"{system}\n\n{prompt}\n\nReturn only one JSON object with the key "
                f"`{key}` (an array of strings). No tools, prose or Markdown fences.\n",
                encoding="utf-8",
            )
            return run_direct_session(self.provider, path, key)


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
    evidence = agent.run(query, on_progress=lambda m: print(f"[Librarian] {m}", file=sys.stderr))
    debug = agent.last_run_debug
    paragraphs = debug.get("paragraph_count", 0)
    print(f"[Librarian] queries={debug.get('query_count', 0)} paragraphs={paragraphs} relevant={len(evidence)}")
    if not paragraphs:
        print("No paragraphs retrieved — nothing to judge. Report the empty result.")
        return 0
    summary = (
        f"{debug['query_count']} sub-queries → {paragraphs} paragraphs "
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

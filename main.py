"""Run the librarian agent from the command line.

Usage:
    python main.py "does metformin extend lifespan in mammals?"

Configure the LLM backend in a `.env` file (copy `.env.example`) or via env vars:
    LLM_BASE_URL   e.g. http://localhost:8000/v1   (default)
    LLM_MODEL      the model name to request
    LLM_API_KEY    bearer token (defaults to "EMPTY" for keyless vLLM)

    LLM_REASONING_EFFORT  "low" (default here) / "high" / "max"
"""

import json
import os
import sys

from dotenv import load_dotenv

from librarian import LibrarianAgent
from librarian.progress import Spinner

# Load LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from a .env file if present.
load_dotenv()

# Default the reasoning effort to "low" unless the environment already picked one.
#
# Reasoning models charge their hidden trace against the same max_tokens budget as
# the answer. With no effort set, GLM-5.3-class models reason without a ceiling and
# can consume the whole of _FILTER_MAX_TOKENS before emitting any JSON, so the
# relevance judge sees an empty or truncated reply and drops the batch. Measured on
# a vLLM-served glm-5.3-flash, same query repeated: 413s returning 0-3 passages with
# it unset, 60-82s returning 54-58 passages at "low". Retrieval is a judging task,
# not a writing one, so "low" is the right default; set LLM_REASONING_EFFORT to
# override.
os.environ.setdefault("LLM_REASONING_EFFORT", "low")

DEFAULT_QUERY = "What is the role of telomere shortening in cellular senescence?"


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    # On a terminal, show a live spinner and keep the agent's own logs quiet so
    # they don't fight the spinner. When piped/redirected, fall back to plain logs.
    interactive = sys.stderr.isatty()
    agent = LibrarianAgent(verbose=not interactive)
    if interactive:
        with Spinner() as spinner:
            passages = agent.run(query, on_progress=spinner.update)
    else:
        passages = agent.run(query)

    print(f"\n=== {len(passages)} evidence passages for: {query!r} ===\n")
    for i, passage in enumerate(passages, 1):
        print(f"[{i}] {passage['title']} ({passage['year']})  PMID: {passage['pmid']}")
        for snippet in passage["evidence_snippets"]:
            print(f"    - {snippet}")
        print()

    # Full structured output (what a downstream synthesis layer would consume).
    print("=== raw passages (JSON) ===")
    print(json.dumps(passages, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

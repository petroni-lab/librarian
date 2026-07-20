"""Run the librarian agent from the command line.

Usage:
    python main.py "does metformin extend lifespan in mammals?"

Configure the LLM backend with env vars (see librarian/llm_client.py):
    LLM_BASE_URL   e.g. http://localhost:8000/v1   (default)
    LLM_MODEL      the model name to request
    LLM_API_KEY    bearer token (defaults to "EMPTY" for keyless vLLM)
"""

import json
import sys

from librarian import LibrarianAgent

DEFAULT_QUERY = "What is the role of telomere shortening in cellular senescence?"


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    agent = LibrarianAgent(verbose=True)
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

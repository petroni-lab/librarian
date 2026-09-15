"""Run the librarian agent from the command line.

Usage:
    python main.py "does metformin extend lifespan in mammals?"
    python main.py "..." --retrieval-only    # ranked evidence, no synthesized answer

By default the retrieved evidence is synthesized into a cited answer. Pass
``--retrieval-only`` to stop after retrieval and print the ranked passages plus
the raw JSON instead.

Configure the LLM backend in a `.env` file (copy `.env.example`) or via env vars:
    LLM_BASE_URL   e.g. http://localhost:8000/v1   (default)
    LLM_MODEL      the model name to request
    LLM_API_KEY    bearer token (defaults to "EMPTY" for keyless vLLM)
"""

import argparse
import json
import sys
from typing import Any

from dotenv import load_dotenv

from librarian import LibrarianAgent, SynthesisAgent, load_runtime_config
from librarian.citations import citation_keys
from librarian.progress import Spinner

# Load LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from a .env file if present.
load_dotenv()

DEFAULT_QUERY = "What is the role of telomere shortening in cellular senescence?"


def _parse_args() -> argparse.Namespace:
    """Read the query and the one mode flag off the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Answer a research question from Europe PMC evidence. "
            "Synthesizes a cited answer unless --retrieval-only is passed."
        )
    )
    # nargs="*" so an unquoted multi-word question still works, as it always has.
    parser.add_argument(
        "query",
        nargs="*",
        help=f"the research question (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="stop after retrieval: print ranked passages and raw JSON, no answer",
    )
    return parser.parse_args()


def _print_passages(query: str, passages: list[dict[str, Any]]) -> None:
    """The retrieval-only view: one block per passage, then the full JSON."""
    print(f"\n=== {len(passages)} evidence passages for: {query!r} ===\n")
    for i, passage in enumerate(passages, 1):
        print(f"[{i}] {passage['title']} ({passage['year']})  PMID: {passage['pmid']}")
        for snippet in passage["evidence_snippets"]:
            print(f"    - {snippet}")
        print()

    # Full structured output (what a downstream synthesis layer would consume).
    print("=== raw passages (JSON) ===")
    print(json.dumps(passages, indent=2, ensure_ascii=False))


def _print_references(passages: list[dict[str, Any]]) -> None:
    """Resolve the answer's inline citation keys to titles and links.

    The keys come from ``citation_keys`` — the same function that built the
    ``Cite as:`` lines the model copied — so this list and the answer's inline
    citations always agree. The prompt forbids the model from writing its own
    bibliography, which is what makes printing one here deterministic rather
    than a second, possibly conflicting, set of references.
    """
    if not passages:
        return

    keys = citation_keys(passages)
    # Indent the detail lines to the width of the widest key, so each entry
    # reads as one block rather than a ragged left edge.
    indent = max(len(key) for key in keys) + 5

    print("\nReferences")
    for key, passage in zip(keys, passages):
        identifiers = []
        if passage["pmid"]:
            identifiers.append(f"PMID {passage['pmid']}")
        if passage["doi"]:
            identifiers.append(f"doi:{passage['doi']}")
        venue = " ".join(part for part in (passage["journal"], passage["year"]) if part)
        detail = " · ".join(part for part in (venue, " · ".join(identifiers)) if part)

        print(f"{'  [' + key + ']':<{indent}}{passage['title']}")
        if detail:
            print(f"{'':<{indent}}{detail}")
        if passage["url"]:
            print(f"{'':<{indent}}{passage['url']}")


def main() -> None:
    args = _parse_args()
    query = " ".join(args.query).strip() or DEFAULT_QUERY

    # Keep the agents' own logs quiet on a terminal so they don't fight the
    # spinner; piped or redirected, those logs are all the progress there is.
    interactive = sys.stderr.isatty()
    # Tuning knobs come from librarian/config.toml; edit that file to change
    # them, or dataclasses.replace() the loaded config for a one-off run.
    librarian = LibrarianAgent(
        runtime_config=load_runtime_config(), verbose=not interactive
    )

    # Spinner disables itself off a TTY and update() is then inert, so both
    # modes drive it unconditionally. Printing happens after it is torn down,
    # or the spinner's line would interleave with the output.
    summary = ""
    with Spinner() as spinner:
        passages = librarian.run(query, on_progress=spinner.update)
        if not args.retrieval_only:
            spinner.update("Synthesizing answer")
            summary = SynthesisAgent(verbose=not interactive).run(query, passages)

    if args.retrieval_only:
        _print_passages(query, passages)
        return

    print()
    print(summary)
    _print_references(passages)


if __name__ == "__main__":
    main()

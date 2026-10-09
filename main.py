"""Run the librarian agent from the command line.

Usage:
    python main.py "does metformin extend lifespan in mammals?"
    python main.py "..." --retrieval-only    # ranked evidence, no synthesized answer
    python main.py "..." --supplementary     # also search supplementary PDF/Word files
    python main.py "..." --save-json out.json  # also save this run's evidence + answer

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
from dataclasses import replace
from pathlib import Path
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
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help=(
            "also retrieve from papers' supplementary PDF/Word files "
            "(overrides supplementary_enrichment in config.toml for this run)"
        ),
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        help=(
            "also write this run's evidence (with the section each snippet came "
            "from), counts, and answer to a JSON file"
        ),
    )
    return parser.parse_args()


def _print_passages(
    query: str, passages: list[dict[str, Any]], debug: dict[str, Any]
) -> None:
    """The retrieval-only view: one block per passage, then the full JSON.

    Each snippet is prefixed with the section it was cited from, so evidence
    taken from a supplementary file reads ``[Supplementary: ...]``.
    """
    print(f"\n=== {len(passages)} evidence passages for: {query!r} ===\n")
    for i, passage in enumerate(passages, 1):
        print(
            f"[{i}] {passage['title']} ({passage['year']})  "
            f"PMID: {passage['pmid'] or '-'}  PMCID: {passage.get('pmcid') or '-'}  "
            f"DOI: {passage['doi'] or '-'}"
        )
        sections = passage.get("evidence_sections") or []
        for j, snippet in enumerate(passage["evidence_snippets"]):
            section = sections[j] if j < len(sections) else ""
            prefix = f"[{section}] " if section else ""
            print(f"    - {prefix}{snippet}")
        print()

    print(
        "Supplementary paragraphs in the Stage-2 pool: "
        f"{debug.get('supplementary_paragraph_count', 0)} "
        f"of {debug.get('paragraph_count', 0)}; "
        "evidence spans cited from supplementary files: "
        f"{debug.get('supplementary_evidence_count', 0)}\n"
    )

    # Full structured output (what a downstream synthesis layer would consume).
    print("=== raw passages (JSON) ===")
    print(json.dumps(passages, indent=2, ensure_ascii=False))


def _print_retrieved(passages: list[dict[str, Any]]) -> None:
    """List every retrieved paper under the key the answer cites it by.

    The keys come from ``citation_keys`` — the same function that built the
    ``Cite as:`` lines the model copied — so any inline citation in the answer
    resolves here to the paper it names. This is every paper retrieved, not
    only the cited ones: the answer may lean on three of twenty and all twenty
    are listed, which is why the heading is not "References". The prompt
    forbids the model from writing its own bibliography, so this stays the
    only list printed rather than one of two that could disagree.
    """
    if not passages:
        return

    keys = citation_keys(passages)
    # Indent the detail lines to the width of the widest key, so each entry
    # reads as one block rather than a ragged left edge.
    indent = max(len(key) for key in keys) + 5

    print("\nRetrieved papers")
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


def _save_json(
    path: Path,
    query: str,
    passages: list[dict[str, Any]],
    debug: dict[str, Any],
    summary: str,
) -> None:
    """Write one run's evidence, counts and answer, so they can be checked later.

    ``supplementary_evidence`` lists every snippet cited from a supplementary
    file (empty when none was used); ``passages`` is the full evidence the
    answer was synthesized from, each snippet paired with its section.
    """
    supplementary = [
        {
            "pmid": passage["pmid"],
            "pmcid": passage.get("pmcid", ""),
            "doi": passage["doi"],
            "title": passage["title"],
            "section": section,
            "snippet": snippet,
        }
        for passage in passages
        for section, snippet in zip(
            passage.get("evidence_sections") or [], passage["evidence_snippets"]
        )
        if section.startswith("Supplementary")
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query": query,
                "debug": debug,
                "supplementary_evidence": supplementary,
                "passages": passages,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"Saved run to {path}: {len(passages)} passages, "
        f"{len(supplementary)} snippet(s) from supplementary files",
        file=sys.stderr,
    )


def main() -> None:
    args = _parse_args()
    query = " ".join(args.query).strip() or DEFAULT_QUERY

    # Keep the agents' own logs quiet on a terminal so they don't fight the
    # spinner; piped or redirected, those logs are all the progress there is.
    interactive = sys.stderr.isatty()
    # Tuning knobs come from librarian/config.toml; edit that file to change
    # them, or dataclasses.replace() the loaded config for a one-off run.
    config = load_runtime_config()
    if args.supplementary:
        config = replace(config, supplementary_enrichment=True)
    librarian = LibrarianAgent(runtime_config=config, verbose=not interactive)

    # Spinner disables itself off a TTY and update() is then inert, so both
    # modes drive it unconditionally. Printing happens after it is torn down,
    # or the spinner's line would interleave with the output.
    summary = ""
    with Spinner() as spinner:
        passages = librarian.run(query, on_progress=spinner.update)
        if not args.retrieval_only:
            spinner.update("Synthesizing answer")
            summary = SynthesisAgent(verbose=not interactive).run(query, passages)

    if args.save_json:
        _save_json(
            Path(args.save_json), query, passages, librarian.last_run_debug, summary
        )

    if args.retrieval_only:
        _print_passages(query, passages, librarian.last_run_debug)
        return

    print()
    print(summary)
    _print_retrieved(passages)


if __name__ == "__main__":
    main()

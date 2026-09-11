#!/usr/bin/env python3
"""Step 4 — cited sentence ids → ranked papers → the report.

The tail of the pipeline, calling the agent's own code:

  ``LibrarianAgent._papers_from_cited_sentences``  cited ids → papers, ranked by
                                                   first citation, evidence spans
                                                   grouped in reading order
  ``LibrarianAgent._passage_from_paper``          the returned evidence record

The sentence registry is not carried over from step 3: ``build_judge_items`` is a
pure function of the paragraph pool, so rebuilding it from ``02_paragraphs.json``
yields the identical ids the judge was shown.

    python3 step4_finalize.py --run DIR

Writes ``04_evidence.json`` and ``04_report.md``, and prints the report — that
printed report is what the root agent answers from.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _runs
from librarian.llm_client import parse_json_response


def collect_ranked_ids(run_dir: Path, batch_count: int) -> Tuple[List[str], List[int]]:
    """Read every batch's ``relevant_ids``, concatenated in batch order.

    Batch order is what ``_judge_items_in_batches`` preserves, and what makes the
    merged most→least-relevant ranking deterministic. Batches whose session output
    is missing or malformed are reported rather than silently dropped — the caller
    re-runs step 3 for them.

    :param run_dir: The run directory.
    :type run_dir: Path
    :param batch_count: How many batches step 3 rendered.
    :type batch_count: int
    :return: ``(ranked_sentence_ids, pending_batch_indices)``.
    :rtype: tuple[list[str], list[int]]
    """
    ranked: List[str] = []
    pending: List[int] = []
    for index in range(batch_count):
        output = _runs.batch_paths(run_dir, index)["output"]
        payload = None
        if output.exists():
            payload = parse_json_response(output.read_text(encoding="utf-8"))
        ids = payload.get("relevant_ids") if isinstance(payload, dict) else None
        if isinstance(ids, list):
            ranked.extend(str(identifier) for identifier in ids)
        else:
            pending.append(index)
    return ranked, pending


def _short_authors(authors: str, keep: int = 3) -> str:
    """First ``keep`` author names, then ``et al.`` — the report stays scannable."""
    names = [name.strip() for name in str(authors).split(",") if name.strip()]
    if len(names) <= keep:
        return ", ".join(names)
    return ", ".join(names[:keep]) + ", et al."


def _first_author_surname(authors: str) -> str:
    """Surname of the first author, initials dropped.

    Europe PMC writes each author as ``Surname II``, so the trailing all-caps
    initials block is what gets cut; multi-word surnames ("van der Meer") stay.
    """
    first_author = str(authors).split(",")[0].strip()
    parts = first_author.split()
    initials_trail = len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3
    if initials_trail:
        parts = parts[:-1]
    return " ".join(parts)


def citation_keys(evidence: List[Dict[str, Any]]) -> List[str]:
    """The author-year citation key for each paper, in report order.

    The summarizer cites papers by these keys rather than by list position, so
    they are computed here once: deriving a surname, spotting that two papers
    share an author and year, and picking the a/b suffix are all things Python
    can do exactly and a model cannot.

    :param evidence: The evidence records, in the order the report prints them.
    :type evidence: list[dict[str, Any]]
    :return: One key per record, e.g. ``["Chen 2023a", "Chen 2023b", "Kuo 2012"]``.
    :rtype: list[str]
    """
    bases: List[str] = []
    for record in evidence:
        surname = _first_author_surname(record["authors"])
        year = str(record["year"]).strip()
        identifier = record["pmid"] or record["doi"]
        # Author-year when we have it; an identifier is the fallback so a key is
        # never empty and never two papers' key at once.
        bases.append(
            " ".join(part for part in (surname, year) if part)
            or (f"PMID {identifier}" if record["pmid"] else f"doi:{identifier}")
            or "unattributed source"
        )

    shared = Counter(bases)
    used: Counter = Counter()
    keys: List[str] = []
    for base in bases:
        if shared[base] == 1:
            keys.append(base)
            continue
        # ponytail: 26 same-author-same-year papers in one run would run past 'z';
        # switch to a numeric suffix if that ever shows up.
        keys.append(f"{base}{chr(ord('a') + used[base])}")
        used[base] += 1
    return keys


def render_report(query: str, evidence: List[Dict[str, Any]], summary: str) -> str:
    """One compact block per paper: citation, link, and the judge-cited spans.

    The heading is the ready-made markdown citation for the paper, so the
    summarizer copies it inline instead of building one from the metadata.

    The spans are the whole point of the run, so they are printed in full and
    uncapped — the judge already decided which sentences answer the question.
    """
    lines = [f"# Librarian results — {query}", "", summary, ""]
    if not evidence:
        lines.append("No papers survived the relevance judge.")
        return "\n".join(lines) + "\n"

    for key, record in zip(citation_keys(evidence), evidence):
        identifiers = []
        if record["pmid"]:
            identifiers.append(f"PMID {record['pmid']}")
        if record["doi"]:
            identifiers.append(f"doi:{record['doi']}")
        venue = " ".join(part for part in (record["journal"], record["year"]) if part)
        source = "full text available" if record["has_fulltext"] else "abstract only"
        citation = f"[{key}]({record['url']})" if record["url"] else f"[{key}]"
        lines.append(f"## {record['title']}")
        lines.append(f"- Cite as: {citation}")
        lines.append(f"- {_short_authors(record['authors'])}")
        lines.append(
            "- "
            + " · ".join(
                part for part in (venue, " · ".join(identifiers), source) if part
            )
        )
        for span in record["evidence_snippets"]:
            lines.append(f"- Evidence: {span}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge the judge verdicts and compose the result (step 4 of 4)."
    )
    parser.add_argument("--run", required=True, help="run directory from step 1")
    args = parser.parse_args()

    run_dir = _runs.resolve_run(args.run)
    manifest = _runs.read_manifest(run_dir)
    query = manifest["query"]
    config = _runs.config_from_manifest(manifest)
    stages = manifest.get("stages", {})
    batch_count = int(stages.get("step3_judge_prompts", {}).get("batch_count", 0))
    if not batch_count:
        raise SystemExit(
            "No judge batches recorded — run step3_judge_prompts.py first."
        )

    paragraphs: List[Dict[str, Any]] = _runs.read_json(run_dir / _runs.PARAGRAPHS_NAME)
    # Same pool, same order → the same sentence ids the judge was shown.
    _, registry, paper_by_id = _runs.build_judge_items(paragraphs)

    ranked_ids, pending = collect_ranked_ids(run_dir, batch_count)

    agent = _runs.build_agent(config)
    relevant_papers = agent._papers_from_cited_sentences(
        ranked_ids, registry, paper_by_id
    )
    evidence = [agent._passage_from_paper(paper) for paper in relevant_papers]

    subqueries = stages.get("step2_retrieve", {}).get("search_queries", [])
    summary = (
        f"{len(subqueries)} sub-queries → {len(paragraphs)} paragraphs from "
        f"{len(paper_by_id)} papers → {len(evidence)} papers cited by the judge"
    )
    report = render_report(query, evidence, summary)

    _runs.write_json(run_dir / _runs.EVIDENCE_NAME, evidence)
    (run_dir / _runs.REPORT_NAME).write_text(report, encoding="utf-8")
    _runs.record_stage(
        run_dir,
        "step4_finalize",
        {
            "cited_sentence_count": len(ranked_ids),
            "relevant_count": len(evidence),
            "final_pmids": [record["pmid"] for record in evidence],
            "pending_batches": pending,
        },
    )

    # Judge batches run inside step 3, so the only recovery is re-running that step;
    # the report below is still valid, just missing whatever those batches would have cited.
    if pending:
        print(f"PENDING BATCHES {pending} — no usable relevant_ids.")
        print("Re-run step 3, then re-run this script.")
        print()

    print(
        f"[Librarian] paragraphs={len(paragraphs)} papers={len(paper_by_id)} "
        f"cited_sentences={len(ranked_ids)} relevant={len(evidence)}"
    )
    print(f"EVIDENCE      {run_dir / _runs.EVIDENCE_NAME}")
    print(f"REPORT        {run_dir / _runs.REPORT_NAME}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

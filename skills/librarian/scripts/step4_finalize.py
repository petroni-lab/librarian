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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _runs
from librarian.citations import render_report
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

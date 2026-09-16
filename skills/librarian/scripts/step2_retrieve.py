#!/usr/bin/env python3
"""Step 2 — validate the sub-queries, search Europe PMC, BM25-rank the paragraphs.

The whole non-LLM half of the pipeline, calling the agent's own Stage-2 code:

  ``LibrarianAgent._validate_queries``       drop broken Europe PMC syntax
  ``LibrarianAgent._paragraphs_for_subquery`` search, decompose papers into
                                              paragraphs, BM25-rank against that
                                              sub-query, keep its top k
  ``agent._merge_selected_paragraphs``        merge duplicate/overlapping
                                              selections across sub-queries

    python3 step2_retrieve.py --run DIR

Reads ``01_queries.json`` (the step-1 session's output), writes
``02_paragraphs.json``, and prints the run's provenance counters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import _runs
from librarian.agent import _available_cpus, _merge_selected_paragraphs
from librarian.llm_client import parse_json_response


def load_queries(run_dir: Path, query: str, max_queries: int) -> List[str]:
    """Read the step-1 session's queries, dedup (preserving order) and cap.

    Mirrors the tail of ``_generate_queries``, including its fallback: a response
    that yields no queries falls back to the raw question rather than searching
    nothing. A bare JSON array is tolerated — the brief asks for an object, but a
    parseable list is still the answer we asked for.
    """
    path = run_dir / _runs.QUERIES_NAME
    queries: List[Any] = []
    if path.exists():
        parsed = parse_json_response(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            candidate = parsed.get("queries", [])
            queries = candidate if isinstance(candidate, list) else []
        elif isinstance(parsed, list):
            queries = parsed

    if not queries:
        queries = [query]

    seen, unique = set(), []
    for candidate in queries:
        candidate = str(candidate).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique[:max_queries] or [query]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search Europe PMC and BM25-rank the paragraphs (step 2 of 4)."
    )
    parser.add_argument("--run", required=True, help="run directory from step 1")
    args = parser.parse_args()

    run_dir = _runs.resolve_run(args.run)
    manifest = _runs.read_manifest(run_dir)
    query = manifest["query"]
    config = _runs.config_from_manifest(manifest)

    agent = _runs.build_agent(config)
    agent.full_text_enrichment = True
    agent.verbose = False

    queries = agent._validate_queries(
        load_queries(run_dir, query, config.num_subqueries)
    )

    # One thread per sub-query, capped at the CPUs actually available — the same
    # fan-out ``agent._run`` does. No shared state: each thread searches, chunks and
    # ranks on its own, and the pools are merged afterwards.
    stage2_workers = min(len(queries), _available_cpus())
    with ThreadPoolExecutor(max_workers=stage2_workers) as pool:
        per_subquery = pool.map(agent._paragraphs_for_subquery, queries)
    paragraphs: List[Dict[str, Any]] = _merge_selected_paragraphs(
        [paragraph for subquery_paras in per_subquery for paragraph in subquery_paras]
    )

    paper_ids = {
        str(paragraph["paper"].get("pmid") or paragraph["paper"].get("epmcId") or "")
        for paragraph in paragraphs
    }
    _runs.write_json(run_dir / _runs.PARAGRAPHS_NAME, paragraphs)
    _runs.record_stage(
        run_dir,
        "step2_retrieve",
        {
            "search_queries": queries,
            "query_count": len(queries),
            "paragraph_count": len(paragraphs),
            "paper_count": len(paper_ids),
            "full_text_enrichment": agent.full_text_enrichment,
        },
    )

    print(
        f"[Librarian] queries={len(queries)} paragraphs={len(paragraphs)} "
        f"papers={len(paper_ids)}"
    )
    for subquery in queries:
        print(f"  query: {subquery}")
    print(f"PARAGRAPHS    {run_dir / _runs.PARAGRAPHS_NAME}")
    if not paragraphs:
        print()
        print(
            "No paragraphs retrieved — nothing to judge. Report the empty result "
            "and the sub-queries above; do not run steps 3 and 4."
        )
        return 0
    print()
    print("Then run:")
    print(
        f"  {Path(__file__).with_name('run.sh')} step3_judge_prompts "
        f"--run {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

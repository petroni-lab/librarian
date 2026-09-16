#!/usr/bin/env python3
"""Step 1 — render and run the Stage-1 Europe PMC query-generation prompt.

``agent.py::_generate_queries`` minus the LLM call: the same substitutions on the
same ``prompts/stage_1_europe_pmc_query_generation.md`` template, written out as a
direct CLI input. A fresh Claude Code or Codex session receives that file through
stdin and its JSON response is written back into the run.

    python3 step1_query_prompt.py --provider claude "<research question>"

Prints the run directory and query output path.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import _runs
from _direct_session import run_direct_session
from librarian.agent import _QUERY_PROMPT_PATH


def render_query_prompt(query: str, budget_guidance: str, max_queries: int) -> str:
    """Fill the Stage-1 template exactly as ``_generate_queries`` does.

    ``{max_queries}`` goes into the budget guidance first, from ``num_subqueries``,
    so the cap the planner is told about and the cap step 2 enforces are one knob.
    """
    today = datetime.date.today()
    budget = budget_guidance.replace("{max_queries}", str(max_queries))
    template = _QUERY_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{today_date}", today.isoformat())
        .replace("{today_year}", str(today.year))
        .replace("{query_budget_guidance}", budget)
        .replace("{conversation}", f"User: {query}")
        .replace("{additional_context}", "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Europe PMC queries in a fresh CLI session (step 1 of 4)."
    )
    parser.add_argument("--provider", choices=("claude", "codex"), required=True)
    parser.add_argument("query", help="the user's research question, verbatim")
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        parser.error("query must not be empty")

    config = _runs.load_runtime_config()
    run_dir = _runs.create_run(query, config)

    body = render_query_prompt(
        query,
        config.query_budget_guidance,
        config.num_subqueries,
    )
    output_path = run_dir / _runs.QUERIES_NAME
    prompt_path = _runs.write_prompt(
        run_dir / _runs.QUERY_PROMPT_NAME,
        title="Stage 1 — Europe PMC query generation",
        system_prompt=_runs.QUERY_SYSTEM_PROMPT,
        body=body,
        output_key="queries",
    )
    run_direct_session(
        args.provider,
        prompt_path,
        output_path,
        "queries",
    )

    _runs.record_stage(
        run_dir,
        "step1_query_prompt",
        {
            "prompt_path": str(prompt_path),
            "prompt_chars": len(body),
            "output_path": str(output_path),
            "provider": args.provider,
        },
    )

    print(f"RUN_DIR       {run_dir}")
    print(f"PROMPT        {prompt_path}")
    print(f"OUTPUT        {output_path}")
    print(f"MAX_QUERIES   {config.num_subqueries}")
    # The planner's queries, shown before step 2 spends a Europe PMC round-trip on
    # them. Step 2 prints them again after validation, which may drop or cap some.
    print("QUERIES")
    for position, generated_query in enumerate(
        _runs.read_json(output_path)["queries"], start=1
    ):
        print(f"  {position}. {generated_query}")
    print()
    print("Then run:")
    print(
        f"  {Path(__file__).with_name('run.sh')} step2_retrieve "
        f"--run {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

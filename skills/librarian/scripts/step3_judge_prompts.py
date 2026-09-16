#!/usr/bin/env python3
"""Step 3 — render and run the Stage-3 relevance judge in fresh CLI sessions.

The batch-building half of ``agent.py::_relevance_filter`` plus
``_evaluate_batch``'s substitution: one judge item per paragraph, batched
``paragraphs_per_judge_batch`` paragraphs per call — one LLM call there, one
fresh Claude Code or Codex CLI session here. Prod sizes that batch to hold the
whole pool (48), so a normal run makes one globally ranked judge call.

    python3 step3_judge_prompts.py --run DIR --provider claude

Reads ``02_paragraphs.json``, writes ``03_judge/batch_NN.md``, and writes the
corresponding ``relevant_ids`` JSON output for every batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import _runs
from _direct_session import run_direct_session
from librarian.agent import _FILTER_PROMPT_PATH


def _run_judge_batch(
    provider: str,
    prompt_path: Path,
    output_path: Path,
) -> None:
    """Run one rendered judge prompt in an isolated direct CLI session."""
    run_direct_session(provider, prompt_path, output_path, "relevant_ids")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge paragraph relevance in fresh CLI sessions (step 3 of 4)."
    )
    parser.add_argument("--run", required=True, help="run directory from step 1")
    parser.add_argument("--provider", choices=("claude", "codex"), required=True)
    args = parser.parse_args()

    run_dir = _runs.resolve_run(args.run)
    manifest = _runs.read_manifest(run_dir)
    query = manifest["query"]
    config = _runs.config_from_manifest(manifest)

    paragraphs_path = run_dir / _runs.PARAGRAPHS_NAME
    if not paragraphs_path.exists():
        raise SystemExit(
            f"{paragraphs_path.name} missing — run step2_retrieve.py first."
        )
    paragraphs: List[Dict[str, Any]] = _runs.read_json(paragraphs_path)
    if not paragraphs:
        raise SystemExit("No paragraphs retrieved — nothing to judge.")

    items, _, _ = _runs.build_judge_items(paragraphs)

    today = datetime.date.today()
    template = _FILTER_PROMPT_PATH.read_text(encoding="utf-8")
    dated_template = template.replace("{today_date}", today.isoformat()).replace(
        "{today_year}", str(today.year)
    )

    batch_size = config.paragraphs_per_judge_batch
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    judge_paths: List[tuple[Path, Path]] = []
    for index, batch_items in enumerate(batches):
        paths = _runs.batch_paths(run_dir, index)
        # json.dumps(..., indent=2) is the serialization _evaluate_batch sends —
        # the judge has only ever seen its paragraphs in that shape.
        body = dated_template.replace("{user_query}", query).replace(
            "{paragraphs_batch}", json.dumps(batch_items, indent=2)
        )
        _runs.write_prompt(
            paths["prompt"],
            title=(
                f"Stage 3 — relevance judge, batch {index + 1}/{len(batches)} "
                f"({len(batch_items)} paragraph{'' if len(batch_items) == 1 else 's'})"
            ),
            system_prompt=_runs.JUDGE_SYSTEM_PROMPT,
            body=body,
            output_key="relevant_ids",
        )
        judge_paths.append((paths["prompt"], paths["output"]))

    with ThreadPoolExecutor(max_workers=len(judge_paths)) as executor:
        futures = [
            executor.submit(
                _run_judge_batch,
                args.provider,
                prompt_path,
                output_path,
            )
            for prompt_path, output_path in judge_paths
        ]
        for future in futures:
            future.result()

    _runs.record_stage(
        run_dir,
        "step3_judge_prompts",
        {
            "paragraph_count": len(items),
            "batch_size": batch_size,
            "batch_count": len(batches),
            # Recorded, not applied: a one-shot CLI session exposes no temperature control.
            "filter_temperature": config.filter_temperature,
            "provider": args.provider,
        },
    )

    print(f"[Librarian] paragraphs={len(items)} batches={len(batches)} of {batch_size}")
    print()
    print("Then run:")
    print(
        f"  {Path(__file__).with_name('run.sh')} step4_finalize "
        f"--run {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Format agent predictions for ScholarQABench rubric evaluation.

Reads predictions JSON (with input/output/ctxs) from run_scholarqa.py,
joins with test_configs_snippets.json to get case_id, and writes a JSONL
file in the format expected by rubric_eval.py:
  {"case_id": "...", "answer_text": "..."}

Usage:
    python format_rubric.py \
        --predictions predictions.json \
        --test-config ../scholarqabench/data/scholarqa_cs/test_configs_snippets.json \
        --output rubric_input.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_records(path: str) -> List[Dict]:
    """Load a JSON list, JSONL file, or {"data": [...]} wrapper."""
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".jsonl":
        with open(path_obj) as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path_obj) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported test config format: {path}")
    return data


def _build_question_to_case_id(test_config_path: str) -> dict:
    """Build a mapping from question text → case_id from test config."""
    config = _load_records(test_config_path)

    mapping = {}
    for entry in config:
        # scholarqa_cs uses initial_prompt & case_id
        # scholarqa_multi uses input & id
        question = entry.get(
            "initial_prompt", entry.get("input", entry.get("question", ""))
        ).strip()
        case_id = entry.get("case_id", entry.get("id", entry.get("idx", "")))
        if question and case_id:
            mapping[question] = case_id

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Format predictions for rubric eval.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--test-config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Load question → case_id mapping
    q_to_cid = _build_question_to_case_id(args.test_config)
    print(f"Loaded {len(q_to_cid)} questions from test config.")

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions.")

    # Match and write
    matched = 0
    unmatched = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_out:
        for pred in predictions:
            question = pred.get("input", "").strip()
            answer = pred.get("output", "").strip()

            case_id = q_to_cid.get(question)
            if case_id is None:
                # Try fuzzy match — strip trailing punctuation
                for q, cid in q_to_cid.items():
                    if q.rstrip("?.!") == question.rstrip("?.!"):
                        case_id = cid
                        break

            if case_id:
                f_out.write(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "answer_text": answer,
                        }
                    )
                    + "\n"
                )
                matched += 1
            else:
                unmatched += 1

    print(f"Written {matched} entries to {output_path}")
    if unmatched:
        print(f"WARNING: {unmatched} predictions could not be matched to a case_id")


if __name__ == "__main__":
    main()

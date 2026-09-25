"""Recompute LAB-Bench accuracy/precision/coverage from the raw prediction logs.

Trusts only ``predictions.jsonl``. The derived files in a result directory
(``metrics.json``, ``summary.md``, ``run_config.json``) are read only to check
them against those rows.

Checks performed per result directory:

- the three rates, recomputed, with the ``total / attempted / correct`` counts
  beside them;
- the identity ``accuracy == precision * coverage``, exact by construction;
- ``run_config.json``'s ``model`` against the model recorded on every row;
- rows lost to ``error``, and whether ``metrics.json``'s denominator matches the
  number of rows in the log;
- retrieval health: ``retrieval_error`` and zero-evidence questions;
- raw U+2028-class separators, which a ``splitlines()``-based reader cannot read
  back.

Usage:
    python -m evals.Literature.LabBench.audit_results [results_dir] [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
IDENTITY_TOLERANCE = 0.01
# str.splitlines() breaks on these, and json.dumps(ensure_ascii=False) does not
# escape them, so any occurrence in predictions.jsonl breaks such a reader.
SPLITLINES_ONLY = ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "", "")


def load_rows(predictions_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return (rows, n_unparseable). Splits on "\\n" only — see SPLITLINES_ONLY."""
    rows: list[dict[str, Any]] = []
    unparseable = 0
    for line in predictions_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            unparseable += 1
    return rows, unparseable


def audit_run(run_dir: Path) -> dict[str, Any] | None:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return None
    rows, unparseable = load_rows(predictions_path)

    def read_json(name: str) -> dict[str, Any]:
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    config = read_json("run_config.json")
    cached = read_json("metrics.json")

    # Last row wins, so a resumed duplicate id is counted once.
    by_id = {r.get("id"): r for r in rows}
    unique = list(by_id.values())
    total = len(unique)
    attempted = sum(1 for r in unique if r.get("sure"))
    correct = sum(1 for r in unique if r.get("correct"))
    accuracy = correct / total if total else 0.0
    precision = correct / attempted if attempted else 0.0
    coverage = attempted / total if total else 0.0

    metadata = [(r.get("metadata") or {}) for r in unique]
    raw_text = predictions_path.read_text(encoding="utf-8")
    return {
        "dir": run_dir.name,
        "total": total,
        "attempted": attempted,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "coverage": coverage,
        "identity_gap": abs(accuracy - precision * coverage),
        "task": ",".join(sorted({str(r.get("task")) for r in unique})),
        "mode": ",".join(sorted({str(r.get("mode")) for r in unique})),
        "config_model": config.get("model"),
        "row_models": sorted({str(r.get("model")) for r in unique if r.get("model")}),
        "duplicate_ids": len(rows) - total,
        "unparseable_lines": unparseable,
        "n_errors": sum(1 for r in unique if r.get("error")),
        "n_parse_failures": sum(1 for r in unique if r.get("parse_failure")),
        "retrieval_errors": sum(1 for m in metadata if m.get("retrieval_error")),
        "zero_evidence": sum(1 for m in metadata if m.get("n_evidence_papers") == 0),
        "cached": cached,
        # Rows a splitlines()-based reader would shred.
        "splitlines_fragments": len([x for x in raw_text.splitlines() if x.strip()])
        - len([x for x in raw_text.split("\n") if x.strip()]),
    }


def top_retrieval_errors(run_dir: Path, limit: int = 3) -> list[tuple[int, str]]:
    rows, _ = load_rows(run_dir / "predictions.jsonl")
    messages = [
        (r.get("metadata") or {}).get("retrieval_error")
        for r in rows
        if (r.get("metadata") or {}).get("retrieval_error")
    ]
    counter = collections.Counter(re.sub(r"\d+", "N", str(m))[:100] for m in messages)
    return [(n, m) for m, n in counter.most_common(limit)]


def report(results_dir: Path) -> list[dict[str, Any]]:
    audits = [
        a for d in sorted(results_dir.iterdir()) if d.is_dir() and (a := audit_run(d))
    ]

    header = (
        f"{'dir':38s} {'task':11s} {'mode':9s} {'tot':>4s} {'att':>4s} {'cor':>4s} "
        f"{'ACC':>6s} {'PREC':>6s} {'COV':>6s} {'ID':>4s}"
    )
    print(header)
    print("-" * len(header))
    for a in audits:
        ok = "ok" if a["identity_gap"] <= IDENTITY_TOLERANCE else "BAD"
        print(
            f"{a['dir']:38s} {a['task']:11s} {a['mode']:9s} {a['total']:4d} "
            f"{a['attempted']:4d} {a['correct']:4d} {a['accuracy']:6.3f} "
            f"{a['precision']:6.3f} {a['coverage']:6.3f} {ok:>4s}"
        )

    print("\n" + "=" * 78 + "\nWARNINGS\n" + "=" * 78)
    clean = True
    for a in audits:
        notes: list[str] = []
        if a["total"] == 0:
            notes.append(
                "EMPTY predictions.jsonl, so there is nothing to audit here"
            )
        if a["unparseable_lines"]:
            notes.append(f"{a['unparseable_lines']} unparseable line(s)")
        if a["duplicate_ids"]:
            notes.append(f"{a['duplicate_ids']} duplicate id(s) (kept last)")
        if a["splitlines_fragments"]:
            notes.append(
                f"contains raw U+2028-class separators ({a['splitlines_fragments']} "
                "extra fragments under splitlines())"
            )
        if (
            a["config_model"]
            and a["row_models"]
            and a["config_model"] not in a["row_models"]
        ):
            notes.append(
                f"MODEL MISLABELLED: run_config says {a['config_model']!r} but rows "
                f"were answered by {a['row_models']}"
            )
        if a["identity_gap"] > IDENTITY_TOLERANCE:
            notes.append(f"identity broken: |A - P*C| = {a['identity_gap']:.4f}")
        cached = a["cached"]
        if cached and cached.get("n_total") not in (None, a["total"]):
            notes.append(
                f"metrics.json denominator is {cached['n_total']} but the log holds "
                f"{a['total']} rows -> cached A={cached.get('accuracy', 0):.3f} vs "
                f"true A={a['accuracy']:.3f}"
            )
        if a["n_errors"]:
            notes.append(f"{a['n_errors']} errored row(s) (re-run with --resume)")
        if a["mode"] == "knowledge" and a["retrieval_errors"]:
            share = a["retrieval_errors"] / a["total"] if a["total"] else 0
            label = "SEVERE" if share > 0.25 else "note"
            notes.append(
                f"{label}: retrieval failed on {a['retrieval_errors']}/{a['total']} "
                "question(s)"
            )
            for n, message in top_retrieval_errors(results_dir / a["dir"]):
                notes.append(f"    {n}x  {message}")
        if a["mode"] == "knowledge":
            silent = a["zero_evidence"] - a["retrieval_errors"]
            if silent > 0.1 * max(a["total"], 1):
                notes.append(
                    f"{silent} question(s) got zero evidence with no error "
                    "raised, and were answered parametrically"
                )
        if notes:
            clean = False
            print(f"\n{a['dir']}:")
            for note in notes:
                print(f"  - {note}")
    if clean:
        print("\nNone.")
    return audits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--json", dest="json_out", default=None, help="Also write raw audit JSON here."
    )
    args = parser.parse_args(argv)

    audits = report(Path(args.results_dir).expanduser().resolve())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(audits, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()

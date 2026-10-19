"""Tables from SuppQA run directories (see run_suppqa_eval.py).

Usage:
    python -m evals.Literature.SuppQA.analyze_suppqa RUN_DIR [RUN_DIR ...]

Each directory holds one arm's predictions.jsonl. Accuracy and timing compare
the arms over the questions every answering arm finished without error, so each
arm is scored on the same set. Prints Markdown.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

from evals.Literature.SuppQA.run_suppqa_eval import KEY_PASSAGE_THRESHOLD, _read_rows

FOUND = KEY_PASSAGE_THRESHOLD


def _pct(n: float, d: float) -> str:
    return f"{n / d:.1%}" if d else "-"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def preflight_report(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    in_supp = [r for r in rows if r["key_in_supp"] >= FOUND]
    counts = [
        ("questions (paper has a PMC id)", n),
        ("full text fetched", sum(r["fulltext_ok"] for r in rows)),
        ("JATS declares supplementary material", sum(r["declares_supp"] for r in rows)),
        ("supplementary download failed / over size cap", sum(bool(r["supp_error"]) for r in rows)),
        ("≥1 readable supplementary file (PDF/Word)", sum(bool(r["supp_files"]) for r in rows)),
        ("key passage in extracted supplement", len(in_supp)),
        ("…but lost to the per-paper record cap", sum(
            r["key_in_supp"] < FOUND <= r["key_in_supp_uncapped"] for r in rows)),
        ("key passage in main-text body", sum(r["key_in_body"] >= FOUND for r in rows)),
        ("key passage ONLY in supplement (ceiling for supp gain)", sum(
            r["key_in_body"] < FOUND for r in in_supp)),
    ]
    return "## Preflight: where each key passage can be found\n\n" + _table(
        ["", "n", "% of questions"], [[k, v, _pct(v, n)] for k, v in counts]
    )


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0


def accuracy_report(arms: dict[str, dict[str, dict]], ids: set[str]) -> str:
    rows = []
    for arm, by_id in arms.items():
        scored = [by_id[i] for i in ids]
        correct = sum(r["correct"] for r in scored)
        sure = sum(r["sure"] for r in scored)
        rows.append([arm, len(scored), _pct(correct, len(scored)), _pct(correct, sure),
                     _pct(sure, len(scored)), correct])
    out = f"## Accuracy (n = {len(ids)} questions every arm answered)\n\n" + _table(
        ["arm", "n", "accuracy", "precision", "coverage", "correct"], rows
    )
    if "main" in arms and "supp" in arms:
        main, supp = arms["main"], arms["supp"]
        gained = [i for i in ids if supp[i]["correct"] and not main[i]["correct"]]
        lost = [i for i in ids if main[i]["correct"] and not supp[i]["correct"]]
        both = sum(main[i]["correct"] and supp[i]["correct"] for i in ids)
        cites_supp = lambda i: supp[i]["retrieval"]["counts"].get("cited_supp", 0) > 0  # noqa: E731
        out += "\n\n### main → supp, paired\n\n" + _table(
            ["", "n"],
            [
                ["right in both", both],
                ["wrong → right (gained)", len(gained)],
                ["right → wrong (lost)", len(lost)],
                ["gained, and supp arm cited supplementary text", sum(map(cites_supp, gained))],
                ["exact McNemar p", f"{_mcnemar_p(len(gained), len(lost)):.3f}"],
            ],
        )
        for label, subset in (
            ("cited ≥1 supplementary span", [i for i in ids if cites_supp(i)]),
            ("cited none", [i for i in ids if not cites_supp(i)]),
        ):
            out += (
                f"\n\nsupp arm, questions where the judge {label}: n = {len(subset)}, "
                f"accuracy {_pct(sum(supp[i]['correct'] for i in subset), len(subset))} "
                f"(main arm on the same: "
                f"{_pct(sum(main[i]['correct'] for i in subset), len(subset))})"
            )
    return out


def timing_report(arms: dict[str, dict[str, dict]], ids: set[str]) -> str:
    """Retrieval seconds per question; Stage 1 is left out of the headline since
    a run with --queries-from skips the query-planning LLM call."""
    spans = ["librarian.stage2_paragraphs", "librarian.epmc_search",
             "librarian.epmc_fulltext", "librarian.bm25_rank", "librarian.filter_relevance"]
    rows = []
    for arm, by_id in arms.items():
        stats = [by_id[i]["retrieval"] for i in ids if by_id[i].get("retrieval")]
        if not stats:
            continue
        retrieval = [s["run_seconds"] - s["stage1_seconds"] for s in stats]
        counts = Counter()
        for s in stats:
            counts.update(s["counts"])
        rows.append(
            [arm, f"{statistics.median(retrieval):.1f}", f"{_quantile(retrieval, 0.9):.1f}",
             f"{statistics.mean(retrieval):.1f}"]
            + [f"{statistics.mean(s['span_seconds'].get(n, 0.0) for s in stats):.1f}" for n in spans]
            + [f"{counts['supp_fetch_errors']}/{counts['supp_fetches']}"]
        )
    return (
        "## Retrieval time per question, seconds (Stage 2 + 3)\n\n"
        "Sub-query spans (search, fulltext, bm25) are summed over the parallel "
        "sub-query threads, so they are thread-seconds, not wall-clock. fulltext "
        "includes the supplementary downloads; bm25 includes parsing them.\n\n"
        + _table(
            ["arm", "median", "p90", "mean", "stage 2 wall", "search Σ", "fulltext Σ",
             "bm25+parse Σ", "judge", "supp fetch errors"],
            rows,
        )
    )


def evidence_report(arms: dict[str, dict[str, dict]], ids: set[str]) -> str:
    out = []
    for arm in ("main", "supp"):
        if arm not in arms:
            continue
        stats = [arms[arm][i]["retrieval"] for i in ids]
        c = Counter()
        for s in stats:
            c.update(s["counts"])
        n = len(stats)
        judged_main = c["judged_total"] - c["judged_supp"]
        rows = [
            ["candidate pool (before BM25, Σ sub-queries)", c["pool_main"], c["pool_supp"],
             _pct(c["pool_supp"], c["pool_main"] + c["pool_supp"])],
            ["kept by BM25 (Σ sub-queries)", c["bm25_main"], c["bm25_supp"],
             _pct(c["bm25_supp"], c["bm25_main"] + c["bm25_supp"])],
            ["sent to the judge (merged)", judged_main, c["judged_supp"],
             _pct(c["judged_supp"], c["judged_total"])],
            ["cited by the judge (spans)", c["cited_main"], c["cited_supp"],
             _pct(c["cited_supp"], c["cited_main"] + c["cited_supp"])],
            ["shown to the answering model (spans)", c["shown_main"], c["shown_supp"],
             _pct(c["shown_supp"], c["shown_main"] + c["shown_supp"])],
        ]
        rates = [
            ["BM25 keep rate (kept / pool)", _pct(c["bm25_main"], c["pool_main"]),
             _pct(c["bm25_supp"], c["pool_supp"])],
            ["citation rate (cited / sent to judge)", _pct(c["cited_main"], judged_main),
             _pct(c["cited_supp"], c["judged_supp"])],
        ]
        out.append(
            f"## Evidence by source, `{arm}` arm (totals over {n} questions)\n\n"
            + _table(["stage", "main text", "supplementary", "supp share"], rows)
            + "\n\n" + _table(["rate", "main text", "supplementary"], rates)
            + f"\n\nQuestions citing ≥1 supplementary span: "
            f"{sum(s['counts'].get('cited_supp', 0) > 0 for s in stats)} / {n}"
        )
    return "\n\n".join(out)


def target_report(arms: dict[str, dict[str, dict]], ids: set[str]) -> str:
    """How far each question's own paper, and its key passage, got."""
    header = ["stage", *[a for a in ("main", "supp") if a in arms]]
    checks = [
        ("target paper in candidate pool", lambda s: s["counts"].get("target_pool", 0) > 0),
        ("target's supplement in candidate pool", lambda s: s["counts"].get("target_pool_supp", 0) > 0),
        ("key passage in pool (main text)", lambda s: s["key"].get("key_pool_main", 0) >= FOUND),
        ("key passage in pool (supplement)", lambda s: s["key"].get("key_pool_supp", 0) >= FOUND),
        ("key passage kept by BM25 (main text)", lambda s: s["key"].get("key_bm25_main", 0) >= FOUND),
        ("key passage kept by BM25 (supplement)", lambda s: s["key"].get("key_bm25_supp", 0) >= FOUND),
        ("target paper cited by the judge", lambda s: s["target_cited"]),
        ("key passage cited by the judge", lambda s: s["key"].get("key_cited", 0) >= FOUND),
        ("key passage shown to the answering model", lambda s: s["key"].get("key_shown", 0) >= FOUND),
    ]
    rows = []
    for label, check in checks:
        row = [label]
        for arm in header[1:]:
            hits = sum(check(arms[arm][i]["retrieval"]) for i in ids)
            row.append(f"{hits} ({_pct(hits, len(ids))})")
        rows.append(row)
    out = "## Target-paper funnel (does each question's own paper and key passage get through?)\n\n"
    out += _table(header, rows)
    for arm in header[1:]:
        shown = [i for i in ids if arms[arm][i]["retrieval"]["key"].get("key_shown", 0) >= FOUND]
        rest = ids - set(shown)
        out += (
            f"\n\n`{arm}`: accuracy when the key passage reached the model "
            f"{_pct(sum(arms[arm][i]['correct'] for i in shown), len(shown))} (n={len(shown)}), "
            f"otherwise {_pct(sum(arms[arm][i]['correct'] for i in rest), len(rest))} (n={len(rest)})"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)

    sections = []
    arms: dict[str, dict[str, dict]] = {}
    for run_dir in args.run_dirs:
        # One row per question: the last one wins, as with a resumed run.
        rows = list({r["id"]: r for r in _read_rows(Path(run_dir) / "predictions.jsonl")}.values())
        if not rows:
            continue
        arm = rows[0]["arm"]
        if arm == "preflight":
            sections.append(preflight_report(rows))
        else:
            arms[arm] = {r["id"]: r for r in rows if not r.get("error")}
    if arms:
        ids = set.intersection(*(set(by_id) for by_id in arms.values()))
        sections.append(accuracy_report(arms, ids))
        retrieval_arms = {a: v for a, v in arms.items() if a != "baseline"}
        if retrieval_arms:
            sections += [
                timing_report(retrieval_arms, ids),
                evidence_report(retrieval_arms, ids),
                target_report(retrieval_arms, ids),
            ]
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()

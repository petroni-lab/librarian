"""Run LAB-Bench SuppQA with and without supplementary-material retrieval.

Every SuppQA question is answered from one paper's supplementary material. One
run is one arm, chosen with ``--arm``:

  preflight  No LLM. Fetch each question's own paper (full text and
             supplementary files) and record where its gold ``key-passage`` can
             be found: the ceiling of what supplementary support can add.
  baseline   The answering model alone, from its own knowledge.
  main       The model grounded in the librarian, abstracts + full-text bodies
             (``supplementary_enrichment`` off).
  supp       The same, plus supplementary PDF/Word files
             (``supplementary_enrichment`` on).

Questions whose paper has no PMC id in Europe PMC are skipped by every arm. As
upstream, the question is prefixed with the paper title and DOI, and the same
text is the librarian's query. ``--queries-from <supp run dir>`` makes the main
arm reuse the supp arm's Stage-1 sub-queries, so the two differ only in the
supplementary files. The relevance filter runs at temperature 0 for the same
reason.

Choices are shuffled, prompted, parsed and scored as in ``LabBench``. Each run
writes ``predictions.jsonl`` to ``--out-dir``; ``analyze_suppqa.py`` turns run
directories into tables.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from evals.Literature.LabBench.labbench_compat import (  # noqa: E402
    build_mcq_prompt,
    parse_answer,
    randomize_choices,
    score_prediction,
)
from librarian.agent import LibrarianAgent  # noqa: E402
from librarian.config import load_runtime_config  # noqa: E402
from librarian.jats import (  # noqa: E402
    extract_body_paragraphs,
    extract_supplementary_captions,
)
from librarian.literature_search import (  # noqa: E402
    declares_supplementary,
    fetch_fulltext,
    fetch_supplementary,
    normalize_pmcid,
    search_scientific_literature_structured,
)
from librarian.supplementary import (  # noqa: E402
    SUPPLEMENTARY_SECTION_TYPE,
    extract_supplementary_records,
)

LOGGER = logging.getLogger("suppqa")

DATA_URL = (
    "https://raw.githubusercontent.com/Future-House/LAB-Bench/main/"
    "SuppQA/suppqa-v1-public.jsonl"
)
# Share of the key passage's distinct words a text must contain to count as
# containing it. Below 1 so PDF extraction noise (hyphenation, ligatures,
# dropped symbols) does not hide a real match.
KEY_PASSAGE_THRESHOLD = 0.6


# ── Questions ────────────────────────────────────────────────────────────────


def _key_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


def key_overlap(key_passage: str, text: str) -> float:
    """Share of the key passage's distinct words found in ``text`` (0..1)."""
    key = _key_tokens(key_passage)
    return len(key & _key_tokens(text)) / len(key) if key else 0.0


def _best_overlap(key_passage: str, texts: list[str]) -> float:
    return max((key_overlap(key_passage, t) for t in texts), default=0.0)


def resolve_pmcid(doi: str) -> str:
    """The PMC id Europe PMC lists for ``doi``, or "" when it has none."""
    for paper in search_scientific_literature_structured(f'DOI:"{doi}"', page_size=5):
        if str(paper.get("pmcid") or "").strip():
            return normalize_pmcid(paper["pmcid"])
    return ""


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    """SuppQA rows → shuffled MCQs; rows whose paper has no PMC id are dropped."""
    if args.data_file:
        text = Path(args.data_file).read_text(encoding="utf-8")
    else:
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        text = response.text
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]

    questions = []
    skipped = 0
    for index, row in enumerate(rows):
        if args.max_examples and len(questions) >= args.max_examples:
            break
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", row["source"].strip())
        pmcid = resolve_pmcid(doi)
        if not pmcid:
            skipped += 1
            continue
        choices, answer, unsure = randomize_choices(
            row["ideal"], list(row["distractors"]), seed=args.seed + index
        )
        # Upstream SuppQA's prefix (LAB-Bench SuppQA/task.py).
        question = (
            f"Paper title: {row['paper-title']}\nDOI: {row['source']}\n"
            f"{row['question'].strip()}"
        )
        questions.append(
            {
                "id": str(row["id"]),
                "query": question,
                "prompt": build_mcq_prompt(question, choices),
                "n_choices": len(choices),
                "answer_letter": answer,
                "unsure_letter": unsure,
                "doi": doi.lower(),
                "pmcid": pmcid,
                "key_passage": row["key-passage"],
            }
        )
    LOGGER.info("%d questions loaded; %d skipped (no PMC id).", len(questions), skipped)
    return questions


# ── Preflight ────────────────────────────────────────────────────────────────


def preflight_row(args: argparse.Namespace, q: dict[str, Any]) -> dict[str, Any]:
    """Where the question's key passage sits in its own paper, with no retrieval."""
    config = load_runtime_config()
    row: dict[str, Any] = {
        "id": q["id"],
        "arm": "preflight",
        "pmcid": q["pmcid"],
        "fulltext_ok": False,
        "declares_supp": False,
        "supp_files": [],
        "supp_error": "",
        "supp_records": 0,
        "key_in_body": 0.0,
        "key_in_supp": 0.0,
        "key_in_supp_uncapped": 0.0,
    }
    fulltext = fetch_fulltext(q["pmcid"])
    if not fulltext.ok:
        row["error_detail"] = fulltext.error
        return row
    row["fulltext_ok"] = True
    body = [r["text"] for r in extract_body_paragraphs(fulltext.xml)]
    row["key_in_body"] = _best_overlap(q["key_passage"], body)
    row["declares_supp"] = declares_supplementary(fulltext.xml)
    if not row["declares_supp"]:
        return row
    try:
        files = fetch_supplementary(q["pmcid"], config.max_supplementary_bytes)
    except (requests.exceptions.RequestException, ValueError) as exc:
        row["supp_error"] = str(exc) or type(exc).__name__
        return row
    captions = extract_supplementary_captions(fulltext.xml)
    capped = extract_supplementary_records(
        files, captions, config.max_supplementary_records_per_paper
    )
    uncapped = extract_supplementary_records(files, captions, 10_000)
    row["supp_files"] = sorted(files)
    row["supp_records"] = len(capped)
    row["key_in_supp"] = _best_overlap(q["key_passage"], [r["text"] for r in capped])
    row["key_in_supp_uncapped"] = _best_overlap(
        q["key_passage"], [r["text"] for r in uncapped]
    )
    return row


# ── Instrumented librarian ───────────────────────────────────────────────────


class SpanTimer:
    """TracingPort that sums wall-clock seconds per span name.

    Sub-query spans run one per thread, so their sums are thread-time, not
    wall-clock; ``librarian.stage2_paragraphs`` is Stage 2's wall-clock.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    @contextmanager
    def start_span(self, name, *, attributes=None):
        start = time.perf_counter()
        try:
            yield None
        finally:
            with self._lock:
                self.seconds[name] += time.perf_counter() - start

    def set_span_attributes(self, span, attributes) -> None:
        pass

    def mark_span_error(self, span, exc) -> None:
        pass

    def bind_current_trace_context(self, fn):
        return fn


def _kind(record: dict[str, Any]) -> str:
    return "supp" if record.get("section_type") == SUPPLEMENTARY_SECTION_TYPE else "main"


class InstrumentedAgent(LibrarianAgent):
    """LibrarianAgent that counts, per run, where its paragraphs come from.

    Pool (before BM25) and BM25-kept counts are summed over sub-queries, so a
    paper two sub-queries both find counts twice in each, and the two stay
    comparable. Optionally replays fixed Stage-1 sub-queries.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stats_lock = threading.Lock()
        self.begin({}, None)

    def begin(self, question: dict[str, Any], fixed_queries: list[str] | None) -> None:
        """Reset the per-run counters for ``question``."""
        self.question = question
        self.fixed_queries = fixed_queries
        self.counts: Counter = Counter()
        self.key: dict[str, float] = defaultdict(float)
        self.stage1_seconds = 0.0
        self.passages: list[dict[str, Any]] = []
        self._tracer.seconds.clear()

    def _is_target(self, paper: dict[str, Any]) -> bool:
        doi = str(paper.get("doi") or "").lower()
        pmcid = str(paper.get("pmcid") or "").strip()
        return bool(self.question) and (
            doi == self.question["doi"]
            or (bool(pmcid) and normalize_pmcid(pmcid) == self.question["pmcid"])
        )

    def _note(self, stage: str, paper: dict[str, Any], records: list[dict]) -> None:
        with self._stats_lock:
            for record in records:
                self.counts[f"{stage}_{_kind(record)}"] += 1
        if not self._is_target(paper):
            return
        with self._stats_lock:
            self.counts[f"target_{stage}"] += 1
            for record in records:
                kind = _kind(record)
                self.counts[f"target_{stage}_{kind}"] += 1
                overlap = key_overlap(self.question["key_passage"], record["text"])
                name = f"key_{stage}_{kind}"
                self.key[name] = max(self.key[name], overlap)

    def _generate_queries(self, query, previous_evidences=None):
        start = time.perf_counter()
        try:
            if self.fixed_queries:
                return list(self.fixed_queries)
            return super()._generate_queries(query, previous_evidences)
        finally:
            self.stage1_seconds = time.perf_counter() - start

    def _paragraph_records_for_paper(self, paper, fulltexts):
        records = super()._paragraph_records_for_paper(paper, fulltexts)
        self._note("pool", paper, records)
        return records

    def _paragraphs_for_subquery(self, subquery):
        top = super()._paragraphs_for_subquery(subquery)
        by_paper: dict[int, list[dict]] = defaultdict(list)
        for record in top:
            by_paper[id(record["paper"])].append(record)
        for records in by_paper.values():
            self._note("bm25", records[0]["paper"], records)
        return top

    def _supplementary_paragraphs(self, pmcid, fulltext):
        with self._stats_lock:
            self.counts["supp_fetches"] += 1
            if getattr(fulltext, "supplementary_error", ""):
                self.counts["supp_fetch_errors"] += 1
            if getattr(fulltext, "supplementary", None):
                self.counts["supp_fetches_with_files"] += 1
        return super()._supplementary_paragraphs(pmcid, fulltext)


def retrieval_stats(agent: InstrumentedAgent, budget: int) -> dict[str, Any]:
    """One run's counts, timings and key-passage funnel, for predictions.jsonl."""
    counts = Counter(agent.counts)
    key = dict(agent.key)
    target = None
    for passage in agent.passages:
        sections = passage.get("evidence_sections") or []
        start = 0
        for i, snippet in enumerate(passage["evidence_snippets"]):
            kind = "supp" if i < len(sections) and sections[i].startswith(
                "Supplementary: "
            ) else "main"
            counts[f"cited_{kind}"] += 1
            # format_evidence joins a paper's snippets with spaces and keeps the
            # first `budget` characters: a span starting past that is never seen.
            if start < budget:
                counts[f"shown_{kind}"] += 1
            start += len(snippet) + 1
        if str(passage.get("doi") or "").lower() == agent.question["doi"] or (
            passage.get("pmcid") == agent.question["pmcid"]
        ):
            target = passage
    if target is not None:
        joined = " ".join(target["evidence_snippets"])
        key["key_cited"] = key_overlap(agent.question["key_passage"], joined)
        key["key_shown"] = key_overlap(agent.question["key_passage"], joined[:budget])
    debug = agent.last_run_debug
    counts["judged_total"] = debug.get("paragraph_count", 0)
    counts["judged_supp"] = debug.get("supplementary_paragraph_count", 0)
    seconds = dict(agent._tracer.seconds)
    return {
        "search_queries": debug.get("search_queries", []),
        "stage1_seconds": round(agent.stage1_seconds, 3),
        "run_seconds": round(seconds.get("librarian.run", 0.0), 3),
        "span_seconds": {k: round(v, 3) for k, v in seconds.items()},
        "counts": dict(counts),
        "target_cited": target is not None,
        "key": {k: round(v, 3) for k, v in key.items()},
    }


# ── Answering ────────────────────────────────────────────────────────────────

_thread_local = threading.local()


def _worker(args: argparse.Namespace):
    """One (agent, solver) per thread: the agent keeps per-run state on itself."""
    if getattr(_thread_local, "solver", None) is not None:
        return _thread_local.agent, _thread_local.solver
    from evals.Literature.LabBench.solvers import BaselineSolver, KnowledgeLayerSolver

    common = dict(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
    )
    agent = None
    if args.arm == "baseline":
        solver = BaselineSolver(**common)
    else:
        config = replace(
            load_runtime_config(),
            supplementary_enrichment=args.arm == "supp",
            filter_temperature=0.0,
        )
        agent = InstrumentedAgent(
            runtime_config=config,
            llm_base_url=args.agent_base_url,
            llm_model_name=args.agent_model,
            tracer=SpanTimer(),
            verbose=True,  # "[Librarian] ..." progress lines in the log
        )

        def search(query: str) -> dict[str, Any]:
            agent.passages = agent.run(query)
            return {
                "evidence": [
                    {
                        "pmid": p.get("pmid"),
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "evidence": p.get("evidence_snippets") or [],
                    }
                    for p in agent.passages
                ]
            }

        solver = KnowledgeLayerSolver(
            search_fn=search, char_budget=args.evidence_char_budget, **common
        )
    if args.reasoning_effort:
        # The LabBench chat client has no reasoning knob; without one a
        # reasoning model thinks at full length on every question.
        completions = solver.client.client.chat.completions
        completions.create = functools.partial(
            completions.create, reasoning_effort=args.reasoning_effort
        )
    _thread_local.agent, _thread_local.solver = agent, solver
    return agent, solver


def answer_row(args: argparse.Namespace, q: dict[str, Any]) -> dict[str, Any]:
    agent, solver = _worker(args)
    if agent is not None:
        agent.begin(q, args.fixed_queries.get(q["id"]))
    try:
        result = solver.answer(q["prompt"], retrieval_query=q["query"])
        raw, metadata = result["raw_output"], result["metadata"]
    except Exception as exc:  # network / rate limit: excluded, re-run by --resume
        LOGGER.exception("Question %s failed.", q["id"])
        raw, metadata = "", {"error": str(exc)}
    predicted = parse_answer(raw, q["n_choices"])
    correct, sure = score_prediction(predicted, q["answer_letter"], q["unsure_letter"])
    return {
        "id": q["id"],
        "arm": args.arm,
        "model": args.model,
        "pmcid": q["pmcid"],
        "target_choice": q["answer_letter"],
        "predicted_choice": predicted,
        "correct": correct,
        "sure": sure,
        # A failed retrieval would silently become a baseline answer; count it
        # as an error instead so the arms stay comparable.
        "error": bool(metadata.get("error") or metadata.get("retrieval_error")),
        "raw_output": raw,
        "metadata": metadata,
        "retrieval": (
            retrieval_stats(agent, args.evidence_char_budget) if agent else None
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # "\n" only: ensure_ascii=False leaves U+2028 raw, which splitlines() breaks on.
    return [json.loads(line) for line in path.read_text("utf-8").split("\n") if line.strip()]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = out_dir / "predictions.jsonl"
    args.fixed_queries = {
        r["id"]: r["retrieval"]["search_queries"]
        for r in _read_rows(Path(args.queries_from) / "predictions.jsonl")
        if r.get("retrieval") and not r.get("error")
    } if args.queries_from else {}

    questions = load_questions(args)
    kept = [r for r in _read_rows(predictions) if not r.get("error")] if args.resume else []
    predictions.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
    )
    done = {r["id"] for r in kept}
    pending = [q for q in questions if q["id"] not in done]
    LOGGER.info("%d done, %d to run.", len(done), len(pending))

    config = load_runtime_config()
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in vars(args).items() if k not in {"fixed_queries", "api_key"}},
                "n_questions": len(questions),
                "librarian_config": asdict(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    work = preflight_row if args.arm == "preflight" else answer_row
    with predictions.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as pool:
        futures = [pool.submit(work, args, q) for q in pending]
        for n, future in enumerate(as_completed(futures), len(done) + 1):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            LOGGER.info("[%d/%d] %s", n, len(questions), row["id"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=["preflight", "baseline", "main", "supp"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-file", help="Local suppqa-v1-public.jsonl (default: fetched from GitHub).")
    parser.add_argument("--model", help="Answering model (not needed for preflight).")
    parser.add_argument("--base-url", help="OpenAI-compatible URL of the answering model.")
    parser.add_argument("--api-key", help="Answering model key (default: OPENAI_API_KEY).")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("LLM_REASONING_EFFORT"),
        help="reasoning_effort sent to the answering model (default: LLM_REASONING_EFFORT).",
    )
    parser.add_argument("--agent-model", help="Librarian LLM (default: LLM_MODEL).")
    parser.add_argument("--agent-base-url", help="Librarian LLM URL (default: LLM_BASE_URL).")
    parser.add_argument(
        "--evidence-char-budget",
        type=int,
        default=6000,
        help="Characters of evidence per paper shown to the answering model. "
        "Supplementary spans come last in a paper's evidence, so LabBench's 1500 "
        "cuts them first.",
    )
    parser.add_argument("--queries-from", help="Run dir whose Stage-1 sub-queries to replay.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Keep finished rows, re-run errored ones.")
    args = parser.parse_args(argv)
    if args.arm != "preflight" and not args.model:
        parser.error("--model is required for every arm except preflight")
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(parse_args(argv))


if __name__ == "__main__":
    main()

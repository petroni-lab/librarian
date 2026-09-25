"""Run a text-only LAB-Bench multiple-choice benchmark.

A single, task-agnostic runner for the LAB-Bench MCQ subtasks our text knowledge
layer can serve. Pick the subtask with ``--task``:

  DbQA         Retrieving information from biological databases   (520 Qs)
  SeqQA        Manipulating biological sequences                  (600 Qs)
  ProtocolQA   Troubleshooting biological protocols               (108 Qs)
  CloningScenarios  Molecular cloning workflows                   (33 Qs)
  TableQA      Reading data tables reported in the literature     (244 Qs)

(TableQA runs retrieval-only — its table image and source DOI are ignored, so
the model must find the right table in retrieved passages. FigQA needs figure
images and is out of scope; LitQA2 and SuppQA are excluded by design — see
README.)

Evaluates an answering LLM (default ``gpt-4o-2024-05-13``) in one of two modes:

  --mode baseline    The LAB-Bench setup: the model answers each MCQ from its own
                     parametric knowledge (reproduces the paper's numbers).

  --mode knowledge   The same model, grounded in the LibrarianAgent as a knowledge
                     layer (single-shot RAG): the agent retrieves once on the
                     question, its passages are injected, the model answers in one
                     call.

Run both with the same --seed to compare accuracy / precision / coverage over
identically shuffled options.

The prompt template, option layout, refusal option, answer parsing, and metrics
all match the upstream LAB-Bench harness (see labbench_compat.py).

Examples:
    # Knowledge-layer GPT-4o on SeqQA
    python -m evals.Literature.LabBench.run_labbench_eval \
        --task SeqQA --mode knowledge --model gpt-4o \
        --out-dir results/seqqa_knowledge

    # Parametric GPT-4o baseline on DbQA (same options/seed)
    python -m evals.Literature.LabBench.run_labbench_eval \
        --task DbQA --mode baseline --model gpt-4o \
        --out-dir results/dbqa_baseline

Credentials: set OPENAI_API_KEY (or --api-key). Pass --base-url for the
answering model's endpoint and --agent-base-url for the knowledge agent's LLM
endpoint (and the LLM_* env vars LLMClient reads).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load credentials from the repo .env (OPENAI_API_KEY for the answering model,
# LLM_* for the knowledge agent's own LLM) into the process environment so the
# OpenAI SDK and LLMClient can read them.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()  # also honour a .env in the current working directory
except ImportError:
    pass


from evals.Literature.LabBench.labbench_compat import (  # noqa: E402
    TASKS,
    build_mcq_prompt,
    build_retrieval_query,
    compute_metrics,
    load_questions,
    parse_answer,
    score_prediction,
)
from evals.Literature.LabBench.solvers import (  # noqa: E402
    BaselineSolver,
    KnowledgeLayerSolver,
)

LOGGER = logging.getLogger("bioagents.eval.labbench")

_RESULTS_DIR = Path(__file__).parent / "results"


def _model_slug(name: str) -> str:
    """Filesystem-safe slug for a model name.

    Strips snapshot date suffixes (e.g. -2024-05-13) then replaces
    non-alphanumeric runs with underscores.
    """
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _auto_out_dir(args: argparse.Namespace) -> Path:
    """Canonical output directory derived from task, mode, and model names."""
    name = f"{args.task}_{args.mode}_{_model_slug(args.model)}"
    if args.mode != "baseline" and args.agent_model:
        name += f"__{_model_slug(args.agent_model)}"
    return _RESULTS_DIR / name


def _make_progress_bar(total: int, enabled: bool, desc: str):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, ncols=0)


def _evidence_entries(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape librarian per-paper records into what ``format_evidence`` reads.

    The librarian returns a flat list whose chunks live under
    ``evidence_snippets``; ``format_evidence`` wants entries with those chunks
    under ``evidence``. The in-process and --via-api paths receive the same
    record shape and share this.
    """
    return [
        {
            "pmid": paper.get("pmid"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "evidence": paper.get("evidence_snippets") or [],
        }
        for paper in papers
    ]


def build_search_fn(args: argparse.Namespace):
    """Build the knowledge-layer search_fn(query) -> result dict.

    The LibrarianAgent is imported lazily so --mode baseline never needs its deps.
    """
    if args.via_api:
        # The orchestrator ships LibrarianAgent.run()'s records verbatim, so
        # both paths use the same adapter.
        from evals.Literature import orchestrator_client

        def search_via_api(query: str) -> dict[str, Any]:
            """search_fn backed by the orchestrator instead of a local agent."""
            papers = orchestrator_client.librarian_evidence(
                query,
                base_url=args.api_base_url,
                source="labbench-eval",
                session_prefix="labbench",
            )
            return {"evidence": _evidence_entries(papers)}

        return search_via_api

    from librarian.agent import LibrarianAgent
    from librarian.config import load_runtime_config

    agent = LibrarianAgent(
        runtime_config=load_runtime_config(),
        llm_base_url=args.agent_base_url,
        llm_model_name=args.agent_model,
        full_text_enrichment=args.full_text,
        verbose=args.verbose,
    )

    def search(query: str) -> dict[str, Any]:
        """Adapt LibrarianAgent's output to the ``search_fn`` result-dict contract.

        ``LibrarianAgent.run`` returns a flat list of per-paper passages;
        ``format_evidence`` wants ``{"evidence": [...]}``.
        """
        return {"evidence": _evidence_entries(agent.run(query))}

    return search


def build_solver(args: argparse.Namespace):
    """Construct the baseline or knowledge-layer solver from CLI args."""
    if args.mode == "baseline":
        return BaselineSolver(
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
        )

    return KnowledgeLayerSolver(
        search_fn=build_search_fn(args),
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        char_budget=args.evidence_char_budget,
        grounding=args.grounding,
        parametric_fallback=args.parametric_fallback,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )


_thread_local = threading.local()


def _solver_for_thread(args: argparse.Namespace):
    """One solver (and its agent) per worker thread.

    The knowledge-layer agent keeps per-run state on ``self``, so it must not be
    shared across threads. ThreadPoolExecutor reuses threads, so at most
    ``--max-workers`` agents are built in total.
    """
    solver = getattr(_thread_local, "solver", None)
    if solver is None:
        solver = build_solver(args)
        _thread_local.solver = solver
    return solver


def _answer_question(args: argparse.Namespace, question) -> dict[str, Any]:
    """Run one MCQ end-to-end and return its result row (thread-safe)."""
    solver = _solver_for_thread(args)
    prompt = build_mcq_prompt(question.question, question.choices)
    retrieval_query = build_retrieval_query(
        question.retrieval_question, strip_sequences=args.strip_query_sequences
    )
    try:
        # Retrieve on the (sequence-stripped) question; the MCQ prompt the model
        # answers keeps the full sequence. (Baseline ignores the retrieval query.)
        result = solver.answer(prompt, retrieval_query=retrieval_query)
        raw_output = result["raw_output"]
        metadata = result["metadata"]
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.exception("Question %s failed.", question.id)
        raw_output = ""
        metadata = {"error": str(exc), "mode": args.mode}

    errored = bool(metadata.get("error"))
    predicted = parse_answer(raw_output, len(question.choices))
    correct, sure = score_prediction(
        predicted, question.answer_letter, question.unsure_letter
    )
    return {
        "id": question.id,
        "task": args.task,
        "mode": args.mode,
        "model": args.model,
        "question": question.question[:500],
        "choices": question.choices,
        "target_choice": question.answer_letter,
        "unsure_choice": question.unsure_letter,
        "predicted_choice": predicted,
        "correct": correct,
        "sure": sure,
        "parse_failure": predicted is None,
        # An infra failure (rate limit, timeout, network) rather than a model
        # answer: excluded from metrics, and re-run by --resume.
        "error": errored,
        "raw_output": raw_output,
        "metadata": metadata,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.md"
    run_config_path = out_dir / "run_config.json"

    questions = load_questions(
        args.task,
        data_file=args.data_file,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    if not questions:
        raise SystemExit(f"No {args.task} questions loaded.")
    LOGGER.info("Loaded %d %s question(s).", len(questions), args.task)

    # Resume: keep only successfully-answered rows, and drop errored ones so
    # they are re-run.
    rows: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if args.resume and predictions_path.exists():
        kept: list[dict[str, Any]] = []
        dropped = 0
        # Split on "\n" only, never str.splitlines(): rows are written with
        # ensure_ascii=False, which leaves U+2028/U+2029/U+0085 raw inside JSON
        # strings, and splitlines() treats those as line breaks, shredding one
        # row into several.
        for line in predictions_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                dropped += 1
                continue
            kept.append(r)
            existing_ids.add(str(r["id"]))
        # Refuse a resume into rows answered by a different model, before
        # anything on disk is rewritten.
        prior_models = {str(r["model"]) for r in kept if r.get("model")}
        if prior_models and str(args.model) not in prior_models:
            raise SystemExit(
                f"Refusing to resume {out_dir}: its rows were answered by "
                f"{sorted(prior_models)}, but --model is {args.model!r}. "
                "Use a fresh --out-dir."
            )
        rows = kept
        # Rewrite the file without the errored rows so they re-run cleanly.
        predictions_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            encoding="utf-8",
        )
        LOGGER.info(
            "Resume: %d already scored, %d errored row(s) will be re-run.",
            len(kept),
            dropped,
        )
    else:
        predictions_path.write_text("", encoding="utf-8")

    pending = [q for q in questions if q.id not in existing_ids]
    write_lock = threading.Lock()

    progress = _make_progress_bar(len(questions), not args.no_progress, desc=args.task)
    if progress is not None and len(pending) < len(questions):
        progress.update(len(questions) - len(pending))  # account for resumed rows
    try:
        with predictions_path.open("a", encoding="utf-8") as out_handle:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(_answer_question, args, q): q for q in pending}
                for future in as_completed(futures):
                    # solver.answer failures are captured into the row; only an
                    # agent-construction failure propagates here.
                    row = future.result()
                    with write_lock:
                        rows.append(row)
                        out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_handle.flush()
                        LOGGER.info(
                            "[%d/%d] %s pred=%s gold=%s correct=%s sure=%s queries=%d",
                            len(rows),
                            len(questions),
                            row["id"],
                            row["predicted_choice"],
                            row["target_choice"],
                            row["correct"],
                            row["sure"],
                            len(row["metadata"].get("literature_queries", [])),
                        )
                        if progress is not None:
                            progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    scored_rows = [r for r in rows if not r.get("error")]
    n_errors = len(rows) - len(scored_rows)
    metrics = compute_metrics(scored_rows)
    metrics["n_errors"] = n_errors
    # Full dataset size; every rate above is computed over scored_rows only.
    metrics["n_questions"] = len(questions)
    if n_errors:
        LOGGER.warning(
            "%d question(s) failed (rate limit / network) and were EXCLUDED from "
            "metrics. Re-run with --resume to retry just those.",
            n_errors,
        )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_path.write_text(render_summary(args, metrics), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "mode": args.mode,
                "model": args.model,
                "base_url": args.llm_base_url,
                "grounding": args.grounding,
                "parametric_fallback": args.parametric_fallback,
                "agent_model": args.agent_model,
                "agent_base_url": args.agent_base_url,
                "strip_query_sequences": args.strip_query_sequences,
                "full_text_enrichment": args.full_text,
                "evidence_char_budget": args.evidence_char_budget,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "seed": args.seed,
                "n_examples": len(rows),
                "data_file": str(args.data_file) if args.data_file else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Done. task=%s accuracy=%.3f precision=%.3f coverage=%.3f (n=%d)",
        args.task,
        metrics["accuracy"],
        metrics["precision"],
        metrics["coverage"],
        metrics["n_total"],
    )
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }


def render_summary(args: argparse.Namespace, metrics: dict[str, Any]) -> str:
    return (
        f"# {args.task} evaluation (LAB-Bench)\n\n"
        f"- Task: `{args.task}` — {TASKS[args.task].description}\n"
        f"- Mode: `{args.mode}`\n"
        f"- Answering model: `{args.model}`\n"
        + (
            f"- Knowledge layer: `librarian` agent"
            f" (LLM: `{args.agent_model or 'default'}`)\n"
            if args.mode == "knowledge"
            else ""
        )
        + f"- Questions: {metrics['n_total']}"
        + (
            f" of {metrics['n_questions']} — **{metrics['n_errors']} question(s)"
            " errored and are EXCLUDED from the rates below. Re-run with"
            " `--resume`.**"
            if metrics.get("n_errors")
            else ""
        )
        + "\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Accuracy | {metrics['accuracy']:.3f} |\n"
        f"| Precision | {metrics['precision']:.3f} |\n"
        f"| Coverage | {metrics['coverage']:.3f} |\n"
        f"| Correct | {metrics['n_correct']} |\n"
        f"| Answered (sure) | {metrics['n_sure']} |\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        required=True,
        help="Which LAB-Bench subtask to run (DbQA, SeqQA, ProtocolQA, "
        "CloningScenarios, TableQA).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "knowledge", "librarian"],
        required=True,
        help="baseline = parametric LLM; knowledge = LLM grounded in our knowledge "
        "layer. ('librarian' is a deprecated alias of 'knowledge'.)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/{Task}_{mode}_{model}[__{agent}].",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-05-13",
        help="Answering model id. Default pins the gpt-4o snapshot the LAB-Bench "
        "paper's `gpt-4o` alias resolved to (released 2024-05-13).",
    )
    parser.add_argument(
        "--base-url",
        "--llm-base-url",
        dest="llm_base_url",
        default=None,
        help="OpenAI-compatible base URL for the answering model.",
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env)."
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="OpenAI client retries. Raise this if you hit 429 rate limits — the "
        "SDK honours Retry-After and waits out the per-minute (TPM) window.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout (seconds) for the answering model.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Choice-shuffle seed (use the same for both modes).",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Local <task>-v1-public.jsonl. Omit to load from HuggingFace.",
    )
    # Knowledge-layer options (the retrieval agent's own LLM and retrieval knobs).
    parser.add_argument(
        "--agent-model",
        "--librarian-model",
        dest="agent_model",
        default=None,
        help="LLM the knowledge-layer agent uses internally (query planning, "
        "relevance filtering). Separate from the answering --model.",
    )
    parser.add_argument(
        "--agent-base-url",
        "--librarian-base-url",
        dest="agent_base_url",
        default=None,
        help="OpenAI-compatible base URL for the knowledge-layer agent's LLM.",
    )
    parser.add_argument(
        "--strip-query-sequences",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Strip long DNA/RNA sequence runs from the literature *search query* "
        "(the answering prompt keeps the full sequence). Default depends on the "
        "task: on for SeqQA and CloningScenarios, off otherwise.",
    )
    parser.add_argument(
        "--evidence-char-budget",
        type=int,
        default=1500,
        help="Max characters of raw passage text passed to the LLM per paper.",
    )
    parser.add_argument(
        "--grounding",
        choices=["augment", "strict"],
        default="augment",
        help="How the model uses the evidence: 'augment' (default) treats it as "
        "helpful context and still reasons to an answer; 'strict' allows ONLY the "
        "evidence and refuses ('Insufficient information') when it is unsupported.",
    )
    parser.add_argument(
        "--parametric-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When retrieval returns no evidence, answer from the plain MCQ "
        "(baseline) instead of injecting an empty-evidence block. On by default; "
        "use --no-parametric-fallback to always inject the evidence block.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-k passages the librarian returns."
    )
    parser.add_argument(
        "--full-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Full-text enrichment (on by default; use --no-full-text to disable).",
    )
    parser.add_argument(
        "--via-api",
        action="store_true",
        help=(
            "Retrieve through the orchestrator (POST /run-agent/stream, "
            "agent=librarian) instead of an in-process LibrarianAgent."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("LIBRARIAN_API_URL", "http://localhost:8080"),
        help="Orchestrator API base for --via-api (env: LIBRARIAN_API_URL).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose librarian logging."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Questions answered concurrently. Each worker gets its own agent, "
        "and Europe PMC is queried live on every question, so raise this only as "
        "far as that API tolerates.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.mode == "librarian":
        # Back-compat: the old --mode librarian meant "use the knowledge layer".
        args.mode = "knowledge"
    if args.strip_query_sequences is None:
        # Fall back to the task's default when the flag is not given.
        args.strip_query_sequences = TASKS[args.task].strip_query_sequences
    # --via-api uses the librarian the server was deployed with, so the per-run
    # knobs below cannot apply to it.
    if args.via_api:
        conflicts = [
            flag
            for flag, is_set in (
                ("--agent-model", args.agent_model is not None),
                ("--agent-base-url", args.agent_base_url is not None),
                ("--no-full-text", not args.full_text),
            )
            if is_set
        ]
        if conflicts:
            parser.error(f"--via-api cannot be combined with: {', '.join(conflicts)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.out_dir is None:
        args.out_dir = str(_auto_out_dir(args))
        LOGGER.info("Auto out-dir: %s", args.out_dir)
    result = run_evaluation(args)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

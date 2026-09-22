"""SQA bench evaluation using the new two-agent literature stack.

Drives SynthesisAgent (summarisation via GPT-4o or any model) over
LibrarianAgent (retrieval via a self-hosted vLLM endpoint, or any other model),
so the two agents can use different models independently. Synthesis runs through
NumberedCitationSynthesisAgent, because Citation F1 is scored on [n]-style
citations -- see numbered_citations.py.

Output format is the predictions-file shape the AutoAIS/citation-eval scripts
under ``code/scripts/`` read.

Example — synthesis on GPT-4o, librarian on the local vLLM GLM-5 model:

    python evals/Literature/SQA_bench/run_sqa_new_stack.py \\
        --bench bio \\
        --synthesis-open-ai \\
        --synthesis-model gpt-4o

The librarian then reads LLM_BASE_URL / LLM_MODEL from the environment (or the
librarian/config.toml) to pick up the local model automatically.
"""

import argparse
import datetime
import glob as _glob
import json
import os
import random
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


from librarian.config import load_runtime_config  # noqa: E402
from evals.Literature import orchestrator_client  # noqa: E402
from librarian.agent import LibrarianAgent  # noqa: E402
from evals.Literature.SQA_bench.numbered_citations import (  # noqa: E402
    NumberedCitationSynthesisAgent,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BENCHMARKS = ["bio", "neuro", "cs", "multi"]

parser = argparse.ArgumentParser(
    description="Run SQA bench with the new librarian + synthesis agent stack."
)
parser.add_argument(
    "--bench",
    choices=BENCHMARKS,
    default="bio",
    help="Which ScholarQA benchmark split to run (default: bio).",
)
parser.add_argument(
    "--data-file",
    default=None,
    help="Override the benchmark data file path (JSON/JSONL).",
)

# Synthesis-side model config
parser.add_argument(
    "--synthesis-open-ai",
    action="store_true",
    help="Route synthesis calls through the OpenAI API (sets base URL to api.openai.com).",
)
parser.add_argument(
    "--synthesis-model",
    default=None,
    help="Model name for the synthesis agent. Defaults to gpt-4o when --synthesis-open-ai is set.",
)
parser.add_argument(
    "--synthesis-base-url",
    default=None,
    help="Explicit base URL for the synthesis agent (overrides --synthesis-open-ai).",
)

# Librarian-side model config (independent of synthesis)
parser.add_argument(
    "--librarian-model",
    default=None,
    help=(
        "Model name for the librarian agent. "
        "Defaults to librarian/config.toml / LLM_MODEL env var (i.e. the local model)."
    ),
)
parser.add_argument(
    "--librarian-base-url",
    default=None,
    help=(
        "Base URL for the librarian agent. "
        "Defaults to librarian/config.toml / LLM_BASE_URL env var."
    ),
)

parser.add_argument(
    "--thinking",
    action="store_true",
    help="Enable thinking/reasoning mode on the synthesis agent.",
)
parser.add_argument(
    "--librarian-num-subqueries",
    type=int,
    default=None,
    help="Override the librarian's num_subqueries. Default: the librarian/config.toml value.",
)
parser.add_argument(
    "--librarian-paragraphs-per-subquery",
    type=int,
    default=None,
    help="Override the librarian's paragraphs_per_subquery. Default: the librarian/config.toml value.",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=None,
    help="Passages per question for the --bm25_retrieval path only (default 20). "
    "The librarian returns a variable-size Stage-3 set and ignores this.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip questions that already have an output in a previous predictions file.",
)
parser.add_argument(
    "--retry",
    action="store_true",
    help="Re-run only questions that previously failed (reads *_errors.json).",
)
parser.add_argument(
    "--max-retries",
    type=int,
    default=2,
    help="Auto-retry failed questions up to N times (default: 2).",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Only process the first N questions (for testing).",
)
parser.add_argument(
    "--sample",
    type=float,
    default=None,
    help="Random sample as a fraction, e.g. 0.1 for 10%% (seeded for reproducibility).",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed for --sample (default: 42)."
)
parser.add_argument(
    "--max-workers",
    type=int,
    default=1,
    help=(
        "Questions answered concurrently (default: 1 — serial). "
        "Each worker gets its own agent; their vLLM calls are continuous-batched. "
        "Europe PMC is queried live on every question; raise it only as far as "
        "that API tolerates, or it starts rate-limiting."
    ),
)
parser.add_argument(
    "--via-api",
    action="store_true",
    help=(
        "Run each question on the orchestrator (POST /run-agent/stream) instead "
        "of an in-process agent. The run lands on whichever replica has a free "
        "MAX_CONCURRENT_AGENTS slot, so Stage-2 BM25 uses pod CPUs and each "
        "question costs one round trip instead of one per LLM call. "
        "Incompatible with the librarian ablation flags."
    ),
)
parser.add_argument(
    "--api-base-url",
    default=os.environ.get("LIBRARIAN_API_URL", "http://localhost:8080"),
    help="Orchestrator API base for --via-api (env: LIBRARIAN_API_URL).",
)
parser.add_argument(
    "--api-timeout",
    type=float,
    default=1800.0,
    help="Read timeout in seconds for one --via-api question (default: 1800).",
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable agent verbose logging."
)
parser.add_argument(
    "--skip-citation-eval",
    action="store_true",
    help="Skip the AutoAIS citation scoring step after predictions are saved.",
)
parser.add_argument(
    "--autoais-chunk-size",
    type=int,
    default=200,
    help="Chunk size for AutoAIS (default: 200). Larger = more GPU memory.",
)
parser.add_argument(
    "--no-librarian",
    action="store_true",
    help=(
        "Skip retrieval entirely and answer each question directly from the "
        "synthesis model's own knowledge (LLM-only baseline)."
    ),
)
parser.add_argument(
    "--no-librarian-full-text",
    action="store_true",
    help=(
        "Disable librarian full-text enrichment (abstract-only retrieval); the "
        "abstract-vs-full-text ablation arm."
    ),
)
parser.add_argument(
    "--bm25-retrieval",
    action="store_true",
    help=(
        "Pure BM25 baseline: retrieve passages from OSDS Elasticsearch "
        "(no LLM query generation, no relevance filter), then synthesize."
    ),
)
parser.add_argument(
    "--es-fulltext-url",
    default="http://localhost:9202",
    help="Elasticsearch URL for the fulltext chunks index (default: http://localhost:9202).",
)
parser.add_argument(
    "--es-url",
    default="http://localhost:9201",
    help="Elasticsearch URL for the abstracts index (default: http://localhost:9201).",
)
args = parser.parse_args()

# The orchestrator runs whatever librarian config its pods were deployed with,
# so no per-run knob can be honoured on that path. Fail now rather than report
# numbers for an ablation arm that silently never ran.
if args.via_api:
    _api_conflicts = [
        flag
        for flag, is_set in (
            ("--no-librarian", args.no_librarian),
            ("--bm25-retrieval", args.bm25_retrieval),
            ("--no-librarian-full-text", args.no_librarian_full_text),
            ("--librarian-num-subqueries", args.librarian_num_subqueries is not None),
            (
                "--librarian-paragraphs-per-subquery",
                args.librarian_paragraphs_per_subquery is not None,
            ),
        )
        if is_set
    ]
    if _api_conflicts:
        parser.error(f"--via-api cannot be combined with: {', '.join(_api_conflicts)}")

# ---------------------------------------------------------------------------
# Resolve model / base-URL for each agent
# ---------------------------------------------------------------------------
synthesis_base_url: str | None
if args.synthesis_base_url:
    synthesis_base_url = args.synthesis_base_url
elif args.synthesis_open_ai:
    synthesis_base_url = "https://api.openai.com/v1"
else:
    synthesis_base_url = None

synthesis_model: str | None
if args.synthesis_model:
    synthesis_model = args.synthesis_model
elif args.synthesis_open_ai:
    synthesis_model = "gpt-4o"
else:
    synthesis_model = None

librarian_base_url: str | None = args.librarian_base_url
librarian_model: str | None = args.librarian_model

# Ablation-sweep overrides for individual librarian knobs (e.g.
# --librarian-num-subqueries), applied as a dataclasses.replace() on top of
# librarian/config.toml — no env vars.
_librarian_overrides = {
    k: v
    for k, v in {
        "num_subqueries": args.librarian_num_subqueries,
        "paragraphs_per_subquery": args.librarian_paragraphs_per_subquery,
    }.items()
    if v is not None
}
librarian_runtime_config = load_runtime_config()
if _librarian_overrides:
    librarian_runtime_config = replace(librarian_runtime_config, **_librarian_overrides)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def _sanitize(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


# --via-api ignores --synthesis-model/--librarian-model entirely: the pods run
# their own deployed model. Tagging the file with the flags anyway produced
# predictions named `lib-glm-5-fp8` that were actually glm-5.3-flash, and made
# --resume reuse in-process answers for an API run (and vice versa), silently
# mixing two retrievers in one predictions file. "api" is what we can honestly
# say from here -- the serving model is the pods', not ours to name.
if args.via_api:
    synth_tag = lib_tag = "api"
else:
    synth_tag = _sanitize(synthesis_model or "default")
    if args.no_librarian:
        lib_tag = "none"
    elif args.bm25_retrieval:
        lib_tag = "bm25-osds"
    else:
        lib_tag = _sanitize(librarian_model or "default")
model_file_tag = f"synth-{synth_tag}_lib-{lib_tag}"

OUTPUT_DIR = os.environ.get(
    "SCHOLARQA_OUTPUT_DIR",
    str(
        PROJECT_ROOT
        / "evals"
        / "Literature"
        / "SQA_bench"
        / "output"
        / "new_stack"
        / f"scholarqa_{args.bench}"
    ),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
predictions_path = os.path.join(OUTPUT_DIR, f"pred_{model_file_tag}_wip.json")
errors_path = os.path.join(OUTPUT_DIR, f"pred_{model_file_tag}_wip_errors.json")

# ---------------------------------------------------------------------------
# Load benchmark data
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "SCHOLARQA_DATA_ROOT",
    str(PROJECT_ROOT / "evals" / "Literature" / "SQA_bench" / "data"),
)
BENCH_CONFIG = {
    "bio": {"file": "scholarqabench_bio.jsonl", "format": "jsonl", "qfield": "input"},
    "neuro": {
        "file": "scholarqabench_neuro.jsonl",
        "format": "jsonl",
        "qfield": "input",
    },
    "cs": {"file": "qa_metadata_all.jsonl", "format": "jsonl", "qfield": "question"},
    "multi": {"file": "human_answers.json", "format": "json", "qfield": "input"},
}

cfg = BENCH_CONFIG[args.bench]
if args.data_file:
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
else:
    data_file = Path(DATA_ROOT) / f"scholarqa_{args.bench}" / cfg["file"]

if not data_file.exists():
    print(f"ERROR: benchmark file not found: {data_file}")
    sys.exit(1)

data_format = cfg["format"]
if args.data_file:
    data_format = "jsonl" if str(data_file).endswith(".jsonl") else "json"

with open(data_file) as f:
    raw_questions = (
        [json.loads(line) for line in f if line.strip()]
        if data_format == "jsonl"
        else json.load(f)
    )

all_questions = []
for i, q in enumerate(raw_questions):
    norm = dict(q)
    if cfg["qfield"] != "input":
        norm["input"] = q[cfg["qfield"]]
    if "id" not in norm:
        norm["id"] = norm.get("idx", f"{args.bench}_{i}")
    all_questions.append(norm)

if args.sample:
    rng = random.Random(args.seed)
    k = max(1, int(len(all_questions) * args.sample))
    all_questions = rng.sample(all_questions, k)
    print(f"SAMPLE MODE: {k}/{len(raw_questions)} questions (seed={args.seed})")

if args.limit:
    all_questions = all_questions[: args.limit]

print(f"Benchmark:  scholarqa_{args.bench}")
print(f"Data file:  {data_file}")
print(f"Questions:  {len(all_questions)}")

# ---------------------------------------------------------------------------
# Helpers needed before the run loop
# ---------------------------------------------------------------------------


def _is_empty_result(entry: dict) -> bool:
    """True when the agent searched but found nothing — a soft failure to retry.

    Works on both the raw agent result (key: 'passages', a list) and a saved
    prediction entry (key: 'passage_count', an int).
    """
    if entry.get("action") != "search":
        return False
    passages = entry.get("passages")
    if passages is not None:
        return len(passages) == 0
    return entry.get("passage_count", 0) == 0


# ---------------------------------------------------------------------------
# Resume / retry state
# ---------------------------------------------------------------------------
def _find_pred_file(out_dir, tag):
    wip = os.path.join(out_dir, f"pred_{tag}_wip.json")
    if os.path.exists(wip):
        return wip
    pattern = os.path.join(out_dir, f"pred_{tag}_[0-9]*.json")
    candidates = [
        f for f in _glob.glob(pattern) if "_errors" not in f and ".autoais_" not in f
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _load_predictions(path):
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass
    # partial recovery
    with open(path) as f:
        raw = f.read()
    decoder = json.JSONDecoder()
    items, idx = [], raw.find("[") + 1
    while idx < len(raw):
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw) or raw[idx] == "]":
            break
        try:
            entry, idx = decoder.raw_decode(raw, idx)
            if isinstance(entry, dict):
                items.append(entry)
        except json.JSONDecodeError:
            break
        while idx < len(raw) and raw[idx] in " \t\n,":
            idx += 1
    return items


if args.retry:
    pred_file = _find_pred_file(OUTPUT_DIR, model_file_tag)
    err_file = pred_file.replace(".json", "_errors.json") if pred_file else None
    if not pred_file or not (err_file and os.path.exists(err_file)):
        print("ERROR: --retry requires existing predictions and errors files.")
        sys.exit(1)
    with open(pred_file) as f:
        predictions = json.load(f)
    with open(err_file) as f:
        prev_errors = json.load(f)
    failed_inputs = {e["input"] for e in prev_errors}
    questions_to_run = [q for q in all_questions if q["input"] in failed_inputs]
    pred_by_input = {p["input"]: i for i, p in enumerate(predictions)}
    answered_inputs = set()
    print(f"RETRY MODE: {len(questions_to_run)} questions")
elif args.resume:
    pred_file = _find_pred_file(OUTPUT_DIR, model_file_tag)
    predictions = _load_predictions(pred_file) if pred_file else []
    if pred_file:
        print(f"Resuming from: {pred_file}")
    answered_inputs = {
        p["input"] for p in predictions if p.get("output") and not _is_empty_result(p)
    }
    questions_to_run = [q for q in all_questions if q["input"] not in answered_inputs]
    pred_by_input = {p["input"]: i for i, p in enumerate(predictions)}
    print(
        f"RESUME MODE: {len(answered_inputs)} done, {len(questions_to_run)} remaining."
    )
else:
    questions_to_run = all_questions
    predictions = []
    pred_by_input = {}
    answered_inputs = set()


# ---------------------------------------------------------------------------
# BM25-only retrieval (no LLM in the retrieval loop)
# ---------------------------------------------------------------------------
def _run_bm25(question: str) -> dict:
    """The OpenScholar-style BM25 baseline, which this repository does not ship.

    It read ~250-word chunks straight out of an OpenScholar datastore (OSDS)
    Elasticsearch index, with no paper-level aggregation and no LLM filter. That
    index and its retriever are internal infrastructure, not part of the
    librarian, so only the +Librarian row is reproducible here.
    """
    raise RuntimeError(
        "--bm25-retrieval needs an OpenScholar datastore Elasticsearch index, "
        "which is not part of this repository. Drop the flag to run the "
        "librarian (what SQA_bench/run_sqa.sh does)."
    )


# ---------------------------------------------------------------------------
# Agent factory (one per worker thread via thread-local storage)
# ---------------------------------------------------------------------------
# Both agents hold an HTTP client and the librarian keeps per-run state, so a
# pair is built per worker thread rather than shared. With max_workers=1
# (the default) that is one pair for the whole run.
_thread_local = threading.local()


class _LibrarianSynthesisPipeline:
    """Retrieve with the librarian, then write a cited answer over what it found.

    The two agents are separate in this repository -- ``LibrarianAgent`` returns
    ranked passages and ``SynthesisAgent`` writes over them -- so this composes
    them into the one ``{"action", "summary", "passages"}`` record the rest of
    this script (``validate_result``, ``build_prediction_entry``) reads, which is
    also what the ``--via-api`` transport returns.

    There is no router: SQA questions are always literature questions, so every
    one retrieves. That is the ``action: "search"`` branch an internal router
    would have picked anyway, and it is what the paper's row measured.
    """

    def __init__(self) -> None:
        self.librarian = LibrarianAgent(
            runtime_config=librarian_runtime_config,
            llm_base_url=librarian_base_url,
            llm_model_name=librarian_model,
            full_text_enrichment=not args.no_librarian_full_text,
            verbose=args.verbose,
        )
        # Numbered citations, NOT the shipped agent's author-year links: the
        # AutoAIS scorer only extracts [n] / [n.k] forms, so an author-year
        # answer scores as having no citations at all. See numbered_citations.py.
        self.synthesis = NumberedCitationSynthesisAgent(
            llm_base_url=synthesis_base_url,
            llm_model_name=synthesis_model,
            verbose=args.verbose,
        )

    @property
    def llm(self):
        """The synthesis client, which is the one ``--no-librarian`` answers on."""
        return self.synthesis.llm

    def _summarize(self, question: str, passages: list) -> str:
        return self.synthesis.run(question, passages)

    def run(self, question: str) -> dict:
        passages = self.librarian.run(question)
        return {
            "action": "search",
            "summary": self.synthesis.run(question, passages),
            "passages": passages,
        }


def _agent_for_thread() -> _LibrarianSynthesisPipeline:
    if getattr(_thread_local, "agent", None) is None:
        _thread_local.agent = _LibrarianSynthesisPipeline()
    return _thread_local.agent


# ---------------------------------------------------------------------------
# API transport (--via-api)
# ---------------------------------------------------------------------------
# The alternative to the in-process agents above; the rationale and the
# concurrency measurements live in evals/Literature/orchestrator_client.py.
# SQA asks for `literature_synthesis` rather than `librarian` because citation
# F1 is scored on the written summary, which the librarian agent never produces.
def _run_via_api(question: str) -> dict:
    """Answer one question through the orchestrator instead of in-process.

    The wire contract drops ``passages`` and ``evidence`` (they restate
    ``papers``), so ``papers`` is returned under the ``passages`` key that
    :func:`build_prediction_entry` and :func:`validate_result` already read --
    it carries the same ``title`` / ``year`` / ``evidence_snippets`` they use.

    :param question: The benchmark question text.
    :returns: ``{"action", "summary", "passages"}``, shaped like ``run()``.
    """
    results = orchestrator_client.run_agent(
        question,
        agent="literature_synthesis",
        base_url=args.api_base_url,
        source="sqa-eval",
        session_prefix=f"sqa-{args.bench}",
        read_timeout=args.api_timeout,
    )
    return {
        "action": results.get("action", "search"),
        "summary": results.get("summary", ""),
        "passages": results.get("papers") or [],
    }


print(
    f"Transport:  {'orchestrator API ' + args.api_base_url if args.via_api else 'in-process agents'}"
)
print(f"Workers:    {args.max_workers}")
print(f"Output dir: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path, payload):
    out_dir = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=out_dir
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_prediction_entry(question_input: str, result: dict) -> dict:
    passages = result.get("passages", [])
    ctxs = [
        {
            "title": p.get("title", ""),
            "text": " ".join(p.get("evidence_snippets") or []).strip(),
            "year": p.get("year", ""),
        }
        for p in passages
    ]
    return {
        "input": question_input,
        "output": result.get("summary", ""),
        "action": result.get("action", ""),
        "ctxs": ctxs,
        "passage_count": len(passages),
    }


def validate_result(
    result: dict, question_input: str, require_passages: bool = True
) -> None:
    summary = result.get("summary", "") if isinstance(result, dict) else ""
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError(f"Empty summary for: {question_input[:150]}")
    if require_passages and _is_empty_result(result):
        raise RuntimeError(f"No passages retrieved for: {question_input[:150]}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
in_docker = os.path.exists("/.dockerenv")
errors = []
total = len(questions_to_run)
overall_total = len(all_questions)
run_start = time.time()
write_lock = threading.Lock()
# Seed counters from already-loaded predictions so the postfix shows totals,
# not just this session's counts. answered_inputs holds the good ones;
# anything loaded but not in answered_inputs was a soft/hard failure.
n_success = len(answered_inputs)
n_errors = len(predictions) - len(answered_inputs)

progress_bar = None
if not in_docker:
    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=overall_total,
            initial=min(len(answered_inputs), overall_total),
            desc=f"SQA new-stack: {args.bench}",
            unit="q",
            dynamic_ncols=True,
        )
    except ImportError:
        pass


def _run_one(q: dict) -> None:
    """Run a single question and append the result under the write lock."""
    global n_success, n_errors
    try:
        if args.no_librarian:
            answer = _agent_for_thread().llm.chat_completion(
                [{"role": "user", "content": q["input"]}],
                temperature=0.5,
                max_tokens=8192,
            )
            result = {"action": "reply", "summary": answer, "passages": []}
        elif args.bm25_retrieval:
            result = _run_bm25(q["input"])
        elif args.via_api:
            result = _run_via_api(q["input"])
        else:
            result = _agent_for_thread().run(q["input"])
        validate_result(result, q["input"], require_passages=not args.no_librarian)
        entry = build_prediction_entry(q["input"], result)
        with write_lock:
            if (args.retry or args.resume) and q["input"] in pred_by_input:
                predictions[pred_by_input[q["input"]]] = entry
            else:
                predictions.append(entry)
            n_success += 1
            if in_docker:
                print(
                    f"  [{n_success + n_errors}/{total}] action={result['action']} passages={len(result.get('passages', []))}",
                    flush=True,
                )
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(success=n_success, errors=n_errors)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ERROR ({q['input'][:60]}): {e}\n{tb}")
        with write_lock:
            errors.append(
                {
                    "input": q["input"],
                    "id": q.get("id", ""),
                    "error": str(e),
                    "traceback": tb,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            if not args.retry and not (args.resume and q["input"] in pred_by_input):
                predictions.append(
                    {
                        "input": q["input"],
                        "output": "",
                        "action": "",
                        "ctxs": [],
                        "passage_count": 0,
                    }
                )
            n_errors += 1
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(success=n_success, errors=n_errors)


with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futures = {pool.submit(_run_one, q): q for q in questions_to_run}
    for i, future in enumerate(as_completed(futures), 1):
        future.result()  # re-raise any agent-construction failure (fatal)
        if i % 5 == 0 or i == total:
            with write_lock:
                _atomic_write_json(predictions_path, predictions)

# ---------------------------------------------------------------------------
# Auto-retry
# ---------------------------------------------------------------------------
for retry_round in range(1, args.max_retries + 1):
    if not errors:
        break
    print(
        f"\n{'=' * 60}\nAUTO-RETRY {retry_round}/{args.max_retries}: {len(errors)} questions\n{'=' * 60}"
    )
    time.sleep(5)

    still_failing = []
    for j, err_entry in enumerate(errors):
        q_input = err_entry["input"]
        print(f"  Retrying [{j + 1}/{len(errors)}]: {q_input[:80]}...")
        try:
            if args.no_librarian:
                answer = _agent_for_thread().llm.chat_completion(
                    [{"role": "user", "content": q_input}],
                    temperature=0.5,
                    max_tokens=8192,
                )
                result = {"action": "reply", "summary": answer, "passages": []}
            elif args.bm25_retrieval:
                result = _run_bm25(q_input)
            elif args.via_api:
                result = _run_via_api(q_input)
            else:
                result = _agent_for_thread().run(q_input)
            validate_result(result, q_input, require_passages=not args.no_librarian)
            entry = build_prediction_entry(q_input, result)
            for idx, p in enumerate(predictions):
                if p["input"] == q_input and not p["output"]:
                    predictions[idx] = entry
                    break
            print("  ✓ succeeded")
        except Exception as e:
            print(f"  ✗ failed: {e}")
            still_failing.append(
                {
                    "input": q_input,
                    "id": err_entry.get("id", ""),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "retry_round": retry_round,
                }
            )
    errors = still_failing
    _atomic_write_json(predictions_path, predictions)

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
if progress_bar is not None:
    progress_bar.close()

_atomic_write_json(predictions_path, predictions)
_atomic_write_json(errors_path, errors)

final_stem = f"pred_{model_file_tag}_{len(predictions)}_{run_ts}"
final_path = os.path.join(OUTPUT_DIR, f"{final_stem}.json")
final_errors_path = os.path.join(OUTPUT_DIR, f"{final_stem}_errors.json")
os.replace(predictions_path, final_path)
os.replace(errors_path, final_errors_path)

n_success = sum(1 for p in predictions if p.get("output"))
print(f"\n{'=' * 60}")
print("DONE")
print(f"  Total:       {len(predictions)}")
print(f"  Successful:  {n_success}")
print(f"  Failed:      {len(predictions) - n_success}")
print(f"  Predictions: {final_path}")
print(f"  Errors:      {final_errors_path}")
print(f"{'=' * 60}")

# ---------------------------------------------------------------------------
# Citation eval (AutoAIS)
# ---------------------------------------------------------------------------
if not args.skip_citation_eval:
    import subprocess

    citation_eval_script = str(
        PROJECT_ROOT
        / "evals"
        / "Literature"
        / "SQA_bench"
        / "code"
        / "scripts"
        / "citation_correctness_eval.py"
    )
    per_q_path = final_path + ".autoais_per_question.json"
    citation_cmd = [
        sys.executable,
        citation_eval_script,
        "--f",
        final_path,
        "--citations",
        "--autoais_chunk_size",
        str(args.autoais_chunk_size),
        "--autoais_reload_model_per_chunk",
        "--per_question_output",
        per_q_path,
    ]
    print(f"\n{'=' * 60}")
    print("CITATION EVAL (AutoAIS)")
    print(f"  Predictions: {final_path}")
    print(f"{'=' * 60}")
    completed = subprocess.run(
        citation_cmd,
        cwd=str(
            PROJECT_ROOT / "evals" / "Literature" / "SQA_bench" / "code" / "scripts"
        ),
    )
    if completed.returncode != 0:
        # The scoring stack is torch + transformers against your own CUDA, so it
        # is installed separately from the harness. Predictions are already on
        # disk; say how to score them rather than losing the run to a traceback.
        print(
            f"\nERROR: AutoAIS scoring failed (exit {completed.returncode}).\n"
            "       If the cause above is a missing torch/transformers, the local\n"
            "       scoring stack is not installed. It is deliberately separate --\n"
            "       it builds against your CUDA:\n\n"
            "         uv pip install -r evals/Literature/SQA_bench/code/requirements.txt\n\n"
            f"       The predictions are already written to\n         {final_path}\n"
            "       so re-running the same command with --resume scores them\n"
            "       without answering the questions again.",
            file=sys.stderr,
        )
        sys.exit(1)
    score_path = final_path + ".score_post_fix"
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)
        print(f"\n{'=' * 60}")
        print("CITATION SCORES")
        print(f"  F1:          {scores.get('citation_f1', 'N/A'):.2f}")
        print(f"  Recall:      {scores.get('citation_rec', 'N/A'):.2f}")
        print(f"  Precision:   {scores.get('citation_prec', 'N/A'):.2f}")
        print(f"  Score file:  {score_path}")
        print(f"  Per-question:{per_q_path}")
        print(f"{'=' * 60}")

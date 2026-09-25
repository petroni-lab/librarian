"""ProClaim-eval harness for the librarian-backed ProClaim verification pipeline.

`verifier.py` (arm 1) evaluates the librarian's retrieved evidence directly,
asking a model for a verdict. This harness instead drives the full ProClaim
verification pipeline — ``proclaim.verification.evidence_programming_direct`` —
with the librarian retrieval backend (``retrieval_backend: librarian``). Each
claim is verified by librarian search, sparse-evidence recovery, fact
extraction, and a final verdict. If the configured recovery budget ends with no
extracted facts, the final verdict is produced from the accumulated librarian
evidence records directly, using the same ProClaim label semantics as
``verifier.py``.

Per claim it runs ``evidence_programming_direct`` as a subprocess, then reads
the verdict and ``EvidenceState`` it persisted. It reports the same metrics as
``proclaim.py``. The librarian path uses a bounded sparse-evidence recovery
loop: it starts with one librarian search, then refines failed extractions until
at least one SUPPORT or REFUTE fact is available or ``max_iterations`` is
reached. The librarian configs set that budget to two iterations.

Outputs (in --out-dir): predictions.jsonl, metrics.json, confusion_matrix.csv,
summary.md, latex_table.tex, run_config.json, and per-claim run directories
under runs/<subset>/<id>/. Each claim workspace also stores
``librarian_evidence_by_search_iteration.json`` with retrieved evidence grouped
by librarian search iteration.

Usage:
    python -m evals.ProClaim.proclaim_librarian \
        --proclaim-path evals/ProClaim \
        --subset all \
        --max-examples 3 \
        --out-dir results/proclaim_librarian_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency.
    tqdm = None

from evals.Literature.ProClaim.verifier import (  # noqa: E402
    DEFAULT_FALLBACK_LABEL,
    ProClaimExample,
    ProClaimPrediction,
    compute_metrics_by_subset,
    load_proclaim_examples,
    normalize_label,
    render_confusion_matrix_csv,
    render_latex_table,
    render_summary_markdown,
    _append_jsonl,
    _load_prediction_rows,
    _prediction_row,
    _update_progress,
    _write_jsonl,
)

LOGGER = logging.getLogger("bioagents.eval.proclaim_librarian")

DEFAULT_PROCLAIM_SRC = (
    PROJECT_ROOT / "evals" / "Literature" / "ProClaim" / "ProClaim_src"
)
# Ours, in this repository — they select the librarian backend, which exists
# nowhere upstream, so they were never ProClaim's to carry. Relative to this
# file's directory, not to the clone.
DEFAULT_SUBSET_CONFIGS = {
    "signor": "configs/signor_librarian.yaml",
    "connectomedb": "configs/connectomedb_librarian.yaml",
}
DEFAULT_CLAIM_TIMEOUT_SECONDS = 1800


def _truncate_log_value(value: str, limit: int = 500) -> str:
    """Return a compact single-line value for logs."""
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _read_config_summary(path: Path) -> dict[str, Any]:
    """Read non-secret run-critical fields from a YAML config for logging."""
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"path": str(path), "read_error": str(exc)}

    llm = raw.get("llm") or {}
    return {
        "path": str(path),
        "retrieval_backend": raw.get("retrieval_backend"),
        "librarian_llm_model": raw.get("librarian_llm_model"),
        "librarian_llm_base_url": raw.get("librarian_llm_base_url"),
        "agent_model": llm.get("model"),
        "agent_base_url": llm.get("agent_base_url"),
        "subagent_model": llm.get("subagent_model"),
        "subagent_base_url": llm.get("subagent_base_url"),
    }


def _patch_config_endpoint(
    config_path: Path,
    out_dir: Path,
    base_url: str | None,
    model: str | None,
) -> Path:
    """Copy *config_path* with its librarian endpoint overridden.

    The full-pipeline arm retrieves in-process and takes its librarian endpoint
    from the YAML, which pins one host. That made the arm unrunnable whenever
    that host was not serving — there was no way to point it elsewhere without
    editing a checked-in file, unlike ``subagent_base_url`` which has always
    been overridable. The patched copy is written into the run directory rather
    than a temp path, so the config a run actually used stays with its results.

    :param config_path: The YAML the run would otherwise use.
    :param out_dir: Run directory; the copy lands in ``configs/`` beneath it.
    :param base_url: Replacement ``librarian_llm_base_url``, or None to keep.
    :param model: Replacement ``librarian_llm_model``, or None to keep.
    :returns: The patched copy, or *config_path* unchanged when no override.
    """
    if not base_url and not model:
        return config_path
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if base_url:
        raw["librarian_llm_base_url"] = base_url
    if model:
        raw["librarian_llm_model"] = model
    patched_dir = out_dir / "configs"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched = patched_dir / config_path.name
    patched.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    LOGGER.info(
        "Librarian endpoint overridden for %s: %s (%s)",
        config_path.name,
        base_url or raw.get("librarian_llm_base_url"),
        model or raw.get("librarian_llm_model"),
    )
    return patched


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from disk, returning None when absent or unreadable."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Could not read JSON: %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _safe_dir_name(value: str) -> str:
    """Turn an example id into a filesystem-safe run-directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "claim"


def _safe_int(value: Any, default: int = 0) -> int:
    """Parse an artifact integer without letting one bad field stop the run."""
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse an artifact float without letting one bad field stop the run."""
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_succeeded(row: dict[str, Any]) -> bool:
    """Return True when a saved prediction row is a durable success.

    A row is reusable on ``--resume`` only if the claim produced a real
    verdict. Rows from a subprocess crash, timeout, or missing/garbled verdict
    count as failures so ``--resume`` (with ``--retry-failed``) re-runs them.
    """
    if row.get("parse_failure"):
        return False
    metadata = row.get("metadata") or {}
    if metadata.get("parse_failure") or metadata.get("subprocess_failed"):
        return False
    return True


def _make_librarian_progress_bar(total: int, enabled: bool):
    """Create the librarian eval progress bar with an explicit colour."""
    if not enabled or tqdm is None:
        return None
    kwargs = {
        "total": total,
        "desc": "ProClaim+librarian",
        "unit": "claim",
        "dynamic_ncols": True,
    }
    try:
        return tqdm(**kwargs, colour="cyan")
    except TypeError:  # pragma: no cover - older tqdm versions.
        return tqdm(**kwargs)


def _live_progress_text(rows: list[dict[str, Any]], last_id: str | None = None) -> str:
    """Return a compact live metric string for tqdm/log updates."""
    if not rows:
        return "AGR=-- seen=0"
    metrics = compute_metrics_by_subset(rows).get("combined", {})
    seen = int(metrics.get("n") or 0)
    agr = metrics.get("agr")
    correct = sum(
        1 for row in rows if row.get("predicted_label") == row.get("gold_label")
    )
    failures = int(metrics.get("parse_failures") or 0)
    prefix = f"last={last_id} " if last_id else ""
    agr_text = "--" if agr is None else f"{float(agr):.3f}"
    return f"{prefix}AGR={agr_text} {correct}/{seen} failures={failures}"


def _prediction_from_run_dir(
    run_dir: Path,
    *,
    latency: float,
    launch_error: str,
) -> ProClaimPrediction:
    """Build a ProClaimPrediction from the artifacts of one verification run.

    Reads ``workspace/verdict.json`` for the verdict and ``evidence_state.json``
    for the retrieval pass count and the evidence gathered.
    """
    workspace = run_dir / "workspace"
    verdict = _read_json(workspace / "verdict.json")
    state = _read_json(workspace / "evidence_state.json") or {}
    run_error = _read_json(workspace / "run_error.json")
    token_usage = _read_json(run_dir / "token_usage.json")

    iterations = _safe_int(state.get("iteration"))
    papers = state.get("papers") or {}
    facts = state.get("facts") or []
    parse_failure = bool(launch_error)
    if verdict and verdict.get("verdict"):
        label = normalize_label(verdict.get("verdict"), fallback=None)
        if label is None:
            label = DEFAULT_FALLBACK_LABEL
            parse_failure = True
        reasoning = str(verdict.get("reasoning") or "")
        key_evidence = verdict.get("key_evidence") or []
        confidence = _safe_float(verdict.get("confidence"))
    else:
        label = DEFAULT_FALLBACK_LABEL
        reasoning = (
            (run_error or {}).get("message")
            or launch_error
            or "No verdict emitted by the verification pipeline."
        )
        key_evidence = []
        confidence = 0.0
        parse_failure = True

    citations = [{"evidence": str(item)} for item in key_evidence if item]

    metadata: dict[str, Any] = {
        "iterations": iterations,
        "retrieved_paper_count": len(papers) if isinstance(papers, dict) else 0,
        "fact_count": len(facts) if isinstance(facts, list) else 0,
        "confidence": round(confidence, 4),
        "latency_seconds": round(latency, 3),
        "inference_mode": "proclaim_librarian",
        "parse_failure": parse_failure,
        "run_dir": str(run_dir),
    }
    if token_usage:
        metadata["token_usage"] = token_usage
    if launch_error:
        metadata["error"] = launch_error
        metadata["subprocess_failed"] = True
    if run_error:
        metadata["run_error"] = run_error

    return ProClaimPrediction(
        predicted_label=label,
        reasoning=reasoning,
        citations=citations,
        raw_output=json.dumps(verdict or {}, ensure_ascii=False),
        metadata=metadata,
        parse_failure=parse_failure,
    )


def run_librarian_verification(
    example: ProClaimExample,
    *,
    proclaim_src: Path,
    config_path: Path,
    run_dir: Path,
    timeout: int,
    resume: bool,
) -> ProClaimPrediction:
    """Verify one claim by running evidence_programming_direct as a subprocess.

    On ``resume`` a run directory is reused only when it holds a ``verdict.json``
    (a successful completion); a directory with only ``run_error.json`` is
    re-run from scratch so transient failures are not cached.
    """
    workspace = run_dir / "workspace"
    completed = (workspace / "verdict.json").exists()

    started = time.perf_counter()
    launch_error = ""

    if resume and completed:
        LOGGER.info("Claim %s: reusing completed run at %s", example.id, run_dir)
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Our entry point, not ProClaim's. It takes the same arguments —
        # it reuses ProClaim's own CLI parser — but runs the librarian
        # one-shot path instead of upstream's PubMed dispatch, which is what
        # lets the clone stay pristine. See direct_entry.py.
        cmd = [
            sys.executable,
            "-m",
            "evals.Literature.ProClaim.direct_entry",
            "--config",
            str(config_path),
            "--claim",
            example.claim,
            "--output-dir",
            str(run_dir),
        ]
        env = os.environ.copy()
        env["RETRIEVAL_BACKEND"] = "librarian"
        # The subprocess runs with cwd inside the clone, and needs to import
        # two trees: proclaim (a src layout, so its source root rather than the
        # clone root) and this repository, which carries direct_entry and the
        # librarian backend. Neither is installed into the environment.
        parts = [str(proclaim_src / "src"), str(PROJECT_ROOT)]
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        (run_dir / "subprocess_command.json").write_text(
            json.dumps(
                {
                    "started_at": datetime.now().isoformat(),
                    "cwd": str(proclaim_src),
                    "timeout_seconds": timeout,
                    "command": cmd,
                    "retrieval_backend": env["RETRIEVAL_BACKEND"],
                    "pythonpath": env["PYTHONPATH"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info(
            "Claim %s: launching subprocess (subset=%s, config=%s, run_dir=%s)",
            example.id,
            example.subset,
            config_path,
            run_dir,
        )
        LOGGER.debug("Claim %s command: %s", example.id, " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(proclaim_src),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            (run_dir / "subprocess_stdout.log").write_text(
                proc.stdout or "", encoding="utf-8"
            )
            (run_dir / "subprocess_stderr.log").write_text(
                proc.stderr or "", encoding="utf-8"
            )
            LOGGER.debug(
                "Claim %s stdout tail: %s",
                example.id,
                _truncate_log_value((proc.stdout or "")[-1000:]),
            )
            LOGGER.debug(
                "Claim %s stderr tail: %s",
                example.id,
                _truncate_log_value((proc.stderr or "")[-1000:]),
            )
            if proc.returncode != 0:
                launch_error = (
                    f"subprocess exit {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[-600:]}"
                )
                LOGGER.error("Claim %s failed: %s", example.id, launch_error)
        except subprocess.TimeoutExpired:
            launch_error = f"timeout after {timeout}s"
            LOGGER.error("Claim %s timed out after %ss.", example.id, timeout)
        except Exception as exc:  # pragma: no cover - environment dependent
            launch_error = f"subprocess error: {exc}"
            LOGGER.exception("Claim %s subprocess error.", example.id)

    latency = time.perf_counter() - started
    prediction = _prediction_from_run_dir(
        run_dir, latency=latency, launch_error=launch_error
    )
    metadata = prediction.metadata or {}
    log_fn = LOGGER.warning if prediction.parse_failure else LOGGER.info
    log_fn(
        "Claim %s complete: label=%s confidence=%.4f papers=%s facts=%s "
        "iterations=%s latency=%.1fs parse_failure=%s run_dir=%s",
        example.id,
        prediction.predicted_label,
        float(metadata.get("confidence") or 0.0),
        metadata.get("retrieved_paper_count"),
        metadata.get("fact_count"),
        metadata.get("iterations"),
        latency,
        prediction.parse_failure,
        run_dir,
    )
    return prediction


def run_librarian_evaluation(
    examples: list[ProClaimExample],
    *,
    proclaim_src: Path,
    subset_configs: dict[str, Path],
    out_dir: Path,
    timeout: int,
    resume: bool,
    retry_failed: bool,
    keep_going: bool,
    progress: bool,
    workers: int = 1,
) -> dict[str, Any]:
    """Verify every example through the librarian pipeline and write artifacts."""
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    predictions_path = out_path / "predictions.jsonl"
    metrics_path = out_path / "metrics.json"
    confusion_path = out_path / "confusion_matrix.csv"
    summary_path = out_path / "summary.md"
    latex_path = out_path / "latex_table.tex"
    run_config_path = out_path / "run_config.json"
    log_path = out_path / "evaluation.log"

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == log_path
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)

    ordered_examples = list(examples)
    selected_ids = {example.id for example in ordered_examples}
    config_summaries = {
        subset: _read_config_summary(path) for subset, path in subset_configs.items()
    }
    LOGGER.info("Evaluation output directory: %s", out_path)
    LOGGER.info("Evaluation log: %s", log_path)
    LOGGER.info("Predictions file: %s", predictions_path)
    LOGGER.info("Claim timeout: %ss", timeout)
    LOGGER.info(
        "Resume=%s retry_failed=%s keep_going=%s progress=%s",
        resume,
        retry_failed,
        keep_going,
        progress,
    )
    for subset, summary in config_summaries.items():
        LOGGER.info("Config[%s]: %s", subset, json.dumps(summary, sort_keys=True))

    existing_rows = _load_prediction_rows(predictions_path) if resume else {}
    rows_by_id = {
        key: row
        for key, row in existing_rows.items()
        if key in selected_ids and (_row_succeeded(row) or not retry_failed)
    }

    if resume:
        retried = sum(1 for key in existing_rows if key in selected_ids) - len(
            rows_by_id
        )
        reused = len(rows_by_id)
        LOGGER.info(
            "Resume cache: loaded=%d reusable=%d re_run=%d selected=%d",
            len(existing_rows),
            reused,
            retried,
            len(selected_ids),
        )
        if retried:
            LOGGER.info("Resume: re-running %d previously-failed claim(s).", retried)
        # Rewrite predictions.jsonl with only the rows we keep; dropped failed
        # rows are re-appended when their claims are re-run.
        _write_jsonl(predictions_path, list(rows_by_id.values()))
    else:
        predictions_path.write_text("", encoding="utf-8")

    total = len(ordered_examples)
    progress_bar = _make_librarian_progress_bar(total, progress)
    write_lock = threading.Lock()
    abort_event = threading.Event()

    if progress_bar is not None and rows_by_id:
        progress_bar.set_postfix_str(
            _live_progress_text(list(rows_by_id.values())),
            refresh=False,
        )

    def _run_one(index_example: tuple[int, ProClaimExample]) -> None:
        index, example = index_example
        if abort_event.is_set():
            return
        if resume and example.id in rows_by_id:
            metadata = rows_by_id[example.id].get("metadata") or {}
            LOGGER.info(
                "[%d/%d] %s reused from predictions.jsonl: label=%s "
                "confidence=%s papers=%s facts=%s iterations=%s",
                index,
                total,
                example.id,
                rows_by_id[example.id].get("predicted_label"),
                metadata.get("confidence"),
                metadata.get("retrieved_paper_count"),
                metadata.get("fact_count"),
                metadata.get("iterations"),
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    _live_progress_text(
                        list(rows_by_id.values()),
                        last_id=example.id,
                    ),
                    refresh=False,
                )
            _update_progress(progress_bar)
            return

        if progress_bar is not None:
            progress_bar.set_postfix_str(f"running={example.id}", refresh=False)
        LOGGER.info(
            "[%d/%d] starting %s (%s): %s",
            index,
            total,
            example.id,
            example.subset,
            _truncate_log_value(example.claim, limit=240),
        )
        run_dir = out_path / "runs" / example.subset / _safe_dir_name(example.id)
        prediction = run_librarian_verification(
            example,
            proclaim_src=proclaim_src,
            config_path=subset_configs[example.subset],
            run_dir=run_dir,
            timeout=timeout,
            resume=resume,
        )
        row = _prediction_row(example, prediction)
        with write_lock:
            rows_by_id[example.id] = row
            _append_jsonl(predictions_path, row)
            live_progress = _live_progress_text(
                list(rows_by_id.values()),
                last_id=example.id,
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(live_progress, refresh=False)
        if prediction.parse_failure and not keep_going:
            abort_event.set()
        _update_progress(progress_bar)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, (i, ex)): ex
                for i, ex in enumerate(ordered_examples, start=1)
            }
            for future in as_completed(futures):
                future.result()  # re-raise any unexpected exception
        if abort_event.is_set():
            example_id = next(
                (
                    ex.id
                    for ex in ordered_examples
                    if rows_by_id.get(ex.id, {}).get("parse_failure")
                ),
                "unknown",
            )
            raise RuntimeError(
                f"Stopping after failed claim {example_id}. "
                "Use --keep-going to continue after failures."
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    ordered_rows = [
        rows_by_id[example.id]
        for example in ordered_examples
        if example.id in rows_by_id
    ]
    _write_jsonl(predictions_path, ordered_rows)

    metrics = compute_metrics_by_subset(ordered_rows)
    failures = sum(1 for row in ordered_rows if row.get("parse_failure"))
    LOGGER.info(
        "Evaluation complete: rows=%d failures=%d metrics=%s",
        len(ordered_rows),
        failures,
        json.dumps(metrics.get("combined", {}), sort_keys=True),
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion_path.write_text(render_confusion_matrix_csv(metrics), encoding="utf-8")
    summary_path.write_text(render_summary_markdown(metrics), encoding="utf-8")
    latex_path.write_text(render_latex_table(metrics), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(
            {
                "harness": "proclaim_librarian",
                "proclaim_src": str(proclaim_src),
                "subset_configs": {
                    name: str(path) for name, path in subset_configs.items()
                },
                "config_summaries": config_summaries,
                "claim_timeout_seconds": timeout,
                "n_examples": total,
                "resume": resume,
                "retry_failed": retry_failed,
                "keep_going": keep_going,
                "log_path": str(log_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_path),
        "summary_path": str(summary_path),
        "latex_table_path": str(latex_path),
        "log_path": str(log_path),
        "metrics": metrics,
    }


HERE = Path(__file__).resolve().parent


def _resolve_config(value: str | Path) -> Path:
    """Resolve a librarian config path, allowing paths relative to this directory.

    Relative to *here*, not to the clone: these configs are ours. The clone is
    a pristine checkout of ProClaim at its pinned commit and has nowhere for
    them to live.
    """
    path = Path(value)
    if not path.is_absolute():
        path = (HERE / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Librarian config not found: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the librarian-backed ProClaim verification pipeline "
        "over ProClaim-eval claims and report all metrics.",
    )
    parser.add_argument(
        "--proclaim-path",
        type=Path,
        required=True,
        help="Directory containing the ProClaim-eval data CSVs.",
    )
    parser.add_argument(
        "--subset", choices=["signor", "connectomedb", "all"], required=True
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--proclaim-src",
        type=Path,
        default=DEFAULT_PROCLAIM_SRC,
        help="Path to the ProClaim_src project root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Librarian config applied to every subset, overriding the "
        "per-subset defaults (signor/connectomedb librarian configs).",
    )
    parser.add_argument(
        "--librarian-base-url",
        default=os.environ.get("PROCLAIM_LIBRARIAN_URL"),
        help="Override librarian_llm_base_url in whichever config is used "
        "(env: PROCLAIM_LIBRARIAN_URL). The checked-in YAML is left alone; a "
        "patched copy is written into the run directory.",
    )
    parser.add_argument(
        "--librarian-model",
        default=os.environ.get("PROCLAIM_LIBRARIAN_MODEL"),
        help="Override librarian_llm_model (env: PROCLAIM_LIBRARIAN_MODEL).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CLAIM_TIMEOUT_SECONDS,
        help="Per-claim subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse predictions.jsonl rows and completed per-claim run dirs.",
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On --resume, re-run claims whose saved row was a failure "
        "(subprocess crash, timeout, or no verdict). --no-retry-failed keeps "
        "failed rows untouched.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a claim subprocess fails. By default the full "
        "evaluation stops at the first failed claim to avoid computing metrics "
        "from fallback labels.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of claims to verify in parallel (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    proclaim_src = args.proclaim_src.resolve()
    if not (proclaim_src / "src" / "proclaim").is_dir():
        raise SystemExit(
            f"ProClaim_src not found at {proclaim_src}.\n"
            "Run ./evals/Literature/setup.sh --bench proclaim."
        )

    if args.config:
        override = _resolve_config(args.config)
        subset_configs = {name: override for name in DEFAULT_SUBSET_CONFIGS}
    else:
        subset_configs = {
            name: _resolve_config(rel) for name, rel in DEFAULT_SUBSET_CONFIGS.items()
        }

    if args.librarian_base_url or args.librarian_model:
        out_root = Path(args.out_dir).expanduser().resolve()
        subset_configs = {
            name: _patch_config_endpoint(
                path, out_root, args.librarian_base_url, args.librarian_model
            )
            for name, path in subset_configs.items()
        }

    examples = load_proclaim_examples(
        args.proclaim_path,
        subset=args.subset,
        max_examples=args.max_examples,
    )
    if not examples:
        raise SystemExit("No ProClaim examples selected.")

    LOGGER.info("Verifying %d claim(s) through the librarian pipeline.", len(examples))
    result = run_librarian_evaluation(
        examples,
        proclaim_src=proclaim_src,
        subset_configs=subset_configs,
        out_dir=args.out_dir,
        timeout=args.timeout,
        resume=args.resume,
        retry_failed=args.retry_failed,
        keep_going=args.keep_going,
        progress=not args.no_progress,
        workers=args.workers,
    )

    LOGGER.info("Predictions: %s", result["predictions_path"])
    LOGGER.info("Metrics: %s", result["metrics_path"])
    LOGGER.info("Log: %s", result["log_path"])
    combined = result["metrics"].get("combined", {})
    LOGGER.info(
        "Combined AGR=%s mean_iterations=%s",
        combined.get("agr"),
        combined.get("avg_iterations"),
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

"""Run the librarian's AstaBench literature ablations via Inspect's Python API."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "agents").is_dir()),
    Path(__file__).resolve().parents[3],
)  # repo root = first ancestor containing agents/ (move-proof)
VENDORED_ASTABENCH_ROOT = Path(__file__).resolve().parent / "vendor"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if str(VENDORED_ASTABENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_ASTABENCH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if load_dotenv is not None:
    # Always prefer the repository .env for eval runs so refreshed keys
    # replace stale shell exports from previous sessions.
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _normalize_huggingface_env() -> None:
    # Prefer HUGGINGFACE_HUB_TOKEN (datasets-native), then fall back.
    # Mirror to all aliases so downstream libraries resolve consistently.
    token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    if not token:
        return

    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ["HF_TOKEN"] = token
    os.environ["HF_ACCESS_TOKEN"] = token


_normalize_huggingface_env()


def _normalize_google_env() -> None:
    google_key = os.getenv("GOOGLE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # Inspect's Google client warns when both are set; normalize to GOOGLE_API_KEY.
    if google_key and gemini_key:
        os.environ.pop("GEMINI_API_KEY", None)
        return

    if not google_key and gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        os.environ.pop("GEMINI_API_KEY", None)


_normalize_google_env()


def _infer_prediction_model_name(
    cli_model_name: str | None,
    solver_name: str,
) -> str:
    if solver_name == "openscholar_api":
        return "openscholar-8b"
    if solver_name == "llm_web_search" and cli_model_name:
        return cli_model_name
    if cli_model_name:
        return cli_model_name

    provider_env = os.getenv("LLM_PROVIDER", "")
    if ":" in provider_env:
        _, provider_model = provider_env.split(":", 1)
        if provider_model:
            return provider_model

    env_model = os.getenv("LLM_MODEL")
    if env_model:
        return env_model

    base_url = (os.getenv("LLM_BASE_URL") or "").lower()
    if "api.z.ai" in base_url:
        return "glm-4.7"
    return "glm-5-fp8"


def _sanitize_filename_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "unknown-model"


def _require_asta_tool_key_for_standard_tooling(
    config_names: list[str],
    solver_name: str,
    task_labels: list[str] | None = None,
) -> None:
    if solver_name in {"llm_only", "openscholar_api"}:
        return
    if "standard_tooling" not in config_names:
        return
    if task_labels and set(task_labels) <= PUBLIC_SEARCH_TASK_LABELS:
        return
    if os.getenv("ASTA_TOOL_KEY"):
        return

    raise RuntimeError(
        "Config `standard_tooling` requires `ASTA_TOOL_KEY` for the vendored "
        "Asta MCP literature tools. `HUGGINGFACE_HUB_TOKEN` only covers gated "
        "dataset access, and `S2_API_KEY` is not automatically used here. "
        "Either add `ASTA_TOOL_KEY=...` to `.env`, or run with "
        "`--config custom_tooling` or `--config llm_only`."
    )


import inspect_ai
from inspect_ai import task_with
from inspect_ai.dataset import MemoryDataset

from evals.Literature.AstaBench.bio_agent_wrapper import bio_agent_solver
from evals.Literature.AstaBench.llm_only_solver import llm_only_solver
from evals.Literature.AstaBench.llm_web_search_solver import llm_web_search_solver
from evals.Literature.AstaBench.openscholar_solver import openscholar_solver
from evals.Literature.AstaBench.litqa2_open_judge import (
    DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
    litqa2_open_judge_task,
)
from evals.Literature.AstaBench.pubmedqa_open_judge import (
    DEFAULT_PUBMEDQA_DATASET,
    DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
    pubmedqa_open_judge_task,
)


ConfigName = Literal["custom_tooling", "standard_tooling", "llm_only"]
SplitName = Literal["validation", "test"]
LitQA2OpenJudgeSplit = Literal["validation", "test", "full", "europepmc_fulltext"]


def _subsample_task(task: object, fraction: float, seed: int) -> object:
    """Return ``task`` with a seeded random ``fraction`` of its dataset.

    Shuffles the samples with ``seed`` (reproducible across runs) and keeps the
    first ``ceil(fraction * n)`` — at least one sample. Rebuilds the task via
    ``task_with`` + ``MemoryDataset``, the same idiom the LitQA2 subset uses.
    """
    samples = list(task.dataset)
    keep = max(1, math.ceil(len(samples) * fraction))
    random.Random(seed).shuffle(samples)
    subset = samples[:keep]
    print(
        f"  sample-fraction {fraction:g} (seed {seed}): {len(subset)}/{len(samples)} samples",
        flush=True,
    )
    return task_with(task, dataset=MemoryDataset(subset))


@dataclass(frozen=True)
class EvalSpec:
    label: str
    task_type: str
    task_factory: Callable[[], object]
    include_by_default: bool = True


TASK_LABELS = (
    "PaperFindingBench",
    "LitQA2-FullText",
    "LitQA2-FullText-OpenJudge",
    "LitQA2-FullText-OpenJudge-Full",
    "LitQA2-FullText-OpenJudge-EuropePMCFullText",
    "PubMedQA-OpenJudge",
    "LitQA2-FullText-Search",
    "ScholarQA-CS2",
    "ArxivDIGESTables-Clean",
)
PUBLIC_SEARCH_TASK_LABELS = {"PubMedQA-OpenJudge"}

SOLVER_LABELS = (
    "bio_agent",
    "llm_only",
    "llm_web_search",
    "openscholar_api",
)


def _paper_finder_task(split: SplitName, with_asta_search: bool) -> object:
    task_module = importlib.import_module("astabench.evals.paper_finder.task")
    paper_finder_test = task_module.paper_finder_test
    paper_finder_validation = task_module.paper_finder_validation

    if split == "validation":
        return paper_finder_validation(with_search_tools=with_asta_search)
    return paper_finder_test(with_search_tools=with_asta_search)


def _paper_finder_litqa2_task(split: SplitName, with_asta_search: bool) -> object:
    task_module = importlib.import_module("astabench.evals.paper_finder.task")
    paper_finder_litqa2_test = task_module.paper_finder_litqa2_test
    paper_finder_litqa2_validation = task_module.paper_finder_litqa2_validation

    if split == "validation":
        return paper_finder_litqa2_validation(with_search_tools=with_asta_search)
    return paper_finder_litqa2_test(with_search_tools=with_asta_search)


def _litqa2_task(split: SplitName, with_asta_search: bool) -> object:
    task_module = importlib.import_module("astabench.evals.labbench.litqa2.task")
    litqa2_test = task_module.litqa2_test
    litqa2_validation = task_module.litqa2_validation

    if split == "validation":
        return litqa2_validation(with_search_tools=with_asta_search)
    return litqa2_test(with_search_tools=with_asta_search)


def _litqa2_open_judge_task(
    split: LitQA2OpenJudgeSplit,
    with_asta_search: bool,
    judge_model: str,
) -> object:
    return litqa2_open_judge_task(
        split=split,
        with_search_tools=with_asta_search,
        judge_model=judge_model,
    )


def _pubmedqa_open_judge_task(
    dataset_path: str,
    judge_model: str,
) -> object:
    return pubmedqa_open_judge_task(
        dataset_path=dataset_path,
        judge_model=judge_model,
    )


def _sqa_task(
    split: SplitName,
    with_asta_search: bool,
    scorer_model: str | None = None,
) -> object:
    task_module = importlib.import_module("astabench.evals.sqa.task")
    sqa = task_module.sqa

    benchmark_split = "dev" if split == "validation" else "test"
    kwargs: dict[str, object] = {
        "split": benchmark_split,
        "with_search_tools": with_asta_search,
    }
    if scorer_model:
        kwargs["scorer_model"] = scorer_model
    return sqa(**kwargs)


def _arxivdigestables_task(split: SplitName, with_asta_search: bool) -> object:
    task_module = importlib.import_module("astabench.evals.arxivdigestables.task")
    arxivdigestables_test = task_module.arxivdigestables_test
    arxivdigestables_validation = task_module.arxivdigestables_validation

    if split == "validation":
        return arxivdigestables_validation(
            with_snippet_search_tool=with_asta_search,
        )
    return arxivdigestables_test(with_snippet_search_tool=with_asta_search)


def _build_eval_specs(
    config_name: ConfigName,
    split: SplitName,
    sqa_scorer_model: str | None = None,
    litqa2_open_judge_model: str = DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
    pubmedqa_open_judge_model: str = DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
    pubmedqa_dataset: str = str(DEFAULT_PUBMEDQA_DATASET),
) -> list[EvalSpec]:
    with_asta_search = config_name == "standard_tooling"
    if split == "validation":
        return [
            EvalSpec(
                label="PaperFindingBench",
                task_type="paper_finder",
                task_factory=lambda: _paper_finder_task(split, with_asta_search),
            ),
            EvalSpec(
                label="LitQA2-FullText",
                task_type="litqa2",
                task_factory=lambda: _litqa2_task(split, with_asta_search),
            ),
            EvalSpec(
                label="LitQA2-FullText-OpenJudge",
                task_type="litqa2_open"
                if config_name != "llm_only"
                else "litqa2_open_llm_only",
                task_factory=lambda: _litqa2_open_judge_task(
                    split,
                    with_asta_search,
                    litqa2_open_judge_model,
                ),
                include_by_default=False,
            ),
            EvalSpec(
                label="LitQA2-FullText-OpenJudge-Full",
                task_type="litqa2_open"
                if config_name != "llm_only"
                else "litqa2_open_llm_only",
                task_factory=lambda: _litqa2_open_judge_task(
                    "full",
                    with_asta_search,
                    litqa2_open_judge_model,
                ),
                include_by_default=False,
            ),
            EvalSpec(
                label="LitQA2-FullText-OpenJudge-EuropePMCFullText",
                task_type="litqa2_open"
                if config_name != "llm_only"
                else "litqa2_open_llm_only",
                task_factory=lambda: _litqa2_open_judge_task(
                    "europepmc_fulltext",
                    with_asta_search,
                    litqa2_open_judge_model,
                ),
                include_by_default=False,
            ),
            EvalSpec(
                label="PubMedQA-OpenJudge",
                task_type="pubmedqa_open",
                task_factory=lambda: _pubmedqa_open_judge_task(
                    pubmedqa_dataset,
                    pubmedqa_open_judge_model,
                ),
                include_by_default=False,
            ),
            EvalSpec(
                label="LitQA2-FullText-Search",
                task_type="paper_finder",
                task_factory=lambda: _paper_finder_litqa2_task(split, with_asta_search),
            ),
            EvalSpec(
                label="ScholarQA-CS2",
                task_type="sqa",
                task_factory=lambda: _sqa_task(
                    split,
                    with_asta_search,
                    scorer_model=sqa_scorer_model,
                ),
            ),
            EvalSpec(
                label="ArxivDIGESTables-Clean",
                task_type="arxivdigestables",
                task_factory=lambda: _arxivdigestables_task(split, with_asta_search),
            ),
        ]

    return [
        EvalSpec(
            label="PaperFindingBench",
            task_type="paper_finder",
            task_factory=lambda: _paper_finder_task(split, with_asta_search),
        ),
        EvalSpec(
            label="LitQA2-FullText",
            task_type="litqa2",
            task_factory=lambda: _litqa2_task(split, with_asta_search),
        ),
        EvalSpec(
            label="LitQA2-FullText-OpenJudge",
            task_type="litqa2_open"
            if config_name != "llm_only"
            else "litqa2_open_llm_only",
            task_factory=lambda: _litqa2_open_judge_task(
                split,
                with_asta_search,
                litqa2_open_judge_model,
            ),
            include_by_default=False,
        ),
        EvalSpec(
            label="LitQA2-FullText-OpenJudge-Full",
            task_type="litqa2_open"
            if config_name != "llm_only"
            else "litqa2_open_llm_only",
            task_factory=lambda: _litqa2_open_judge_task(
                "full",
                with_asta_search,
                litqa2_open_judge_model,
            ),
            include_by_default=False,
        ),
        EvalSpec(
            label="LitQA2-FullText-OpenJudge-EuropePMCFullText",
            task_type="litqa2_open"
            if config_name != "llm_only"
            else "litqa2_open_llm_only",
            task_factory=lambda: _litqa2_open_judge_task(
                "europepmc_fulltext",
                with_asta_search,
                litqa2_open_judge_model,
            ),
            include_by_default=False,
        ),
        EvalSpec(
            label="PubMedQA-OpenJudge",
            task_type="pubmedqa_open",
            task_factory=lambda: _pubmedqa_open_judge_task(
                pubmedqa_dataset,
                pubmedqa_open_judge_model,
            ),
            include_by_default=False,
        ),
        EvalSpec(
            label="LitQA2-FullText-Search",
            task_type="paper_finder",
            task_factory=lambda: _paper_finder_litqa2_task(split, with_asta_search),
        ),
        EvalSpec(
            label="ScholarQA-CS2",
            task_type="sqa",
            task_factory=lambda: _sqa_task(
                split,
                with_asta_search,
                scorer_model=sqa_scorer_model,
            ),
        ),
        EvalSpec(
            label="ArxivDIGESTables-Clean",
            task_type="arxivdigestables",
            task_factory=lambda: _arxivdigestables_task(split, with_asta_search),
        ),
    ]


def _select_eval_specs(
    config_name: ConfigName,
    split: SplitName,
    task_labels: list[str] | None,
    sqa_scorer_model: str | None = None,
    litqa2_open_judge_model: str = DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
    pubmedqa_open_judge_model: str = DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
    pubmedqa_dataset: str = str(DEFAULT_PUBMEDQA_DATASET),
) -> list[EvalSpec]:
    specs = _build_eval_specs(
        config_name=config_name,
        split=split,
        sqa_scorer_model=sqa_scorer_model,
        litqa2_open_judge_model=litqa2_open_judge_model,
        pubmedqa_open_judge_model=pubmedqa_open_judge_model,
        pubmedqa_dataset=pubmedqa_dataset,
    )
    if not task_labels:
        return [spec for spec in specs if spec.include_by_default]

    requested = set(task_labels)
    return [spec for spec in specs if spec.label in requested]


def _run_score(log_dir: Path) -> None:
    astabench_bin = shutil.which("astabench")
    if not astabench_bin:
        raise RuntimeError(
            "Could not find the `astabench` executable on PATH. "
            "Run this script from the AstaBench environment or disable --score-after-run."
        )
    subprocess.run([astabench_bin, "score", str(log_dir)], check=True)


def _format_missing_dependency_error(exc: ModuleNotFoundError) -> str:
    missing_module = exc.name or "unknown module"
    return (
        f"Missing Python dependency '{missing_module}' while loading AstaBench "
        "from `evals/Literature/AstaBench/vendor` (created by "
        "`evals/Literature/setup.sh`). Install the missing runtime dependency "
        "into this environment and retry."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Which AstaBench split to run (default: validation).",
    )
    parser.add_argument(
        "--config",
        choices=["custom_tooling", "standard_tooling", "llm_only", "both"],
        default="standard_tooling",
        help=(
            "Which retrieval config to run. The default is `standard_tooling`, "
            "which uses AstaBench's task-provided corpus tools. "
            "Use `llm_only` to run the LLM with no retrieval (parametric knowledge baseline)."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=SOLVER_LABELS,
        default="bio_agent",
        help=(
            "Which solver to use. `bio_agent` runs the librarian wrapper. "
            "`llm_only` runs a direct LLM baseline. "
            "`openscholar_api` uses the OpenScholar API (ScholarQA-CS2 only)."
        ),
    )
    parser.add_argument(
        "--web-search-tool-choice",
        choices=["auto", "required"],
        default="required",
        help="Tool choice for web search solver (default: required).",
    )
    parser.add_argument(
        "--web-search-context-size",
        choices=["low", "medium", "high"],
        default="medium",
        help="Search context size for web search solver (default: medium).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "evals" / "AstaBench" / "results"),
        help="Directory where Inspect logs will be written.",
    )
    parser.add_argument(
        "--openscholar-api-base-url",
        default=os.getenv("OPENSCHOLAR_API_BASE_URL", "https://openscilm.allen.ai"),
        help="Base URL for the OpenScholar API (default: https://openscilm.allen.ai).",
    )
    parser.add_argument(
        "--openscholar-poll-interval",
        type=int,
        default=8,
        help="Seconds between OpenScholar poll requests (default: 8).",
    )
    parser.add_argument(
        "--openscholar-max-polls",
        type=int,
        default=40,
        help="Maximum OpenScholar polling attempts per query (default: 40).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional Inspect sample limit for smoke runs (first N samples).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help=(
            "Run a random fraction of the dataset (e.g. 0.1 for 10%%). Sampled with "
            "--sample-seed for reproducibility. Applied before --limit."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for --sample-fraction (default: 42).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum concurrent samples for Inspect (default: 10).",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Maximum concurrent model API requests per model (default: 10).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="low",
        help=(
            "GLM-5.3+ reasoning effort for the solver's LLM calls. Higher levels "
            "spend more tokens on the hidden trace. Defaults to low."
        ),
    )
    parser.add_argument(
        "--verbose-agent",
        action="store_true",
        help="Enable verbose librarian internals.",
    )
    parser.add_argument(
        "--bio-agent-full-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable librarian full-text enrichment (default: enabled).",
    )
    parser.add_argument(
        "--bio-agent-query-planner",
        default="default",
        help=(
            "Librarian query planner to use. Built-ins: default, "
            "simple_bm25, raw_question, epmc_full_interface. You can also pass "
            "the stem of a .md prompt under agents/literature/prompts, e.g. "
            "europepmc_claude_librarian."
        ),
    )
    parser.add_argument(
        "--bio-agent-max-query-count",
        type=int,
        default=None,
        help="Optional hard cap for generated librarian search queries.",
    )
    parser.add_argument(
        "--bio-agent-num-subqueries",
        type=int,
        default=None,
        help=(
            "Override LibrarianAgent's num_subqueries (LibrarianAgent path "
            "only, e.g. --solver bio_agent with the new librarian). Default: "
            "the librarian/config.toml value."
        ),
    )
    parser.add_argument(
        "--bio-agent-paragraphs-per-subquery",
        type=int,
        default=None,
        help="Override LibrarianAgent's paragraphs_per_subquery (LibrarianAgent path only).",
    )
    parser.add_argument(
        "--bio-agent-paragraphs-per-judge-batch",
        type=int,
        default=None,
        help="Override LibrarianAgent's paragraphs_per_judge_batch (LibrarianAgent path only).",
    )
    parser.add_argument(
        "--bio-agent-simple-bm25-max-queries",
        type=int,
        default=7,
        help="Maximum plain BM25 queries for --bio-agent-query-planner simple_bm25.",
    )
    parser.add_argument(
        "--bio-agent-epmc-full-interface-max-queries",
        type=int,
        default=7,
        help=(
            "Maximum EuropePMC advanced queries for "
            "--bio-agent-query-planner epmc_full_interface."
        ),
    )
    parser.add_argument(
        "--bio-agent-retrieval-policy",
        choices=[
            "synthesis",
            "localized_evidence",
            "localized_evidence_bm25",
            "localized_evidence_bm25_per_paper",
            "cascade_bm25",
            "upfront_fulltext",
            "upfront_fulltext_bm25_filter",
            "simple_fulltext_bm25",
        ],
        default=os.getenv("LITERATURE_RETRIEVAL_POLICY", "synthesis"),
        help=(
            "Librarian retrieval policy. `localized_evidence` uses semantic "
            "localized full-text prefiltering with the heuristic paragraph "
            "scorer; `localized_evidence_bm25_per_paper` mirrors the same "
            "pipeline end-to-end but substitutes per-paragraph BM25 for the "
            "heuristic (clean scoring-function A/B); `localized_evidence_bm25` "
            "uses a separate global-BM25 passage pipeline with paper bundles."
        ),
    )
    parser.add_argument(
        "--score-after-run",
        action="store_true",
        help="Run `astabench score <log_dir>` after each task finishes.",
    )
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Ask Inspect to surface debug traces in the eval logs.",
    )
    parser.add_argument(
        "--inspect-display",
        choices=["full", "conversation", "rich", "plain", "none"],
        default="rich",
        help=(
            "Inspect live progress display mode (default: rich). "
            "Use `full` for the most detailed live task UI."
        ),
    )
    parser.add_argument(
        "--inspect-no-score-display",
        action="store_true",
        help="Disable realtime scorer metric updates in Inspect output.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_LABELS,
        help=(
            "Optional dataset label to run. Pass multiple times to run a subset, "
            "for example `--task ScholarQA-CS2 --task LitQA2-FullText`."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "Optional override for the librarian LLM base URL. "
            "Example: http://127.0.0.1:8000/v1"
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional override for the librarian LLM model name.",
    )
    parser.add_argument(
        "--sqa-scorer-model",
        default=None,
        help=(
            "Optional ScholarQA judge model override. "
            "If omitted, the task default is used (Gemini in upstream AstaBench)."
        ),
    )
    parser.add_argument(
        "--litqa2-open-judge-model",
        default=os.getenv(
            "ASTABENCH_LITQA2_OPEN_JUDGE_MODEL",
            DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
        ),
        help=(
            "Judge model for `LitQA2-FullText-OpenJudge` answer-to-choice "
            f"mapping (default: {DEFAULT_LITQA2_OPEN_JUDGE_MODEL})."
        ),
    )
    parser.add_argument(
        "--pubmedqa-dataset",
        default=str(DEFAULT_PUBMEDQA_DATASET),
        help="Path to the PubMedQA PQA-L ori_pqal.json file.",
    )
    parser.add_argument(
        "--pubmedqa-open-judge-model",
        default=os.getenv(
            "ASTABENCH_PUBMEDQA_OPEN_JUDGE_MODEL",
            DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
        ),
        help=(
            "Judge model for `PubMedQA-OpenJudge` open-answer yes/no/maybe "
            f"mapping (default: {DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL})."
        ),
    )
    parser.add_argument(
        "--inspect-log-file-pattern",
        default=None,
        help=(
            "Optional override for INSPECT_EVAL_LOG_FILE_PATTERN. "
            "Default now includes the prediction model, e.g. "
            "`{task}_{id}_pred-glm-5-fp8`."
        ),
    )
    args = parser.parse_args()

    def _build_solver_for_spec(spec: EvalSpec, config_name: ConfigName) -> object:
        if args.solver == "bio_agent":
            return bio_agent_solver(
                config_name=config_name,
                task_type=spec.task_type,
                reasoning_effort=args.reasoning_effort,
                verbose=args.verbose_agent,
                llm_base_url=args.llm_base_url,
                llm_model_name=args.llm_model,
                full_text_enrichment=args.bio_agent_full_text,
                query_planner=args.bio_agent_query_planner,
                simple_bm25_max_queries=args.bio_agent_simple_bm25_max_queries,
                epmc_full_interface_max_queries=(
                    args.bio_agent_epmc_full_interface_max_queries
                ),
                max_query_count_override=args.bio_agent_max_query_count,
                librarian_num_subqueries_override=args.bio_agent_num_subqueries,
                librarian_paragraphs_per_subquery_override=(
                    args.bio_agent_paragraphs_per_subquery
                ),
                librarian_paragraphs_per_judge_batch_override=(
                    args.bio_agent_paragraphs_per_judge_batch
                ),
                retrieval_policy=args.bio_agent_retrieval_policy,
            )
        if args.solver == "llm_only":
            return llm_only_solver(
                task_type=spec.task_type,
                llm_base_url=args.llm_base_url,
                llm_model_name=args.llm_model,
                reasoning_effort=args.reasoning_effort,
            )
        if args.solver == "llm_web_search":
            return llm_web_search_solver(
                task_type=spec.task_type,
                base_url=args.llm_base_url,
                model=args.llm_model,
                tool_choice=args.web_search_tool_choice,
                search_context_size=args.web_search_context_size,
            )
        if args.solver == "openscholar_api":
            return openscholar_solver(
                task_type=spec.task_type,
                api_base_url=args.openscholar_api_base_url,
                poll_interval=args.openscholar_poll_interval,
                max_poll_attempts=args.openscholar_max_polls,
            )
        raise ValueError(f"Unknown solver '{args.solver}'.")

    config_names: list[ConfigName]
    if args.config == "both":
        config_names = ["standard_tooling", "custom_tooling"]
    else:
        config_names = [args.config]

    _require_asta_tool_key_for_standard_tooling(
        config_names,
        args.solver,
        task_labels=args.task,
    )

    prediction_model_tag = _sanitize_filename_component(
        _infer_prediction_model_name(args.llm_model, args.solver)
    )
    inspect_log_pattern = (
        args.inspect_log_file_pattern or f"{{task}}_{{id}}_pred-{prediction_model_tag}"
    )
    os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = inspect_log_pattern

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Inspect log filename pattern: {inspect_log_pattern}", flush=True)
    print(f"Prediction model results folder: {prediction_model_tag}", flush=True)
    print(f"Inspect display mode: {args.inspect_display}", flush=True)

    for config_name in config_names:
        config_dir = results_root / args.split / config_name / prediction_model_tag
        config_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running {config_name} on {args.split} ===", flush=True)

        selected_specs = _select_eval_specs(
            config_name=config_name,
            split=args.split,
            task_labels=args.task,
            sqa_scorer_model=args.sqa_scorer_model,
            litqa2_open_judge_model=args.litqa2_open_judge_model,
            pubmedqa_open_judge_model=args.pubmedqa_open_judge_model,
            pubmedqa_dataset=args.pubmedqa_dataset,
        )
        if args.solver == "openscholar_api":
            unsupported = [
                spec.label
                for spec in selected_specs
                if spec.task_type
                not in {
                    "sqa",
                    "litqa2_open",
                    "litqa2_open_llm_only",
                    "pubmedqa_open",
                }
            ]
            if unsupported:
                raise RuntimeError(
                    "OpenScholar API solver only supports ScholarQA-CS2 and "
                    "open-answer judge tasks. Remove unsupported tasks: "
                    f"{', '.join(unsupported)}."
                )
        if args.solver == "llm_web_search":
            unsupported = [
                spec.label
                for spec in selected_specs
                if spec.task_type
                not in {
                    "litqa2_open",
                    "litqa2_open_llm_only",
                    "pubmedqa_open",
                    "paper_finder",
                }
            ]
            if unsupported:
                raise RuntimeError(
                    "LLM web search solver only supports LitQA2-FullText-OpenJudge "
                    "PubMedQA-OpenJudge, and paper-finder tasks. "
                    f"Remove unsupported tasks: {', '.join(unsupported)}."
                )
        for spec in selected_specs:
            task_log_dir = config_dir / spec.label
            task_log_dir.mkdir(parents=True, exist_ok=True)
            os.environ["INSPECT_LOG_DIR"] = str(task_log_dir)

            print(
                f"\n--- {spec.label} ({config_name}) → {task_log_dir} ---",
                flush=True,
            )

            try:
                task = spec.task_factory()
                if args.sample_fraction is not None:
                    task = _subsample_task(task, args.sample_fraction, args.sample_seed)
                inspect_ai.eval(
                    task,
                    model=None,
                    solver=_build_solver_for_spec(spec, config_name),
                    limit=args.limit,
                    max_samples=args.max_samples,
                    max_connections=args.max_connections,
                    display=args.inspect_display,
                    score_display=not args.inspect_no_score_display,
                    fail_on_error=False,
                    debug_errors=args.debug_errors,
                )
            except ModuleNotFoundError as exc:
                raise RuntimeError(_format_missing_dependency_error(exc)) from exc

            if args.score_after_run:
                _run_score(task_log_dir)


if __name__ == "__main__":
    main()

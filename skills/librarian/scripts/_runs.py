"""Shared plumbing for the four step scripts: run directories and CLI session briefs.

The in-process agent keeps the whole pipeline's state in locals. Here the pipeline
is four script invocations with one-shot CLI sessions in between, so the same state
lives in files under one run directory:

    run.json                manifest: question, resolved config, per-stage counters
    01_query_prompt.md      rendered Stage-1 prompt              (CLI session input)
    01_queries.json         {"queries": [...]}                   (CLI session output)
    02_paragraphs.json      the Stage-2 paragraph pool
    03_judge/batch_NN.md    rendered Stage-3 judge batch         (CLI session input)
    03_judge/batch_NN.json  {"relevant_ids": [...]}              (CLI session output)
    04_evidence.json        the evidence records ``agent.run()`` returns
    04_report.md            the same records as the root agent's report

Everything else — prompts, EPMC search, full-text fetch, BM25, sentence splitting,
evidence assembly — is imported from ``librarian.agent``; nothing is
reimplemented here.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# skills/librarian/scripts/_runs.py -> the Librarian project root is 3 levels up.
# Anchored on this file, not on the working directory or a git root, so the skill
# works from a plugin cache, a clone, or a vendored copy inside another repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from librarian.agent import (  # noqa: E402  (needs REPO_ROOT on sys.path)
    _ABSTRACT_BM25_FIELDS,
    _build_sentence_items,
    _candidate_id,
    _Sentence,
    LibrarianAgent,
)
from librarian.config import (  # noqa: E402
    LibrarianRuntimeConfig,
    load_runtime_config,
)

MANIFEST_NAME = "run.json"
QUERY_PROMPT_NAME = "01_query_prompt.md"
QUERIES_NAME = "01_queries.json"
PARAGRAPHS_NAME = "02_paragraphs.json"
JUDGE_DIR_NAME = "03_judge"
EVIDENCE_NAME = "04_evidence.json"
REPORT_NAME = "04_report.md"

# The two system prompts agent.py pairs with these prompt bodies, quoted into the
# briefs (agent.py::_generate_queries and agent.py::_evaluate_batch).
QUERY_SYSTEM_PROMPT = "You are a helpful assistant."
JUDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Return only valid JSON "
    "with a 'relevant_ids' array ordered most→least relevant."
)

# The LLM endpoint is never contacted: Stage 1 and Stage 3 run as one-shot CLI sessions, and
# the stages these scripts do call (search, BM25, evidence assembly) never touch
# ``agent.llm``. LibrarianAgent still builds a client in __init__, so pin it at an
# unroutable URL — passing base_url explicitly also stops create_llm_client from
# picking up an ``LLM_BASE_URL`` out of the ambient environment.
_UNUSED_LLM_URL = "http://librarian-skill.invalid/v1"


# ── Config ──────────────────────────────────────────────────────────────────


def build_agent(config: LibrarianRuntimeConfig) -> LibrarianAgent:
    """A ``LibrarianAgent`` for the mechanical stages, with no live LLM endpoint.

    Constructing one is how the scripts reuse Stage 2 (``_paragraphs_for_subquery``)
    and the Stage-3 tail (``_papers_from_cited_sentences``, ``_passage_from_paper``)
    verbatim. The client banner ``LLMClient.__init__`` prints goes to stderr so it
    cannot corrupt a script's stdout, which is the step protocol.

    :param config: The runtime config for this run (see ``librarian.config``).
    :type config: LibrarianRuntimeConfig
    :return: An agent whose LLM stages must not be called.
    :rtype: LibrarianAgent
    """
    with contextlib.redirect_stdout(sys.stderr):
        return LibrarianAgent(
            runtime_config=config,
            llm_base_url=_UNUSED_LLM_URL,
            llm_model_name=config.default_model_name,
        )


# ── Judge items ─────────────────────────────────────────────────────────────


def build_judge_items(
    paragraphs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, _Sentence], Dict[str, Dict[str, Any]]]:
    """Paragraph pool → Stage-3 judge items, sentence registry, papers by id.

    The one piece of ``agent.py::_relevance_filter`` that is inline there rather
    than a reusable helper, so it is adapted here (on the same
    ``_build_sentence_items`` / ``_candidate_id`` / ``_ABSTRACT_BM25_FIELDS``
    primitives) instead of being reimplemented differently.

    Purely a function of the pool and its order — ids are ``paragraph_<index>``
    plus a sentence suffix — so step 4 rebuilds the identical registry from
    ``02_paragraphs.json`` rather than step 3 having to serialize it.

    :param paragraphs: Stage-2 paragraph records, each carrying its ``paper``.
    :type paragraphs: list[dict]
    :return: ``(items, registry, paper_by_id)``.
    :rtype: tuple[list[dict], dict[str, _Sentence], dict[str, dict]]
    """
    registry: Dict[str, _Sentence] = {}
    paper_by_id: Dict[str, Dict[str, Any]] = {}
    items: List[Dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        paper = paragraph["paper"]
        paper_id = _candidate_id(paper)
        paper_by_id[paper_id] = paper
        paragraph_id = f"paragraph_{paragraph_index}"
        metadata = {field: paper.get(field, "N/A") for field in _ABSTRACT_BM25_FIELDS}
        items.append(
            {
                "id": paragraph_id,
                **metadata,
                "sentences": _build_sentence_items(
                    paragraph_id,
                    paper_id,
                    int(paragraph["source_paragraph_index"]),
                    int(paragraph["word_start"]),
                    str(paragraph.get("text") or "").strip(),
                    registry,
                ),
            }
        )
    return items, registry, paper_by_id


# ── Run directory ───────────────────────────────────────────────────────────


def work_dir() -> Path:
    """Base directory holding every run (``LIBRARIAN_WORK_DIR``, else a temp dir)."""
    configured = os.environ.get("LIBRARIAN_WORK_DIR", "").strip()
    base = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "librarian-runs"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _slug(text: str, max_len: int = 40) -> str:
    """Filename-safe fragment of the question, for a recognizable run directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len].rstrip("-") or "query"


def create_run(query: str, config: LibrarianRuntimeConfig) -> Path:
    """Create a fresh run directory for ``query`` and write its manifest.

    :param query: The user's research question, verbatim.
    :type query: str
    :param config: The runtime config recorded for every later step to reuse.
    :type config: LibrarianRuntimeConfig
    :return: The new run directory.
    :rtype: Path
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = work_dir() / f"{stamp}-{_slug(query)}"
    suffix = 2
    while run_dir.exists():
        run_dir = work_dir() / f"{stamp}-{_slug(query)}-{suffix}"
        suffix += 1
    (run_dir / JUDGE_DIR_NAME).mkdir(parents=True)
    write_json(
        run_dir / MANIFEST_NAME,
        {
            "query": query,
            "created_at": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "config": asdict(config),
            "stages": {},
        },
    )
    return run_dir


def resolve_run(argument: str) -> Path:
    """Resolve and validate the run directory passed by the previous step."""
    run_dir = Path(argument).expanduser()
    if not (run_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"Not a librarian run directory: {run_dir}")
    return run_dir


def read_json(path: Path) -> Any:
    """Parse a JSON file written by an earlier step."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write ``value`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_manifest(run_dir: Path) -> Dict[str, Any]:
    """The run manifest (question, config, per-stage counters)."""
    return read_json(run_dir / MANIFEST_NAME)


def config_from_manifest(manifest: Dict[str, Any]) -> LibrarianRuntimeConfig:
    """Rebuild the config recorded at step 1, so every step of a run agrees on it."""
    return LibrarianRuntimeConfig(**manifest["config"])


def record_stage(run_dir: Path, stage: str, payload: Dict[str, Any]) -> None:
    """Merge ``payload`` into ``manifest["stages"][stage]``.

    The file-based equivalent of ``agent.last_run_debug``: each step writes the
    counters it produced as it completes.
    """
    manifest = read_manifest(run_dir)
    manifest.setdefault("stages", {})[stage] = {
        "completed_at": datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        **payload,
    }
    write_json(run_dir / MANIFEST_NAME, manifest)


def batch_paths(run_dir: Path, index: int) -> Dict[str, Path]:
    """The brief and output paths for judge batch ``index``."""
    judge_dir = run_dir / JUDGE_DIR_NAME
    judge_dir.mkdir(parents=True, exist_ok=True)
    return {
        "prompt": judge_dir / f"batch_{index:02d}.md",
        "output": judge_dir / f"batch_{index:02d}.json",
    }


# ── Direct model prompts ────────────────────────────────────────────────────

# The root agent sees only this file's path. A local launcher streams the rendered
# prompt to a fresh CLI session, then saves the model's JSON response itself.
_PROMPT_TEMPLATE = """# {title}

{system_prompt}

<!-- BEGIN PROMPT -->

{body}

<!-- END PROMPT -->

## Response contract

Return only one JSON object with the key `{output_key}`. Its value must be an
array of strings. Do not use tools, write files, add prose, or use Markdown fences.
"""


def write_prompt(
    path: Path,
    title: str,
    system_prompt: str,
    body: str,
    output_key: str,
) -> Path:
    """Write a rendered prompt for a fresh direct CLI model session.

    :param path: Where the brief is written.
    :type path: Path
    :param title: One-line heading naming the stage and batch.
    :type title: str
    :param system_prompt: The system prompt ``agent.py`` pairs with this body.
    :type system_prompt: str
    :param body: The rendered prompt body, verbatim.
    :type body: str
    :param output_key: The single key that JSON object must carry.
    :type output_key: str
    :return: ``path``, for chaining.
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _PROMPT_TEMPLATE.format(
            title=title,
            system_prompt=system_prompt,
            body=body,
            output_key=output_key,
        ),
        encoding="utf-8",
    )
    return path

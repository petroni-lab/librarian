"""LAB-Bench multiple-choice compatibility layer (task-agnostic).

Self-contained reimplementation of the pieces of the upstream LAB-Bench harness
(https://github.com/Future-House/LAB-Bench) that this evaluation needs, so the
eval runs without depending on the `labbench`/`chembench` packages. The same
prompt template, deterministic choice shuffling, answer parser, and metrics,
driven by a ``TaskSpec`` so it serves every text-only LAB-Bench multiple-choice
task:

  - **DbQA** (520)        — retrieving information from biological databases
  - **SeqQA** (600)       — manipulating biological sequences
  - **ProtocolQA** (108)  — troubleshooting biological protocols
  - **CloningScenarios** (33) — molecular cloning workflows
  - **TableQA** (244)     — reading data tables reported in the literature

TableQA runs through the same retrieval flow as the others; the dataset's table
image and source DOI are not read.

All five registered tasks share the same row schema (``id, question, ideal,
distractors``); ProtocolQA additionally carries a separate ``protocol`` field,
which upstream prepends to the question (``input.question = protocol +
question``) — reproduced here in ``build_effective_question``.

Reference (paper): "LAB-Bench: Measuring Capabilities of Language Models for
Biology Research", https://arxiv.org/abs/2407.10362.
"""

from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALPHABET = string.ascii_uppercase

# The fixed refusal option LAB-Bench injects into every MCQ. Selecting it counts
# as "unsure" (it lowers coverage but never costs precision).
REFUSE_CHOICE = "Insufficient information to answer the question"

# chembench.constant.COT_PROMPT, used verbatim by labbench.zero_shot.
COT_PROMPT = "Think step by step."

# labbench.zero_shot.MCQ_INSTRUCT_TEMPLATE, reproduced verbatim.
MCQ_INSTRUCT_TEMPLATE = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.{cot}

Question: {question}

Options:
{answers}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""


@dataclass(frozen=True)
class TaskSpec:
    """Per-task knobs for a LAB-Bench multiple-choice subtask.

    ``hf_config`` is the HuggingFace config name (and the local public-JSONL
    basename). ``strip_query_sequences`` is the *default* for whether long raw
    DNA/RNA runs are stripped from the literature search query (see
    ``build_retrieval_query``); the CLI can override it. ``has_protocol`` marks
    tasks (ProtocolQA) whose rows carry a separate ``protocol`` field to prepend.
    """

    hf_config: str
    description: str
    strip_query_sequences: bool
    has_protocol: bool = False


# The LAB-Bench MCQ tasks this runner supports.
TASKS: dict[str, TaskSpec] = {
    "DbQA": TaskSpec(
        hf_config="DbQA",
        description="Retrieving information from biological databases",
        strip_query_sequences=False,
    ),
    "SeqQA": TaskSpec(
        hf_config="SeqQA",
        description="Manipulating biological sequences",
        strip_query_sequences=True,
    ),
    "ProtocolQA": TaskSpec(
        hf_config="ProtocolQA",
        description="Troubleshooting biological protocols",
        strip_query_sequences=False,
        has_protocol=True,
    ),
    "CloningScenarios": TaskSpec(
        hf_config="CloningScenarios",
        description="Molecular cloning workflows",
        strip_query_sequences=True,
    ),
    # Retrieval-only: the question is the search query, and the row's table
    # image and source DOI are not read.
    "TableQA": TaskSpec(
        hf_config="TableQA",
        description="Reading data tables reported in the literature",
        strip_query_sequences=False,
    ),
}


@dataclass
class LabBenchQuestion:
    """One LAB-Bench MCQ with its presented choices and gold answer."""

    id: str
    question: str  # the effective question shown to the model (protocol prepended)
    retrieval_question: str  # question used for search — never includes the protocol
    choices: list[str]  # e.g. ["(A) ...", "(B) ...", ...]
    answer_letter: str  # letter of the ideal answer
    unsure_letter: str  # letter of the REFUSE_CHOICE option
    ideal: str
    distractors: list[str] = field(default_factory=list)


# Runs of >=20 nucleotide letters: the long plasmid/oligo sequences in the
# CloningScenarios and SeqQA questions. Protein sequences are not matched, since
# they share the ordinary uppercase alphabet.
_SEQUENCE_RUN_RE = re.compile(r"[ACGTUN]{20,}", re.IGNORECASE)


def build_effective_question(row: dict[str, Any], spec: TaskSpec) -> str:
    """Return the question text shown to the model for one raw dataset row.

    For ProtocolQA (``spec.has_protocol``) the separate ``protocol`` field is
    prepended to the question, matching upstream (``input.question = protocol +
    question``); a blank line is inserted for readability.
    """
    question = str(row.get("question", ""))
    if spec.has_protocol:
        protocol = str(row.get("protocol") or "").strip()
        if protocol:
            return f"{protocol}\n\n{question}"
    return question


def build_retrieval_query(question: str, strip_sequences: bool = True) -> str:
    """Derive the literature-search query from a question.

    When ``strip_sequences`` is set, long raw DNA/RNA runs are replaced with a
    short placeholder, so the retrieval agent searches on the conceptual text
    (enzymes, method, assay). Only the query is cleaned; the answering prompt
    keeps the full sequence.
    """
    if not strip_sequences:
        return question
    cleaned = _SEQUENCE_RUN_RE.sub("[sequence]", question)
    return " ".join(cleaned.split())


def build_mcq_prompt(question: str, choices: list[str], use_cot: bool = True) -> str:
    """Render the verbatim LAB-Bench MCQ prompt for one question."""
    cot = ("\n" + COT_PROMPT) if use_cot else ""
    return MCQ_INSTRUCT_TEMPLATE.format(
        cot=cot, question=question, answers="\n".join(choices)
    )


def randomize_choices(
    ideal: str, distractors: list[str], seed: int
) -> tuple[list[str], str, str]:
    """Deterministically shuffle [ideal, refuse, *distractors] into lettered choices.

    Mirrors ``labbench.utils.randomize_choices`` exactly, except the shuffle is
    seeded, so the option ordering is reproducible and identical between the
    baseline and knowledge-layer runs.

    Returns (lettered_choices, answer_letter, unsure_letter).
    """
    choices = [ideal, REFUSE_CHOICE, *distractors]
    if len(choices) > len(ALPHABET):
        raise ValueError("Too many choices for the alphabet.")

    perm = list(range(len(choices)))
    random.Random(seed).shuffle(perm)

    lettered = [f"({ALPHABET[i]}) {choices[perm[i]]}" for i in range(len(perm))]
    answer = ALPHABET[perm.index(0)]  # slot now holding the ideal answer
    unsure = ALPHABET[perm.index(1)]  # slot now holding the refusal option
    return lettered, answer, unsure


# ── Answer parsing ──────────────────────────────────────────────────────────

_ANSWER_BLOCK_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL | re.IGNORECASE)
_LETTER_RE = re.compile(r"[A-Z]")

# Fallback patterns for thinking models (e.g. GLM-5, DeepSeek-R1) that do not
# reliably emit the [ANSWER]...[/ANSWER] tags. Tried in order; first match wins.
_FALLBACK_ANSWER_RES = [
    # "the answer is (C)" / "Answer: C" / "correct answer is C"
    re.compile(
        r"(?:the\s+)?(?:correct\s+)?(?:final\s+)?(?:answer|choice)\s*(?:is|:)\s*[\(\[]?([A-Z])[\)\].]?",
        re.IGNORECASE,
    ),
    # Chinese equivalents: 答案是C / 正确答案：C / 选择C
    re.compile(r"(?:正确)?(?:答案|选项|选择)[是:：]?\s*[\(\[]?([A-Z])[\)\].]?"),
    # Parenthesised letter on its own line: "(C)" or "(C) some text"
    re.compile(r"^\s*\(([A-Z])\)", re.MULTILINE),
    # Bold letter at the end of a line: "**C**" or "*C*"
    re.compile(r"\*{1,2}([A-Z])\*{1,2}"),
]


def parse_answer(text: str, n_choices: int) -> str | None:
    """Extract the answer letter from a model response.

    First looks for the last ``[ANSWER]...[/ANSWER]`` block (the official LAB-Bench
    format). If none is found, falls back to a set of common free-text patterns so
    that thinking models that don't strictly follow the tag format still get scored.
    Returns ``None`` only when no valid letter can be found anywhere.
    """
    if not text:
        return None
    valid = set(ALPHABET[:n_choices])

    # Primary: [ANSWER]X[/ANSWER] tag (upstream LAB-Bench format).
    blocks = _ANSWER_BLOCK_RE.findall(text)
    if blocks:
        match = _LETTER_RE.search(blocks[-1].upper())
        if match and match.group(0) in valid:
            return match.group(0)

    # Fallback: scan the last 800 characters for common free-text answer
    # formats, so an earlier mention cannot override the final answer.
    tail = text[-800:]
    for pattern in _FALLBACK_ANSWER_RES:
        for m in pattern.finditer(tail):
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    return None


# ── Metrics (labbench.Evaluator.compute_metrics) ──────────────────────────────


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Accuracy / precision / coverage over scored prediction rows.

    - accuracy  = correct / total
    - precision = correct / sure          (accuracy among answered questions)
    - coverage  = sure / total            (fraction the model chose to answer)

    "sure" means the model did not pick the refusal option (and produced a
    parseable letter).
    """
    n_total = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    n_sure = sum(1 for r in rows if r.get("sure"))
    return {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "precision": n_correct / n_sure if n_sure else 0.0,
        "coverage": n_sure / n_total if n_total else 0.0,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_sure": n_sure,
    }


def score_prediction(
    predicted_letter: str | None, answer_letter: str, unsure_letter: str
) -> tuple[bool, bool]:
    """Return (correct, sure) for one prediction, matching labbench.Evaluator.

    A missing/unparseable letter is treated as "sure but wrong" (the upstream
    harness behaviour), so it lowers both accuracy and precision.
    """
    correct = predicted_letter == answer_letter
    sure = predicted_letter != unsure_letter
    return correct, sure


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_questions(
    task: str,
    *,
    data_file: str | Path | None = None,
    hf_repo: str = "futurehouse/lab-bench",
    seed: int = 0,
    max_examples: int | None = None,
) -> list[LabBenchQuestion]:
    """Load a LAB-Bench task's examples and pre-compute their shuffled choices.

    ``task`` must be a key of ``TASKS``. If ``data_file`` is given, reads a local
    JSONL (the ``<task>-v1-public.jsonl`` shipped in the LAB-Bench repo).
    Otherwise loads the task's config from the HuggingFace dataset
    (``futurehouse/lab-bench``) — the dataset is gated, so a HuggingFace login
    may be required.

    Each row's options are shuffled deterministically with ``seed + index`` so
    repeated runs (and the baseline vs knowledge-layer comparison) see the same
    layout.
    """
    spec = TASKS.get(task)
    if spec is None:
        raise ValueError(
            f"Unknown task {task!r}. Choose one of: {', '.join(sorted(TASKS))}."
        )

    raw_rows = _read_raw_rows(data_file, hf_repo, spec.hf_config)

    questions: list[LabBenchQuestion] = []
    for index, row in enumerate(raw_rows):
        ideal = row.get("ideal")
        distractors = list(row.get("distractors") or [])
        if not ideal or ideal == "null" or not distractors:
            # Skip open-answer / malformed rows; this eval is multiple choice.
            continue
        choices, answer_letter, unsure_letter = randomize_choices(
            ideal, distractors, seed=seed + index
        )
        questions.append(
            LabBenchQuestion(
                id=str(row.get("id", index)),
                question=build_effective_question(row, spec),
                # Use the raw question (without protocol) for retrieval so ProtocolQA
                # searches on the troubleshooting question, not the protocol text.
                retrieval_question=str(row.get("question", "")),
                choices=choices,
                answer_letter=answer_letter,
                unsure_letter=unsure_letter,
                ideal=ideal,
                distractors=distractors,
            )
        )
        if max_examples is not None and len(questions) >= max_examples:
            break
    return questions


def _read_raw_rows(
    data_file: str | Path | None, hf_repo: str, hf_config: str
) -> list[dict[str, Any]]:
    """Return raw dataset rows from a local JSONL or the HuggingFace hub."""
    if data_file is not None:
        rows: list[dict[str, Any]] = []
        with open(data_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "The `datasets` package is required to load LAB-Bench from "
            "HuggingFace. Add it to evals/Literature/envs/labbench.in and re-run "
            "setup.sh --relock labbench, or pass --data-file with a local "
            "<task>-v1-public.jsonl."
        ) from exc

    dataset = datasets.load_dataset(hf_repo, hf_config)["train"]
    return [dict(row) for row in dataset]

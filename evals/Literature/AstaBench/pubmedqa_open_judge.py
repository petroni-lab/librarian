"""Open-answer PubMedQA task with post-hoc yes/no/maybe judging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field

from evals.Literature.AstaBench.compat import extract_json_from_response

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "agents").is_dir()),
    Path(__file__).resolve().parents[3],
)  # repo root = first ancestor containing agents/ (move-proof)
DEFAULT_PUBMEDQA_DATASET = (
    PROJECT_ROOT / "evals" / "disabled" / "PubmedQA" / "data" / "ori_pqal.json"
)
DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL = "openai/gpt-4o-2024-11-20"

PubMedQALabel = Literal["yes", "no", "maybe", "unanswered"]


class PubMedQAOpenJudgeResponse(BaseModel):
    predicted_label: PubMedQALabel = Field(
        description=(
            "The label implied by the model answer: yes, no, maybe, or unanswered."
        )
    )
    is_answered: bool = Field(
        description="False only when the answer does not commit to yes/no/maybe."
    )
    supporting_text_from_model_answer: str = Field(
        description="A short quote or paraphrase from the model answer."
    )
    reason: str = Field(description="Brief explanation of the label judgement.")


pubmedqa_open_judge_schema = ResponseSchema(
    name="pubmedqa_open_judge",
    json_schema=json_schema(PubMedQAOpenJudgeResponse),
)


def pubmedqa_open_judge_task(
    dataset_path: str | Path = DEFAULT_PUBMEDQA_DATASET,
    judge_model: str = DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
) -> Task:
    """PubMedQA PQA-L scored by judging open answers into yes/no/maybe labels."""

    return Task(
        dataset=_load_pubmedqa_dataset(dataset_path),
        solver=_not_implemented_solver(),
        scorer=score_pubmedqa_open_judge(judge_model),
        name="pubmedqa_open_judge",
        metadata={
            "pubmedqa_variant": "open_answer_label_judge",
            "dataset_path": str(dataset_path),
            "judge_model": judge_model,
        },
    )


def _load_pubmedqa_dataset(dataset_path: str | Path) -> MemoryDataset:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples: list[Sample] = []
    for pmid, entry in raw_data.items():
        question = str(entry.get("QUESTION") or "").strip()
        target = str(entry.get("final_decision") or "").strip().lower()
        if not question or target not in {"yes", "no", "maybe"}:
            continue

        contexts = [
            str(context).strip()
            for context in entry.get("CONTEXTS", [])
            if str(context).strip()
        ]
        samples.append(
            Sample(
                id=str(pmid),
                input=question,
                target=target,
                metadata={
                    "pmid": str(pmid),
                    "long_answer": str(entry.get("LONG_ANSWER") or "").strip(),
                    "contexts": contexts,
                    "year": entry.get("YEAR"),
                    "labels": entry.get("LABELS", []),
                    "meshes": entry.get("MESHES", []),
                },
            )
        )

    return MemoryDataset(samples)


def _not_implemented_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        raise NotImplementedError(
            "This is a placeholder solver. Please specify a real solver using "
            "--solver from the CLI."
        )

    return solve


@metric
def coverage() -> Metric:
    """Fraction of samples where the open answer commits to a PubMedQA label."""

    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        score_vals = [cast(dict[str, Any], score.score.value) for score in scores]
        return sum(bool(value.get("is_answered")) for value in score_vals) / len(
            score_vals
        )

    return metric_fn


@metric
def macro_f1() -> Metric:
    """Macro F1 over yes/no/maybe labels present in the evaluated targets."""

    def metric_fn(scores: list[SampleScore]) -> float:
        score_rows = [
            {
                "gold_label": str(score.score.metadata.get("gold_label", "")),
                "predicted_label": str(score.score.metadata.get("predicted_label", "")),
            }
            for score in scores
        ]
        labels = ("yes", "no", "maybe")
        present_labels = [
            label
            for label in labels
            if any(row["gold_label"] == label for row in score_rows)
        ]
        if not present_labels:
            return 0.0

        f1_values = []
        for label in present_labels:
            true_positive = sum(
                row["gold_label"] == label and row["predicted_label"] == label
                for row in score_rows
            )
            false_positive = sum(
                row["gold_label"] != label and row["predicted_label"] == label
                for row in score_rows
            )
            false_negative = sum(
                row["gold_label"] == label and row["predicted_label"] != label
                for row in score_rows
            )
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / max(1, true_positive + false_negative)
            if precision + recall == 0:
                f1_values.append(0.0)
                continue
            f1_values.append(2 * precision * recall / (precision + recall))

        return sum(f1_values) / len(f1_values)

    return metric_fn


@scorer(
    metrics=[
        {"is_correct": [accuracy(), stderr()]},
        coverage(),
        macro_f1(),
    ]
)
def score_pubmedqa_open_judge(
    judge_model: str = DEFAULT_PUBMEDQA_OPEN_JUDGE_MODEL,
) -> Scorer:
    grader_model = None

    async def score(state: TaskState, target: Target) -> Score:
        nonlocal grader_model
        if grader_model is None:
            grader_model = get_model(judge_model)

        question = str(state.input).strip()
        gold_label = target.text.strip().lower()
        open_answer = state.output.completion.strip()

        raw_judge_output = await grader_model.generate(
            _judge_prompt(question=question, open_answer=open_answer),
            config=GenerateConfig(
                temperature=0,
                max_tokens=512,
                response_schema=pubmedqa_open_judge_schema,
            ),
        )
        parsed = _parse_judge_output(raw_judge_output.completion)
        predicted_label = _normalize_pubmedqa_label(parsed.get("predicted_label"))
        fallback_label = _extract_pubmedqa_label(open_answer)
        judge_parse_failed = predicted_label is None
        if predicted_label is None:
            predicted_label = fallback_label or "unanswered"

        is_answered = _coerce_bool(
            parsed.get("is_answered"),
            default=predicted_label in {"yes", "no", "maybe"},
        )
        is_correct = is_answered and predicted_label == gold_label
        return Score(
            value={
                "is_correct": is_correct,
                "is_answered": is_answered,
            },
            answer=open_answer,
            explanation=str(parsed.get("reason") or ""),
            metadata={
                "judge_model": judge_model,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "fallback_label": fallback_label,
                "judge_parse_failed": judge_parse_failed,
                "judge_is_answered": is_answered,
                "judge_supporting_text": str(
                    parsed.get("supporting_text_from_model_answer") or ""
                ),
                "judge_raw_output": raw_judge_output.completion,
            },
        )

    return score


def _judge_prompt(question: str, open_answer: str) -> str:
    return (
        "Classify an open-form PubMedQA answer into the PubMedQA labels.\n"
        "Use only the question and the evaluated model answer. Do not use outside "
        "biomedical knowledge and do not decide whether the model is factually "
        "correct; only infer the label the model answer commits to.\n\n"
        "LABEL RULES:\n"
        "- `yes`: the answer affirms the question's proposition.\n"
        "- `no`: the answer rejects or contradicts the question's proposition.\n"
        "- `maybe`: the answer says the evidence is mixed, uncertain, limited, "
        "conditional, or insufficient for a yes/no conclusion.\n"
        "- `unanswered`: the answer does not commit to any of yes/no/maybe.\n\n"
        f"Question:\n{question}\n\n"
        f"Open answer from the evaluated model:\n{open_answer}\n\n"
        "Return JSON only with keys `predicted_label`, `is_answered`, "
        "`supporting_text_from_model_answer`, and `reason`."
    )


def _parse_judge_output(raw_output: str) -> dict[str, Any]:
    parsed = extract_json_from_response(raw_output)
    if isinstance(parsed, dict):
        return parsed
    try:
        loaded = json.loads(raw_output)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _normalize_pubmedqa_label(value: Any) -> PubMedQALabel | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"yes", "no", "maybe", "unanswered"}:
        return normalized  # type: ignore[return-value]
    return None


def _extract_pubmedqa_label(text: str) -> Literal["yes", "no", "maybe"] | None:
    matches = re.findall(r"<(yes|no|maybe)>|\b(yes|no|maybe)\b", text, re.I)
    labels = [tag or bare for tag, bare in matches]
    if not labels:
        return None
    return labels[-1].lower()  # type: ignore[return-value]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if value is None:
        return default
    return bool(value)

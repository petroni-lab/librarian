"""Open-answer LitQA2 variant with post-hoc gold-answer judging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from inspect_ai import Task, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy as _accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field
from typing import cast

from astabench.evals.labbench.litqa2.task import (
    coverage,
    litqa2,
    litqa2_test,
    litqa2_validation,
    precision,
)
from evals.Literature.AstaBench.compat import extract_json_from_response

try:
    from astabench.evals.labbench.litqa2.task import load_litqa2 as _load_litqa2
except Exception:  # pragma: no cover
    _load_litqa2 = None  # type: ignore[assignment]


DEFAULT_LITQA2_OPEN_JUDGE_MODEL = "openai/gpt-4o-2024-11-20"
DEFAULT_LITQA2_EUROPEPMC_FULLTEXT_SUBSET = (
    Path(__file__).resolve().parent
    / "data"
    / "litqa2_europepmc_fulltext"
    / "litqa2_full_europepmc_fulltext.json"
)


class LitQA2OpenJudgeResponse(BaseModel):
    contains_correct_answer: bool = Field(
        description="True only when the open answer clearly states the gold answer."
    )
    is_answered: bool = Field(
        description="False when the open answer is too ambiguous or insufficient."
    )
    supporting_text_from_model_answer: str = Field(
        description="A short quote or paraphrase from the open answer that supports the judgement."
    )
    reason: str = Field(description="Brief explanation of the judgement.")


litqa2_open_judge_schema = ResponseSchema(
    name="litqa2_open_judge",
    json_schema=json_schema(LitQA2OpenJudgeResponse),
)


def litqa2_open_judge_task(
    split: Literal["validation", "test", "full", "europepmc_fulltext"],
    with_search_tools: bool,
    judge_model: str = DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
) -> Task:
    """LitQA2 full-text task scored by judging an open answer against the gold answer."""

    key_passage_lookup: dict[str, str] | None = None
    source_pmid_lookup: dict[str, str] | None = None
    if split == "validation":
        base_task = litqa2_validation(with_search_tools=with_search_tools)
        litqa2_data_split = "dev"
    elif split == "test":
        base_task = litqa2_test(with_search_tools=with_search_tools)
        litqa2_data_split = "test"
    elif split == "europepmc_fulltext":
        base_task = litqa2(split="all", with_search_tools=with_search_tools)
        subset_rows = _load_litqa2_europepmc_fulltext_subset()
        subset_ids = {str(row["id"]) for row in subset_rows}
        subset_samples = [
            sample for sample in base_task.dataset if str(sample.id) in subset_ids
        ]
        base_task = task_with(base_task, dataset=MemoryDataset(subset_samples))
        litqa2_data_split = "europepmc_fulltext"
        key_passage_lookup = {
            str(obj["id"]): str(obj.get("key-passage") or "") for obj in subset_rows
        }
        source_pmid_lookup = _extract_source_pmids_from_rows(subset_rows)
    else:
        base_task = litqa2(split="all", with_search_tools=with_search_tools)
        litqa2_data_split = "all"

    if key_passage_lookup is None:
        key_passage_lookup = _build_key_passage_lookup(litqa2_data_split)
    if source_pmid_lookup is None:
        source_pmid_lookup = _build_source_pmid_lookup(litqa2_data_split)
    return task_with(
        base_task,
        dataset=_question_only_dataset(
            base_task, key_passage_lookup, source_pmid_lookup
        ),
        name=f"litqa2_open_judge_{split}",
        scorer=score_litqa2_open_judge(judge_model),
        metadata={
            **(base_task.metadata or {}),
            "litqa2_variant": "open_answer_gold_entailment",
            "litqa2_data_split": litqa2_data_split,
            "judge_model": judge_model,
        },
    )


def _load_litqa2_europepmc_fulltext_subset() -> list[dict[str, Any]]:
    if not DEFAULT_LITQA2_EUROPEPMC_FULLTEXT_SUBSET.exists():
        raise FileNotFoundError(
            "Missing LitQA2 EuropePMC fulltext subset. Generate it with "
            "`python evals/AstaBench/check_litqa2_europepmc_fulltext.py`."
        )
    with DEFAULT_LITQA2_EUROPEPMC_FULLTEXT_SUBSET.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON list in {DEFAULT_LITQA2_EUROPEPMC_FULLTEXT_SUBSET}"
        )
    return [dict(row) for row in payload]


def _build_key_passage_lookup(astabench_split: str) -> dict[str, str]:
    """Return {sample_id: key_passage} for the given split.

    Falls back to an empty dict if the raw data cannot be loaded (e.g. missing
    HF token), so the scorer degrades gracefully to gold-answer-only judging.
    """
    if _load_litqa2 is None:
        return {}
    try:
        raw_data = _load_litqa2(split=astabench_split)  # type: ignore[call-arg]
        return {str(obj["id"]): str(obj.get("key-passage") or "") for obj in raw_data}
    except Exception:  # pragma: no cover
        return {}


def _extract_source_pmids_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Extract {sample_id: source_pmid} from rows.

    Tries the ``sources`` field for ``PMID:...`` entries first, then falls
    back to resolving DOIs via the EuropePMC availability cache.
    """
    lookup: dict[str, str] = {}
    # Try the availability cache for DOI → PMID mapping
    doi_cache: dict[str, str] = {}
    cache_path = (
        Path(__file__).resolve().parent
        / "data"
        / "litqa2_europepmc_fulltext"
        / "source_availability_cache.json"
    )
    if cache_path.exists():
        try:
            with cache_path.open() as handle:
                raw_cache = json.load(handle)
            for doi_key, info in (raw_cache or {}).items():
                if isinstance(info, dict) and info.get("pmid"):
                    doi_cache[doi_key.strip().lower()] = str(info["pmid"]).strip()
        except Exception:
            pass

    for obj in rows:
        obj_id = str(obj["id"])
        sources = obj.get("sources") or []
        for src in sources:
            src_str = str(src).strip()
            # Direct PMID
            if src_str.upper().startswith("PMID:"):
                lookup[obj_id] = src_str[5:].strip()
                break
            # DOI → PMID via cache
            if src_str and doi_cache:
                for doi_key, pmid in doi_cache.items():
                    if src_str.lower() in doi_key or doi_key in src_str.lower():
                        lookup[obj_id] = pmid
                        break
                if obj_id in lookup:
                    break
    return lookup


def _build_source_pmid_lookup(astabench_split: str) -> dict[str, str]:
    """Return {sample_id: source_pmid} for each LitQA2 question.

    The LitQA2 ``sources`` field contains strings like ``"PMID:12345678"``.
    We extract the numeric PMID for matching against EuropePMC search results.

    For the ``europepmc_fulltext`` split this is handled inline via the subset
    rows; this function is called only for standard LitQA2 splits.
    """
    if _load_litqa2 is None:
        return {}
    try:
        raw_data = _load_litqa2(split=astabench_split)  # type: ignore[call-arg]
        return _extract_source_pmids_from_rows(raw_data)
    except Exception:
        return {}


def _question_only_dataset(
    base_task: Task,
    key_passage_lookup: dict[str, str] | None = None,
    source_pmid_lookup: dict[str, str] | None = None,
) -> MemoryDataset:
    """Strip MCQ choices from sample.input and optionally attach metadata."""
    samples = []
    for sample in base_task.dataset:
        clean_sample = sample.model_copy(deep=True)
        clean_sample.input = _question_text_from_input(str(clean_sample.input))
        meta = dict(clean_sample.metadata or {})
        if key_passage_lookup and str(sample.id) in key_passage_lookup:
            passage = key_passage_lookup[str(sample.id)]
            if passage:
                meta["key_passage"] = passage
        if source_pmid_lookup and str(sample.id) in source_pmid_lookup:
            meta["source_pmid"] = source_pmid_lookup[str(sample.id)]
        if meta:
            clean_sample.metadata = meta
        samples.append(clean_sample)
    return MemoryDataset(samples)


@metric
def accuracy() -> Metric:
    """Accuracy is the fraction of correct answers (ignoring sureness)."""

    def metric_fn(scores: list[SampleScore]) -> float:
        score_vals: list[dict[str, bool]] = [
            cast(dict[str, bool], score.score.value) for score in scores
        ]
        return sum(v["is_correct"] for v in score_vals) / max(1, len(score_vals))

    return metric_fn


@scorer(
    metrics=[
        {"is_correct": [_accuracy(), stderr()]},
        accuracy(),
        precision(),
        coverage(),
    ]
)
def score_litqa2_open_judge(
    judge_model: str = DEFAULT_LITQA2_OPEN_JUDGE_MODEL,
) -> Scorer:
    grader_model = get_model(judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        choices = _choices_from_state(state)
        unsure_letter = str(state.metadata.get("unsure_letter") or "").strip().upper()
        target_letter = target.text.strip().upper()
        gold_answer = _answer_text_for_letter(choices, target_letter)
        open_answer = state.output.completion.strip()
        # key_passage is the verbatim text from the target paper that answers the
        # question; injected by _question_only_dataset via _build_key_passage_lookup.
        key_passage = str(state.metadata.get("key_passage") or "").strip()

        raw_judge_output = await grader_model.generate(
            _judge_prompt(
                question=_question_from_state(state),
                gold_answer=gold_answer,
                open_answer=open_answer,
                key_passage=key_passage,
            ),
            config=GenerateConfig(
                temperature=0,
                max_tokens=512,
                response_schema=litqa2_open_judge_schema,
            ),
        )
        parsed = _parse_judge_output(raw_judge_output.completion)
        is_correct = _coerce_bool(parsed.get("contains_correct_answer"))
        is_answered = _coerce_bool(parsed.get("is_answered"), default=is_correct)
        is_sure = bool(is_answered or is_correct)
        return Score(
            value={"is_correct": is_correct, "is_sure": is_sure},
            answer=open_answer,
            explanation=str(parsed.get("reason") or ""),
            metadata={
                "judge_model": judge_model,
                "target_answer": target_letter,
                "gold_answer": gold_answer,
                "key_passage": key_passage,
                "unsure_letter": unsure_letter,
                "judge_is_answered": is_answered,
                "judge_contains_correct_answer": is_correct,
                "judge_supporting_text": str(
                    parsed.get("supporting_text_from_model_answer") or ""
                ),
                "judge_raw_output": raw_judge_output.completion,
            },
        )

    return score


def _choices_from_state(state: TaskState) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    if getattr(state, "choices", None):
        for idx, choice in enumerate(state.choices):
            choices.append((chr(ord("A") + idx), str(choice.value)))
    if choices:
        return choices

    for match in re.finditer(r"^([A-Z])\.\s+(.*)$", str(state.input), re.MULTILINE):
        choices.append((match.group(1), match.group(2).strip()))
    return choices


def _answer_text_for_letter(choices: list[tuple[str, str]], letter: str) -> str:
    for choice_letter, choice_text in choices:
        if choice_letter == letter:
            return choice_text
    return ""


def _question_from_state(state: TaskState) -> str:
    return _question_text_from_input(str(state.input))


def _question_text_from_input(sample_input: str) -> str:
    return sample_input.split("\n\n", 1)[0].strip()


def _judge_prompt(
    question: str,
    gold_answer: str,
    open_answer: str,
    key_passage: str = "",
) -> str:
    """Build the judge prompt.

    When key_passage is available it is included as the authoritative reference
    text from the target paper, which lets the judge assess semantic equivalence
    much more reliably than comparing to the terse MCQ option string alone.
    """
    passage_block = (
        f"Key passage from the target paper:\n{key_passage}\n\n" if key_passage else ""
    )
    return (
        "Evaluate whether an open-form scientific answer contains the known gold answer.\n"
        "Use only the open answer text and the optional key passage. "
        "Do not use outside knowledge.\n\n"
        "SCORING RULES — apply in order:\n"
        "1. Mark `contains_correct_answer` TRUE if the open answer explicitly states "
        "the gold answer, a scientifically equivalent expression, or an unambiguous "
        "paraphrase. Use the key passage (when provided) as the authoritative reference "
        "for what the correct answer looks like in natural language. "
        "Examples of acceptable equivalents:\n"
        "   - Same numeric value in different notation (e.g. '2.7-fold' = '2.7 fold', "
        "'43.6%' = '43.6').\n"
        "   - Core scientific claim present even if an MCQ-style parenthetical label "
        "is absent (e.g. gold is 'Supercoiled and doubly tethered (Y-shape)' and the "
        "answer says 'Supercoiled DNA' — accept if the answer clearly identifies the "
        "correct structure and the missing label was only an MCQ discriminator).\n"
        "   - Gene/protein synonym or alias that unambiguously refers to the same entity.\n"
        "2. Mark `contains_correct_answer` FALSE if the open answer is missing the key "
        "fact, is vague, contradicts the gold answer, or says the evidence is "
        "insufficient.\n"
        "3. Mark `is_answered` FALSE only if the open answer does not commit to any "
        "specific answer at all (pure abstention, 'I don't know', 'evidence is "
        "insufficient', etc.).\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"{passage_block}"
        f"Open answer from the evaluated model:\n{open_answer}\n\n"
        "Return JSON only with keys `contains_correct_answer`, `is_answered`, "
        "`supporting_text_from_model_answer`, and `reason`."
    )


def _parse_judge_output(raw_output: str) -> dict[str, Any]:
    parsed = extract_json_from_response(raw_output)
    if isinstance(parsed, dict):
        return parsed
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if value is None:
        return default
    return bool(value)

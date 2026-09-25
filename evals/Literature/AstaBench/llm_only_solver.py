"""LLM-only baselines for AstaBench literature tasks."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from inspect_ai.solver import Generate, Solver, TaskState, solver

from evals.Literature.AstaBench.bio_agent_wrapper import InspectUsageRecorder
from evals.Literature.AstaBench.compat import (
    build_llm_client,
    extract_json_from_response,
)
from librarian.llm_client import LLMClient

TaskKind = Literal[
    "auto",
    "paper_finder",
    "litqa2",
    "litqa2_open",
    "litqa2_open_llm_only",
    "pubmedqa_open",
    "sqa",
    "arxivdigestables",
]


class LLMOnlyAstaWrapper:
    """Task-aware LLM-only baseline for AstaBench literature tasks."""

    def __init__(
        self,
        task_type: TaskKind = "auto",
        llm_base_url: str | None = None,
        llm_model_name: str | None = None,
        reasoning_effort: str = "low",
    ) -> None:
        self.task_type = task_type
        self.llm_client = build_llm_client(
            base_url=llm_base_url,
            model_name=llm_model_name,
            reasoning_effort=reasoning_effort,
            usage_callback=InspectUsageRecorder(),
        )

    async def solve(self, state: TaskState) -> TaskState:
        task_type = self._infer_task_type(state)
        if task_type == "paper_finder":
            return await self._solve_paper_finder(state)
        if task_type == "litqa2":
            return await self._solve_litqa2(state)
        if task_type in {"litqa2_open", "litqa2_open_llm_only"}:
            return await self._solve_litqa2_open(state)
        if task_type == "pubmedqa_open":
            return await self._solve_pubmedqa_open(state)
        if task_type == "sqa":
            return await self._solve_sqa(state)
        if task_type == "arxivdigestables":
            return await self._solve_arxivdigestables(state)
        raise ValueError(f"Unsupported task type '{task_type}'.")

    def _infer_task_type(self, state: TaskState) -> TaskKind:
        if self.task_type != "auto":
            return self.task_type

        metadata = dict(state.metadata or {})
        sample_input = self._state_input_text(state)
        has_choices = bool(getattr(state, "choices", None))
        try:
            has_choices = has_choices and len(state.choices) > 0
        except TypeError:
            pass
        if metadata.get("raw_query") or sample_input.startswith(
            "Find papers relevant to the following query:"
        ):
            return "paper_finder"
        if metadata.get("unsure_letter") or has_choices:
            return "litqa2"
        if metadata.get("case_id") or metadata.get("initial_prompt"):
            return "sqa"
        if metadata.get("corpus_ids"):
            return "arxivdigestables"
        raise ValueError("Could not infer AstaBench task type from TaskState.")

    async def _solve_paper_finder(self, state: TaskState) -> TaskState:
        # LLM-only baselines cannot retrieve real corpus ids. Emit an empty result set.
        state.output.completion = json.dumps({"output": {"results": []}})
        return state

    async def _solve_litqa2(self, state: TaskState) -> TaskState:
        metadata = dict(state.metadata or {})
        question = self._extract_question_from_multichoice_input(state)
        choices = self._extract_choices(state)

        prompt = (
            "You are answering a multiple-choice scientific literature question.\n"
            "Pick the single best answer letter from the choices.\n"
            'Return JSON only in the form {"answer": "<letter>"}.\n\n'
            f"Question:\n{question}\n\n"
            "Choices:\n" + "\n".join(f"{letter}. {text}" for letter, text in choices)
        )

        raw_output = await asyncio.to_thread(
            self.llm_client.generate_structured_output,
            prompt,
            "You are a careful scientist. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output) or {}
        answer = str(parsed.get("answer") or "").strip().upper()
        valid_letters = {letter for letter, _ in choices}
        if answer not in valid_letters:
            answer = str(metadata.get("unsure_letter") or "A").strip().upper()

        if getattr(state, "choices", None):
            for idx in range(len(state.choices)):
                state.choices.mark_choice(idx, idx == (ord(answer) - ord("A")))

        state.output.completion = json.dumps({"answer": answer})
        return state

    async def _solve_litqa2_open(self, state: TaskState) -> TaskState:
        question = self._extract_question_from_multichoice_input(state)
        prompt = (
            "Answer the following scientific question directly and concisely. "
            "Always commit to a specific answer.\n\n"
            f"Question:\n{question}\n\n"
            "Answer with a short specific phrase or value only."
        )
        answer = await asyncio.to_thread(
            self.llm_client.generate_structured_output,
            prompt,
            "You are a knowledgeable scientist.",
        )
        state.output.completion = answer.strip() or "No answer generated."
        return state

    async def _solve_pubmedqa_open(self, state: TaskState) -> TaskState:
        question = self._state_input_text(state).strip()
        prompt = (
            "Answer the following PubMedQA biomedical question directly. "
            "Give a concise answer that clearly indicates yes, no, or maybe, "
            "with one short sentence of rationale.\n\n"
            f"Question:\n{question}"
        )
        answer = await asyncio.to_thread(
            self.llm_client.generate_structured_output,
            prompt,
            "You are a careful biomedical scientist.",
        )
        state.output.completion = answer.strip() or "No answer generated."
        return state

    async def _solve_sqa(self, state: TaskState) -> TaskState:
        metadata = dict(state.metadata or {})
        question = str(
            metadata.get("initial_prompt") or self._extract_user_query(state)
        ).strip()
        prompt = (
            "Generate a ScholarQA-style response. Return JSON with a top-level "
            "`sections` list. Each section must include `title`, `text`, and "
            "`citations` (a list). If you cannot provide citations, return an "
            "empty list.\n\n"
            f"Question:\n{question}\n\n"
            "Return JSON only."
        )

        raw_output = await asyncio.to_thread(
            self.llm_client.generate_structured_output,
            prompt,
            "You are a precise report formatter. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output) or {}
        normalized = self._normalize_sqa_response(parsed, fallback_text=raw_output)
        state.output.completion = json.dumps(normalized, indent=2, ensure_ascii=False)
        return state

    async def _solve_arxivdigestables(self, state: TaskState) -> TaskState:
        sample_input = self._state_input_text(state)
        prompt = (
            "You are building an ArxivDIGESTables-style comparison table.\n"
            'Return JSON only with the schema {"cell_values": '
            '[{"paper_id": ..., "column_name": ..., "cell_value": ...}, ...]}.\n\n'
            f"Task input:\n{sample_input}"
        )

        raw_output = await asyncio.to_thread(
            self.llm_client.generate_structured_output,
            prompt,
            "You are a careful table formatter. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output)
        if not parsed or "cell_values" not in parsed:
            parsed = {"cell_values": []}
        state.output.completion = json.dumps(parsed, indent=2, ensure_ascii=False)
        return state

    @staticmethod
    def _normalize_sqa_response(
        parsed: dict[str, Any],
        fallback_text: str,
    ) -> dict[str, Any]:
        if isinstance(parsed, dict) and parsed.get("sections"):
            return parsed
        return {
            "sections": [
                {
                    "title": "Answer",
                    "text": fallback_text.strip() or "No answer generated.",
                    "citations": [],
                }
            ]
        }

    @staticmethod
    def _state_input_text(state: TaskState) -> str:
        return str(getattr(state, "input", "") or "")

    def _extract_user_query(self, state: TaskState) -> str:
        sample_input = self._state_input_text(state)
        if sample_input.startswith("Find papers relevant to the following query:"):
            return sample_input.split("\n", 1)[0].split(":", 1)[1].strip()
        return sample_input.strip()

    def _extract_question_from_multichoice_input(self, state: TaskState) -> str:
        sample_input = self._state_input_text(state)
        return sample_input.split("\n\n", 1)[0].strip()

    def _extract_choices(self, state: TaskState) -> list[tuple[str, str]]:
        choices = []
        if getattr(state, "choices", None):
            for idx, choice in enumerate(state.choices):
                value = getattr(choice, "value", None)
                if value is None and hasattr(choice, "text"):
                    value = choice.text
                choices.append((chr(ord("A") + idx), str(value)))
            if choices:
                return choices

        sample_input = self._state_input_text(state)
        for match in re.finditer(r"^([A-Z])\.\s+(.*)$", sample_input, re.MULTILINE):
            choices.append((match.group(1), match.group(2).strip()))
        return choices


@solver
def llm_only_solver(
    task_type: TaskKind = "auto",
    llm_base_url: str | None = None,
    llm_model_name: str | None = None,
    reasoning_effort: str = "low",
) -> Solver:
    """LLM-only solver for AstaBench literature tasks."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        wrapper = LLMOnlyAstaWrapper(
            task_type=task_type,
            llm_base_url=llm_base_url,
            llm_model_name=llm_model_name,
            reasoning_effort=reasoning_effort,
        )
        return await wrapper.solve(state)

    return solve


__all__ = ["llm_only_solver", "LLMOnlyAstaWrapper"]

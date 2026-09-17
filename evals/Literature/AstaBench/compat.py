"""Small local compatibility helpers for the AstaBench integration."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from inspect_ai.agent import as_solver, bridge
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage
from inspect_ai.model._model import record_and_check_model_usage
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef

try:
    from inspect_ai.event import ModelEvent
except ImportError:  # pragma: no cover - older inspect_ai
    from inspect_ai.log import ModelEvent

from inspect_ai.log import transcript

# The benchmarks' LLM-client surface; see evals/Literature/llm_compat.py for
# what it reconciles. Re-exported here so AstaBench call sites keep one import.
from evals.Literature.llm_compat import build_llm_client  # noqa: F401

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract the outermost JSON object from a model or tool response."""

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    candidates = _json_object_candidates(response)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    logger.debug("No parseable JSON object found in response.")
    return None


def _json_object_candidates(text: str) -> list[str]:
    """Return candidate JSON object substrings from longest to shortest."""

    candidates: list[str] = []
    stack: list[int] = []
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append(idx)
            continue
        if ch == "}" and stack:
            start = stack.pop()
            fragment = text[start : idx + 1].strip()
            if fragment:
                candidates.append(fragment)

    # Try larger fragments first (more likely to be full objects than nested values)
    candidates.sort(key=len, reverse=True)
    return candidates


@solver
def merge_tools_with_state(
    tools: list[Tool],
    prefer_given_tools: bool = False,
    select_fn: Callable[[ToolDef], bool] | None = None,
) -> Solver:
    """Merge solver-provided tools into task state, preferring task tools by default."""

    select_fn = select_fn or (lambda td: True)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        preferred_tools = [ToolDef(t) for t in state.tools]
        other_tools = [ToolDef(t) for t in tools]

        if prefer_given_tools:
            preferred_tools, other_tools = other_tools, preferred_tools

        preferred_tool_names = {tool.name for tool in preferred_tools}
        other_tools = [
            tool for tool in other_tools if tool.name not in preferred_tool_names
        ]
        state.tools = [
            tool.as_tool() for tool in preferred_tools + other_tools if select_fn(tool)
        ]
        return state

    return solve


def record_model_usage_with_inspect(
    model_name: str,
    usage: ModelUsage,
    allow_invalid: bool = False,
) -> None:
    """Record token usage into Inspect summaries for non-Inspect model calls."""

    if not _is_valid_model_usage(usage):
        if allow_invalid:
            logger.warning("Skipping invalid token usage for model '%s'.", model_name)
            return
        raise ValueError(f"Invalid usage payload for model '{model_name}'.")

    event = ModelEvent(
        model=model_name,
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(model=model_name, usage=usage),
        cache=None,
        call=None,
        pending=False,
    )
    transcript()._event(event)
    record_and_check_model_usage(model_name, usage)


def _is_valid_model_usage(usage: ModelUsage) -> bool:
    counts = [usage.input_tokens, usage.output_tokens, usage.total_tokens]
    if not all(isinstance(count, int) and count >= 0 for count in counts):
        return False
    if usage.total_tokens < max(usage.input_tokens, usage.output_tokens):
        return False
    return True


@solver
def full_state_bridge(func: Solver) -> Solver:
    """Allow a bridged solver to work with the full Inspect TaskState."""

    async def outer_solver(state: TaskState, generate: Generate) -> TaskState:
        final_state: TaskState = state

        async def inner_solver(sample: dict[str, Any]) -> dict[str, Any]:
            del sample
            nonlocal final_state
            final_state = await func(state, generate)
            return {"output": ""}

        bridged_solver = as_solver(bridge(inner_solver))
        await bridged_solver(state, generate)
        return final_state

    return outer_solver

"""The LLM-client surface the benchmarks were written against.

The benchmarks were written against a larger internal LLM client.
``librarian.llm_client`` is the version that shipped: same name, same
``chat_completion``, without the conveniences the internal one had. This module
holds the difference, in one place.

Three things differ:

- ``generate_structured_output(prompt, system_message)`` is gone. It was a thin
  wrapper — system message plus user prompt, one ``chat_completion`` — so it is
  reimplemented here verbatim.
- ``thinking=`` / ``reasoning_effort=`` was a constructor argument. It is now an
  attribute the client seeds from ``LLM_REASONING_EFFORT`` and reads on every
  call, so a per-run override is an assignment rather than an argument.
- ``usage_callback=`` has no equivalent. The shipped client returns the
  assistant's text and discards the response's ``usage`` block, so per-call token
  counts are **not recoverable**. It is accepted and ignored, which means a run's
  reported token and cost totals stay empty. Scores are unaffected.
"""

from __future__ import annotations

from typing import Any, Callable

from librarian.llm_client import LLMClient


class EvalLLMClient(LLMClient):
    """``LLMClient`` plus the one convenience method the benchmarks call."""

    def generate_structured_output(
        self, prompt: str, system_message: str = "You are a helpful assistant."
    ) -> str:
        """Answer a single prompt under a system message.

        The benchmarks call this wherever they expect JSON back. Parsing is the
        caller's job: the name describes what the model is asked for, not what
        this method enforces.

        :param prompt: The user prompt.
        :param system_message: Optional system instruction.
        :returns: The assistant's raw text.
        """
        return self.chat_completion(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        )


def as_effort(thinking: bool | str | None) -> str | None:
    """Map the benchmarks' ``--thinking`` flag to a ``reasoning_effort`` value.

    The internal client took ``thinking=`` as either a bool or an effort string
    and normalised it the same way. The benchmarks still pass the bool.

    :param thinking: ``True``/``False``, an effort string, or ``None``.
    :returns: The effort string, or ``None`` to leave the client's own alone.
    """
    if thinking is None:
        return None
    if isinstance(thinking, str):
        return thinking
    return "max" if thinking else "low"


def build_llm_client(
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    reasoning_effort: bool | str | None = None,
    usage_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvalLLMClient:
    """Build the client the benchmarks expect.

    :param base_url: OpenAI-compatible base URL, or ``None`` to let the client
        resolve ``LLM_BASE_URL``.
    :param model_name: Model id, or ``None`` to let it resolve ``LLM_MODEL``.
    :param api_key: Bearer token, or ``None`` to let it resolve ``LLM_API_KEY``.
    :param reasoning_effort: Per-run effort level, as an effort string or the
        benchmarks' ``--thinking`` bool (see :func:`as_effort`). ``None`` keeps
        whatever the constructor read from ``LLM_REASONING_EFFORT``.
    :param usage_callback: Accepted for call-site compatibility; ignored, see
        the module docstring.
    """
    client = EvalLLMClient(base_url=base_url, api_key=api_key, model_name=model_name)
    effort = as_effort(reasoning_effort)
    if effort:
        client.reasoning_effort = effort
    return client


def _self_check() -> None:
    """Assert the shim's shape without opening a socket."""
    client = build_llm_client(
        "http://localhost:1/v1", "m", reasoning_effort="high", usage_callback=lambda *a: None
    )
    assert client.reasoning_effort == "high"
    assert client.model_name == "m"

    captured: dict[str, Any] = {}
    client.chat_completion = lambda messages, **kw: captured.update(messages=messages) or "ok"
    assert client.generate_structured_output("Q", system_message="S") == "ok"
    assert captured["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
    ], captured

    # The benchmarks pass a bool, which must become an effort string rather
    # than reaching the API as `"reasoning_effort": true`.
    assert as_effort(True) == "max" and as_effort(False) == "low"
    assert as_effort("high") == "high" and as_effort(None) is None
    assert build_llm_client("http://localhost:1/v1", "m", reasoning_effort=True).reasoning_effort == "max"

    # An omitted effort must not clobber what the environment set.
    import os

    os.environ["LLM_REASONING_EFFORT"] = "low"
    assert build_llm_client("http://localhost:1/v1", "m").reasoning_effort == "low"
    print("llm_compat self-check ok")


if __name__ == "__main__":
    _self_check()

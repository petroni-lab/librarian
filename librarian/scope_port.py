"""Scope port the synthesis agent depends on — no model or backend implied.

Defines the contract :class:`~librarian.synthesis.SynthesisAgent` calls to decide
whether a question is one it should answer (:class:`ScopePort`), the verdict it gets
back (:class:`ScopeVerdict`), and :class:`ZeroShotScopeGuard`, a ready-made adapter
that asks the LLM once, with the plain list of accepted questions in
``prompts/scope.md``.

A guard can accept a question, refuse it, or accept only part of it: the agent then
answers that part and adds a sentence about the rest.

Nothing is checked unless a guard is passed: ``SynthesisAgent(scope_guard=...)``.
To use your own rules or your own model, implement ``ScopePort`` and pass that
instead — nothing in this module knows your implementation exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from librarian.llm_client import create_llm_client
from librarian.synthesis import _fill, _render_history_block

_SCOPE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "scope.md"

# The guard is a yes/no decision, so it is deterministic. The JSON is short, but a
# model that reasons before answering spends tokens first: too small a budget cuts
# the JSON off, and an unreadable verdict lets the question through.
_SCOPE_TEMPERATURE = 0.0
_SCOPE_MAX_TOKENS = 2048

# Said when a guard refuses a question without saying what to reply. Fixed text, so it
# cannot follow the question's language; a guard should supply its own reply.
DEFAULT_REDIRECT = (
    "I can only help with scientific research questions. Ask me about a paper, "
    "a finding or a research topic."
)

# Said after the answer when a guard accepts only part of a question and gives no
# sentence of its own. Fixed text, like DEFAULT_REDIRECT.
DEFAULT_NOTE = (
    "I answered the scientific research part of your message. "
    "I can only help with scientific research questions."
)


# The prompt wraps the question and the earlier turns in these tags, so that what the
# user wrote is plainly data. A copy of a tag inside that text would end the block
# early, so any such copy is made inert before the prompt is filled.
_DELIMITER_RE = re.compile(r"<\s*/?\s*(?:question|conversation)\s*>", re.IGNORECASE)


def _neutralize(text: str) -> str:
    """Make any copy of the prompt's own delimiters inside ``text`` inert.

    :param text: Text written by the user (a question or an earlier turn).
    :type text: str
    :return: ``text`` with the angle brackets of a delimiter tag replaced by
        look-alike characters; everything else is unchanged.
    :rtype: str
    """
    return _DELIMITER_RE.sub(
        lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text
    )


def _comparable(text: str) -> str:
    """Reduce ``text`` to its lower-cased words, for comparing two texts.

    Case, punctuation and spacing are set aside, so ``"What is CRISPR?"`` and
    ``"what is crispr"`` compare equal.

    :param text: Any text.
    :type text: str
    :return: The letters and digits of ``text``, lower-cased, words separated by a
        single space.
    :rtype: str
    """
    return " ".join("".join(c if c.isalnum() else " " for c in text.casefold()).split())


def _read_verdict(response: Any) -> dict[str, Any] | None:
    """Find the verdict in a model answer: its last JSON object with ``in_scope``.

    Reads every JSON object in the answer instead of the span from the first ``{`` to
    the last ``}``, so reasoning written before the verdict, even reasoning that
    contains braces, does not hide it. The last such object wins: the answer comes
    after the reasoning.

    :param response: The model's answer.
    :type response: object
    :return: The verdict object, or ``None`` when the answer has none.
    :rtype: dict or None
    """
    if not isinstance(response, str):
        return None
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    pos = response.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(response, pos)
        except (ValueError, RecursionError):
            pos = response.find("{", pos + 1)
            continue
        if isinstance(obj, dict) and "in_scope" in obj:
            found = obj
        pos = response.find("{", end)
    return found


@dataclass(frozen=True)
class ScopeVerdict:
    """What a guard decided about one question.

    :param in_scope: ``True`` when the question is one the agent should answer.
    :type in_scope: bool
    :param reply: The text to give the user instead of an answer when ``in_scope``
        is ``False``. A blank reply is replaced by ``DEFAULT_REDIRECT``, so a refused
        question is never answered with nothing.
    :type reply: str
    :param label: A short reason, for traces and evaluations (for example
        ``"in_scope"``, ``"out_of_scope"``, ``"partial"`` or ``"unreadable"``).
    :type label: str
    :param query: Only when ``in_scope`` is ``True`` and part of the question is not
        accepted: the accepted part, copied from the question. The agent answers
        this instead of the whole question. Empty means the whole question goes on.
    :type query: str
    :param note: With ``query``: one sentence, shown after the answer, saying that
        the rest was not answered. A blank note is replaced by ``DEFAULT_NOTE``;
        without ``query`` there is no note.
    :type note: str
    """

    in_scope: bool
    reply: str = ""
    label: str = ""
    query: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.in_scope:
            if not self.reply.strip():
                object.__setattr__(self, "reply", DEFAULT_REDIRECT)
            object.__setattr__(self, "query", "")
            object.__setattr__(self, "note", "")
        elif self.query.strip():
            object.__setattr__(self, "query", self.query.strip())
            if not self.note.strip():
                object.__setattr__(self, "note", DEFAULT_NOTE)
        else:
            object.__setattr__(self, "query", "")
            object.__setattr__(self, "note", "")


class ScopePort(Protocol):
    """Contract the synthesis agent depends on to decide whether to answer."""

    def check(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> ScopeVerdict:
        """Decide whether ``query`` is one the agent should answer.

        :param query: The user's question.
        :type query: str
        :param conversation_history: The earlier turns, oldest first, as
            ``{"role": ..., "content": ...}`` dicts, or ``None``.
        :type conversation_history: list[dict[str, str]] or None
        :return: The verdict. Errors from the model or the network are not caught
            here: they reach the caller of ``run``, as the router's do.
        :rtype: ScopeVerdict
        """
        ...


class ZeroShotScopeGuard:
    """Ready-made guard: one zero-shot model call against ``prompts/scope.md``."""

    def __init__(self, llm_client: Any | None = None):
        """Build a guard that asks ``llm_client`` whether a question is accepted.

        :param llm_client: Anything with ``chat_completion(messages,
            temperature=..., max_tokens=...) -> str``, like
            ``llm_client.LLMClient``. When omitted one is built from the ``LLM_*``
            environment variables, as :class:`~librarian.synthesis.SynthesisAgent`
            does.
        :type llm_client: object or None
        """
        self.llm = llm_client if llm_client is not None else create_llm_client()
        self._prompt = _SCOPE_PROMPT_PATH.read_text(encoding="utf-8")

    def check(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ScopeVerdict:
        """Ask the model whether ``query`` is accepted.

        An answer that cannot be read as a verdict lets the question through, with
        the label ``"unreadable"``: a guard that blocks on its own malfunction would
        refuse real research questions.

        The verdict is the last JSON object in the answer that has an ``in_scope``
        key, so reasoning written before it does not hide it. The question is
        accepted in part only when the model gives both the part (``query``) and the
        sentence about the rest (``note``), and the part differs from the question
        once case and punctuation are set aside: a ``query`` that only echoes the
        question is not a restriction.

        :param query: The user's question.
        :type query: str
        :param conversation_history: The earlier turns, oldest first.
        :type conversation_history: list[dict[str, str]] or None
        :return: The verdict, labelled ``"in_scope"``, ``"out_of_scope"``,
            ``"partial"`` (part of the question accepted) or ``"unreadable"``.
        :rtype: ScopeVerdict
        """
        history = _render_history_block(conversation_history)
        history_block = (
            f"<conversation>\n{_neutralize(history)}\n</conversation>\n\n"
            if history
            else ""
        )
        prompt = _fill(
            self._prompt,
            {"{history_block}": history_block, "{query}": _neutralize(query)},
        )
        response = self.llm.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=_SCOPE_TEMPERATURE,
            max_tokens=_SCOPE_MAX_TOKENS,
        )
        parsed = _read_verdict(response)
        flag = parsed.get("in_scope") if parsed is not None else None
        if isinstance(flag, str):
            flag = {"true": True, "false": False}.get(flag.strip().lower())
        if not isinstance(flag, bool):
            return ScopeVerdict(in_scope=True, label="unreadable")
        if flag:
            accepted = str(parsed.get("query") or "").strip()
            note = str(parsed.get("note") or "").strip()
            if note and accepted and _comparable(accepted) != _comparable(query):
                return ScopeVerdict(
                    in_scope=True, label="partial", query=accepted, note=note
                )
            return ScopeVerdict(in_scope=True, label="in_scope")
        reply = str(parsed.get("reply") or "").strip()
        return ScopeVerdict(in_scope=False, reply=reply, label="out_of_scope")

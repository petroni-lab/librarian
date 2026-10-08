"""SynthesisAgent — answers a question over the literature with a cited answer.

Two ways in, both through ``run``:

* ``run(query, passages)`` — the caller already has the evidence: the records
  :class:`~librarian.agent.LibrarianAgent`'s ``run`` returns, live or
  round-tripped through a run's ``04_evidence.json``. The agent writes the
  grounded answer and nothing else.
* ``run(query)`` on an agent built with ``librarian=...`` — the full path: route
  the question, search the literature when it needs it, then answer.

Either way ``run`` returns the answer text. Everything else that happened (the
route taken, the standalone query, the passages, the sub-queries) is left in
``last_run_debug`` for the caller to read.

``run`` is a thin sequence of small methods — ``_screen_query``, ``_route``,
``_retrieve``, ``_output_formatting``, ``_build_prompt``, ``_summarize`` — and
hands each the arguments it was called with (conversation history, output
channel, progress and stream callbacks), so a subclass can replace one step
without rewriting the flow.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from librarian.citations import render_papers
from librarian.llm_client import create_llm_client, parse_json_response
from librarian.tracing_port import NullTracer, TracingPort

if TYPE_CHECKING:
    from librarian.agent import LibrarianAgent

# Prompts — co-located with the planner and judge prompts. The skill reads the
# summarizer file (skills/librarian/SKILL.md, Step 2), so the citation discipline
# is identical whichever path produced the answer.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_SUMMARIZER_PROMPT_PATH = _PROMPTS_DIR / "summarizer.md"
_ROUTER_PROMPT_PATH = _PROMPTS_DIR / "router.md"

# Synthesis is grounded rewriting of supplied evidence, not creative writing:
# sampling entropy is exactly where a paraphrase drifts onto the wrong paralog.
# Not 0.0 — long answers at 0.0 can degenerate into repetition.
_SYNTHESIS_TEMPERATURE = 0.2
_SYNTHESIS_MAX_TOKENS = 8192

# Routing is a yes/no decision, so it is deterministic. Its JSON is short for a
# "search", but a "reply" carries a whole earlier answer with its citation links:
# too small a budget cuts the JSON off, the parse fails and the agent silently
# falls back to a search.
_ROUTER_TEMPERATURE = 0.0
_ROUTER_MAX_TOKENS = 4096

# How much of the conversation the router and the summarizer get to see. The
# per-message limit must fit a full cited answer: the links sit at the end of
# it, and cutting there loses them.
_HISTORY_MAX_MESSAGES = 6
_HISTORY_MAX_CHARS = 8000

_OUTPUT_CHANNEL = "terminal"
_FORMATTING_GUIDANCE = (
    "Format the answer as plain text for a terminal. Use short paragraphs and "
    "simple '- ' bullets. Write section headings as a bare line of text, not as "
    "Markdown '#' headings. Do not use tables, HTML, or bold/italic markup — a "
    "terminal renders none of it. Leave the markdown citation links exactly as "
    "each paper's 'Cite as:' line gives them: terminals make the URL clickable."
)

# Used when the router did not report a language. It must never name a concrete
# language: a named example primes the model to answer in it regardless of the
# question's own language.
_ANSWER_LANGUAGE = "the language the USER QUERY above is written in"
_NO_HISTORY = "(none — first turn)"

# The search ran and the judge kept nothing. That is a reportable answer, and a
# different thing from the search failing, which propagates as an exception.
_NO_PAPERS_MESSAGE = (
    "The literature search ran but returned no relevant papers for this "
    "question. Try rephrasing or broadening the query."
)

# The router is asked for the *name* of the language, but some models answer with
# an ISO 639-1 code ("en"). The summarizer is told the language outright, so both
# forms are turned into the same name. A code missing from this table passes
# through unchanged.
_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
# A two-letter code, optionally with a region or script ("pt-BR", "zh_Hans").
_LANGUAGE_CODE = re.compile(r"^([A-Za-z]{2})(?:[-_][A-Za-z0-9]+)*$")


def _fill(template: str, values: dict[str, str]) -> str:
    """Substitute every ``{placeholder}`` in ``template`` in a single pass.

    One pass is the point: chained ``str.replace`` calls rescan text they just
    inserted, so a question or a paper containing ``{papers_text}`` would have
    it expanded. Also avoids ``str.format``, which chokes on the literal braces
    in the prompt's own citation examples.

    :param template: The prompt text carrying ``{placeholder}`` tokens.
    :type template: str
    :param values: Placeholder (including braces) to replacement text.
    :type values: dict[str, str]
    :return: The template with every known placeholder substituted.
    :rtype: str
    """
    pattern = re.compile("|".join(re.escape(key) for key in values))
    return pattern.sub(lambda match: values[match.group(0)], template)


def _normalize_language(value: Any) -> str:
    """Turn the language the router reported into a language name.

    ``"en"``, ``"English"`` and ``"english"`` all give ``"English"``; a code
    with a region (``"pt-BR"``) gives the name of its language. A code that is
    not in ``_LANGUAGE_NAMES`` is returned as written; any other text is taken
    for a name and gets its first letter capitalised.

    :param value: The ``language`` field of the router's JSON, as parsed.
    :type value: Any
    :return: The language name, or ``""`` when the router gave none.
    :rtype: str
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    match = _LANGUAGE_CODE.match(text)
    if match:
        return _LANGUAGE_NAMES.get(match.group(1).lower(), text)
    return text[0].upper() + text[1:]


def _render_history_block(
    conversation_history: list[dict[str, str]] | None,
    max_messages: int = _HISTORY_MAX_MESSAGES,
    max_chars: int = _HISTORY_MAX_CHARS,
) -> str:
    """Render the most recent turns as ``role: content`` lines.

    :param conversation_history: Earlier turns, oldest first, as
        ``{"role": ..., "content": ...}`` dicts.
    :type conversation_history: list[dict[str, str]] or None
    :param max_messages: How many of the latest turns to keep.
    :type max_messages: int
    :param max_chars: Each turn is cut to this many characters.
    :type max_chars: int
    :return: The rendered lines, or ``""`` when there is no history.
    :rtype: str
    """
    if not conversation_history:
        return ""
    recent = conversation_history[-max_messages:]
    return "\n".join(
        f"{m.get('role', '')}: {str(m.get('content', ''))[:max_chars]}" for m in recent
    )


class SynthesisAgent:
    """Write a grounded, cited answer to a question about the literature."""

    def __init__(
        self,
        llm_base_url: str | None = None,
        llm_model_name: str | None = None,
        verbose: bool = False,
        tracer: TracingPort | None = None,
        llm_client: Any | None = None,
        librarian: LibrarianAgent | None = None,
    ):
        """Build an agent that answers over passages, or searches for them itself.

        :param llm_base_url: Override for the LLM endpoint; falls back to the
            client's own ``LLM_BASE_URL`` resolution when omitted. Ignored when
            ``llm_client`` is given.
        :type llm_base_url: str or None
        :param llm_model_name: Override for the LLM model name; falls back to
            the client's own ``LLM_MODEL`` resolution when omitted. Ignored when
            ``llm_client`` is given.
        :type llm_model_name: str or None
        :param verbose: If ``True``, print per-stage progress.
        :type verbose: bool
        :param tracer: Span-tracing adapter (see ``tracing_port.py``). Defaults
            to ``NullTracer`` (no-op).
        :type tracer: TracingPort or None
        :param llm_client: A ready-made LLM client to use instead of building
            one. Anything with ``chat_completion(messages, temperature=...,
            max_tokens=...) -> str``, like ``llm_client.LLMClient``.
        :type llm_client: object or None
        :param librarian: The retrieval agent to search with when ``run`` is
            called without passages. Anything with ``run(query, max_loops=...,
            progress_callback=...) -> list[dict]`` and a ``last_run_debug``
            dict, like :class:`~librarian.agent.LibrarianAgent`. Not needed when
            every ``run`` call passes its own passages.
        :type librarian: LibrarianAgent or None
        """
        self.verbose = verbose
        self._tracer: TracingPort = tracer if tracer is not None else NullTracer()
        # Both default to None so the client resolves LLM_BASE_URL / LLM_MODEL
        # from the environment — the same fallback the librarian relies on, since
        # config.toml ships an empty default_model_name.
        self.llm = (
            llm_client
            if llm_client is not None
            else create_llm_client(base_url=llm_base_url, model_name=llm_model_name)
        )
        self.librarian = librarian
        # Filled by every run; see ``run``.
        self.last_run_debug: dict[str, Any] = {}
        self._summarizer_prompt = _SUMMARIZER_PROMPT_PATH.read_text(encoding="utf-8")
        self._router_prompt = _ROUTER_PROMPT_PATH.read_text(encoding="utf-8")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Synthesis] {message}")

    def _trace_json(self, payload: Any) -> str:
        """Serialize compact trace payloads safely for span attributes."""
        return json.dumps(payload, default=str, ensure_ascii=True)

    # ── Step 0: extension point ──────────────────────────────────────────────

    def _screen_query(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> str | None:
        """Extension point, called first in ``run``, before any model call.

        A subclass overrides this to decide whether a request should go on.
        Return ``None`` to continue — what this base class always does — or a
        text to return as the answer instead; the router, the search and the
        summarizer are then skipped.

        :param query: The user's question, as given to ``run``.
        :type query: str
        :param conversation_history: The earlier turns given to ``run``.
        :type conversation_history: list[dict[str, str]] or None
        :return: ``None`` to continue, or the text to return instead.
        :rtype: str or None
        """
        return None

    # ── Step 1: routing ──────────────────────────────────────────────────────

    def _route(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """Decide between searching the literature and replying directly.

        :param query: The user's question.
        :type query: str
        :param conversation_history: Earlier turns, used to resolve a follow-up
            ("try again", a bare "yes") into a standalone search query.
        :type conversation_history: list[dict[str, str]] or None
        :return: ``{"action": "search" | "reply", "answer", "query",
            "language"}``. ``answer`` is the direct reply when the action is
            ``reply`` and empty otherwise; a reply with no text counts as a
            search, like unreadable router output. ``query`` is the standalone
            search query (the question itself when the router gave none);
            ``language`` is the name of the language the router read off the
            question (``"en"`` is turned into ``"English"``), which the
            summarizer then answers in — empty when the model omitted it.
        :rtype: dict[str, str]
        """
        history_block = ""
        rendered_history = _render_history_block(conversation_history)
        if rendered_history:
            history_block = f"Conversation so far:\n{rendered_history}\n\n"

        prompt = _fill(
            self._router_prompt,
            {"{history_block}": history_block, "{query}": query},
        )
        with self._tracer.start_span(
            "synthesis.route",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": self._trace_json({"query": query}),
                "input.mime_type": "application/json",
            },
        ) as span:
            try:
                response = self.llm.chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=_ROUTER_TEMPERATURE,
                    max_tokens=_ROUTER_MAX_TOKENS,
                )
            except Exception as exc:
                self._tracer.mark_span_error(span, exc)
                raise
            parsed = parse_json_response(response)
            language = ""
            reply = ""
            if isinstance(parsed, dict):
                language = _normalize_language(parsed.get("language"))
                if parsed.get("action") == "reply":
                    reply = str(parsed.get("answer") or "").strip()
            if reply:
                result = {
                    "action": "reply",
                    "answer": reply,
                    "query": query,
                    "language": language,
                }
            else:
                # Search is also the fallback for unreadable output and for a
                # "reply" that carries no text: no answer is better than none.
                resolved_query = ""
                if isinstance(parsed, dict):
                    resolved_query = str(parsed.get("query", "")).strip()
                result = {
                    "action": "search",
                    "answer": "",
                    "query": resolved_query or query,
                    "language": language,
                }
            self._tracer.set_span_attributes(
                span,
                {
                    "route.action": result["action"],
                    "output.value": self._trace_json(result),
                    "output.mime_type": "application/json",
                },
            )
        return result

    # ── Step 2: retrieval ────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        max_loops: int = 1,
        on_progress: Callable[[str], None] | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the literature for ``query`` with the librarian given at construction.

        :param query: The standalone search query.
        :type query: str
        :param max_loops: Passed to the librarian.
        :type max_loops: int
        :param on_progress: Passed to the librarian: called with a short status
            string at each retrieval stage.
        :type on_progress: Callable[[str], None] or None
        :param progress_callback: Passed to the librarian: called with the
            retrieval step number.
        :type progress_callback: Callable[[int], None] or None
        :return: The ranked evidence records, as ``LibrarianAgent.run`` returns
            them.
        :rtype: list[dict[str, Any]]
        :raises ValueError: If the agent was built without a ``librarian``.
        """
        if self.librarian is None:
            raise ValueError(
                "This question needs a literature search, but the agent has no "
                "librarian: build it with librarian=..., or pass passages= to run()."
            )
        return self.librarian.run(
            query,
            max_loops=max_loops,
            on_progress=on_progress,
            progress_callback=progress_callback,
        )

    # ── Step 3: summarization ────────────────────────────────────────────────

    def _output_formatting(self, output_channel: str | None) -> tuple[str, str]:
        """Name and formatting guidance of the channel the answer is written for.

        This base class writes for a terminal and has no other channel, so every
        value of ``output_channel`` gets the terminal guidance. A subclass adds
        its own channels by overriding this method.

        :param output_channel: The channel given to ``run``, or ``None``.
        :type output_channel: str or None
        :return: ``(channel, guidance)``: what the summarizer prompt's
            ``{output_channel}`` and ``{formatting_guidance}`` placeholders take.
        :rtype: tuple[str, str]
        """
        return _OUTPUT_CHANNEL, _FORMATTING_GUIDANCE

    def _build_prompt(
        self,
        query: str,
        passages: list[dict[str, Any]],
        conversation_history: list[dict[str, str]] | None = None,
        language: str = "",
        output_channel: str | None = None,
    ) -> str:
        """Fill the summarizer prompt's eight placeholders for this run.

        :param query: The question to answer.
        :type query: str
        :param passages: The evidence records, in citation order.
        :type passages: list[dict[str, Any]]
        :param conversation_history: Earlier turns, shown to the model as
            context only — never as a source.
        :type conversation_history: list[dict[str, str]] or None
        :param language: The language to answer in, as the router reported it;
            when empty the model is told to follow the question's own language.
        :type language: str
        :param output_channel: The channel the answer is written for; see
            ``_output_formatting``.
        :type output_channel: str or None
        :return: The finished prompt.
        :rtype: str
        """
        today = datetime.date.today()
        channel, guidance = self._output_formatting(output_channel)
        return _fill(
            self._summarizer_prompt,
            {
                "{today_date}": today.isoformat(),
                "{today_year}": str(today.year),
                "{output_channel}": channel,
                "{formatting_guidance}": guidance,
                "{conversation_history}": (
                    _render_history_block(conversation_history) or _NO_HISTORY
                ),
                "{user_query}": query,
                "{answer_language}": language or _ANSWER_LANGUAGE,
                "{papers_text}": render_papers(passages),
            },
        )

    def _summarize(
        self,
        query: str,
        passages: list[dict[str, Any]],
        conversation_history: list[dict[str, str]] | None = None,
        language: str = "",
        output_channel: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Write a grounded answer to ``query`` from ``passages``.

        This base class has no token streaming: when ``stream_callback`` is
        given it receives the finished answer in one piece. A subclass whose LLM
        client streams overrides this method and feeds the callback as chunks
        arrive.

        :param query: The question to answer.
        :type query: str
        :param passages: Ranked evidence records, as ``LibrarianAgent.run``
            returns them. Order is the citation order.
        :type passages: list[dict[str, Any]]
        :param conversation_history: Earlier turns, as context for the answer.
        :type conversation_history: list[dict[str, str]] or None
        :param language: The language to answer in; see ``_build_prompt``.
        :type language: str
        :param output_channel: The channel the answer is written for; see
            ``_output_formatting``.
        :type output_channel: str or None
        :param stream_callback: Called with the answer text; see above.
        :type stream_callback: Callable[[str], None] or None
        :return: The answer, citing papers by the ``Cite as:`` keys the
            passages carry. When ``passages`` is empty, a line saying the
            search found nothing — no LLM call is made.
        :rtype: str
        """
        if not passages:
            # Nothing to ground an answer in, so nothing worth an LLM call.
            if stream_callback:
                stream_callback(_NO_PAPERS_MESSAGE)
            return _NO_PAPERS_MESSAGE

        prompt = self._build_prompt(
            query,
            passages,
            conversation_history=conversation_history,
            language=language,
            output_channel=output_channel,
        )
        with self._tracer.start_span(
            "synthesis.summarize",
            attributes={
                "openinference.span.kind": "LLM",
                "paper.count": len(passages),
                "input.value": self._trace_json(
                    {
                        "query": query,
                        "passage_count": len(passages),
                        "passage_titles": [p.get("title") for p in passages[:25]],
                    }
                ),
                "input.mime_type": "application/json",
            },
        ) as span:
            try:
                answer = self.llm.chat_completion(
                    [
                        {
                            "role": "system",
                            "content": "You are a professional scientific summarizer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=_SYNTHESIS_TEMPERATURE,
                    max_tokens=_SYNTHESIS_MAX_TOKENS,
                ).strip()
            except Exception as exc:
                self._tracer.mark_span_error(span, exc)
                raise
            self._log(f"summarized {len(passages)} passages")
            self._tracer.set_span_attributes(
                span,
                {"output.value": answer[:2000], "output.mime_type": "text/plain"},
            )
        if stream_callback:
            stream_callback(answer)
        return answer

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        passages: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        max_loops: int = 1,
        on_progress: Callable[[str], None] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        output_channel: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Answer ``query``, from ``passages`` if given, else by searching first.

        With ``passages`` the agent only summarizes them: no routing, no search.
        Without, it routes the question, searches through its ``librarian`` when
        the router says the question needs the literature, and summarizes what
        came back (or returns the router's direct reply).

        After the call, ``last_run_debug`` holds this run's details, rebuilt on
        every run:

        * ``action`` — ``"summarize"`` (passages were given), ``"search"``,
          ``"reply"`` (the router answered directly) or ``"stopped"``
          (``_screen_query`` returned a text).
        * ``query`` — the query actually used: the router's standalone rewrite
          on the search path, the question as given otherwise.
        * ``language`` — the language the router reported, ``""`` if none.
        * ``passages`` — the evidence records the answer was written from.
        * ``search_queries`` — the sub-queries the librarian ran, ``[]`` when it
          did not run.

        :param query: The user's question.
        :type query: str
        :param passages: Evidence records, as ``LibrarianAgent.run`` returns
            them; their order is the citation order. When given (even empty),
            the router and the search are skipped.
        :type passages: list[dict[str, Any]] or None
        :param conversation_history: Earlier turns, oldest first, as
            ``{"role", "content"}`` dicts. Context for the router and the
            summarizer.
        :type conversation_history: list[dict[str, str]] or None
        :param max_loops: Passed to the librarian when a search runs.
        :type max_loops: int
        :param on_progress: Passed to the librarian when a search runs: called
            with a short status string at each retrieval stage.
        :type on_progress: Callable[[str], None] or None
        :param progress_callback: Passed to the librarian when a search runs,
            then called with ``3`` once its evidence is back.
        :type progress_callback: Callable[[int], None] or None
        :param output_channel: The channel the answer is written for (see
            ``_output_formatting``); ``None`` means the default one.
        :type output_channel: str or None
        :param stream_callback: Called with the answer text. This base class
            delivers it in one piece, whichever way the answer was produced
            (summary, direct reply, nothing found, or a text returned by
            ``_screen_query``); see ``_summarize``.
        :type stream_callback: Callable[[str], None] or None
        :return: The answer text.
        :rtype: str
        :raises ValueError: If the question has to be searched and the agent
            was built without a ``librarian``.
        """
        self.last_run_debug = {}
        with self._tracer.start_span(
            "synthesis.run",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": self._trace_json({"query": query}),
                "input.mime_type": "application/json",
            },
        ) as run_span:
            try:
                stop_text = self._screen_query(query, conversation_history)
                if stop_text is not None:
                    self.last_run_debug = {
                        "action": "stopped",
                        "query": query,
                        "language": "",
                        "passages": [],
                        "search_queries": [],
                    }
                    if stream_callback:
                        stream_callback(stop_text)
                    answer = stop_text
                elif passages is not None:
                    self.last_run_debug = {
                        "action": "summarize",
                        "query": query,
                        "language": "",
                        "passages": passages,
                        "search_queries": [],
                    }
                    answer = self._summarize(
                        query,
                        passages,
                        conversation_history=conversation_history,
                        output_channel=output_channel,
                        stream_callback=stream_callback,
                    )
                else:
                    answer = self._route_retrieve_summarize(
                        query,
                        conversation_history,
                        max_loops,
                        on_progress,
                        progress_callback,
                        output_channel,
                        stream_callback,
                    )
            except Exception as exc:
                self._tracer.mark_span_error(run_span, exc)
                raise
            self._tracer.set_span_attributes(
                run_span,
                {
                    "route.action": self.last_run_debug["action"],
                    "retrieval.passage_count": len(self.last_run_debug["passages"]),
                    "output.value": self._trace_json(
                        {
                            "action": self.last_run_debug["action"],
                            "passage_count": len(self.last_run_debug["passages"]),
                            "summary_preview": answer[:500],
                        }
                    ),
                    "output.mime_type": "application/json",
                },
            )
        return answer

    def _route_retrieve_summarize(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
        max_loops: int,
        on_progress: Callable[[str], None] | None,
        progress_callback: Callable[[int], None] | None,
        output_channel: str | None,
        stream_callback: Callable[[str], None] | None,
    ) -> str:
        """The full path of ``run``: route, then reply directly or search + summarize.

        ``last_run_debug`` is filled as each step completes, so a caller that
        catches an exception from the search still finds the route in it.
        """
        decision = self._route(query, conversation_history)
        # A reply with no text is no reply. _route already turns one into a
        # search; this keeps a subclass's own _route from reaching the user with
        # a blank answer.
        replying = decision["action"] == "reply" and bool(decision.get("answer"))
        search_query = decision.get("query") or query
        language = decision.get("language", "")
        self._log(f"route → {'reply' if replying else 'search'}")
        self.last_run_debug = {
            "action": "reply" if replying else "search",
            "query": search_query,
            "language": language,
            "passages": [],
            "search_queries": [],
        }

        if replying:
            if stream_callback:
                stream_callback(decision["answer"])
            return decision["answer"]

        passages = self._retrieve(
            search_query, max_loops, on_progress, progress_callback
        )
        if progress_callback:
            progress_callback(3)
        self.last_run_debug["passages"] = passages
        # The sub-queries the librarian actually ran: it stashes them in its own
        # last_run_debug, cleared at the start of every run.
        self.last_run_debug["search_queries"] = getattr(
            self.librarian, "last_run_debug", {}
        ).get("search_queries", [])
        return self._summarize(
            search_query,
            passages,
            conversation_history=conversation_history,
            language=language,
            output_channel=output_channel,
            stream_callback=stream_callback,
        )

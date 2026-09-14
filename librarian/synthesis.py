"""SynthesisAgent — turns the librarian's ranked evidence into a cited answer.

Two steps, and no routing: every query retrieves, then summarizes.

  1. retrieve    delegate to the injected :class:`~librarian.agent.LibrarianAgent`.
  2. _summarize  write a grounded answer over the passages it returned.

``run`` returns ``{"summary", "passages", "search_queries"}``.

The librarian is injected rather than constructed here: a caller that offers a
retrieval-only mode needs one anyway, so there stays a single construction site
and a single reader of ``config.toml``.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from librarian.agent import LibrarianAgent
from librarian.citations import render_papers
from librarian.llm_client import create_llm_client
from librarian.tracing_port import NullTracer, TracingPort

# Prompt — co-located with the planner and judge prompts. The skill reads this
# same file (skills/librarian/SKILL.md, Step 5), so the citation discipline is
# identical whichever path produced the answer.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_SUMMARIZER_PROMPT_PATH = _PROMPTS_DIR / "summarizer.md"

# Synthesis is grounded rewriting of supplied evidence, not creative writing:
# sampling entropy is exactly where a paraphrase drifts onto the wrong paralog.
# Not 0.0 — long answers at 0.0 can degenerate into repetition.
_SYNTHESIS_TEMPERATURE = 0.2
_SYNTHESIS_MAX_TOKENS = 8192

_OUTPUT_CHANNEL = "terminal"
_FORMATTING_GUIDANCE = (
    "Format the answer as plain text for a terminal. Use short paragraphs and "
    "simple '- ' bullets. Write section headings as a bare line of text, not as "
    "Markdown '#' headings. Do not use tables, HTML, or bold/italic markup — a "
    "terminal renders none of it. Leave the markdown citation links exactly as "
    "each paper's 'Cite as:' line gives them: terminals make the URL clickable."
)

# The router that would have supplied these was dropped, so they are fixed here.
# The language line must never name a concrete language: a named example primes
# the model to answer in it regardless of the question's own language.
_ANSWER_LANGUAGE = "the language the USER QUERY above is written in"
_NO_HISTORY = "(none — first turn)"

# The search ran and the judge kept nothing. That is a reportable answer, and a
# different thing from the search failing, which propagates as an exception.
_NO_PAPERS_MESSAGE = (
    "The literature search ran but returned no relevant papers for this "
    "question. Try rephrasing or broadening the query."
)


def _fill(template: str, values: Dict[str, str]) -> str:
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


class SynthesisAgent:
    """Retrieve evidence for a question, then write a grounded answer over it."""

    def __init__(
        self,
        librarian: LibrarianAgent,
        llm_base_url: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        verbose: bool = False,
        tracer: Optional[TracingPort] = None,
    ):
        """Build an agent that answers over whatever ``librarian`` retrieves.

        :param librarian: The retrieval agent to delegate to. Injected, so the
            caller owns its configuration and can also use it on its own.
        :type librarian: LibrarianAgent
        :param llm_base_url: Override for the LLM endpoint; falls back to the
            client's own ``LLM_BASE_URL`` resolution when omitted.
        :type llm_base_url: str or None
        :param llm_model_name: Override for the LLM model name; falls back to
            the client's own ``LLM_MODEL`` resolution when omitted.
        :type llm_model_name: str or None
        :param verbose: If ``True``, print per-stage progress.
        :type verbose: bool
        :param tracer: Span-tracing adapter (see ``tracing_port.py``). Defaults
            to ``NullTracer`` (no-op).
        :type tracer: TracingPort or None
        """
        self.librarian = librarian
        self.verbose = verbose
        self._tracer: TracingPort = tracer if tracer is not None else NullTracer()
        # Both default to None so the client resolves LLM_BASE_URL / LLM_MODEL
        # from the environment — the same fallback the librarian relies on, since
        # config.toml ships an empty default_model_name.
        self.llm = create_llm_client(base_url=llm_base_url, model_name=llm_model_name)
        self._summarizer_prompt = _SUMMARIZER_PROMPT_PATH.read_text(encoding="utf-8")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Synthesis] {message}")

    def _trace_json(self, payload: Any) -> str:
        """Serialize compact trace payloads safely for span attributes."""
        return json.dumps(payload, default=str, ensure_ascii=True)

    def _build_prompt(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Fill the summarizer prompt's eight placeholders for this run."""
        today = datetime.date.today()
        return _fill(
            self._summarizer_prompt,
            {
                "{today_date}": today.isoformat(),
                "{today_year}": str(today.year),
                "{output_channel}": _OUTPUT_CHANNEL,
                "{formatting_guidance}": _FORMATTING_GUIDANCE,
                "{conversation_history}": _NO_HISTORY,
                "{user_query}": query,
                "{answer_language}": _ANSWER_LANGUAGE,
                "{papers_text}": render_papers(passages),
            },
        )

    def _summarize(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Write the grounded answer, or the no-papers line when nothing was kept."""
        if not passages:
            # Nothing to ground an answer in, so nothing worth an LLM call.
            return _NO_PAPERS_MESSAGE

        prompt = self._build_prompt(query, passages)
        with self._tracer.start_span(
            "synthesis.summarize",
            attributes={
                "openinference.span.kind": "LLM",
                "paper.count": len(passages),
                "input.value": self._trace_json(
                    {"query": query, "passage_count": len(passages)}
                ),
                "input.mime_type": "application/json",
            },
        ) as span:
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
            self._tracer.set_span_attributes(
                span,
                {"output.value": answer[:2000], "output.mime_type": "text/plain"},
            )
        return answer

    def run(
        self,
        query: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Retrieve evidence for ``query`` and synthesize a cited answer.

        :param query: The user's research question.
        :type query: str
        :param on_progress: Called with the current stage name, forwarded to the
            librarian so retrieval and synthesis report to the same display.
        :type on_progress: callable or None
        :return: ``{"summary", "passages", "search_queries"}`` — the answer, the
            ranked passages it was written from, and the sub-queries the
            librarian actually ran.
        :rtype: dict[str, Any]
        """
        with self._tracer.start_span(
            "synthesis.run",
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": self._trace_json({"query": query}),
                "input.mime_type": "application/json",
            },
        ) as run_span:
            try:
                passages = self.librarian.run(query, on_progress=on_progress)
                if on_progress is not None:
                    on_progress("Synthesizing answer")
                summary = self._summarize(query, passages)
            except Exception as exc:
                self._tracer.mark_span_error(run_span, exc)
                raise
            self._log(f"summarized {len(passages)} passages")
            self._tracer.set_span_attributes(
                run_span,
                {
                    "retrieval.passage_count": len(passages),
                    "output.value": self._trace_json(
                        {
                            "passage_count": len(passages),
                            "summary_preview": summary[:500],
                        }
                    ),
                    "output.mime_type": "application/json",
                },
            )

        # last_run_debug is only assigned once Stage 3 completes, so a run that
        # retrieved nothing leaves the attribute unset entirely.
        librarian_debug = getattr(self.librarian, "last_run_debug", {})
        return {
            "summary": summary,
            "passages": passages,
            "search_queries": librarian_debug.get("search_queries", []),
        }

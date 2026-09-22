"""Numbered-citation synthesis, which is what ScholarQA-Bench scores.

``librarian.synthesis.SynthesisAgent`` cites papers the way a reader wants them:
author-year markdown links copied from each paper's ``Cite as:`` line, e.g.
``[Chen 2023](https://europepmc.org/article/MED/12345678)``.

ScholarQA-Bench's scorer cannot read those. ``code/scripts/citation_correctness_eval.py``
extracts citations with

    CITATION_PATTERN = r"\\[((?:REF_)?\\d+(?:\\.\\d+)?(?:\\s*,\\s*(?:REF_)?\\d+(?:\\.\\d+)?)*)\\]"

which matches ``[2]``, ``[2, 7]``, ``[REF_2]`` and ``[2.3]`` — numbers only. An
answer written in author-year form parses as having NO citations, every sentence
scores unsupported, and Citation F1 collapses to roughly zero. It is a silent
failure: the run completes and reports a number.

So the benchmark keeps the citation contract its metric is defined on. This
module is that contract and nothing else: papers rendered as ``[REF_n]`` blocks
with a lookup table, and the matching summarizer prompt in
``prompts/summarizer_numbered.md``. Both are ports of what produced the paper's
row. Retrieval, ranking and the evidence itself are untouched — only how the
answer names the paper it is leaning on.

Nothing outside this benchmark should use it: the shipped agent's author-year
links are the better output for a human, and the reason for them is that a bare
``[12]`` means nothing once the answer is copied out of its result set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from librarian.synthesis import SynthesisAgent, _fill

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "summarizer_numbered.md"

# The shipped agent's guidance tells the model to copy each paper's ``Cite as:``
# markdown link, which is exactly what has to change here.
_FORMATTING_GUIDANCE = (
    "Format the answer in clear Markdown with concise sections and bullets when "
    "helpful. Keep the structure easy to read in plain text. Keep citation "
    "numbers visible as square-bracket text such as [2], [2, 7], or [2.3] "
    "inside the prose."
)
_OUTPUT_CHANNEL = "generic"
_ANSWER_LANGUAGE = "the language the USER QUERY above is written in"
_NO_HISTORY = "(none — first turn)"


def render_papers_numbered(passages: list[dict[str, Any]]) -> str:
    """Render passages as ``[REF_n]`` blocks plus a reference lookup table.

    A paper whose evidence arrived as several non-contiguous spans gets each span
    tagged ``[n.k]``, so the model can point at the passage actually backing a
    claim; the scorer collapses ``[n.k]`` to paper ``n``. Single-span papers keep
    the plain form — a lone ``[n.1]`` would be noise.

    :param passages: Evidence records as ``LibrarianAgent.run`` returns them, in
        the order they should be numbered.
    :returns: The rendered blocks followed by the lookup table.
    """
    blocks = []
    for i, paper in enumerate(passages, 1):
        authors = paper.get("authors", "N/A")
        if isinstance(authors, list):
            authors = ", ".join(authors) if authors else "N/A"
        snippets = [s.strip() for s in (paper.get("evidence_snippets") or []) if s.strip()]

        cite_line = f"Cite this paper as: [{i}]"
        if len(snippets) > 1:
            evidence_text = "\n" + "\n".join(
                f"  [{i}.{k}] {span}" for k, span in enumerate(snippets, 1)
            )
            cite_line += (
                f". To point at one evidence passage, cite its tag, "
                f"e.g. [{i}.{len(snippets)}]"
            )
        else:
            evidence_text = " " + (snippets[0] if snippets else "No evidence available.")

        blocks.append(
            f"[REF_{i}] Paper #{i}:\n"
            f"Reference Number: {i}\n"
            f"Title: {paper.get('title', 'N/A')}\n"
            f"Authors: {authors}\n"
            f"Journal: {paper.get('journal', 'N/A')} ({paper.get('year', 'N/A')})\n"
            f"Evidence:{evidence_text}\n"
            f"{cite_line}\n"
        )

    papers_text = "\n---\n".join(blocks)
    lookup = ["\n\n===== REFERENCE LOOKUP TABLE (verify every citation) ====="]
    for i, paper in enumerate(passages, 1):
        lookup.append(f'[{i}] = "{paper.get("title", "N/A")}"')
    lookup.append("===== END LOOKUP TABLE =====")
    return papers_text + "\n".join(lookup)


class NumberedCitationSynthesisAgent(SynthesisAgent):
    """``SynthesisAgent`` that cites ``[2]`` instead of ``[Chen 2023](URL)``.

    Only the prompt and the papers rendering differ; ``run()`` — the LLM call,
    temperature, tracing and the empty-evidence path — is inherited unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._summarizer_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def _build_prompt(self, query: str, passages: list[dict[str, Any]]) -> str:
        """Fill the numbered summarizer prompt's eight placeholders."""
        import datetime

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
                "{papers_text}": render_papers_numbered(passages),
            },
        )


def _self_check() -> None:
    """Render a two-paper set and assert the scorer can read what comes back."""
    import re
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "scripts"))

    passages = [
        {"title": "A", "authors": "Chen X", "journal": "J", "year": "2023",
         "evidence_snippets": ["span one"]},
        {"title": "B", "authors": ["Li Y"], "journal": "K", "year": "2021",
         "evidence_snippets": ["span one", "span two"]},
    ]
    text = render_papers_numbered(passages)
    assert "[REF_1] Paper #1:" in text and "[REF_2] Paper #2:" in text, text
    # Multi-span papers get per-passage tags; single-span ones do not.
    assert "[2.1] span one" in text and "[2.2] span two" in text, text
    assert "[1.1]" not in text, text
    assert '[1] = "A"' in text and '[2] = "B"' in text, text

    # The whole point: the benchmark's own extractor must find these.
    pattern = r"\[((?:REF_)?\d+(?:\.\d+)?(?:\s*,\s*(?:REF_)?\d+(?:\.\d+)?)*)\]"
    answer = "Claim one [1]. Claim two [2.2]. Both [1, 2]."
    found = re.findall(pattern, answer)
    assert found == ["1", "2.2", "1, 2"], found

    # And the shipped author-year form must be what fails, or this module has no
    # reason to exist.
    assert re.findall(pattern, "Claim [Chen 2023](https://example.org/1).") == []

    assert _PROMPT_PATH.exists(), _PROMPT_PATH
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "{today_date}", "{today_year}", "{output_channel}", "{formatting_guidance}",
        "{conversation_history}", "{user_query}", "{answer_language}", "{papers_text}",
    ):
        assert placeholder in prompt, placeholder

    print("numbered_citations self-check ok")


if __name__ == "__main__":
    _self_check()

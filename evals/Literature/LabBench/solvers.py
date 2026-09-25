"""Answering models for the LAB-Bench MCQ evals (task-agnostic).

Two solvers, both queried with the verbatim LAB-Bench MCQ prompt:

  - ``BaselineSolver`` — the LAB-Bench setup: send the MCQ to the model and let
    it answer from its own (parametric) knowledge, which is the paper's
    baseline row.

  - ``KnowledgeLayerSolver`` — single-shot RAG. The retrieval agent is run
    **once** on the question, its passages are injected into the prompt, and the
    model answers in **one** API call, told to use the evidence and to answer in
    the required format.

The agent is injected as a ``search_fn(query) -> result_dict`` callable, so the
solver does not depend on which agent produces the evidence. Both
``LiteratureSearchAgent.run(..., include_summary=False)`` and
``LibrarianAgent.run(...)`` return the ``{evidence, papers_raw, ...}`` shape
``format_evidence`` consumes — raw retrieved passages, never a summary.

Both solvers use the OpenAI-compatible chat-completions API, and ``--model``
defaults to ``gpt-4o-2024-05-13``, the snapshot the paper's floating ``gpt-4o``
alias resolved to in mid-2024.

The knowledge-layer system prompts are phrased for biology research generally
rather than for one task, so they fit DbQA, SeqQA and ProtocolQA alike.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from openai import BadRequestError, OpenAI, RateLimitError


class _ChatClient:
    """OpenAI chat wrapper that adapts to model-specific parameter quirks.

    Older models (e.g. gpt-4o) use ``max_tokens``; newer ones (gpt-5.x / o-series)
    require ``max_completion_tokens`` and may reject a non-default ``temperature``.
    The first call that hits a 400 about either parameter switches the request
    shape and remembers it for the rest of the run.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        request_timeout: float,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._token_param = "max_tokens"
        self._send_temperature = True
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=max_retries,
            timeout=request_timeout,
        )

    def complete(self, messages: list[dict]) -> str:
        """Return the assistant text, adapting params to the model on 400s."""
        for attempt in range(10):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                self._token_param: self.max_tokens,
            }
            if self._send_temperature:
                kwargs["temperature"] = self.temperature
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""
                if not content.strip():
                    # Thinking models served via vLLM (e.g. GLM-5, DeepSeek-R1)
                    # sometimes return the answer in `reasoning_content` with an
                    # empty `content`.
                    return reasoning
                # Some vLLM deployments split thinking from the final response:
                # `content` = text after </think>, `reasoning_content` = the
                # thinking trace. Return both, so the parser can find the
                # [ANSWER] tag in whichever part carries it.
                if reasoning.strip() and "[ANSWER]" not in content.upper():
                    return content + "\n" + reasoning
                return content
            except RateLimitError:
                # Sleep a full TPM window, so competing workers drain before
                # this one retries.
                if attempt >= 9:
                    raise
                time.sleep(60)
            except BadRequestError as exc:
                msg = str(exc).lower()
                if self._token_param == "max_tokens" and "max_completion_tokens" in msg:
                    self._token_param = "max_completion_tokens"
                    continue
                if self._send_temperature and "temperature" in msg:
                    # gpt-5.x / o-series accept only the default temperature.
                    self._send_temperature = False
                    continue
                raise
        raise RuntimeError("Chat completion failed after adapting parameters.")


# Two ways to ground the answering model in the retrieved evidence:
#   "augment" (default) — evidence is helpful CONTEXT; the model still reasons and
#                         commits to the best-supported option.
#   "strict"            — evidence is the ONLY allowed basis; refuse if unsupported.
KNOWLEDGE_LAYER_SYSTEM_PROMPTS = {
    "augment": (
        "You are answering a biology-research multiple-choice question.\n"
        "You have been provided with literature evidence retrieved from a curated "
        "scientific-literature knowledge layer (Europe PMC). Use this evidence "
        "ONLY when it directly and decisively resolves which option is correct. "
        "If the evidence is only tangentially related or does not resolve the "
        "question, IGNORE IT COMPLETELY — answer exactly as you would with no "
        "evidence provided. The presence of retrieved evidence does NOT mean it is "
        "useful; treat unhelpful evidence as absent. "
        "In particular, do NOT select the 'Insufficient information' option merely "
        "because the evidence is irrelevant — choose it only when you genuinely "
        "cannot answer from your own knowledge.\n"
        "Give your final answer in the exact required [ANSWER]X[/ANSWER] format."
    ),
    "strict": (
        "You are answering a biology-research multiple-choice question.\n"
        "You MUST NOT answer from your own internal (parametric) knowledge — treat "
        "it as unreliable for this task. Base your answer ONLY on the literature "
        "evidence provided below, which was retrieved for you from a curated "
        "scientific-literature knowledge layer (Europe PMC).\n"
        "If the provided evidence does not support any specific option, choose the "
        "'Insufficient information to answer the question' option.\n"
        "Give your final answer in the exact required [ANSWER]X[/ANSWER] format."
    ),
}

_USER_INTROS = {
    "augment": (
        "The following literature evidence was retrieved from the knowledge layer "
        "(Europe PMC) to help you answer."
    ),
    "strict": (
        "The following literature evidence was retrieved for you from the "
        "knowledge layer (Europe PMC). Use ONLY this evidence to answer — do not "
        "rely on your own knowledge."
    ),
}


def format_evidence(result: dict[str, Any], char_budget: int = 1500) -> str:
    """Render a retrieval-agent result into a compact, citeable evidence block.

    Works for both ``LiteratureSearchAgent`` and ``LibrarianAgent``: the evidence
    entries always carry ``pmid`` and the raw ``evidence`` passages, and paper
    metadata (title/year) is looked up from ``papers_raw`` by PMID when the
    evidence entry doesn't carry it inline.
    """
    evidence = result.get("evidence") or []
    if not evidence:
        return "No literature evidence was found for this query."

    meta_by_pmid = {
        str(paper.get("pmid")): paper
        for paper in (result.get("papers_raw") or [])
        if paper.get("pmid")
    }

    blocks: list[str] = []
    for i, entry in enumerate(evidence, start=1):
        pmid = str(entry.get("pmid") or "")
        meta = meta_by_pmid.get(pmid, {})
        title = entry.get("title") or meta.get("title") or "Untitled"
        year = entry.get("year") or meta.get("year") or "n.d."
        chunks = entry.get("evidence") or []
        snippet = " ".join(str(c) for c in chunks).strip()[:char_budget]
        if not snippet:
            continue
        citation = f"[{i}] {title} ({year})" + (f" PMID:{pmid}" if pmid else "")
        blocks.append(f"{citation}\n{snippet}")

    return (
        "\n\n".join(blocks)
        if blocks
        else "No literature evidence was found for this query."
    )


class BaselineSolver:
    """Plain LAB-Bench-style baseline: model answers from parametric knowledge."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-05-13",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 8,
        request_timeout: float = 120.0,
    ) -> None:
        self.client = _ChatClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

    def answer(self, prompt: str, retrieval_query: str | None = None) -> dict[str, Any]:
        """Return {raw_output, metadata} for one MCQ prompt.

        ``retrieval_query`` is ignored — the baseline answers from parametric
        knowledge; the parameter exists only to match KnowledgeLayerSolver.
        """
        started = time.perf_counter()
        raw = self.client.complete([{"role": "user", "content": prompt}])
        return {
            "raw_output": raw,
            "metadata": {
                "mode": "baseline",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "literature_queries": [],
                "n_evidence_papers": 0,
            },
        }


class KnowledgeLayerSolver:
    """Single-shot RAG: run the retrieval agent once, inject evidence, answer once."""

    def __init__(
        self,
        search_fn: Callable[[str], dict[str, Any]],
        model: str = "gpt-4o-2024-05-13",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        char_budget: int = 1500,
        grounding: str = "augment",
        parametric_fallback: bool = True,
        max_retries: int = 8,
        request_timeout: float = 120.0,
    ) -> None:
        self.search_fn = search_fn
        self.char_budget = char_budget
        self.grounding = grounding if grounding in _USER_INTROS else "augment"
        self.parametric_fallback = parametric_fallback
        self.client = _ChatClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

    def answer(self, prompt: str, retrieval_query: str | None = None) -> dict[str, Any]:
        """Retrieve once on ``retrieval_query``, then answer the MCQ in one call.

        ``prompt`` is the rendered LAB-Bench MCQ; ``retrieval_query`` is the text
        used to query the knowledge layer (the question itself by default).
        """
        started = time.perf_counter()
        query = retrieval_query or prompt

        evidence_block = "No literature evidence was found for this query."
        evidence_papers = 0
        retrieval_error = None
        try:
            result = self.search_fn(query)
            evidence_papers = len(result.get("evidence") or [])
            evidence_block = format_evidence(result, self.char_budget)
        except Exception as exc:  # pragma: no cover - network dependent
            retrieval_error = str(exc)
            evidence_block = f"Literature search failed: {exc}"

        # With nothing retrieved, and parametric_fallback on, answer from the
        # bare MCQ exactly as the baseline does rather than from an empty
        # evidence block.
        used_fallback = self.parametric_fallback and evidence_papers == 0
        if used_fallback:
            messages = [{"role": "user", "content": prompt}]
        else:
            user_content = (
                f"{_USER_INTROS[self.grounding]}\n\n"
                "=== LITERATURE EVIDENCE ===\n"
                f"{evidence_block}\n"
                "=== END OF EVIDENCE ===\n\n"
                "Using the evidence above as context, answer the following "
                "question:\n\n"
                f"{prompt}"
            )
            messages = [
                {
                    "role": "system",
                    "content": KNOWLEDGE_LAYER_SYSTEM_PROMPTS[self.grounding],
                },
                {"role": "user", "content": user_content},
            ]

        raw = self.client.complete(messages)
        return {
            "raw_output": raw,
            "metadata": {
                "mode": "knowledge",
                "grounding": self.grounding,
                "used_parametric_fallback": used_fallback,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "literature_queries": [query],
                "n_evidence_papers": evidence_papers,
                "retrieval_error": retrieval_error,
            },
        }

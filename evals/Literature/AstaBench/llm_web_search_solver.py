"""LLM solver that uses OpenAI Responses web_search for open-answer tasks."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Literal

import requests
from inspect_ai.solver import Generate, Solver, TaskState, solver

from evals.Literature.AstaBench.bio_agent_wrapper import PublicSemanticScholarResolver
from evals.Literature.AstaBench.compat import extract_json_from_response

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

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ResponsesWebSearchClient:
    """Minimal Responses API client for web search."""

    def __init__(
        self,
        base_url: str | None,
        model: str,
        tool_choice: str,
        search_context_size: str,
    ) -> None:
        resolved_base = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        if resolved_base.endswith("/responses"):
            self._responses_url = resolved_base
        else:
            self._responses_url = f"{resolved_base}/responses"
        self._model = model
        self._tool_choice = tool_choice
        self._search_context_size = search_context_size
        self._api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    def query_raw(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "tools": [
                {
                    "type": "web_search",
                    "search_context_size": self._search_context_size,
                }
            ],
            "tool_choice": self._tool_choice,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.post(
            self._responses_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def query_text(self, prompt: str) -> str:
        return _extract_output_text(self.query_raw(prompt))


def _extract_output_text(response: dict[str, Any]) -> str:
    outputs = response.get("output") or []
    if isinstance(outputs, list):
        texts: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = str(content.get("text") or "").strip()
                    if text:
                        texts.append(text)
        if texts:
            return "\n".join(texts)
    if "output_text" in response and isinstance(response["output_text"], str):
        return response["output_text"].strip()
    return ""


def _extract_url_citations(response: dict[str, Any]) -> list[dict[str, str]]:
    outputs = response.get("output") or []
    citations: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    if not isinstance(outputs, list):
        return citations
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                url = str(annotation.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                citations.append(
                    {
                        "url": url,
                        "title": str(annotation.get("title") or "").strip(),
                    }
                )
    return citations


def _extract_question(state: TaskState) -> str:
    sample_input = str(getattr(state, "input", "") or "").strip()
    if "\n\n" in sample_input:
        return sample_input.split("\n\n", 1)[0].strip()
    return sample_input


def _extract_paper_query(state: TaskState) -> str:
    sample_input = str(getattr(state, "input", "") or "").strip()
    if sample_input.startswith("Find papers relevant to the following query:"):
        return sample_input.split("\n", 1)[0].split(":", 1)[1].strip()
    return sample_input


def _pubmedqa_prompt(question: str) -> str:
    return (
        "Use web search to answer the following PubMedQA biomedical question. "
        "Give a concise answer that clearly indicates yes, no, or maybe, "
        "with one short sentence of rationale.\n\n"
        f"Question:\n{question}"
    )


def _parse_doi(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.I)
    return match.group(0) if match else ""


def _normalize_paper_candidates(
    parsed: dict[str, Any] | None,
    citations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    if parsed and isinstance(parsed.get("papers"), list):
        for item in parsed["papers"]:
            if not isinstance(item, dict):
                continue
            papers.append(item)
    if not papers and parsed and isinstance(parsed.get("results"), list):
        for item in parsed["results"]:
            if not isinstance(item, dict):
                continue
            papers.append(item)
    if not papers:
        for citation in citations:
            papers.append(
                {
                    "title": citation.get("title"),
                    "url": citation.get("url"),
                    "evidence": citation.get("url"),
                }
            )
    return papers


def _build_markdown_evidence(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "Untitled").strip() or "Untitled"
    year = str(paper.get("year") or "n.d.").strip() or "n.d."
    evidence = str(paper.get("evidence") or "").strip()
    url = str(paper.get("url") or "").strip()
    body = evidence or (f"Source: {url}" if url else "No evidence available.")
    return f"**{title}** ({year})\n\n{body}"


def _infer_task_type(task_type: TaskKind, state: TaskState) -> TaskKind:
    if task_type != "auto":
        return task_type

    metadata = dict(state.metadata or {})
    sample_input = str(getattr(state, "input", "") or "")
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
    return "litqa2_open"


@solver
def llm_web_search_solver(
    task_type: TaskKind = "auto",
    base_url: str | None = None,
    model: str | None = None,
    tool_choice: str = "required",
    search_context_size: str = "medium",
) -> Solver:
    if not model:
        raise ValueError("`model` must be set for llm_web_search_solver.")

    client = ResponsesWebSearchClient(
        base_url=base_url,
        model=model,
        tool_choice=tool_choice,
        search_context_size=search_context_size,
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        resolved_task = _infer_task_type(task_type, state)
        if resolved_task in {"litqa2_open", "litqa2_open_llm_only", "pubmedqa_open"}:
            question = _extract_question(state)
            prompt = (
                _pubmedqa_prompt(question)
                if resolved_task == "pubmedqa_open"
                else question
            )
            answer = await asyncio.to_thread(client.query_text, prompt)
            state.output.completion = answer.strip() or "No answer generated."
            return state

        if resolved_task == "paper_finder":
            query = _extract_paper_query(state)
            prompt = (
                "You are finding scientific papers relevant to a query. Use web search. "
                'Return JSON only in the form {"papers": [ ... ]}. '
                "Each paper must include: title, year (if known), doi (if known), url, "
                "and a short verbatim evidence snippet or quote from the paper or source.\n\n"
                f"Query:\n{query}\n\n"
                "JSON format:\n"
                '{"papers": ['
                '{"title": "...", "year": 2020, "doi": "...", '
                '"url": "...", "evidence": "..."}'
                "]}"
            )
            raw_response = await asyncio.to_thread(client.query_raw, prompt)
            output_text = _extract_output_text(raw_response)
            parsed = extract_json_from_response(output_text) or {}
            citations = _extract_url_citations(raw_response)
            candidates = _normalize_paper_candidates(parsed, citations)

            resolver = PublicSemanticScholarResolver()
            results: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for candidate in candidates:
                title = str(candidate.get("title") or "").strip()
                url = str(candidate.get("url") or "").strip()
                doi = str(candidate.get("doi") or "").strip()
                doi = (
                    doi
                    or _parse_doi(url)
                    or _parse_doi(str(candidate.get("evidence") or ""))
                )
                paper_stub = {
                    "title": title,
                    "doi": doi,
                    "year": candidate.get("year"),
                }
                corpus_id = await resolver.resolve_corpus_id(paper_stub)
                if not corpus_id or corpus_id in seen_ids:
                    continue
                seen_ids.add(corpus_id)
                candidate["url"] = url
                if doi:
                    candidate["doi"] = doi
                results.append(
                    {
                        "paper_id": corpus_id,
                        "markdown_evidence": _build_markdown_evidence(candidate),
                    }
                )
                if len(results) >= 50:
                    break

            state.output.completion = json.dumps(
                {"output": {"results": results}}, indent=2, ensure_ascii=False
            )
            return state

        state.output.completion = "No answer generated."
        return state

    return solve


__all__ = ["llm_web_search_solver"]

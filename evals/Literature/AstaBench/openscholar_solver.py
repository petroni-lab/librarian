"""OpenScholar (OpenSciLM demo) solver for ScholarQA-CS2 tasks."""

from __future__ import annotations

import asyncio
import json
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.solver import Generate, Solver, TaskState, solver

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

DEFAULT_API_BASE = "https://openscilm.allen.ai"


@dataclass(frozen=True)
class OpenScholarConfig:
    api_base_url: str = DEFAULT_API_BASE
    submit_timeout: int = 30
    poll_timeout: int = 60
    poll_interval: int = 8
    max_poll_attempts: int = 40
    paper_details_batch: int = 50


class OpenScholarClient:
    """Minimal client for the OpenScholar demo API."""

    def __init__(self, config: OpenScholarConfig) -> None:
        self.config = config
        self.query_endpoint = f"{config.api_base_url}/api/query_open_scholar"
        self.paper_details_endpoint = f"{config.api_base_url}/api/paper_details"
        self._ssl_ctx = ssl.create_default_context()

    def query(self, question: str) -> dict[str, Any] | None:
        submit_response = self._submit_query(question)
        if not submit_response.get("task_id"):
            return None
        poll_response = self._poll_result(str(submit_response["task_id"]))
        if poll_response is None:
            return None
        task_result = poll_response.get("task_result") or {}
        combined_output, all_citations = self._flatten_iterations(task_result)

        corpus_ids = list(
            dict.fromkeys(
                citation.get("corpus_id")
                for citation in all_citations
                if citation.get("corpus_id") is not None
            )
        )
        paper_meta = self._fetch_paper_details(corpus_ids) if corpus_ids else {}
        ctxs = self._build_ctxs(all_citations, paper_meta)

        return {"output": combined_output, "ctxs": ctxs}

    def _submit_query(self, question: str) -> dict[str, Any]:
        return self._post_json(
            self.query_endpoint,
            {
                "query": question,
                "opt_in": False,
                "user_id": "librarian-asta-eval",
                "feedback_toggle": True,
            },
            timeout=self.config.submit_timeout,
        )

    def _poll_result(self, task_id: str) -> dict[str, Any] | None:
        for _ in range(self.config.max_poll_attempts):
            time.sleep(self.config.poll_interval)
            try:
                response = self._post_json(
                    self.query_endpoint,
                    {"task_id": task_id, "feedback_toggle": True},
                    timeout=self.config.poll_timeout,
                )
            except Exception:
                continue
            task_result = response.get("task_result")
            if task_result and task_result.get("iterations"):
                return response
        return None

    def _post_json(
        self, url: str, body: dict[str, Any], timeout: int
    ) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Origin": self.config.api_base_url,
                "Referer": f"{self.config.api_base_url}/",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            req, timeout=timeout, context=self._ssl_ctx
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_paper_details(self, corpus_ids: list[int]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        fields = ["title", "authors", "year", "corpusId", "venue", "abstract"]

        for i in range(0, len(corpus_ids), self.config.paper_details_batch):
            batch = corpus_ids[i : i + self.config.paper_details_batch]
            try:
                papers = self._post_json(
                    self.paper_details_endpoint,
                    {"corpus_ids": batch, "fields": fields},
                    timeout=20,
                )
                for paper in papers:
                    corpus_id = paper.get("corpusId")
                    if corpus_id is not None:
                        result[int(corpus_id)] = paper
            except Exception:
                continue
        return result

    @staticmethod
    def _densify_iteration_citations(
        iteration_citations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[int, int]]:
        ordered: list[dict[str, Any]] = []
        id_mapping: dict[int, int] = {}
        fallback_next_id = 0

        for citation in iteration_citations:
            raw_id = OpenScholarClient._parse_citation_id(citation.get("id"))
            if raw_id is None:
                while fallback_next_id in id_mapping:
                    fallback_next_id += 1
                raw_id = fallback_next_id

            if raw_id in id_mapping:
                continue

            id_mapping[raw_id] = len(ordered)
            ordered.append(citation)

        return ordered, id_mapping

    @staticmethod
    def _parse_citation_id(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return None
        match = re.fullmatch(r"\[(\d+)\]", value.strip())
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _fix_citation_indices(
        text: str,
        id_mapping: dict[int, int],
        offset: int = 0,
    ) -> str:
        def _rewrite(match: re.Match[str]) -> str:
            raw_group = match.group(1)
            rewritten: list[str] = []
            for part in raw_group.split(","):
                raw_id = int(part.strip())
                mapped = id_mapping.get(raw_id)
                if mapped is None:
                    rewritten.append(str(raw_id + offset + 1))
                else:
                    rewritten.append(str(mapped + offset + 1))
            return f"[{', '.join(rewritten)}]"

        return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _rewrite, text)

    def _flatten_iterations(
        self, task_result: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        iterations = task_result.get("iterations", [])
        output_parts: list[str] = []
        all_citations: list[dict[str, Any]] = []

        for iteration in iterations:
            raw_text = iteration.get("text", "")
            iteration_citations = iteration.get("citations", [])
            ordered, id_mapping = self._densify_iteration_citations(iteration_citations)
            output_parts.append(
                self._fix_citation_indices(
                    raw_text, id_mapping, offset=len(all_citations)
                )
            )
            all_citations.extend(ordered)

        combined_output = "\n".join(output_parts).strip()
        return combined_output, all_citations

    @staticmethod
    def _build_ctxs(
        citations: list[dict[str, Any]],
        paper_meta: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ctxs: list[dict[str, Any]] = []
        for citation in citations:
            corpus_id = citation.get("corpus_id")
            snippet = str(citation.get("snippet") or "").strip()
            meta = paper_meta.get(int(corpus_id)) if corpus_id is not None else {}

            title = meta.get("title") or citation.get("title") or ""
            year = meta.get("year") or ""
            abstract = str(meta.get("abstract") or "").strip()
            text = snippet or abstract

            ctxs.append(
                {
                    "title": title,
                    "text": text,
                    "year": year,
                    "corpusId": corpus_id,
                }
            )
        return ctxs


def _extract_question(state: TaskState) -> str:
    metadata = dict(state.metadata or {})
    sample_input = str(metadata.get("initial_prompt") or state.input or "").strip()
    if "\n\n" in sample_input:
        return sample_input.split("\n\n", 1)[0].strip()
    return sample_input


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
    raise ValueError("Could not infer AstaBench task type from TaskState.")


def _build_sqa_response(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "sections": [
                {"title": "Answer", "text": "No answer generated.", "citations": []}
            ]
        }

    text = str(payload.get("output") or "").strip() or "No answer generated."
    ctxs = list(payload.get("ctxs") or [])
    citations: list[dict[str, Any]] = []

    for idx, ctx in enumerate(ctxs, start=1):
        metadata: dict[str, Any] = {}
        year = ctx.get("year")
        corpus_id = ctx.get("corpusId")
        if year not in (None, ""):
            metadata["year"] = year
        if corpus_id not in (None, ""):
            metadata["corpusId"] = corpus_id

        citations.append(
            {
                "id": f"[{idx}]",
                "snippets": [str(ctx.get("text") or "")],
                "title": str(ctx.get("title") or "Untitled"),
                "metadata": metadata,
            }
        )

    return {"sections": [{"title": "Answer", "text": text, "citations": citations}]}


@solver
def openscholar_solver(
    task_type: TaskKind = "auto",
    api_base_url: str = DEFAULT_API_BASE,
    submit_timeout: int = 30,
    poll_timeout: int = 60,
    poll_interval: int = 8,
    max_poll_attempts: int = 40,
) -> Solver:
    """OpenScholar solver for ScholarQA-CS2 tasks."""

    config = OpenScholarConfig(
        api_base_url=api_base_url,
        submit_timeout=submit_timeout,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        max_poll_attempts=max_poll_attempts,
    )
    client = OpenScholarClient(config)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        resolved_task = _infer_task_type(task_type, state)
        if resolved_task not in {
            "sqa",
            "litqa2_open",
            "litqa2_open_llm_only",
            "pubmedqa_open",
        }:
            state.output.completion = json.dumps(
                {
                    "sections": [
                        {
                            "title": "Unsupported task",
                            "text": "OpenScholar API only supports ScholarQA-CS2 and open-answer judge tasks.",
                            "citations": [],
                        }
                    ]
                },
                indent=2,
                ensure_ascii=False,
            )
            return state

        question = _extract_question(state)
        payload = await asyncio.to_thread(client.query, question)
        if resolved_task == "sqa":
            response = _build_sqa_response(payload)
            state.output.completion = json.dumps(response, indent=2, ensure_ascii=False)
        else:
            text = str(payload.get("output") if payload else "").strip()
            state.output.completion = text or "No answer generated."
        return state

    return solve


__all__ = ["openscholar_solver", "OpenScholarClient", "OpenScholarConfig"]

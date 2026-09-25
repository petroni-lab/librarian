"""Inspect/AstaBench wrapper for the librarian literature workflows."""
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import httpx

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "agents").is_dir()),
    Path(__file__).resolve().parents[3],
)  # repo root = first ancestor containing agents/ (move-proof)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from inspect_ai.model import ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ContentText, Tool, ToolDef

from evals.Literature.AstaBench.compat import (
    build_llm_client,
    extract_json_from_response,
    merge_tools_with_state,
    record_model_usage_with_inspect,
)
from librarian.literature_search import search_scientific_literature_structured
from librarian.llm_client import LLMClient

try:
    from .prompts import ASTA_LIBRARIAN_SYSTEM_PROMPT
except ImportError:
    from prompts import ASTA_LIBRARIAN_SYSTEM_PROMPT

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
ConfigName = Literal["custom_tooling", "standard_tooling", "llm_only"]
SearchFn = Callable[[str, int], list[dict[str, Any]]]

S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_S2_FIELDS = "title,abstract,authors,year,venue,corpusId,externalIds"
LITERATURE_PROMPTS_DIR = (
    PROJECT_ROOT / "agents" / "deprecated" / "literature" / "prompts"
)
BUILTIN_QUERY_PLANNERS = {
    "default",
    "simple_bm25",
    "raw_question",
    "epmc_full_interface",
}
logger = logging.getLogger(__name__)


def _bind_current_otel_context_for_eval(func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach the current OpenTelemetry context inside eval worker threads."""
    try:
        from opentelemetry.context import attach, detach, get_current
    except Exception:
        return func

    current_context = get_current()
    if current_context is None:
        return func

    @wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        token = attach(current_context)
        try:
            return func(*args, **kwargs)
        finally:
            detach(token)

    return _wrapped


@dataclass(frozen=True)
class BioAgentExecutionConfig:
    """Execution knobs for the AstaBench integration."""

    name: ConfigName
    search_mode: Literal["europepmc", "asta_corpus"]
    europepmc_search_fn: SearchFn | None = None
    europepmc_librarian_prompt: str | None = None
    query_planner: str = "default"
    librarian_prompt_label: str = "default"
    simple_bm25_max_queries: int = 7
    epmc_full_interface_max_queries: int = 7
    max_query_count_override: int | None = None
    # LibrarianAgent-path overrides only (_run_librarian_agent) — plain
    # dataclasses.replace() on the loaded LibrarianRuntimeConfig, no env vars.
    # None means "use the librarian/config.toml value unchanged".
    librarian_num_subqueries_override: int | None = None
    librarian_paragraphs_per_subquery_override: int | None = None
    librarian_paragraphs_per_judge_batch_override: int | None = None
    retrieval_policy: Literal[
        "synthesis",
        "localized_evidence",
        "localized_evidence_bm25",
        "localized_evidence_bm25_per_paper",
        "cascade_bm25",
        "upfront_fulltext",
        "upfront_fulltext_bm25_filter",
        "simple_fulltext_bm25",
    ] = "synthesis"
    max_paper_finder_results: int = 50
    max_report_papers: int = 10
    snippet_limit: int = 3


class InspectUsageRecorder:
    """Translate OpenAI-compatible usage payloads into Inspect model events."""

    def __call__(self, model_name: str, usage_payload: dict[str, Any]) -> None:
        input_tokens = self._coerce_int(
            usage_payload.get("input_tokens")
            or usage_payload.get("prompt_tokens")
            or usage_payload.get("prompt")
        )
        output_tokens = self._coerce_int(
            usage_payload.get("output_tokens")
            or usage_payload.get("completion_tokens")
            or usage_payload.get("completion")
        )
        total_tokens = self._coerce_int(
            usage_payload.get("total_tokens") or usage_payload.get("total")
        )
        reasoning_tokens = self._coerce_int(
            usage_payload.get("reasoning_tokens") or usage_payload.get("reasoning")
        )

        if None in (input_tokens, output_tokens, total_tokens):
            return

        usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        record_model_usage_with_inspect(model_name, usage, allow_invalid=True)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


#: Placeholder a retrieved paper carries where it has no evidence for a
#: sub-query. Inherited from the retired internal literature agent, whose
#: records these adapters still parse.
_EVIDENCE_GAP_SEPARATOR = "|"

class AstaScientificCorpusAdapter:
    """Thin adapter over task-provided Asta tools."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools_by_name = {ToolDef(tool).name: tool for tool in tools}

    def require(self, name: str) -> Tool:
        if name not in self.tools_by_name:
            available = ", ".join(sorted(self.tools_by_name))
            raise KeyError(f"Required tool '{name}' not available. Found: {available}")
        return self.tools_by_name[name]

    async def search_papers(self, query: str, limit: int) -> list[dict[str, Any]]:
        tool = self.require("search_papers_by_relevance")
        try:
            raw_result = await tool(keyword=query, limit=limit)
        except TypeError:
            # Some tool revisions expose `query` instead of `keyword`.
            raw_result = await tool(query=query, limit=limit)
        payload = self._parse_tool_result(raw_result)
        papers = self._extract_data_records(payload)
        return [self._normalize_search_hit(paper) for paper in papers]

    def search_sync(self, query: str, page_size: int = 50) -> list[dict[str, Any]]:
        return asyncio.run(self.search_papers(query=query, limit=page_size))

    async def search_paper_by_title(self, title: str) -> list[dict[str, Any]]:
        raw_result = await self.require("search_paper_by_title")(title=title)
        payload = self._parse_tool_result(raw_result)
        papers = self._extract_data_records(payload)
        return [self._normalize_search_hit(paper) for paper in papers]

    async def snippet_search(
        self,
        query: str,
        limit: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if paper_ids:
            kwargs["paper_ids"] = ",".join(paper_ids)
        raw_result = await self.require("snippet_search")(**kwargs)
        payload = self._parse_tool_result(raw_result)
        return self._extract_data_records(payload)

    async def enrich_papers_with_snippets(
        self,
        query: str,
        papers: Sequence[dict[str, Any]],
        per_paper_limit: int,
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for paper in papers:
            normalized = dict(paper)
            corpus_id = self._paper_corpus_id(normalized)
            if corpus_id:
                snippets = await self.snippet_search(
                    query=query,
                    limit=per_paper_limit,
                    paper_ids=[f"CorpusId:{corpus_id}"],
                )
                snippet_texts = [
                    snippet.get("text", "").strip()
                    for snippet in snippets
                    if snippet.get("text")
                ]
                if snippet_texts:
                    normalized["support_snippets"] = snippet_texts
                    normalized["full_text_excerpt"] = "\n\n".join(snippet_texts)
            enriched.append(normalized)
        return enriched

    async def resolve_corpus_id(self, paper: dict[str, Any]) -> str | None:
        corpus_id = self._paper_corpus_id(paper)
        if corpus_id:
            return corpus_id

        title = paper.get("title")
        if not title:
            return None

        matches = await self.search_paper_by_title(str(title))
        target_title = _normalize_title(str(title))
        for candidate in matches:
            candidate_title = _normalize_title(str(candidate.get("title", "")))
            if candidate_title == target_title and self._paper_corpus_id(candidate):
                return self._paper_corpus_id(candidate)

        if matches:
            return self._paper_corpus_id(matches[0])
        return None

    @staticmethod
    def _paper_corpus_id(paper: dict[str, Any]) -> str | None:
        corpus_id = paper.get("corpus_id") or paper.get("corpusId")
        if corpus_id in (None, ""):
            return None
        return str(corpus_id)

    @staticmethod
    def _normalize_search_hit(paper: dict[str, Any]) -> dict[str, Any]:
        authors = paper.get("authors") or []
        author_names = []
        for author in authors:
            if isinstance(author, dict) and author.get("name"):
                author_names.append(author["name"])
            elif isinstance(author, str):
                author_names.append(author)
        abstract = paper.get("abstract") or ""
        title = paper.get("title") or "No title"
        corpus_id = paper.get("corpusId")
        return {
            "title": title,
            "authors": author_names,
            "pmid": "",
            "pmcid": "",
            "doi": "",
            "source": "Asta Scientific Corpus",
            "pageContent": f"Title: {title}\n\nAbstract: {abstract}".strip(),
            "abstract": abstract,
            "journal": paper.get("venue", ""),
            "year": paper.get("year", ""),
            "isOpenAccess": paper.get("isOpenAccess", False),
            "hasPDF": False,
            "inEPMC": False,
            "hasFreeFullText": False,
            "corpus_id": str(corpus_id) if corpus_id is not None else "",
            "corpusId": str(corpus_id) if corpus_id is not None else "",
        }

    @staticmethod
    def _parse_tool_result(raw_result: Any) -> Any:
        if isinstance(raw_result, list) and raw_result:
            if all(isinstance(item, ContentText) for item in raw_result):
                if len(raw_result) == 1:
                    return json.loads(raw_result[0].text)
                return [json.loads(item.text) for item in raw_result]
        if isinstance(raw_result, str):
            parsed = extract_json_from_response(raw_result)
            if parsed is not None:
                return parsed
        return raw_result

    @staticmethod
    def _extract_data_records(payload: Any) -> list[dict[str, Any]]:
        """Normalize Asta tool payload variants into a flat list of record dicts."""

        def looks_like_record(obj: dict[str, Any]) -> bool:
            record_keys = {
                "paperId",
                "corpusId",
                "title",
                "abstract",
                "text",
                "snippet",
                "authors",
            }
            return bool(record_keys & set(obj.keys()))

        def extract_from_item(item: Any) -> list[dict[str, Any]]:
            if not isinstance(item, dict):
                return []
            data = item.get("data")
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            if isinstance(data, dict):
                return [data]
            if looks_like_record(item):
                return [item]
            if "error" in item:
                logger.debug("Asta tool payload error: %s", item.get("error"))
            return []

        if isinstance(payload, dict):
            return extract_from_item(payload)

        if isinstance(payload, list):
            records: list[dict[str, Any]] = []
            for item in payload:
                records.extend(extract_from_item(item))
            return records

        return []


class PublicSemanticScholarResolver:
    """Fallback resolver for turning titles/DOIs into corpus IDs."""

    def __init__(self) -> None:
        self._api_key = os.getenv("S2_API_KEY") or os.getenv("ASTA_TOOL_KEY", "")
        self._cache: dict[str, str | None] = {}
        self._last_request_time = 0.0
        self._min_request_interval = float(os.getenv("S2_MIN_INTERVAL_SECONDS", "1.1"))

    async def resolve_corpus_id(self, paper: dict[str, Any]) -> str | None:
        existing_corpus_id = paper.get("corpus_id") or paper.get("corpusId")
        if existing_corpus_id:
            return str(existing_corpus_id)

        doi = str(paper.get("doi") or "").strip()
        query = doi or str(paper.get("title") or "").strip()
        if not query:
            return None
        cache_key = query.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if doi:
                corpus_id = await self._resolve_by_doi(doi)
                if corpus_id:
                    self._cache[cache_key] = corpus_id
                    return corpus_id

            corpus_id = await self._resolve_by_search(query, paper)
            self._cache[cache_key] = corpus_id
            return corpus_id
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            logger.warning(
                "Semantic Scholar corpus-id resolution failed with HTTP %s for %r.",
                status_code,
                query,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Semantic Scholar corpus-id resolution failed for %r: %s",
                query,
                exc,
            )
        self._cache[cache_key] = None
        return None

    async def _resolve_by_doi(self, doi: str) -> str | None:
        payload = await self._get_json(
            f"{S2_API_BASE_URL}/paper/DOI:{doi}",
            params={"fields": DEFAULT_S2_FIELDS},
            allow_not_found=True,
        )
        if not isinstance(payload, dict):
            return None
        corpus_id = payload.get("corpusId")
        return str(corpus_id) if corpus_id not in (None, "") else None

    async def _resolve_by_search(
        self,
        query: str,
        paper: dict[str, Any],
    ) -> str | None:
        payload = await self._get_json(
            f"{S2_API_BASE_URL}/paper/search",
            params={
                "query": query,
                "limit": 5,
                "fields": DEFAULT_S2_FIELDS,
            },
        )
        candidates = payload.get("data", []) if isinstance(payload, dict) else []
        expected_title = _normalize_title(str(paper.get("title", "")))
        expected_year = str(paper.get("year") or "").strip()
        for candidate in candidates:
            if _normalize_title(str(candidate.get("title", ""))) != expected_title:
                continue
            candidate_year = str(candidate.get("year") or "").strip()
            if expected_year and candidate_year and expected_year != candidate_year:
                continue
            corpus_id = candidate.get("corpusId")
            if corpus_id not in (None, ""):
                return str(corpus_id)

        if candidates:
            corpus_id = candidates[0].get("corpusId")
            if corpus_id not in (None, ""):
                return str(corpus_id)
        return None

    async def _get_json(
        self,
        url: str,
        params: dict[str, Any],
        allow_not_found: bool = False,
    ) -> Any:
        headers = {"x-api-key": self._api_key} if self._api_key else {}
        async with httpx.AsyncClient(timeout=20.0) as client:
            for attempt in range(4):
                await self._respect_rate_limit()
                response = await client.get(url, params=params, headers=headers)
                if allow_not_found and response.status_code == 404:
                    return None
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    return response.json()

                if attempt == 3:
                    response.raise_for_status()

                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2.0**attempt
                else:
                    delay = 2.0**attempt
                await asyncio.sleep(delay)
        return None

    async def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.monotonic()


class BioAgentAstaWrapper:
    """Task-aware wrapper that adapts librarian outputs to AstaBench schemas."""

    def __init__(
        self,
        config: BioAgentExecutionConfig,
        task_type: TaskKind = "auto",
        llm_base_url: str | None = None,
        llm_model_name: str | None = None,
        full_text_enrichment: bool = True,
        reasoning_effort: str = "low",
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.task_type = task_type
        self.llm_base_url = llm_base_url
        self.llm_model_name = llm_model_name
        self.full_text_enrichment = full_text_enrichment
        # GLM-5.3+ effort level; LLMClient and the deprecated literature agent
        # both accept it in place of the old thinking bool.
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self.usage_recorder = InspectUsageRecorder()
        self.public_resolver = PublicSemanticScholarResolver()

    async def solve(self, state: TaskState) -> TaskState:
        task_type = self._infer_task_type(state)
        if task_type == "paper_finder":
            return await self._solve_paper_finder(state)
        if task_type == "litqa2":
            return await self._solve_litqa2(state)
        if task_type == "litqa2_open":
            return await self._solve_litqa2_open(state)
        if task_type == "litqa2_open_llm_only":
            return await self._solve_litqa2_open_llm_only(state)
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
        metadata = dict(state.metadata or {})
        query = str(metadata.get("raw_query") or self._extract_user_query(state))
        retrieval = await self._run_literature_agent(query=query, state=state)
        papers = list(retrieval.get("papers_raw", []))
        corpus_adapter = self._make_asta_adapter_if_available(state)
        results = []
        seen_corpus_ids: set[str] = set()

        for paper in papers:
            corpus_id = await self._resolve_corpus_id(
                paper=paper,
                corpus_adapter=corpus_adapter,
            )
            if not corpus_id or corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)
            results.append(
                {
                    "paper_id": corpus_id,
                    "markdown_evidence": _build_markdown_evidence(paper),
                }
            )
            if len(results) >= self.config.max_paper_finder_results:
                break

        state.output.completion = json.dumps(
            {"output": {"results": results}},
            indent=2,
            ensure_ascii=False,
        )
        return state

    async def _solve_litqa2(self, state: TaskState) -> TaskState:
        metadata = dict(state.metadata or {})
        question = self._extract_question_from_multichoice_input(state)
        choices = self._extract_choices(state)
        retrieval = await self._run_literature_agent(query=question, state=state)
        evidence = self._format_sources_for_prompt(
            retrieval.get("papers_raw", [])[: self.config.max_report_papers]
        )

        prompt = (
            "You are answering a multiple-choice scientific literature question.\n"
            "Use only the evidence below.\n\n"
            f"Question:\n{question}\n\n"
            "Choices:\n"
            + "\n".join(f"{letter}. {text}" for letter, text in choices)
            + "\n\nEvidence:\n"
            + evidence
            + '\n\nReturn JSON only in the form {"answer": "<letter>"}.'
        )

        selector = self._make_llm_client()
        raw_output = await asyncio.to_thread(
            selector.generate_structured_output,
            prompt,
            "You are a careful scientist. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output) or {}
        answer = str(parsed.get("answer") or "").strip().upper()
        valid_letters = {letter for letter, _ in choices}
        if answer not in valid_letters:
            answer = str(metadata.get("unsure_letter", "A")).strip().upper()

        if getattr(state, "choices", None):
            for idx in range(len(state.choices)):
                state.choices.mark_choice(idx, idx == (ord(answer) - ord("A")))

        state.output.completion = json.dumps({"answer": answer})
        return state

    async def _solve_litqa2_open(self, state: TaskState) -> TaskState:
        """Run retrieval and synthesize an OpenJudge answer from focused evidence."""
        question = self._extract_question_from_multichoice_input(state)
        retrieval = await self._run_literature_agent(
            query=question,
            state=state,
            additional_context=self._litqa2_open_retrieval_context(question),
            include_summary=False,
        )
        self._attach_retrieval_debug_metadata(
            state=state,
            query=question,
            retrieval=retrieval,
        )
        answer = await self._build_litqa2_open_answer(
            question=question,
            retrieval=retrieval,
        )
        state.output.completion = answer or "No answer generated."
        return state

    async def _build_litqa2_open_answer(
        self,
        question: str,
        retrieval: dict[str, Any],
    ) -> str:
        """Construct the evaluated answer from evidence snippets, not summaries."""
        evidence = self._format_litqa2_open_evidence_for_prompt(retrieval)
        if not evidence:
            summary = str(retrieval.get("summary") or "").strip()
            return summary

        prompt = self._litqa2_open_answer_prompt(question=question, evidence=evidence)
        system_content = (
            "You are a careful biomedical scientist. Extract the direct "
            "answer from provided evidence and return valid JSON only."
        )
        llm = self._make_llm_client()
        raw_answer = await asyncio.to_thread(
            llm.chat_completion,
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        return self._render_litqa2_open_answer(raw_answer)

    @staticmethod
    def _litqa2_open_retrieval_context(question: str) -> str:
        return (
            "LitQA2 OpenJudge retrieval mode: this is an exact-answer biomedical "
            "question, often asking for a short value, gene/protein, residue range, "
            "TM helix, mutation, percentage, fold-change, or named structure. "
            "Preserve every named entity, organism, method, cell line, protein, and "
            "phenotype from the question in search queries. Prefer primary papers "
            "and passages that contain the exact relation asked by the question. "
            "Do not broaden to reviews unless primary evidence is unavailable.\n\n"
            f"Question to preserve exactly:\n{question}"
        )

    @staticmethod
    def _litqa2_open_answer_prompt(question: str, evidence: str) -> str:
        return (
            "Use only the retrieved evidence to answer the LitQA2 question.\n"
            "Return JSON only with this schema:\n"
            '{"answer": "<short exact answer>", "support": "<one evidence-backed '
            'sentence>", "confidence": "high|medium|low"}\n\n'
            "Rules:\n"
            "- The hidden gold answer is usually a short string. Put that short "
            "string in `answer`: a number, percentage, fold-change, gene/protein, "
            "residue range, mutation, helix, structure label, or concise conclusion.\n"
            "- `answer` must be the first thing a judge would need to see; keep it "
            "under 20 words whenever possible.\n"
            "- Do not write chain-of-thought, search commentary, markdown, citations, "
            "or a literature review.\n"
            "- For 'which of the following' questions, the options are intentionally "
            "hidden. Infer the single best option text from the evidence instead of "
            "complaining that options are missing.\n"
            "- If several entities are mentioned, choose the one most directly linked "
            "to the relation in the question. Do not list background pathway members "
            "unless the question asks for multiple answers.\n"
            "- Prefer exact strings in the evidence over paraphrases: e.g. `43.6%`, "
            "`2.7-fold`, `residues 344-360`, `D215W`, `TM3`, `GSDMD`.\n"
            "- Use `confidence: high` only when the evidence explicitly contains the "
            "answer. Use `medium` for a well-supported inference. Use `low` only when "
            "the evidence is weak, but still give the best specific answer if one is "
            "present.\n"
            "- Set `answer` to `Insufficient evidence` only when no retrieved evidence "
            "addresses the question at all.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "JSON:"
        )

    @staticmethod
    def _render_litqa2_open_answer(raw_answer: str) -> str:
        raw_answer = raw_answer.strip()
        parsed = extract_json_from_response(raw_answer)
        if not isinstance(parsed, dict):
            try:
                parsed = json.loads(raw_answer)
            except json.JSONDecodeError:
                parsed = None

        if not isinstance(parsed, dict):
            answer_match = re.search(
                r"(?im)^\s*(?:answer|final answer)\s*:\s*(.+?)\s*$",
                raw_answer,
            )
            if answer_match:
                return answer_match.group(1).strip()
            return raw_answer

        answer = str(parsed.get("answer") or "").strip()
        support = str(parsed.get("support") or "").strip()
        if not answer:
            return raw_answer

        if answer.lower() in {
            "insufficient evidence",
            "insufficient information",
            "not enough evidence",
        }:
            return "Insufficient evidence to answer from the retrieved papers."

        # Append support sentence when present (v1 only; v2+ omit it).
        if support and support.lower() != answer.lower():
            return f"{answer}. {support}"
        return answer

    async def _solve_litqa2_open_llm_only(self, state: TaskState) -> TaskState:
        """LLM-only baseline: answer from parametric knowledge, no retrieval.

        The question is sent directly to the LLM without running the literature
        agent. This isolates what the model already knows from what retrieval adds.
        The model must commit to a specific answer — the judge evaluates it
        against the gold answer without seeing the MCQ options.
        """
        question = self._extract_question_from_multichoice_input(state)
        prompt = (
            "Answer the following biomedical / scientific question directly and "
            "specifically based on your knowledge. Always commit to a specific answer "
            "— give a short, concrete response (a value, gene name, number, etc.). "
            "If uncertain, still give your best answer.\n\n"
            f"Question:\n{question}\n\n"
            "Answer with a short specific phrase or value only."
        )
        llm = self._make_llm_client()
        answer = await asyncio.to_thread(
            llm.generate_structured_output,
            prompt,
            "You are a knowledgeable scientist. Answer with a short specific phrase or value.",
        )
        state.output.completion = answer.strip() or "No answer generated."
        return state

    async def _solve_pubmedqa_open(self, state: TaskState) -> TaskState:
        """Run public biomedical retrieval and return an open PubMedQA answer."""
        question = self._state_input_text(state).strip()
        retrieval = await self._run_literature_agent(
            query=question,
            state=state,
            force_public_search=True,
            include_summary=True,
        )
        summary = str(retrieval.get("summary") or "").strip()
        state.output.completion = summary or "No answer generated."
        return state

    async def _solve_sqa(self, state: TaskState) -> TaskState:
        metadata = dict(state.metadata or {})
        question = str(
            metadata.get("initial_prompt") or self._extract_user_query(state)
        )
        retrieval = await self._run_literature_agent(query=question, state=state)
        deterministic = self._build_deterministic_sqa_response(retrieval)
        if deterministic is not None:
            state.output.completion = json.dumps(
                deterministic, indent=2, ensure_ascii=False
            )
            return state

        parsed = await self._format_sqa_with_llm(question=question, retrieval=retrieval)
        state.output.completion = json.dumps(parsed, indent=2, ensure_ascii=False)
        return state

    async def _format_sqa_with_llm(
        self,
        question: str,
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are writing a structured ScholarQA-style report.\n"
            "Use only the provided evidence and do not invent citations.\n\n"
            f"Question:\n{question}\n\n"
            "Return valid JSON with a top-level `sections` list. Each section must "
            "have `title`, `text`, and `citations`. Each citation must have `id`, "
            "`snippets`, `title`, and optional `metadata`. Every citation id used in "
            "the text must appear in that section's `citations` list.\n\n"
            "Evidence:\n"
            + self._format_sources_for_prompt(
                retrieval.get("papers_raw", [])[: self.config.max_report_papers]
            )
        )

        formatter = self._make_llm_client()
        raw_output = await asyncio.to_thread(
            formatter.generate_structured_output,
            prompt,
            "You are a precise report formatter. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output)
        if not parsed or "sections" not in parsed:
            repair_prompt = (
                "Convert the following text into valid JSON for ScholarQA.\n"
                "Return JSON only with a top-level `sections` list.\n"
                "Each section must contain `title`, `text`, and `citations`.\n"
                "Each citation must contain `id`, `snippets`, `title`, and optional `metadata`.\n\n"
                "Text to repair:\n"
                f"{raw_output}"
            )
            repaired_output = await asyncio.to_thread(
                formatter.generate_structured_output,
                repair_prompt,
                "You are a strict JSON repair assistant. Return valid JSON only.",
            )
            parsed = extract_json_from_response(repaired_output)
        if not parsed or "sections" not in parsed:
            return {
                "sections": [
                    {
                        "title": "Answer",
                        "text": retrieval.get("summary", ""),
                        "citations": [],
                    }
                ]
            }
        return parsed

    def _build_deterministic_sqa_response(
        self,
        retrieval: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build ScholarQA response without a second formatter LLM call."""
        summary_text = str(retrieval.get("summary") or "").strip()
        papers = list(retrieval.get("papers_raw") or [])
        if not summary_text and not papers:
            return None

        normalized_text = self._normalize_inline_numeric_citations(summary_text)
        cited_indices = self._extract_cited_indices(normalized_text)

        # If the summary missed inline markers, add a minimal evidence anchor.
        if not cited_indices and papers:
            fallback_count = min(max(self.config.max_report_papers, 1), len(papers))
            cited_indices = list(range(1, fallback_count + 1))
            cite_suffix = "".join(f"[{idx}]" for idx in cited_indices)
            if normalized_text:
                normalized_text = (
                    f"{normalized_text}\n\nRepresentative evidence: {cite_suffix}"
                )
            else:
                normalized_text = f"Representative evidence: {cite_suffix}"

        citations = [
            self._build_sqa_citation(idx, papers[idx - 1])
            for idx in cited_indices
            if 1 <= idx <= len(papers)
        ]
        if not normalized_text or not citations:
            return None

        return {
            "sections": [
                {
                    "title": "Answer",
                    "text": normalized_text,
                    "citations": citations,
                }
            ]
        }

    @staticmethod
    def _normalize_inline_numeric_citations(text: str) -> str:
        """Expand grouped citations like [8, 9] into [8][9] for exact id matching."""
        if not text:
            return ""

        def _expand_group(match: re.Match[str]) -> str:
            body = match.group(1)
            tokens = [
                token.strip() for token in re.split(r"[,;]", body) if token.strip()
            ]
            expanded: list[int] = []
            for token in tokens:
                range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", token)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if end >= start and (end - start) <= 20:
                        expanded.extend(range(start, end + 1))
                        continue
                    return match.group(0)
                if token.isdigit():
                    expanded.append(int(token))
                    continue
                return match.group(0)
            if not expanded:
                return match.group(0)
            return "".join(f"[{idx}]" for idx in expanded)

        return re.sub(r"\[([0-9,\s;–-]+)\]", _expand_group, text)

    @staticmethod
    def _extract_cited_indices(text: str) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for match in re.finditer(r"\[(\d+)\]", text or ""):
            idx = int(match.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)
        return ordered

    def _build_sqa_citation(self, idx: int, paper: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        year = paper.get("year")
        journal = paper.get("journal")
        corpus_id = paper.get("corpus_id") or paper.get("corpusId")
        doi = paper.get("doi")
        url = paper.get("url")
        authors = paper.get("authors")
        if year:
            metadata["year"] = year
        if journal:
            metadata["venue"] = journal
        if corpus_id:
            metadata["corpusId"] = str(corpus_id)
        if doi:
            metadata["doi"] = str(doi)
        if url:
            metadata["url"] = str(url)
        if authors:
            if isinstance(authors, list):
                metadata["authors"] = [str(author) for author in authors[:8]]
            else:
                metadata["authors"] = str(authors)

        return {
            "id": f"[{idx}]",
            "snippets": self._citation_snippets_for_paper(paper),
            "title": str(paper.get("title") or "Untitled"),
            "metadata": metadata,
        }

    def _citation_snippets_for_paper(self, paper: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        support_snippets = paper.get("support_snippets")
        if isinstance(support_snippets, list):
            candidates.extend(str(snippet) for snippet in support_snippets)
        elif isinstance(support_snippets, str):
            candidates.append(support_snippets)

        full_text_excerpt = str(paper.get("full_text_excerpt") or "").strip()
        if full_text_excerpt:
            candidates.extend(part for part in full_text_excerpt.split("\n\n") if part)

        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            candidates.append(abstract)

        if not candidates:
            title = str(paper.get("title") or "").strip()
            if title:
                candidates.append(title)

        snippets: list[str] = []
        seen: set[str] = set()
        max_snippets = max(1, self.config.snippet_limit)
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if len(normalized) > 600:
                normalized = f"{normalized[:597].rstrip()}..."
            snippets.append(normalized)
            if len(snippets) >= max_snippets:
                break

        return snippets or ["No snippet available."]

    def _format_litqa2_open_evidence_for_prompt(
        self,
        retrieval: dict[str, Any],
    ) -> str:
        """Render ranked evidence for LitQA2 OpenJudge answer synthesis."""
        papers = list(retrieval.get("papers_raw") or [])
        evidence_entries = [
            entry
            for entry in list(retrieval.get("evidence") or [])
            if isinstance(entry, dict)
        ]
        if not papers and not evidence_entries:
            return ""

        paper_by_id: dict[str, dict[str, Any]] = {}
        for paper in papers:
            for key in self._paper_identifier_keys(paper):
                paper_by_id.setdefault(key, paper)

        max_items = max(1, self.config.max_report_papers)
        rendered: list[str] = []
        used_paper_ids: set[int] = set()
        for idx in range(max(len(papers), len(evidence_entries))):
            if len(rendered) >= max_items:
                break
            entry = evidence_entries[idx] if idx < len(evidence_entries) else {}
            paper = papers[idx] if idx < len(papers) else None
            entry_id = self._first_present_identifier(entry)
            if entry_id and entry_id in paper_by_id:
                paper = paper_by_id[entry_id]
            if paper is not None:
                used_paper_ids.add(id(paper))

            block = self._render_litqa2_open_evidence_block(
                rank=len(rendered) + 1,
                paper=paper,
                evidence_entry=entry,
            )
            if block:
                rendered.append(block)

        if len(rendered) < max_items:
            for paper in papers:
                if id(paper) in used_paper_ids:
                    continue
                block = self._render_litqa2_open_evidence_block(
                    rank=len(rendered) + 1,
                    paper=paper,
                    evidence_entry={},
                )
                if block:
                    rendered.append(block)
                if len(rendered) >= max_items:
                    break

        return "\n\n".join(rendered)

    def _render_litqa2_open_evidence_block(
        self,
        rank: int,
        paper: dict[str, Any] | None,
        evidence_entry: dict[str, Any],
    ) -> str:
        snippets = self._dedupe_texts(
            [
                *self._extract_evidence_entry_texts(evidence_entry),
                *self._extract_paper_evidence_texts(paper or {}),
            ]
        )
        if not snippets:
            return ""

        if paper is None:
            title = "Untitled"
            year = "n.d."
            identifiers = self._identifier_label(evidence_entry)
        else:
            title = str(paper.get("title") or "Untitled")
            year = str(paper.get("year") or "n.d.")
            identifiers = self._identifier_label(paper)

        snippet_lines = []
        for snippet in snippets[:6]:
            compact = self._truncate_prompt_text(snippet, 700)
            if compact:
                snippet_lines.append(f"- {compact}")

        if not snippet_lines:
            return ""

        id_suffix = f" [{identifiers}]" if identifiers else ""
        return f"[{rank}] {title} ({year}){id_suffix}\n" + "\n".join(snippet_lines)

    @staticmethod
    def _extract_evidence_entry_texts(entry: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for key in ("evidence", "evidence_abstract", "evidence_fulltext"):
            BioAgentAstaWrapper._extend_texts(texts, entry.get(key))
        return texts

    @staticmethod
    def _extract_paper_evidence_texts(paper: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for key in (
            "support_snippets",
            "evidence_sentences_abstract",
            "evidence_sentences_full_text",
            "full_text_excerpt",
            "abstract",
        ):
            value = paper.get(key)
            if key == "full_text_excerpt" and isinstance(value, str):
                parts = [part.strip() for part in value.split("\n\n") if part.strip()]
                BioAgentAstaWrapper._extend_texts(texts, parts or value)
                continue
            BioAgentAstaWrapper._extend_texts(texts, value)
        return texts

    @staticmethod
    def _extend_texts(target: list[str], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = " ".join(value.split())
            if text and text != _EVIDENCE_GAP_SEPARATOR:
                target.append(text)
            return
        if isinstance(value, list):
            for item in value:
                BioAgentAstaWrapper._extend_texts(target, item)
            return
        if isinstance(value, dict):
            for item in value.values():
                BioAgentAstaWrapper._extend_texts(target, item)

    @staticmethod
    def _dedupe_texts(texts: Sequence[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for text in texts:
            normalized = " ".join(str(text).split())
            if not normalized:
                continue
            fingerprint = normalized.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _paper_identifier_keys(paper: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        for raw_key in (
            "_evidence_id",
            "pmid",
            "pmcid",
            "doi",
            "corpus_id",
            "corpusId",
        ):
            value = str(paper.get(raw_key) or "").strip()
            if value:
                keys.append(value)
        return keys

    @staticmethod
    def _first_present_identifier(obj: dict[str, Any]) -> str:
        for raw_key in (
            "_evidence_id",
            "pmid",
            "pmcid",
            "doi",
            "corpus_id",
            "corpusId",
        ):
            value = str(obj.get(raw_key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _identifier_label(obj: dict[str, Any]) -> str:
        labels: list[str] = []
        for label, raw_key in (
            ("PMID", "pmid"),
            ("PMCID", "pmcid"),
            ("DOI", "doi"),
            ("CorpusId", "corpus_id"),
            ("CorpusId", "corpusId"),
        ):
            value = str(obj.get(raw_key) or "").strip()
            if value and f"{label}: {value}" not in labels:
                labels.append(f"{label}: {value}")
        return "; ".join(labels)

    @staticmethod
    def _truncate_prompt_text(text: str, limit: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3].rstrip()}..."

    async def _solve_arxivdigestables(self, state: TaskState) -> TaskState:
        metadata = dict(state.metadata or {})
        sample_input = self._state_input_text(state)
        corpus_adapter = self._make_asta_adapter_if_available(state)
        extra_snippets = ""
        if corpus_adapter is not None and metadata.get("corpus_ids"):
            caption_query = self._extract_arxiv_caption(sample_input)
            snippets = []
            for corpus_id in metadata["corpus_ids"]:
                snippet_hits = await corpus_adapter.snippet_search(
                    query=caption_query,
                    limit=self.config.snippet_limit,
                    paper_ids=[f"CorpusId:{corpus_id}"],
                )
                snippet_texts = [
                    hit.get("text", "").strip()
                    for hit in snippet_hits
                    if hit.get("text")
                ]
                if snippet_texts:
                    snippets.append(
                        {
                            "paper_id": str(corpus_id),
                            "snippets": snippet_texts,
                        }
                    )
            if snippets:
                extra_snippets = "\n\nAdditional snippet evidence:\n" + json.dumps(
                    snippets, indent=2, ensure_ascii=False
                )

        prompt = (
            "You are building an ArxivDIGESTables-style comparison table.\n"
            "Use the provided task input and any optional snippet evidence.\n"
            'Return JSON only with the schema {"cell_values": [{"paper_id": ..., '
            '"column_name": ..., "cell_value": ...}, ...]}.\n\n'
            f"Task input:\n{sample_input}"
            f"{extra_snippets}"
        )

        formatter = self._make_llm_client()
        raw_output = await asyncio.to_thread(
            formatter.generate_structured_output,
            prompt,
            "You are a careful table formatter. Return valid JSON only.",
        )
        parsed = extract_json_from_response(raw_output)
        if not parsed or "cell_values" not in parsed:
            parsed = {"cell_values": []}
        state.output.completion = json.dumps(parsed, indent=2, ensure_ascii=False)
        return state

    async def _run_librarian_agent(self, query: str) -> dict[str, Any]:
        """Retrieve evidence via the standalone two-filter LibrarianAgent and adapt
        its passages into the {papers_raw, evidence} contract the LitQA2 answer
        formatter expects. Enabled by LITERATURE_USE_LIBRARIAN_AGENT=true.
        Applies self.config's librarian_*_override knobs, if any, on top of the
        loaded config before constructing the librarian."""
        from librarian.agent import LibrarianAgent as _LibrarianClass

        from dataclasses import replace as _replace

        from librarian.config import load_runtime_config

        runtime_config = load_runtime_config()
        overrides = {
            "num_subqueries": self.config.librarian_num_subqueries_override,
            "paragraphs_per_subquery": (
                self.config.librarian_paragraphs_per_subquery_override
            ),
            "paragraphs_per_judge_batch": (
                self.config.librarian_paragraphs_per_judge_batch_override
            ),
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            runtime_config = _replace(runtime_config, **overrides)

        librarian = _LibrarianClass(
            runtime_config=runtime_config,
            full_text_enrichment=self.full_text_enrichment,
            verbose=self.verbose,
            llm_base_url=self.llm_base_url,
            llm_model_name=self.llm_model_name,
        )
        # Per-run effort override; the constructor seeds this from
        # LLM_REASONING_EFFORT and chat_completion reads it on every call.
        if self.reasoning_effort:
            librarian.llm.reasoning_effort = self.reasoning_effort
        # No usage_callback: the shipped client discards the response's usage
        # block, so the librarian's own token counts are not recoverable here.
        # See compat.build_llm_client.
        passages = await asyncio.to_thread(librarian.run, query)
        dbg = getattr(librarian, "last_run_debug", {}) or {}
        return self._shape_librarian_result(passages, dbg)

    async def _run_librarian_via_api(self, query: str) -> dict[str, Any]:
        """Same contract as :meth:`_run_librarian_agent`, retrieved remotely.

        The orchestrator ships ``LibrarianAgent.run``'s records verbatim, so the
        shaping below is shared with the in-process path. Two things are thinner
        here and deliberately so: there is no ``usage_callback``, because token
        counts are per-pod rather than per-call (they remain recoverable from
        Phoenix via the run's user label), and ``last_run_debug`` is limited to
        ``search_queries`` -- the payload carries no ``final_pmids`` or per-stage
        counts. Enabled by LITERATURE_VIA_API=true.
        """
        from evals.Literature import orchestrator_client

        results = await asyncio.to_thread(
            orchestrator_client.run_agent,
            query,
            agent="librarian",
            base_url=os.environ.get(
                "LIBRARIAN_API_URL", orchestrator_client.DEFAULT_BASE_URL
            ),
            source="litqa2-eval",
            session_prefix="litqa2",
        )
        papers = results.get("papers") or []
        return self._shape_librarian_result(
            papers, {"search_queries": results.get("search_queries") or []}
        )

    def _shape_librarian_result(
        self, passages: list[dict[str, Any]], dbg: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapt librarian records into the {papers_raw, evidence} LitQA2 contract.

        :param passages: Per-paper records from ``LibrarianAgent.run`` (or the
            identical records the orchestrator returns).
        :param dbg: The agent's ``last_run_debug``, or the subset the API exposes.
        """
        papers_raw: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for p in passages:
            pid = str(p.get("pmid") or p.get("paper_id") or "")
            # Prefer the filter-cited sentence lists (answer-bearing); fall back to
            # the new LibrarianAgent's `evidence_snippets`. Keep it as a LIST of
            # spans — the answer formatter renders up to 6 snippets/paper, so
            # joining them into one string would waste that budget and then get
            # truncated to a single ~700-char blob.
            ft_sents = list(p.get("evidence_sentences_full_text") or [])
            ab_sents = list(p.get("evidence_sentences_abstract") or [])
            if not ft_sents and not ab_sents:
                ft_sents = [s for s in (p.get("evidence_snippets") or []) if s.strip()]
            support = ft_sents + ab_sents
            papers_raw.append(
                {
                    "pmid": p.get("pmid"),
                    "paperId": pid,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "journal": p.get("journal"),
                    "authors": p.get("authors"),
                    "full_text_excerpt": p.get("text"),
                    "evidence_sentences_full_text": ft_sents,
                    "evidence_sentences_abstract": ab_sents,
                    "support_snippets": support,
                }
            )
            evidence.append(
                {"pmid": p.get("pmid"), "paperId": pid, "support_snippets": support}
            )
        # Surface librarian diagnostics so eval debug shows retrieval activity
        # (query count, candidates retrieved, final pmids) for parity analysis.
        # LibrarianAgent.last_run_debug carries search_queries, query_count,
        # paragraph_count, relevant_count and final_pmids. It has never carried
        # "candidate_pmids"; reading that key reported raw_search_pmid_count=0
        # on every run, which reads as "Europe PMC returned nothing". Pass the
        # whole dict through _bio_agent_debug so the real per-stage narrowing
        # (paragraphs -> relevant) shows up in eval debug instead.
        return {
            "papers_raw": papers_raw,
            "evidence": evidence,
            "summary": "",
            "search_queries": dbg.get("search_queries") or [],
            "pipeline_pmids": dbg.get("final_pmids") or [],
            "_bio_agent_debug": dbg,
        }

    async def _run_literature_agent(
        self,
        query: str,
        state: TaskState,
        force_public_search: bool = False,
        additional_context: str = "",
        include_summary: bool | None = None,
    ) -> dict[str, Any]:
        # Optional: route retrieval through the standalone two-filter LibrarianAgent
        # (the lightweight replacement for the legacy LiteratureSearchAgent). Used
        # to verify the refactor reproduces the bm25_per_paper eval. Europe PMC only.
        if (
            os.environ.get("LITERATURE_USE_LIBRARIAN_AGENT", "").strip().lower()
            in ("1", "true", "yes")
            and self.config.search_mode == "europepmc"
            and not force_public_search
        ):
            # LITERATURE_VIA_API routes the same librarian to the orchestrator.
            # The librarian_*_override knobs only exist in-process, so a run
            # asking for both keeps the local agent rather than silently
            # dropping the override it was configured with.
            wants_api = os.environ.get("LITERATURE_VIA_API", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            has_local_only_config = bool(
                self.config.librarian_num_subqueries_override is not None
                or self.config.librarian_paragraphs_per_subquery_override is not None
                or self.config.librarian_paragraphs_per_judge_batch_override is not None
            )
            if wants_api and not has_local_only_config:
                return await self._run_librarian_via_api(query)
            return await self._run_librarian_agent(query)

        # The pre-librarian retrieval stack (LiteratureSearchAgent, and the Asta
        # corpus adapter layered on it) is not part of this repository — it was
        # an internal agent the librarian replaced. Only the +Librarian row is
        # reproducible here, which is what run_litqa2.sh runs.
        raise RuntimeError(
            "Only the librarian retrieval path is available in this repository. "
            "Set LITERATURE_USE_LIBRARIAN_AGENT=true and use --config "
            "custom_tooling (what AstaBench/run_litqa2.sh does); "
            f"got search_mode={self.config.search_mode!r}, "
            f"force_public_search={force_public_search}."
        )

    def _attach_retrieval_debug_metadata(
        self,
        state: TaskState,
        query: str,
        retrieval: dict[str, Any],
        limit: int = 30,
    ) -> None:
        metadata = dict(state.metadata or {})
        metadata["bio_agent_retrieval_debug"] = self._build_retrieval_debug(
            query=query,
            retrieval=retrieval,
            limit=limit,
        )
        state.metadata = metadata

    def _build_retrieval_debug(
        self,
        query: str,
        retrieval: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        papers = list(retrieval.get("papers_raw") or [])
        agent_debug = dict(retrieval.get("_bio_agent_debug") or {})
        summary = str(retrieval.get("summary") or "")
        evidence = list(retrieval.get("evidence") or [])
        raw_pmids = (
            retrieval.get("raw_search_pmids")
            or agent_debug.get("raw_search_pmids")
            or []
        )
        # Flatten: set of all unique PMIDs across all queries
        all_raw_pmids = sorted(
            set(pmid for sublist in raw_pmids for pmid in sublist if pmid)
        )
        return {
            "query": query,
            "paper_count": len(papers),
            "evidence_count": len(evidence),
            "summary_chars": len(summary),
            "raw_search_pmids_all": all_raw_pmids,
            # Only the legacy bio-agent path exposes a pre-judge candidate pool.
            # The librarian path has none, so report None ("not measured here")
            # rather than 0, which reads as "search returned nothing".
            "raw_search_pmid_count": len(all_raw_pmids) if raw_pmids else None,
            "pipeline_pmids": retrieval.get("pipeline_pmids")
            or agent_debug.get("pipeline_pmids")
            or {},
            **agent_debug,
            "papers": [
                self._compact_retrieved_paper(rank=rank, paper=paper)
                for rank, paper in enumerate(papers[:limit], start=1)
            ],
        }

    @staticmethod
    def _compact_retrieved_paper(rank: int, paper: dict[str, Any]) -> dict[str, Any]:
        support_snippets = paper.get("support_snippets")
        if isinstance(support_snippets, list):
            support_snippet_count = len(support_snippets)
        elif support_snippets:
            support_snippet_count = 1
        else:
            support_snippet_count = 0

        abstract = str(paper.get("abstract") or "").strip()
        full_text_excerpt = str(paper.get("full_text_excerpt") or "").strip()
        return {
            "rank": rank,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "journal": paper.get("journal"),
            "pmid": paper.get("pmid"),
            "pmcid": paper.get("pmcid"),
            "doi": paper.get("doi"),
            "corpus_id": paper.get("corpusId") or paper.get("corpus_id"),
            "has_free_full_text": paper.get("hasFreeFullText")
            or paper.get("has_free_full_text"),
            "full_text_used": bool(full_text_excerpt),
            "full_text_excerpt_chars": len(full_text_excerpt),
            "full_text_reason": paper.get("full_text_reason"),
            "support_snippet_count": support_snippet_count,
            "abstract_chars": len(abstract),
            "abstract_preview": abstract[:240],
            "abstract_full": abstract,
            "full_text_preview": full_text_excerpt[:240],
            "full_text_full": full_text_excerpt,
        }

    async def _resolve_corpus_id(
        self,
        paper: dict[str, Any],
        corpus_adapter: AstaScientificCorpusAdapter | None,
    ) -> str | None:
        if corpus_adapter is not None:
            corpus_id = await corpus_adapter.resolve_corpus_id(paper)
            if corpus_id:
                return corpus_id
        return await self.public_resolver.resolve_corpus_id(paper)

    def _make_llm_client(self) -> LLMClient:
        # Allow a separate model for the final answer step while the retrieval
        # pipeline keeps using LLM_MODEL/LLM_BASE_URL.  Falls back to the
        # retrieval model when the answer-specific vars are not set.
        answer_base_url = (
            os.environ.get("LITQA2_ANSWER_LLM_BASE_URL") or self.llm_base_url
        )
        answer_model = os.environ.get("LITQA2_ANSWER_LLM_MODEL") or self.llm_model_name
        return build_llm_client(
            base_url=answer_base_url,
            model_name=answer_model,
            reasoning_effort=self.reasoning_effort,
            usage_callback=self.usage_recorder,
        )

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

    def _make_asta_adapter(self, state: TaskState) -> AstaScientificCorpusAdapter:
        if not state.tools:
            raise ValueError(
                "Config B requires task-provided Asta tools in state.tools, but none were provided."
            )
        return AstaScientificCorpusAdapter(state.tools)

    def _make_asta_adapter_if_available(
        self,
        state: TaskState,
    ) -> AstaScientificCorpusAdapter | None:
        if not state.tools:
            return None
        tool_names = {ToolDef(tool).name for tool in state.tools}
        if (
            "snippet_search" not in tool_names
            and "search_papers_by_relevance" not in tool_names
        ):
            return None
        return AstaScientificCorpusAdapter(state.tools)

    @staticmethod
    def _extract_arxiv_caption(sample_input: str) -> str:
        match = re.search(
            r"caption:\s*(?P<caption>.*?)\.\s*Return the table",
            sample_input,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return " ".join(match.group("caption").split())
        return "paper comparison details"

    @staticmethod
    def _format_sources_for_prompt(papers: Sequence[dict[str, Any]]) -> str:
        if not papers:
            return "No papers were retrieved."

        rendered = []
        for idx, paper in enumerate(papers, start=1):
            title = str(paper.get("title") or "Untitled")
            year = str(paper.get("year") or "n.d.")
            venue = str(paper.get("journal") or "Unknown venue")
            authors = paper.get("authors") or []
            if isinstance(authors, str):
                authors_text = authors
            else:
                authors_text = ", ".join(str(author) for author in authors[:6])
            snippets = paper.get("support_snippets") or []
            if not snippets and paper.get("full_text_excerpt"):
                snippets = [str(paper["full_text_excerpt"])]
            if not snippets and paper.get("abstract"):
                snippets = [str(paper["abstract"])]
            snippet_block = "\n".join(
                f"  - {snippet.strip()}" for snippet in snippets if snippet.strip()
            )
            rendered.append(
                f"[{idx}] {title} ({year})\n"
                f"Authors: {authors_text}\n"
                f"Venue: {venue}\n"
                f"Snippets:\n{snippet_block or '  - No snippet available.'}"
            )
        return "\n\n".join(rendered)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _build_markdown_evidence(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "Untitled")
    year = str(paper.get("year") or "n.d.")
    abstract = str(paper.get("abstract") or "").strip()
    snippets = paper.get("support_snippets") or []
    snippet_text = "\n\n".join(
        snippet.strip() for snippet in snippets if snippet.strip()
    )
    evidence_body = (
        snippet_text or abstract or str(paper.get("pageContent") or "").strip()
    )
    evidence_body = evidence_body or "No abstract or body-text evidence available."
    return f"**{title}** ({year})\n\n{evidence_body}"


def _resolve_librarian_prompt_selection(
    query_planner: str,
    librarian_prompt_override: str | None,
) -> tuple[str, str | None, str]:
    """Map built-in planners or prompt-file names to agent constructor inputs."""
    selected = (query_planner or "default").strip()
    if not selected:
        selected = "default"
    if selected in BUILTIN_QUERY_PLANNERS:
        return selected, librarian_prompt_override, selected
    if librarian_prompt_override is not None:
        return "default", librarian_prompt_override, selected

    prompt_ref = Path(selected)
    if prompt_ref.is_absolute():
        prompt_path = prompt_ref
    elif prompt_ref.parent != Path("."):
        prompt_path = PROJECT_ROOT / prompt_ref
    else:
        prompt_name = selected if prompt_ref.suffix == ".md" else f"{selected}.md"
        prompt_path = LITERATURE_PROMPTS_DIR / prompt_name

    resolved_prompt_path = prompt_path.resolve()
    resolved_prompts_dir = LITERATURE_PROMPTS_DIR.resolve()
    try:
        resolved_prompt_path.relative_to(resolved_prompts_dir)
    except ValueError as exc:
        raise ValueError(
            f"Custom librarian prompts must live under {LITERATURE_PROMPTS_DIR}."
        ) from exc
    if not resolved_prompt_path.is_file():
        available = ", ".join(
            sorted(path.stem for path in LITERATURE_PROMPTS_DIR.glob("*.md"))
        )
        raise FileNotFoundError(
            f"Unknown librarian prompt '{selected}'. "
            f"Available prompt names: {available}"
        )
    return "default", resolved_prompt_path.read_text(), resolved_prompt_path.stem


# Assemble the BioAgentExecutionConfig for config_name, resolving the librarian
# prompt selection once and threading every knob straight through.
def _build_execution_config(
    config_name: ConfigName,
    europepmc_librarian_prompt: str | None = None,
    query_planner: str = "default",
    simple_bm25_max_queries: int = 7,
    epmc_full_interface_max_queries: int = 7,
    max_query_count_override: int | None = None,
    librarian_num_subqueries_override: int | None = None,
    librarian_paragraphs_per_subquery_override: int | None = None,
    librarian_paragraphs_per_judge_batch_override: int | None = None,
    retrieval_policy: Literal[
        "synthesis",
        "localized_evidence",
        "localized_evidence_bm25",
        "localized_evidence_bm25_per_paper",
        "cascade_bm25",
        "upfront_fulltext",
        "upfront_fulltext_bm25_filter",
        "simple_fulltext_bm25",
    ] = "synthesis",
) -> BioAgentExecutionConfig:
    agent_query_planner, prompt_override, prompt_label = (
        _resolve_librarian_prompt_selection(
            query_planner=query_planner,
            librarian_prompt_override=europepmc_librarian_prompt,
        )
    )
    if config_name == "custom_tooling":
        return BioAgentExecutionConfig(
            name=config_name,
            search_mode="europepmc",
            europepmc_search_fn=search_scientific_literature_structured,
            europepmc_librarian_prompt=prompt_override,
            query_planner=agent_query_planner,
            librarian_prompt_label=prompt_label,
            simple_bm25_max_queries=simple_bm25_max_queries,
            epmc_full_interface_max_queries=epmc_full_interface_max_queries,
            max_query_count_override=max_query_count_override,
            librarian_num_subqueries_override=librarian_num_subqueries_override,
            librarian_paragraphs_per_subquery_override=librarian_paragraphs_per_subquery_override,
            librarian_paragraphs_per_judge_batch_override=librarian_paragraphs_per_judge_batch_override,
            retrieval_policy=retrieval_policy,
        )
    if config_name == "standard_tooling":
        return BioAgentExecutionConfig(
            name=config_name,
            search_mode="asta_corpus",
            europepmc_librarian_prompt=prompt_override,
            query_planner=agent_query_planner,
            librarian_prompt_label=prompt_label,
            simple_bm25_max_queries=simple_bm25_max_queries,
            epmc_full_interface_max_queries=epmc_full_interface_max_queries,
            max_query_count_override=max_query_count_override,
            librarian_num_subqueries_override=librarian_num_subqueries_override,
            librarian_paragraphs_per_subquery_override=librarian_paragraphs_per_subquery_override,
            librarian_paragraphs_per_judge_batch_override=librarian_paragraphs_per_judge_batch_override,
            retrieval_policy=retrieval_policy,
        )
    # llm_only: retrieval is skipped at the task-type level; config just needs
    # a valid search_mode so the dataclass doesn't raise.
    if config_name == "llm_only":
        return BioAgentExecutionConfig(
            name=config_name,
            search_mode="europepmc",
            europepmc_librarian_prompt=prompt_override,
            query_planner=agent_query_planner,
            librarian_prompt_label=prompt_label,
            simple_bm25_max_queries=simple_bm25_max_queries,
            epmc_full_interface_max_queries=epmc_full_interface_max_queries,
            max_query_count_override=max_query_count_override,
            librarian_num_subqueries_override=librarian_num_subqueries_override,
            librarian_paragraphs_per_subquery_override=librarian_paragraphs_per_subquery_override,
            librarian_paragraphs_per_judge_batch_override=librarian_paragraphs_per_judge_batch_override,
            retrieval_policy=retrieval_policy,
        )
    raise ValueError(f"Unknown config_name '{config_name}'.")


def _create_solver_tools(tool_options: dict[str, Any]) -> list[Tool]:
    """Accept known solver flags without coupling this wrapper to AstaBench internals."""

    normalized_options = {
        key: value for key, value in tool_options.items() if value not in (None, False)
    }
    if not normalized_options:
        return []
    unsupported_flags = ", ".join(sorted(normalized_options))
    raise ValueError(
        "This self-contained librarian wrapper does not create extra solver "
        f"tools from AstaBench flags. Unsupported options: {unsupported_flags}."
    )


@solver
def bio_agent_solver(
    config_name: ConfigName = "custom_tooling",
    task_type: TaskKind = "auto",
    reasoning_effort: str = "low",
    verbose: bool = False,
    llm_base_url: str | None = None,
    llm_model_name: str | None = None,
    full_text_enrichment: bool = True,
    europepmc_librarian_prompt: str | None = None,
    query_planner: str = "default",
    simple_bm25_max_queries: int = 7,
    epmc_full_interface_max_queries: int = 7,
    max_query_count_override: int | None = None,
    librarian_num_subqueries_override: int | None = None,
    librarian_paragraphs_per_subquery_override: int | None = None,
    librarian_paragraphs_per_judge_batch_override: int | None = None,
    retrieval_policy: Literal[
        "synthesis",
        "localized_evidence",
        "localized_evidence_bm25",
        "localized_evidence_bm25_per_paper",
        "cascade_bm25",
        "upfront_fulltext",
        "upfront_fulltext_bm25_filter",
        "simple_fulltext_bm25",
    ] = "synthesis",
    **tool_options: Any,
) -> Solver:
    """Generic librarian solver for AstaBench literature tasks.

    Most parameters configure the task/routing/model selection and are exposed
    as the ``--bio-agent-*`` CLI flags in ``run_evals.py``. Documented below
    are the LibrarianAgent-path-only ablation overrides, applied as a
    ``dataclasses.replace()`` on the ``librarian/config.toml`` in
    ``_run_librarian_agent`` — they have no effect on the legacy
    ``_run_literature_agent`` path.

    :param librarian_num_subqueries_override: Override for
        ``LibrarianRuntimeConfig.num_subqueries``; ``None`` leaves the
        ``librarian/config.toml`` value unchanged.
    :type librarian_num_subqueries_override: int or None
    :param librarian_paragraphs_per_subquery_override: Override for
        ``LibrarianRuntimeConfig.paragraphs_per_subquery``.
    :type librarian_paragraphs_per_subquery_override: int or None
    :param librarian_paragraphs_per_judge_batch_override: Override for
        ``LibrarianRuntimeConfig.paragraphs_per_judge_batch``.
    :type librarian_paragraphs_per_judge_batch_override: int or None
    :return: The configured Inspect solver.
    :rtype: Solver
    """

    solver_tools = _create_solver_tools(tool_options)
    merge_solver = merge_tools_with_state(solver_tools)
    execution_config = _build_execution_config(
        config_name=config_name,
        europepmc_librarian_prompt=europepmc_librarian_prompt,
        query_planner=query_planner,
        simple_bm25_max_queries=simple_bm25_max_queries,
        epmc_full_interface_max_queries=epmc_full_interface_max_queries,
        max_query_count_override=max_query_count_override,
        librarian_num_subqueries_override=librarian_num_subqueries_override,
        librarian_paragraphs_per_subquery_override=librarian_paragraphs_per_subquery_override,
        librarian_paragraphs_per_judge_batch_override=librarian_paragraphs_per_judge_batch_override,
        retrieval_policy=retrieval_policy,
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await merge_solver(state, generate)
        wrapper = BioAgentAstaWrapper(
            config=execution_config,
            task_type=task_type,
            llm_base_url=llm_base_url,
            llm_model_name=llm_model_name,
            full_text_enrichment=full_text_enrichment,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
        )
        return await wrapper.solve(state)

    return solve


@solver
def bio_agent_paper_finder(
    config_name: ConfigName = "custom_tooling",
    **kwargs: Any,
) -> Solver:
    return bio_agent_solver(
        config_name=config_name,
        task_type="paper_finder",
        **kwargs,
    )


@solver
def bio_agent_litqa2(
    config_name: ConfigName = "custom_tooling",
    **kwargs: Any,
) -> Solver:
    return bio_agent_solver(
        config_name=config_name,
        task_type="litqa2",
        **kwargs,
    )


@solver
def bio_agent_litqa2_open(
    config_name: ConfigName = "custom_tooling",
    **kwargs: Any,
) -> Solver:
    return bio_agent_solver(
        config_name=config_name,
        task_type="litqa2_open",
        **kwargs,
    )


@solver
def bio_agent_litqa2_open_llm_only(
    **kwargs: Any,
) -> Solver:
    """LLM-only baseline for LitQA2-FullText-OpenJudge (no retrieval)."""
    return bio_agent_solver(
        config_name="llm_only",
        task_type="litqa2_open_llm_only",
        **kwargs,
    )


@solver
def bio_agent_sqa(
    config_name: ConfigName = "custom_tooling",
    **kwargs: Any,
) -> Solver:
    return bio_agent_solver(
        config_name=config_name,
        task_type="sqa",
        **kwargs,
    )


@solver
def bio_agent_arxivdigestables(
    config_name: ConfigName = "custom_tooling",
    **kwargs: Any,
) -> Solver:
    return bio_agent_solver(
        config_name=config_name,
        task_type="arxivdigestables",
        **kwargs,
    )


__all__ = [
    "ASTA_LIBRARIAN_SYSTEM_PROMPT",
    "BioAgentAstaWrapper",
    "bio_agent_solver",
    "bio_agent_paper_finder",
    "bio_agent_litqa2",
    "bio_agent_litqa2_open",
    "bio_agent_sqa",
    "bio_agent_arxivdigestables",
]

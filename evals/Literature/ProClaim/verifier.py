"""ProClaim-eval harness for the librarian retrieval workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.etree import ElementTree as ET

import requests

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
SUBSETS = ("signor", "connectomedb")
DEFAULT_FALLBACK_LABEL = "UNCERTAIN"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
LOGGER = logging.getLogger("bioagents.eval.proclaim")
MAX_SYNTHESIS_EVIDENCE_PAPERS = 20
MAX_SYNTHESIS_SNIPPETS_PER_PAPER = 5
MAX_SYNTHESIS_SNIPPET_CHARS = 1500


INFERENCE_PROMPT_TEMPLATE = """You are evaluating a biomedical scientific claim using evidence from the scientific literature.

Claim:
{claim}

Retrieve and assess relevant scientific evidence. Determine the consensus verdict using the following labels:

SUPPORT:
The retrieved evidence directly corroborates the claim. The evidence is sufficient to conclude that the claim is true or highly likely true.

REFUTE:
The retrieved evidence directly contradicts the claim, or a sufficiently thorough search finds no evidence that substantiates the claim.

UNCERTAIN:
Relevant evidence exists, but it is ambiguous, or conflicting, so neither SUPPORT nor REFUTE is warranted.

Return JSON only with this schema:
{{
  "verdict": "SUPPORT" | "REFUTE" | "UNCERTAIN",
  "reasoning": "...",
  "citations": [
    {{
      "title": "...",
      "url": "...",
      "pmid": "...",
      "doi": "...",
      "evidence": "short quoted or paraphrased evidence"
    }}
  ]
}}"""


@dataclass(frozen=True)
class ProClaimExample:
    """One ProClaim-eval claim with only the fields needed by this harness."""

    id: str
    subset: str
    claim: str
    gold_label: str


@dataclass
class EvalConfig:
    """Runtime knobs for ProClaim inference."""

    llm_base_url: str | None = None
    llm_model_name: str | None = None
    thinking: bool = False
    retriever: str = "europepmc"
    es_url: str = "http://elasticsearch:9200"
    elastic_source: str = "abstracts"
    full_text_enrichment: bool = False
    fallback_label: str = DEFAULT_FALLBACK_LABEL
    cache_enabled: bool = False
    cache_path: Path | None = None
    verbose: bool = False
    max_retries: int = 1
    progress: bool = True
    no_agent: bool = False
    librarian_agent: bool = False
    # Retrieve through the orchestrator instead of an in-process LibrarianAgent.
    # Only affects retrieval: the verdict LLM is unchanged either way.
    via_api: bool = False
    web_search: bool = False
    web_search_tool_choice: str = "required"
    web_search_context_size: str = "medium"
    # PubMed+S2 abstract retrieval modality. When set, retrieval swaps the
    # librarian for the ProClaim evidence-programmer primitives: top-k
    # relevance-sorted PubMed abstracts plus top-k Semantic Scholar abstracts,
    # then one verdict-synthesis LLM call (no iteration, no full text).
    pubmed_s2: bool = False
    pubmed_s2_top_k: int = 5
    # Verdict model for --librarian-agent mode. When set, the SUPPORT/REFUTE/
    # UNCERTAIN classification runs on this model instead of the librarian's
    # retrieval model, so retrieval and verdict can use different LLMs (e.g.
    # glm-5 librarian + claude-sonnet-4-6 verdict). Falls back to the librarian
    # llm_base_url/llm_model_name when unset. verdict_api_key is required for
    # remote providers (e.g. ANTHROPIC_API_KEY for Anthropic's OpenAI-compatible
    # endpoint https://api.anthropic.com/v1); self-hosted vLLM ignores it.
    verdict_base_url: str | None = None
    verdict_model_name: str | None = None
    verdict_api_key: str | None = None
    bioagent_runner: (
        Callable[[str], dict[str, Any] | str | "ProClaimPrediction"] | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass
class ProClaimPrediction:
    """Structured prediction emitted by the librarian adapter."""

    predicted_label: str
    reasoning: str
    citations: list[dict[str, Any]]
    raw_output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parse_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "reasoning": self.reasoning,
            "citations": self.citations,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
            "parse_failure": self.parse_failure,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProClaimPrediction":
        return cls(
            predicted_label=normalize_label(
                payload.get("predicted_label"),
                fallback=DEFAULT_FALLBACK_LABEL,
            ),
            reasoning=str(payload.get("reasoning") or ""),
            citations=_normalize_citations(payload.get("citations") or []),
            raw_output=str(payload.get("raw_output") or ""),
            metadata=dict(payload.get("metadata") or {}),
            parse_failure=bool(payload.get("parse_failure", False)),
        )


@dataclass(frozen=True)
class ParsedOutput:
    label: str
    reasoning: str
    citations: list[dict[str, Any]]
    parse_failure: bool


def build_inference_prompt(claim: str) -> str:
    """Build the only model-facing input for a benchmark example."""
    return INFERENCE_PROMPT_TEMPLATE.format(claim=claim.strip())


def normalize_label(
    value: Any,
    fallback: str | None = DEFAULT_FALLBACK_LABEL,
) -> str | None:
    """Map raw labels and common textual variants to canonical ProClaim labels."""
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    normalized = re.sub(r"[\s_-]+", " ", text.upper()).strip()
    direct_map = {
        "SUPPORT": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "SUPPORTING": "SUPPORT",
        "CORROBORATED": "SUPPORT",
        "CORROBORATES": "SUPPORT",
        "TRUE": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTED": "REFUTE",
        "REFUTES": "REFUTE",
        "REFUTING": "REFUTE",
        "CONTRADICT": "REFUTE",
        "CONTRADICTED": "REFUTE",
        "CONTRADICTS": "REFUTE",
        "CONTRADICTORY": "REFUTE",
        "WRONG": "REFUTE",
        "FALSE": "REFUTE",
        "UNSUPPORTED": "REFUTE",
        "UNCERTAIN": "UNCERTAIN",
        "INCONCLUSIVE": "UNCERTAIN",
        "AMBIGUOUS": "UNCERTAIN",
        "CONFLICTING": "UNCERTAIN",
        "NEI": "UNCERTAIN",
        "NOT ENOUGH INFORMATION": "UNCERTAIN",
        "INSUFFICIENT EVIDENCE": "UNCERTAIN",
    }
    if normalized in direct_map:
        return direct_map[normalized]

    lowered = text.lower()
    phrase_patterns = [
        (
            r"\b(insufficient evidence|not enough evidence|not enough information|nei)\b",
            "UNCERTAIN",
        ),
        (
            r"\b(inconclusive|unclear|ambiguous|mixed evidence|conflicting evidence)\b",
            "UNCERTAIN",
        ),
        (r"\b(no direct evidence|limited evidence|cannot determine)\b", "UNCERTAIN"),
        (
            r"\b(no evidence that substantiates|no evidence supports|not supported|unsupported)\b",
            "REFUTE",
        ),
        (r"\b(refute[sd]?|contradict(?:s|ed|ory)?|falsified|false)\b", "REFUTE"),
        (
            r"\b(support(?:s|ed|ing)?|corroborat(?:e|es|ed)|substantiated)\b",
            "SUPPORT",
        ),
    ]
    for pattern, label in phrase_patterns:
        if re.search(pattern, lowered):
            return label

    return fallback


def parse_bioagent_output(
    raw_output: Any,
    *,
    fallback_label: str = DEFAULT_FALLBACK_LABEL,
) -> ParsedOutput:
    """Parse model output into a verdict, recording failures when no label is found."""
    fallback = normalize_label(fallback_label, fallback=DEFAULT_FALLBACK_LABEL)
    raw_text = "" if raw_output is None else str(raw_output)
    payload = _extract_json_object(raw_text)

    if isinstance(payload, dict):
        raw_label = (
            payload.get("verdict")
            or payload.get("label")
            or payload.get("predicted_label")
            or payload.get("answer")
            or payload.get("classification")
        )
        label = normalize_label(raw_label, fallback=None)
        reasoning = str(payload.get("reasoning") or payload.get("rationale") or "")
        citations = _normalize_citations(
            payload.get("citations") or payload.get("evidence") or []
        )
        if label is not None:
            return ParsedOutput(
                label=label,
                reasoning=reasoning or raw_text.strip(),
                citations=citations,
                parse_failure=False,
            )

    prose_label = _extract_label_from_prose(raw_text)
    if prose_label is not None:
        return ParsedOutput(
            label=prose_label,
            reasoning=raw_text.strip(),
            citations=[],
            parse_failure=False,
        )

    return ParsedOutput(
        label=fallback or DEFAULT_FALLBACK_LABEL,
        reasoning=raw_text.strip(),
        citations=[],
        parse_failure=True,
    )


def load_proclaim_examples(
    proclaim_path: Path | str,
    *,
    subset: str = "all",
    max_examples: int | None = None,
) -> list[ProClaimExample]:
    """Load SIGNOR-Fact, ConnectomeDB-Fact, or both from local CSV files."""
    subset = _normalize_subset(subset)
    root = Path(proclaim_path)
    csv_paths = _resolve_dataset_paths(root)

    selected_subsets = SUBSETS if subset == "all" else (subset,)
    examples_by_subset: list[list[ProClaimExample]] = []
    for subset_name in selected_subsets:
        csv_path = csv_paths[subset_name]
        examples_by_subset.append(_load_subset_csv(csv_path, subset_name))

    if subset == "all":
        examples = _interleave_examples(examples_by_subset)
    else:
        examples = examples_by_subset[0]

    if max_examples is not None:
        examples = examples[: max(0, max_examples)]
    return examples


def predict_proclaim_verdict(
    claim: str,
    *,
    config: EvalConfig,
    bioagent_runner: (
        Callable[[str], dict[str, Any] | str | ProClaimPrediction] | None
    ) = None,
) -> ProClaimPrediction:
    """Run the librarian on a claim and map its answer to a ProClaim verdict."""
    prompt = build_inference_prompt(claim)
    cache_key = _cache_key(claim, config)

    if config.cache_enabled and config.cache_path is not None:
        cached = _read_prediction_cache(config.cache_path).get(cache_key)
        if isinstance(cached, dict):
            prediction = ProClaimPrediction.from_dict(cached)
            prediction.metadata["cache_hit"] = True
            return prediction

    started = time.perf_counter()
    last_error = ""
    for attempt in range(1, max(1, config.max_retries) + 1):
        try:
            runner_output = _run_bioagent(prompt, config, bioagent_runner)
            if isinstance(runner_output, ProClaimPrediction):
                prediction = runner_output
            else:
                prediction = _prediction_from_bioagent_result(
                    runner_output,
                    config=config,
                    started=started,
                )
            prediction.metadata.setdefault("attempts", attempt)
            prediction.metadata.setdefault("cache_hit", False)
            if config.cache_enabled and config.cache_path is not None:
                _write_prediction_cache(config.cache_path, cache_key, prediction)
            return prediction
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
            LOGGER.exception("Prediction attempt %s failed.", attempt)

    latency = time.perf_counter() - started
    return ProClaimPrediction(
        predicted_label=(
            normalize_label(config.fallback_label) or DEFAULT_FALLBACK_LABEL
        ),
        reasoning=f"ERROR: {last_error or 'Librarian inference failed.'}",
        citations=[],
        raw_output="",
        metadata={
            "latency_seconds": round(latency, 3),
            "error": last_error or "Librarian inference failed.",
            "attempts": max(1, config.max_retries),
            "model_name": config.llm_model_name or os.getenv("LLM_MODEL", ""),
        },
        parse_failure=True,
    )


def run_proclaim_evaluation(
    examples: Iterable[ProClaimExample],
    *,
    config: EvalConfig,
    out_dir: Path | str,
    resume: bool = False,
    predict_fn: Callable[[str, EvalConfig], ProClaimPrediction] | None = None,
) -> dict[str, Any]:
    """Evaluate examples, stream predictions, and write all ProClaim artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    predictions_path = out_path / "predictions.jsonl"
    metrics_path = out_path / "metrics.json"
    confusion_path = out_path / "confusion_matrix.csv"
    summary_path = out_path / "summary.md"
    latex_path = out_path / "latex_table.tex"
    run_config_path = out_path / "run_config.json"

    ordered_examples = list(examples)
    existing_rows = _load_prediction_rows(predictions_path) if resume else {}
    selected_ids = {example.id for example in ordered_examples}
    prediction_rows_by_id = {
        key: row for key, row in existing_rows.items() if key in selected_ids
    }
    reusable_bioagent_runner = None
    if predict_fn is None and config.bioagent_runner is None:
        reusable_bioagent_runner = ReusableBioAgentRunner(config)

    predictor = predict_fn or (
        lambda claim, cfg: predict_proclaim_verdict(
            claim,
            config=cfg,
            bioagent_runner=reusable_bioagent_runner,
        )
    )

    if not resume:
        predictions_path.write_text("", encoding="utf-8")

    total = len(ordered_examples)
    progress_bar = _make_progress_bar(total, config.progress)
    try:
        for index, example in enumerate(ordered_examples, start=1):
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"{example.subset}:{example.id}",
                    refresh=False,
                )

            if resume and example.id in prediction_rows_by_id:
                if progress_bar is None:
                    LOGGER.info(
                        "[%d/%d] %s skipped by resume.",
                        index,
                        total,
                        example.id,
                    )
                else:
                    LOGGER.debug(
                        "[%d/%d] %s skipped by resume.",
                        index,
                        total,
                        example.id,
                    )
                _update_progress(progress_bar)
                continue

            if progress_bar is None:
                LOGGER.info("[%d/%d] %s (%s)", index, total, example.id, example.subset)
            else:
                LOGGER.debug(
                    "[%d/%d] %s (%s)", index, total, example.id, example.subset
                )
            prediction = predictor(example.claim, config)
            row = _prediction_row(example, prediction)
            prediction_rows_by_id[example.id] = row
            _append_jsonl(predictions_path, row)
            _update_progress(progress_bar)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    ordered_rows = [
        prediction_rows_by_id[example.id]
        for example in ordered_examples
        if example.id in prediction_rows_by_id
    ]
    _write_jsonl(predictions_path, ordered_rows)

    metrics = compute_metrics_by_subset(ordered_rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion_path.write_text(render_confusion_matrix_csv(metrics), encoding="utf-8")
    summary_path.write_text(render_summary_markdown(metrics), encoding="utf-8")
    latex_path.write_text(render_latex_table(metrics), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(_serializable_config(config), indent=2),
        encoding="utf-8",
    )

    return {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_path),
        "summary_path": str(summary_path),
        "latex_table_path": str(latex_path),
        "metrics": metrics,
    }


def _make_progress_bar(total: int, enabled: bool):
    if not enabled or tqdm is None:
        return None
    return tqdm(
        total=total,
        desc="ProClaim",
        unit="claim",
        dynamic_ncols=True,
    )


def _update_progress(progress_bar: Any) -> None:
    if progress_bar is not None:
        progress_bar.update(1)


def compute_metrics_by_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics for both subsets and the combined benchmark."""
    metrics: dict[str, Any] = {}
    for subset in SUBSETS:
        subset_rows = [row for row in rows if row.get("subset") == subset]
        metrics[subset] = compute_metrics(subset_rows)
    metrics["combined"] = compute_metrics(rows)
    return metrics


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute AGR, macro FPR/FNR, confusion matrix, and per-class PRF."""
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    parse_failures = 0
    latencies: list[float] = []
    costs: list[float] = []
    retrieved_counts: list[float] = []
    iteration_counts: list[float] = []

    for row in rows:
        gold = normalize_label(row.get("gold_label")) or DEFAULT_FALLBACK_LABEL
        pred = normalize_label(row.get("predicted_label")) or DEFAULT_FALLBACK_LABEL
        confusion[gold][pred] += 1
        if row.get("parse_failure") or (row.get("metadata") or {}).get("parse_failure"):
            parse_failures += 1
        metadata = row.get("metadata") or {}
        _append_float(latencies, metadata.get("latency_seconds"))
        _append_float(costs, metadata.get("cost_usd"))
        _append_float(retrieved_counts, metadata.get("retrieved_paper_count"))
        _append_float(iteration_counts, metadata.get("iterations"))

    total = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    per_class: dict[str, dict[str, Any]] = {}
    macro_fpr_values: list[float] = []
    macro_fnr_values: list[float] = []

    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        tn = total - tp - fp - fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        macro_fpr_values.append(fpr)
        macro_fnr_values.append(fnr)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": sum(confusion[label].values()),
        }

    return {
        "n": total,
        "agr": round(_safe_div(correct, total), 4),
        "accuracy": round(_safe_div(correct, total), 4),
        "macro_fpr": round(sum(macro_fpr_values) / len(LABELS), 4),
        "macro_fnr": round(sum(macro_fnr_values) / len(LABELS), 4),
        "parse_failures": parse_failures,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4)
        if latencies
        else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "avg_retrieved_papers": round(sum(retrieved_counts) / len(retrieved_counts), 4)
        if retrieved_counts
        else None,
        "avg_iterations": round(sum(iteration_counts) / len(iteration_counts), 4)
        if iteration_counts
        else None,
    }


def render_confusion_matrix_csv(metrics: dict[str, Any]) -> str:
    rows = [["subset", "gold_label", "predicted_label", "count"]]
    for subset, subset_metrics in metrics.items():
        matrix = subset_metrics.get("confusion_matrix", {})
        for gold in LABELS:
            for pred in LABELS:
                rows.append(
                    [subset, gold, pred, str(matrix.get(gold, {}).get(pred, 0))]
                )
    return "\n".join(",".join(row) for row in rows) + "\n"


def render_summary_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# ProClaim-eval Librarian Summary",
        "",
        "| Subset | n | AGR | Macro FPR | Macro FNR | Mean iters | Parse failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ("signor", "connectomedb", "combined"):
        m = metrics.get(subset, {})
        lines.append(
            f"| {subset} | {m.get('n', 0)} | {_metric_text(m, 'agr')} | "
            f"{_metric_text(m, 'macro_fpr')} | {_metric_text(m, 'macro_fnr')} | "
            f"{_metric_text(m, 'avg_iterations')} | {m.get('parse_failures', 0)} |"
        )
    lines.extend(
        [
            "",
            "AGR is prediction-label agreement, equivalent to accuracy. Macro FPR and "
            "macro FNR average one-vs-rest false positive and false negative rates "
            "over SUPPORT, REFUTE, and UNCERTAIN.",
            "",
            "Leakage constraint: model inference receives only the natural-language "
            "claim and the verification instructions. Gold labels, curated evidence, "
            "PMIDs, entity fields, source database names, and provenance fields are "
            "used only by the harness after prediction.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex_table(metrics: dict[str, Any]) -> str:
    signor = metrics.get("signor", {})
    connectome = metrics.get("connectomedb", {})
    agr_row = (
        "Librarian (ours) & Literature-synthesis agent & "
        f"{_latex_metric(signor, 'agr')} & {_latex_metric(connectome, 'agr')} \\\\"
    )
    full_row = (
        "Librarian (ours) & Literature-synthesis agent\n"
        f"& {_latex_metric(signor, 'agr')} & {_latex_metric(signor, 'macro_fpr')} "
        f"& {_latex_metric(signor, 'macro_fnr')}\n"
        f"& {_latex_metric(connectome, 'agr')} "
        f"& {_latex_metric(connectome, 'macro_fpr')} "
        f"& {_latex_metric(connectome, 'macro_fnr')} \\\\"
    )
    return f"% AGR-only row\n{agr_row}\n\n% AGR / FPR / FNR row\n{full_row}\n"


class ReusableBioAgentRunner:
    """Lazily reuse one ``LibrarianAgent`` for a whole ProClaim run."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self._agent = None
        self._llm = None
        self._web_search_client = None

    def _retrieve_librarian_passages(self, claim: str) -> list:
        """Retrieve evidence for *claim*, in-process or through the orchestrator.

        Both return ``LibrarianAgent.run``'s per-paper records, so the caller and
        the prompt formatter are identical either way.
        """
        if self.config.via_api:
            return _retrieve_librarian_passages_via_api(claim)
        return self._agent.run(query=claim) or []

    def __call__(self, prompt: str) -> dict[str, Any] | str:
        if self.config.web_search:
            if self._web_search_client is None:
                self._web_search_client = ResponsesWebSearchClient(
                    base_url=self.config.llm_base_url,
                    model=self.config.llm_model_name,
                    tool_choice=self.config.web_search_tool_choice,
                    search_context_size=self.config.web_search_context_size,
                )
            return self._web_search_client.query(prompt)
        elif self.config.pubmed_s2:
            if self._llm is None:
                from evals.Literature.llm_compat import build_llm_client

                # Verdict model is independent of retrieval: reuse the verdict
                # overrides so the same verdict LLM can score librarian and
                # PubMed+S2 retrieval for a clean A/B comparison.
                self._llm = build_llm_client(
                    base_url=self.config.verdict_base_url or self.config.llm_base_url,
                    model_name=(
                        self.config.verdict_model_name or self.config.llm_model_name
                    ),
                    api_key=self.config.verdict_api_key,
                    reasoning_effort=self.config.thinking,
                )
            claim = _extract_claim_from_prompt(prompt)
            passages, search_query = _retrieve_pubmed_s2_passages(
                claim, top_k=self.config.pubmed_s2_top_k
            )
            result = _synthesize_proclaim_verdict_from_librarian_passages(
                claim=claim,
                passages=passages,
                llm_client=self._llm,
            )
            result["search_query"] = search_query
            return result
        elif self.config.librarian_agent:
            if self._agent is None and not self.config.via_api:
                self._agent = _build_librarian_agent(self.config)
            if self._llm is None:
                from evals.Literature.llm_compat import build_llm_client

                # Verdict model is independent of the librarian retrieval model:
                # fall back to the librarian llm settings when not overridden.
                self._llm = build_llm_client(
                    base_url=self.config.verdict_base_url or self.config.llm_base_url,
                    model_name=(
                        self.config.verdict_model_name or self.config.llm_model_name
                    ),
                    api_key=self.config.verdict_api_key,
                    reasoning_effort=self.config.thinking,
                )
            claim = _extract_claim_from_prompt(prompt)
            passages = self._retrieve_librarian_passages(claim)
            return _synthesize_proclaim_verdict_from_librarian_passages(
                claim=claim,
                passages=passages,
                llm_client=self._llm,
            )
        elif self.config.no_agent:
            if self._llm is None:
                from evals.Literature.llm_compat import build_llm_client

                self._llm = build_llm_client(
                    base_url=self.config.llm_base_url,
                    model_name=self.config.llm_model_name,
                    reasoning_effort=self.config.thinking,
                )
            return self._llm.generate_structured_output(
                prompt,
                system_message="You are evaluating a biomedical scientific claim.",
            )
        else:
            if self._agent is None:
                self._agent = _build_literature_agent(self.config)
            if self._llm is None:
                from evals.Literature.llm_compat import build_llm_client

                self._llm = build_llm_client(
                    base_url=self.config.llm_base_url,
                    model_name=self.config.llm_model_name,
                    reasoning_effort=self.config.thinking,
                )

            claim = _extract_claim_from_prompt(prompt)
            retrieval_query = (
                "Find scientific evidence that supports or refutes this "
                f"biomedical claim: {claim}"
            )
            result = self._agent.run(
                retrieval_query,
                additional_context=_proclaim_retrieval_context(claim),
                progress_callback=None,
                conversation_history=None,
                output_channel="generic",
                include_summary=False,
            )
            return _synthesize_proclaim_verdict_from_evidence(
                claim=claim,
                retrieval=result,
                llm_client=self._llm,
            )


def _extract_claim_from_prompt(prompt: str) -> str:
    """Extract the benchmark claim from the fixed ProClaim inference prompt."""
    match = re.search(
        r"(?s)\bClaim:\s*(.*?)\n\s*Retrieve and assess relevant scientific evidence",
        prompt,
    )
    if match:
        claim = " ".join(match.group(1).split())
        if claim:
            return claim
    return " ".join(str(prompt).split())


def _proclaim_retrieval_context(claim: str) -> str:
    return (
        "ProClaim retrieval mode: retrieve primary biomedical literature that can "
        "support or refute the exact claim. Preserve all named entities, species, "
        "directionality, interaction type, ligand/receptor or protein relation, "
        "and causal wording from the claim. Include search angles for direct "
        "support, direct contradiction, and closely related evidence that may make "
        "the verdict uncertain. Do not broaden to generic pathway background unless "
        "direct evidence is unavailable.\n\n"
        f"Exact claim:\n{claim}"
    )


def _synthesize_proclaim_verdict_from_evidence(
    *,
    claim: str,
    retrieval: dict[str, Any],
    llm_client: Any,
) -> dict[str, Any]:
    """Build the ProClaim verdict from retrieved evidence, not agent summary."""
    evidence_text = _format_proclaim_evidence_for_prompt(retrieval)
    if not evidence_text:
        raw_output = json.dumps(
            {
                "verdict": "REFUTE",
                "reasoning": (
                    "The literature search did not retrieve evidence that "
                    "substantiates the claim."
                ),
                "citations": [],
            },
            ensure_ascii=False,
        )
    else:
        raw_output = llm_client.chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a biomedical claim-verification expert. Use only "
                        "the provided evidence snippets and return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": _proclaim_verdict_prompt(claim, evidence_text),
                },
            ],
            temperature=0,
            max_tokens=900,
        )

    return {
        "summary": "",
        "raw_output": raw_output,
        "inference_mode": "bio_agent_evidence_synthesis",
        "evidence_prompt_chars": len(evidence_text),
        "evidence": list((retrieval or {}).get("evidence") or []),
        "search_query": (retrieval or {}).get("search_query", ""),
        "retrieved_paper_count": len((retrieval or {}).get("papers_raw") or []),
    }


def _proclaim_verdict_prompt(claim: str, evidence_text: str) -> str:
    return (
        "Evaluate the biomedical claim using only the retrieved evidence snippets.\n\n"
        "Verdict labels:\n"
        "- SUPPORT: the evidence directly corroborates the exact claim, including "
        "the named entities and direction or effect.\n"
        "- REFUTE: the evidence directly contradicts the claim or supports an "
        "incompatible relation or direction.\n"
        "- UNCERTAIN: the evidence is absent, ambiguous, incomplete, indirect, or "
        "conflicting — including when retrieved snippets discuss related biology "
        "but do not address the specific claim.\n\n"
        "Rules:\n"
        "- Do not use outside biomedical knowledge.\n"
        "- Keep reasoning concise and evidence-grounded.\n"
        "- Cite only evidence blocks provided below.\n"
        "- If snippets consistently describe a relation (e.g. protein A stimulates "
        "pathway B which includes protein C), treat that as indirect SUPPORT only "
        "if the chain is explicit and unambiguous in the text.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "verdict": "SUPPORT" | "REFUTE" | "UNCERTAIN",\n'
        '  "reasoning": "one to three concise sentences",\n'
        '  "citations": [\n'
        "    {\n"
        '      "title": "...",\n'
        '      "url": "...",\n'
        '      "pmid": "...",\n'
        '      "doi": "...",\n'
        '      "evidence": "short evidence snippet copied or paraphrased from one block"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Use empty strings for citation fields that are not present in the "
        "evidence block. Do not infer missing title, URL, PMID, or DOI values.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "JSON:"
    )


def _format_proclaim_evidence_for_prompt(
    retrieval: dict[str, Any],
    limit: int = MAX_SYNTHESIS_EVIDENCE_PAPERS,
) -> str:
    evidence_entries = [
        entry
        for entry in list(retrieval.get("evidence") or [])
        if isinstance(entry, dict)
    ]
    if not evidence_entries:
        return ""

    rendered: list[str] = []
    for entry in evidence_entries:
        if len(rendered) >= limit:
            break
        block = _render_proclaim_evidence_block(
            rank=len(rendered) + 1,
            evidence_entry=entry,
        )
        if block:
            rendered.append(block)

    return "\n\n".join(rendered)


def _render_proclaim_evidence_block(
    *,
    rank: int,
    evidence_entry: dict[str, Any],
) -> str:
    snippets = _dedupe_texts(_extract_evidence_entry_texts(evidence_entry))
    snippets = [
        snippet
        for snippet in snippets
        if snippet
        and snippet != "|"
        and "No abstract or full-text excerpt" not in snippet
    ]
    if not snippets:
        return ""

    metadata_parts = []
    identifier_label = _identifier_label(evidence_entry)
    if identifier_label:
        metadata_parts.append(identifier_label)

    snippet_lines = []
    for snippet in snippets[:MAX_SYNTHESIS_SNIPPETS_PER_PAPER]:
        compact = _truncate_prompt_text(snippet, MAX_SYNTHESIS_SNIPPET_CHARS)
        if compact:
            snippet_lines.append(f"- {compact}")
    if not snippet_lines:
        return ""

    header = f"[{rank}]"
    if metadata_parts:
        header += " " + "; ".join(metadata_parts)
    return header + "\n" + "\n".join(snippet_lines)


def _extract_evidence_entry_texts(entry: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    _extend_texts(texts, entry.get("evidence"))
    return texts


def _extend_texts(target: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = " ".join(value.split())
        if text:
            target.append(text)
        return
    if isinstance(value, list):
        for item in value:
            _extend_texts(target, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _extend_texts(target, item)


def _dedupe_texts(texts: list[str]) -> list[str]:
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


def _truncate_prompt_text(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


class ResponsesWebSearchClient:
    """Minimal OpenAI Responses API client for ProClaim web-search baselines."""

    def __init__(
        self,
        base_url: str | None,
        model: str | None,
        tool_choice: str,
        search_context_size: str,
    ) -> None:
        if not model:
            raise ValueError("`llm_model_name` must be set for web search mode.")
        resolved_base = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        if resolved_base.endswith("/responses"):
            self._responses_url = resolved_base
        else:
            self._responses_url = f"{resolved_base}/responses"
        self._model = model
        self._tool_choice = tool_choice
        self._search_context_size = search_context_size
        self._api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    def query(self, prompt: str) -> dict[str, Any]:
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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.post(
            self._responses_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        raw_response = response.json()
        citations = _extract_response_url_citations(raw_response)
        return {
            "summary": _extract_response_output_text(raw_response),
            "citations": citations,
            "papers_raw": [],
            "inference_mode": "openai_web_search",
            "web_search_call_count": _count_response_web_search_calls(raw_response),
            "url_citation_count": len(citations),
        }


def _run_bioagent(
    prompt: str,
    config: EvalConfig,
    bioagent_runner: Callable[[str], dict[str, Any] | str | ProClaimPrediction]
    | None = None,
) -> dict[str, Any] | str | ProClaimPrediction:
    if bioagent_runner is not None:
        return bioagent_runner(prompt)
    if config.bioagent_runner is not None:
        return config.bioagent_runner(prompt)

    runner = ReusableBioAgentRunner(config)
    return runner(prompt)


def _build_literature_agent(config: EvalConfig):
    """The pre-librarian retrieval backend, which this repository does not ship.

    The paper's baseline rows used an internal literature agent (optionally over
    an Elasticsearch index) that the librarian replaced. Only the +Librarian
    rows are reproducible here, which is what ``run_proclaim.sh`` runs.
    """
    raise RuntimeError(
        "Only the librarian retrieval backend is available in this repository. "
        "Pass --librarian-agent (what ProClaim/run_proclaim.sh does); "
        f"got retriever={config.retriever!r}."
    )


def _build_librarian_agent(config: EvalConfig):
    """Construct the new LibrarianAgent (BM25-per-paper strategy) in-process."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    from librarian.agent import LibrarianAgent
    from librarian.config import load_runtime_config

    return LibrarianAgent(
        runtime_config=load_runtime_config(),
        llm_base_url=config.llm_base_url,
        llm_model_name=config.llm_model_name,
        full_text_enrichment=True,
        verbose=config.verbose,
    )


def _retrieve_librarian_passages_via_api(claim: str) -> list:
    """Retrieve librarian passages from the orchestrator instead of in-process.

    The API ships ``LibrarianAgent.run``'s records verbatim, so the caller sees
    the identical list either way. The pods run their deployed config, so
    ``--llm-base-url`` / ``--llm-model-name`` do not apply to retrieval here
    (the verdict model is unaffected -- it is a separate client).
    """
    import sys as _sys

    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))
    from evals.Literature import orchestrator_client

    return orchestrator_client.librarian_evidence(
        claim, source="proclaim-eval", session_prefix="proclaim"
    )


def _format_librarian_passages_for_prompt(passages: list) -> str:
    """Format LibrarianAgent passage dicts into numbered evidence blocks."""
    rendered: list[str] = []
    for passage in passages:
        text = " ".join(passage.get("evidence_snippets") or []).strip()
        if not text:
            continue
        pmid = str(passage.get("pmid") or "").strip()
        title = str(passage.get("title") or "").strip()
        header = f"[{len(rendered) + 1}]"
        if pmid:
            header += f" PMID: {pmid}"
        if title:
            header += f"; {title}"
        rendered.append(f"{header}\n- {text}")
    return "\n\n".join(rendered)


def _synthesize_proclaim_verdict_from_librarian_passages(
    *,
    claim: str,
    passages: list,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the ProClaim verdict from LibrarianAgent passages."""
    evidence_text = _format_librarian_passages_for_prompt(passages)
    if not evidence_text:
        raw_output = json.dumps(
            {
                "verdict": "UNCERTAIN",
                "reasoning": (
                    "The literature search did not retrieve any relevant evidence "
                    "for this claim."
                ),
                "citations": [],
            },
            ensure_ascii=False,
        )
    else:
        user_prompt = (
            "You are evaluating a biomedical scientific claim using evidence "
            "retrieved from the scientific literature.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            "Determine the consensus verdict using the following labels:\n\n"
            "SUPPORT:\n"
            "The retrieved evidence directly corroborates the claim. The evidence "
            "is sufficient to conclude that the claim is true or highly likely true.\n\n"
            "REFUTE:\n"
            "The retrieved evidence directly contradicts the claim, or a sufficiently "
            "thorough search finds no evidence that substantiates the claim.\n\n"
            "UNCERTAIN:\n"
            "Relevant evidence exists, but it is ambiguous, incomplete, "
            "context-dependent, or conflicting, so neither SUPPORT nor REFUTE "
            "is warranted.\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "verdict": "SUPPORT" | "REFUTE" | "UNCERTAIN",\n'
            '  "reasoning": "concise evidence-grounded explanation",\n'
            '  "citations": [\n'
            "    {\n"
            '      "title": "...",\n'
            '      "url": "...",\n'
            '      "pmid": "...",\n'
            '      "doi": "...",\n'
            '      "evidence": "short quoted or paraphrased evidence"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        raw_output = llm_client.chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a biomedical claim-verification expert. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=1500,
        )

    return {
        "summary": "",
        "raw_output": raw_output,
        "inference_mode": "bio_agent_evidence_synthesis",
        "evidence_prompt_chars": len(evidence_text),
        "evidence": [],
        "search_query": claim,
        "retrieved_paper_count": len(passages),
    }


# ---------------------------------------------------------------------------
# PubMed + Semantic Scholar abstract retrieval (ProClaim evidence-programmer
# primitives). This mirrors, without importing the ProClaim_src package:
#   * PubMed: relevance-sorted ("Best Match") search via NCBI E-utilities,
#     matching proclaim.search.custom_pubmed.RelevancePubMedSearcher
#     (usehistory=y + sort=relevance), with structured-abstract rendering as in
#     baselines.single_paper._fetch_abstract.
#   * Semantic Scholar: keyword graph-API search matching
#     proclaim.search.semantic_scholar.S2Client.search.
# Each source contributes its top-k abstracts; results are merged and
# deduplicated by PMID (PubMed wins ties). No full text, no iteration.
# ---------------------------------------------------------------------------

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Per-dataset query processors ported from baselines.retrieval_baseline.
_SIGNOR_CLAIM_RE = re.compile(r"^(?P<base>.*?)\s*\([^)]*\)\s*\.?$")
_CONNECTOMEDB_CLAIM_RE = re.compile(
    r"^(?P<ligand>\S+)\s+as\s+ligand\s+directly\s+interacts"
    r"(?:\s+extracellularly)?\s+with\s+(?P<receptor>\S+)\s+as\s+receptor\.?$"
)


def _proclaim_search_query(claim: str) -> str:
    """Turn a claim into the search query ProClaim would use.

    Replicates the per-dataset query processors from
    ``baselines.retrieval_baseline``: ConnectomeDB claims collapse to the two
    protein names, SIGNOR claims drop the trailing parenthetical, and anything
    else is searched verbatim. Dispatch is by claim pattern (not dataset name)
    so the subset-agnostic runner still matches ProClaim behaviour.
    """
    text = claim.strip()
    connectome = _CONNECTOMEDB_CLAIM_RE.match(text)
    if connectome:
        return (
            f"{connectome.group('ligand')} {connectome.group('receptor')} "
            "protein interaction"
        )
    signor = _SIGNOR_CLAIM_RE.match(text)
    if signor:
        return signor.group("base").strip()
    return text


# Biological verb → noun map and stop words for the PubMed boolean query,
# ported verbatim from proclaim.verification.evidence_api. PubMed's esearch
# ANDs every free-text token, so a natural-language claim ("GNAS directly
# activates ADCY1") returns 0 hits; ProClaim's evidence programmer instead
# formulates an entity-AND boolean query for PubMed (e.g. "GNAS AND ADCY1 AND
# (activation OR up-regulation)"), which is what _formulate_pubmed_query builds.
_BIO_VERB_TO_NOUN: dict[str, str] = {
    "activates": "activation",
    "inhibits": "inhibition",
    "phosphorylates": "phosphorylation",
    "binds": "binding",
    "regulates": "regulation",
    "suppresses": "suppression",
    "promotes": "promotion",
    "blocks": "blocking",
    "induces": "induction",
    "represses": "repression",
    "ubiquitinates": "ubiquitination",
    "methylates": "methylation",
    "acetylates": "acetylation",
    "deactivates": "deactivation",
    "up-regulates": "up-regulation",
    "down-regulates": "down-regulation",
    "upregulates": "up-regulation",
    "downregulates": "down-regulation",
}

_STOP_WORDS: set[str] = {
    "does",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "and",
    "or",
    "that",
    "this",
    "it",
    "by",
    "with",
    "from",
    "for",
    "on",
    "at",
    "be",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "not",
    "no",
    "its",
    "their",
    "our",
    "your",
    "directly",
    "indirectly",
    "via",
    "through",
}


def _formulate_pubmed_query(claim: str) -> str:
    """Convert a claim into a structured PubMed boolean query.

    Verbatim port of ``proclaim.verification.evidence_api.formulate_pubmed_query``:
    extract gene/protein symbols, map biological action verbs to noun forms,
    and combine as ``SYM1 AND SYM2 AND (noun OR noun)``.
    """
    symbols = re.findall(r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b", claim)

    claim_lower = claim.lower().rstrip(".")
    bio_terms: list[str] = []
    for verb, noun in _BIO_VERB_TO_NOUN.items():
        if verb in claim_lower:
            bio_terms.append(noun)

    if not symbols:
        words = re.findall(r"\b\w+\b", claim)
        symbols = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    parts: list[str] = []
    if symbols:
        parts.append(" AND ".join(symbols))
    if bio_terms:
        parts.append("(" + " OR ".join(bio_terms) + ")")

    return " AND ".join(parts) if parts else claim


def _search_pubmed_relevance(query: str, top_k: int) -> list[dict[str, Any]]:
    """Relevance-sorted PubMed search returning up to ``top_k`` abstracts."""
    api_key = os.environ.get("PUBMED_API_KEY")
    email = os.environ.get("PUBMED_EMAIL")

    search_params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": top_k,
        "retmode": "xml",
        "usehistory": "y",  # required to make sort=relevance return results
        "sort": "relevance",  # Best Match
    }
    if api_key:
        search_params["api_key"] = api_key
    if email:
        search_params["email"] = email

    try:
        search_resp = requests.get(_ESEARCH_URL, params=search_params, timeout=30)
        search_resp.raise_for_status()
        ids = [
            node.text for node in ET.fromstring(search_resp.content).findall(".//Id")
        ]
    except (requests.RequestException, ET.ParseError) as exc:
        LOGGER.warning("PubMed esearch failed for %r: %s", query, exc)
        return []
    if not ids:
        return []

    fetch_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(id_ for id_ in ids if id_),
        "retmode": "xml",
    }
    if api_key:
        fetch_params["api_key"] = api_key
    if email:
        fetch_params["email"] = email
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        LOGGER.warning("PubMed efetch failed for %r: %s", query, exc)
        return []

    papers: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        # Structured abstracts split into labelled <AbstractText> sections.
        abstract_parts: list[str] = []
        for abstract_text in article.findall(".//Abstract/AbstractText"):
            label = abstract_text.get("Label", "")
            section = "".join(abstract_text.itertext()).strip()
            if not section:
                continue
            abstract_parts.append(f"{label}: {section}" if label else section)
        year_el = article.find(".//PubDate/Year")
        doi_el = article.find('.//ArticleIdList/ArticleId[@IdType="doi"]')
        papers.append(
            {
                "pmid": pmid_el.text if pmid_el is not None else "",
                "title": (title_el.text if title_el is not None else "") or "",
                "abstract": " ".join(abstract_parts),
                "year": year_el.text if year_el is not None else "",
                "doi": doi_el.text if doi_el is not None else "",
            }
        )
    return papers


def _search_semantic_scholar(query: str, top_k: int) -> list[dict[str, Any]]:
    """Keyword Semantic Scholar search returning up to ``top_k`` abstracts."""
    params = {
        "query": query,
        "fields": "paperId,externalIds,title,abstract,year",
        "limit": min(max(top_k, 1), 100),
    }
    headers: dict[str, str] = {}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    data: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            resp = requests.get(
                _S2_SEARCH_URL, params=params, headers=headers, timeout=30
            )
            if resp.status_code == 429:
                time.sleep(2**attempt * 2)  # 2s, 4s, 8s back-off on rate limit
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            LOGGER.warning("S2 search failed for %r: %s", query, exc)
            return []
    if not data:
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        ext_ids = item.get("externalIds") or {}
        papers.append(
            {
                "pmid": str(ext_ids.get("PubMed") or ""),
                "title": item.get("title") or "",
                "abstract": (item.get("abstract") or "").strip(),
                "year": item.get("year") or "",
                "doi": str(ext_ids.get("DOI") or ""),
            }
        )
    return papers


def _retrieve_pubmed_s2_passages(
    claim: str,
    *,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Retrieve top-k PubMed + top-k S2 abstracts as verdict-ready passages.

    Returns passages in the same shape the librarian modality emits
    (``pmid``, ``title``, ``evidence_snippets``) so the existing
    ``_synthesize_proclaim_verdict_from_librarian_passages`` can score them
    unchanged. Papers without an abstract are skipped; PubMed and S2 hits are
    deduplicated by PMID (falling back to title), PubMed winning ties.

    Each source gets the query form ProClaim uses for it: an entity-AND boolean
    query for PubMed (free text ANDs every token and returns nothing) and the
    free-text cleaned claim for Semantic Scholar's relevance engine.
    """
    pubmed_query = _formulate_pubmed_query(claim)
    s2_query = _proclaim_search_query(claim)
    candidates = _search_pubmed_relevance(
        pubmed_query, top_k
    ) + _search_semantic_scholar(s2_query, top_k)

    passages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in candidates:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        pmid = str(paper.get("pmid") or "").strip()
        title = (paper.get("title") or "").strip()
        key = f"pmid:{pmid}" if pmid else f"title:{title.lower()}"
        if key in seen:
            continue
        seen.add(key)
        passages.append(
            {
                "pmid": pmid,
                "title": title,
                "doi": str(paper.get("doi") or ""),
                "evidence_snippets": [abstract],
            }
        )
    return passages, f"pubmed=[{pubmed_query}] s2=[{s2_query}]"


def _prediction_from_bioagent_result(
    runner_output: dict[str, Any] | str,
    *,
    config: EvalConfig,
    started: float,
) -> ProClaimPrediction:
    if isinstance(runner_output, str):
        result: dict[str, Any] = {"summary": runner_output, "papers_raw": []}
    else:
        result = dict(runner_output or {})

    raw_output = str(result.get("summary") or result.get("raw_output") or "")
    parsed = parse_bioagent_output(raw_output, fallback_label=config.fallback_label)
    evidence = list(result.get("evidence") or [])
    papers = list(result.get("papers_raw") or [])
    inference_mode = str(result.get("inference_mode", "bio_agent"))
    if inference_mode == "bio_agent_evidence_synthesis":
        citations = (
            parsed.citations
            or _normalize_citations(result.get("citations"))
            or _citations_from_evidence_entries(evidence)
        )
    else:
        citations = (
            parsed.citations
            or _normalize_citations(result.get("citations"))
            or _citations_from_papers(papers)
        )
    latency = time.perf_counter() - started
    metadata = {
        "latency_seconds": round(latency, 3),
        "retrieved_paper_count": int(result.get("retrieved_paper_count", len(papers))),
        "evidence_count": len(evidence),
        "model_name": config.llm_model_name or os.getenv("LLM_MODEL", ""),
        "search_query": result.get("search_query", ""),
        "parse_failure": parsed.parse_failure,
        "inference_mode": inference_mode,
    }
    if "evidence_prompt_chars" in result:
        metadata["evidence_prompt_chars"] = result["evidence_prompt_chars"]
    if (
        result.get("full_text_debug")
        and inference_mode != "bio_agent_evidence_synthesis"
    ):
        metadata["full_text_debug"] = result["full_text_debug"]
    if "web_search_call_count" in result:
        metadata["web_search_call_count"] = result["web_search_call_count"]
    if "url_citation_count" in result:
        metadata["url_citation_count"] = result["url_citation_count"]

    return ProClaimPrediction(
        predicted_label=parsed.label,
        reasoning=parsed.reasoning,
        citations=citations,
        raw_output=raw_output,
        metadata=metadata,
        parse_failure=parsed.parse_failure,
    )


def _extract_response_output_text(response: dict[str, Any]) -> str:
    outputs = response.get("output") or []
    if isinstance(outputs, list):
        texts: list[str] = []
        for item in outputs:
            if not isinstance(item, dict) or item.get("type") != "message":
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
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    return ""


def _extract_response_url_citations(response: dict[str, Any]) -> list[dict[str, str]]:
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
                        "evidence": url,
                    }
                )
    return citations


def _count_response_web_search_calls(response: dict[str, Any]) -> int:
    outputs = response.get("output") or []
    if not isinstance(outputs, list):
        return 0
    return sum(
        1
        for item in outputs
        if isinstance(item, dict) and item.get("type") == "web_search_call"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_markdown_fence(text.strip())
    decoder = json.JSONDecoder()
    starts = [0]
    starts.extend(match.start() for match in re.finditer(r"\{", cleaned))
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_label_from_prose(text: str) -> str | None:
    field_match = re.search(
        r"\b(?:verdict|label|classification|answer)\b\s*[:=-]\s*[\"']?([^\n\r.;,}\"]+)",
        text,
        flags=re.IGNORECASE,
    )
    if field_match:
        label = normalize_label(field_match.group(1), fallback=None)
        if label is not None:
            return label

    lowered = text.lower()
    prose_patterns = [
        (
            r"\b(insufficient evidence|not enough evidence|not enough information|inconclusive)\b",
            "UNCERTAIN",
        ),
        (
            r"\b(ambiguous|conflicting evidence|mixed evidence|cannot determine)\b",
            "UNCERTAIN",
        ),
        (
            r"\b(no evidence that substantiates|no evidence supports|unsupported|not supported)\b",
            "REFUTE",
        ),
        (r"\b(refute[sd]?|contradict(?:s|ed|ory)?|falsified)\b", "REFUTE"),
        (r"\b(support(?:s|ed|ing)?|corroborat(?:e|es|ed)|substantiated)\b", "SUPPORT"),
    ]
    for pattern, label in prose_patterns:
        if re.search(pattern, lowered):
            return label

    for label in LABELS:
        if re.search(rf"\b{label}\b", text, flags=re.IGNORECASE):
            return label
    return None


def _normalize_citations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"evidence": value}]
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            citation = {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "pmid": str(item.get("pmid") or ""),
                "doi": str(item.get("doi") or ""),
                "evidence": str(item.get("evidence") or item.get("snippet") or ""),
            }
            extra_keys = set(item) - set(citation)
            for key in sorted(extra_keys):
                if key not in citation:
                    citation[key] = item[key]
            citations.append(citation)
        elif item:
            citations.append({"evidence": str(item)})
    return citations


def _citations_from_papers(
    papers: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for paper in papers[:limit]:
        pmid = str(paper.get("pmid") or "")
        doi = str(paper.get("doi") or "")
        url = str(paper.get("url") or "")
        if not url and pmid:
            url = f"https://europepmc.org/article/MED/{pmid}"
        if not url and doi:
            url = f"https://doi.org/{doi}"
        evidence = _paper_evidence_text(paper)
        citations.append(
            {
                "title": str(paper.get("title") or ""),
                "url": url,
                "pmid": pmid,
                "doi": doi,
                "evidence": evidence,
            }
        )
    return citations


def _citations_from_evidence_entries(
    evidence_entries: list[Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for entry in evidence_entries:
        if len(citations) >= limit:
            break
        if not isinstance(entry, dict):
            continue

        snippets = _dedupe_texts(_extract_evidence_entry_texts(entry))
        if not snippets:
            continue

        citations.append(
            {
                "title": str(entry.get("title") or ""),
                "url": str(entry.get("url") or ""),
                "pmid": str(entry.get("pmid") or ""),
                "doi": str(entry.get("doi") or ""),
                "evidence": _truncate_prompt_text(snippets[0], 600),
            }
        )
    return citations


def _paper_evidence_text(paper: dict[str, Any], limit: int = 600) -> str:
    snippets = paper.get("support_snippets") or []
    if isinstance(snippets, str):
        snippets = [snippets]
    abstract_sentences = paper.get("evidence_sentences_abstract") or []
    if isinstance(abstract_sentences, str):
        abstract_sentences = [abstract_sentences]
    full_text_sentences = paper.get("evidence_sentences_full_text") or []
    if isinstance(full_text_sentences, str):
        full_text_sentences = [full_text_sentences]
    candidates = [
        " ".join(paper.get("evidence_snippets") or []),
        *(str(snippet) for snippet in snippets if snippet),
        *(str(snippet) for snippet in abstract_sentences if snippet),
        *(str(snippet) for snippet in full_text_sentences if snippet),
        str(paper.get("full_text_excerpt") or ""),
        str(paper.get("abstract") or ""),
        str(paper.get("pageContent") or ""),
    ]
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if normalized:
            return normalized[:limit]
    return ""


def _resolve_dataset_paths(root: Path) -> dict[str, Path]:
    # The claim sets are not redistributed here: they ship with ProClaim itself,
    # which `../setup.sh` clones to ProClaim_src/. Looked up there first, so
    # nothing has to be copied out of the clone.
    candidates = [
        root / "ProClaim_src" / "datasets",
        root,
        root / "data",
        root / "datasets",
        root / "evals" / "ProClaim" / "data",
    ]
    for directory in candidates:
        signor_path = directory / "signor.csv"
        connectome_path = directory / "connectomedb.csv"
        if signor_path.exists() and connectome_path.exists():
            return {"signor": signor_path, "connectomedb": connectome_path}
    raise FileNotFoundError(
        "Could not find signor.csv and connectomedb.csv under "
        f"{root}. They ship with ProClaim — run `evals/Literature/setup.sh "
        "--bench proclaim` to clone it, which puts them in "
        "ProClaim/ProClaim_src/datasets/."
    )


def _load_subset_csv(csv_path: Path, subset: str) -> list[ProClaimExample]:
    examples: list[ProClaimExample] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "claim", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=1):
            raw_id = str(row.get("id") or f"{subset}-{row_index}").strip()
            flip_value = str(row.get("flip") or "").strip().lower()
            example_id = (
                f"{raw_id}_flip" if flip_value in {"true", "1", "yes"} else raw_id
            )
            claim = str(row.get("claim") or "").strip()
            if not claim:
                LOGGER.warning(
                    "Skipping %s row %s with empty claim.",
                    csv_path,
                    row_index,
                )
                continue
            examples.append(
                ProClaimExample(
                    id=example_id,
                    subset=subset,
                    claim=claim,
                    gold_label=(
                        normalize_label(row.get("label")) or DEFAULT_FALLBACK_LABEL
                    ),
                )
            )
    return examples


def _interleave_examples(
    examples_by_subset: list[list[ProClaimExample]],
) -> list[ProClaimExample]:
    examples: list[ProClaimExample] = []
    max_len = max(
        (len(subset_examples) for subset_examples in examples_by_subset),
        default=0,
    )
    for index in range(max_len):
        for subset_examples in examples_by_subset:
            if index < len(subset_examples):
                examples.append(subset_examples[index])
    return examples


def _normalize_subset(subset: str) -> str:
    normalized = subset.strip().lower()
    aliases = {
        "all": "all",
        "signor": "signor",
        "signor-fact": "signor",
        "signorfact": "signor",
        "connectomedb": "connectomedb",
        "connectome": "connectomedb",
        "connectomedb-fact": "connectomedb",
        "connectomedbfact": "connectomedb",
    }
    if normalized not in aliases:
        raise ValueError("subset must be one of: signor, connectomedb, all")
    return aliases[normalized]


def _prediction_row(
    example: ProClaimExample,
    prediction: ProClaimPrediction,
) -> dict[str, Any]:
    return {
        "id": example.id,
        "subset": example.subset,
        "claim": example.claim,
        "gold_label": example.gold_label,
        "predicted_label": prediction.predicted_label,
        "reasoning": prediction.reasoning,
        "citations": prediction.citations,
        "raw_output": prediction.raw_output,
        "metadata": prediction.metadata,
        "parse_failure": prediction.parse_failure,
    }


def _load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            if row_id:
                rows[row_id] = row
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_float(values: list[float], value: Any) -> None:
    try:
        if value is None or value == "":
            return
        values.append(float(value))
    except (TypeError, ValueError):
        return


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def _latex_metric(metrics: dict[str, Any], key: str) -> str:
    if not metrics or int(metrics.get("n", 0)) == 0:
        return "--"
    return f"{float(metrics.get(key, 0.0)):.2f}"


def _cache_key(claim: str, config: EvalConfig) -> str:
    payload = {
        "claim": claim,
        "llm_base_url": config.llm_base_url,
        "llm_model_name": config.llm_model_name,
        "thinking": config.thinking,
        "retriever": config.retriever,
        # Retrieval through the orchestrator returns different evidence from the
        # in-process librarian, so a cached verdict from one is not valid for the
        # other. Without this the two transports silently share cache entries.
        "via_api": config.via_api,
        "es_url": config.es_url,
        "elastic_source": config.elastic_source,
        "full_text_enrichment": config.full_text_enrichment,
        "web_search": config.web_search,
        "web_search_tool_choice": config.web_search_tool_choice,
        "web_search_context_size": config.web_search_context_size,
        "pubmed_s2": config.pubmed_s2,
        "pubmed_s2_top_k": config.pubmed_s2_top_k,
        "fallback_label": config.fallback_label,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_prediction_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable prediction cache: %s", path)
        return {}


def _write_prediction_cache(
    path: Path,
    cache_key: str,
    prediction: ProClaimPrediction,
) -> None:
    cache = _read_prediction_cache(path)
    cache[cache_key] = prediction.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _serializable_config(config: EvalConfig) -> dict[str, Any]:
    return {
        "llm_base_url": config.llm_base_url,
        "llm_model_name": config.llm_model_name,
        "thinking": config.thinking,
        "retriever": config.retriever,
        "es_url": config.es_url,
        "elastic_source": config.elastic_source,
        "full_text_enrichment": config.full_text_enrichment,
        "web_search": config.web_search,
        "web_search_tool_choice": config.web_search_tool_choice,
        "web_search_context_size": config.web_search_context_size,
        "pubmed_s2": config.pubmed_s2,
        "pubmed_s2_top_k": config.pubmed_s2_top_k,
        "fallback_label": config.fallback_label,
        "cache_enabled": config.cache_enabled,
        "cache_path": str(config.cache_path) if config.cache_path else None,
        "verbose": config.verbose,
        "max_retries": config.max_retries,
        "progress": config.progress,
    }


def _load_model_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"model config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
        return payload if isinstance(payload, dict) else {}
    except ImportError:
        return _parse_flat_yaml(text)


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip().strip("'\"")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            payload[key.strip()] = lowered == "true"
        else:
            payload[key.strip()] = value
    return payload


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    payload = _load_model_config(args.model_config)

    def get_config(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        return default

    config = EvalConfig(
        llm_base_url=get_config("llm_base_url", "base_url"),
        llm_model_name=get_config("llm_model_name", "model_name", "model"),
        thinking=bool(get_config("thinking", default=False)),
        retriever=str(get_config("retriever", "search_backend", default="europepmc")),
        es_url=str(
            get_config("es_url", "elastic_url", default="http://elasticsearch:9200")
        ),
        elastic_source=str(get_config("elastic_source", default="abstracts")),
        full_text_enrichment=bool(get_config("full_text_enrichment", default=False)),
        fallback_label=normalize_label(
            get_config("fallback_label", default=DEFAULT_FALLBACK_LABEL)
        )
        or DEFAULT_FALLBACK_LABEL,
        cache_enabled=bool(args.cache),
        cache_path=Path(args.out_dir) / "cache.json" if args.cache else None,
        verbose=bool(args.verbose),
        max_retries=max(1, int(get_config("max_retries", default=args.max_retries))),
        progress=not bool(args.no_progress),
        web_search=bool(get_config("web_search", default=False)),
        web_search_tool_choice=str(
            get_config("web_search_tool_choice", default=args.web_search_tool_choice)
        ),
        web_search_context_size=str(
            get_config(
                "web_search_context_size",
                default=args.web_search_context_size,
            )
        ),
        pubmed_s2=bool(get_config("pubmed_s2", default=False)),
        pubmed_s2_top_k=int(get_config("pubmed_s2_top_k", "top_k", default=5)),
    )

    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url
    if args.llm_model:
        config.llm_model_name = args.llm_model

    config.verdict_base_url = args.verdict_base_url or get_config("verdict_base_url")
    config.verdict_model_name = args.verdict_model or get_config("verdict_model")
    verdict_key_env = args.verdict_api_key_env or get_config("verdict_api_key_env")
    if verdict_key_env:
        config.verdict_api_key = os.environ.get(verdict_key_env)
        if not config.verdict_api_key:
            raise SystemExit(f"Verdict API key env var {verdict_key_env} is empty.")
    if args.no_agent:
        config.no_agent = True
    if args.librarian_agent:
        config.librarian_agent = True
    if getattr(args, "via_api", False):
        config.via_api = True
    if args.pubmed_s2:
        config.pubmed_s2 = True
    if args.pubmed_s2_top_k is not None:
        config.pubmed_s2_top_k = args.pubmed_s2_top_k
    if args.web_search:
        config.web_search = True
    if args.thinking:
        config.thinking = True
    if args.full_text:
        config.full_text_enrichment = True
    if args.retriever:
        config.retriever = args.retriever
    if args.es_url:
        config.es_url = args.es_url
    if args.fallback_label:
        config.fallback_label = (
            normalize_label(args.fallback_label) or DEFAULT_FALLBACK_LABEL
        )
    if config.web_search_tool_choice not in {"auto", "required"}:
        config.web_search_tool_choice = "required"
    if config.web_search_context_size not in {"low", "medium", "high"}:
        config.web_search_context_size = "medium"
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the librarian on ProClaim-eval claim verification examples."
    )
    parser.add_argument("--proclaim-path", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=["signor", "connectomedb", "all"],
        required=True,
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--fallback-label", choices=list(LABELS), default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retriever", choices=["europepmc", "elastic"], default=None)
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--via-api",
        action="store_true",
        help=(
            "With --librarian-agent, retrieve through the orchestrator "
            "(POST /run-agent/stream, agent=librarian) instead of an in-process "
            "LibrarianAgent. The verdict model is unaffected."
        ),
    )
    parser.add_argument(
        "--librarian-agent",
        action="store_true",
        help=(
            "Use the new LibrarianAgent (BM25-per-paper) for retrieval, then "
            "synthesize a verdict with one LLM call. No iterative loop."
        ),
    )
    parser.add_argument(
        "--pubmed-s2",
        action="store_true",
        help=(
            "Use the ProClaim evidence-programmer retrieval instead of the "
            "librarian: fetch top-k relevance-sorted PubMed abstracts plus "
            "top-k Semantic Scholar abstracts, then synthesize a verdict with "
            "one LLM call. No iterative loop, no full text."
        ),
    )
    parser.add_argument(
        "--pubmed-s2-top-k",
        type=int,
        default=None,
        help=(
            "Abstracts to retrieve per source in --pubmed-s2 mode (PubMed and "
            "S2 each contribute this many). Default: 5."
        ),
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Bypass the literature agent and evaluate the model directly.",
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help=(
            "Bypass the librarian and use OpenAI Responses API web_search "
            "directly. This is an LLM web-search baseline, not the librarian "
            "retrieval flow."
        ),
    )
    parser.add_argument(
        "--web-search-tool-choice",
        choices=["auto", "required"],
        default="required",
        help="Tool choice for --web-search mode (default: required).",
    )
    parser.add_argument(
        "--web-search-context-size",
        choices=["low", "medium", "high"],
        default="medium",
        help="Search context size for --web-search mode (default: medium).",
    )
    parser.add_argument(
        "--verdict-base-url",
        default=None,
        help=(
            "Base URL for the verdict model in --librarian-agent mode "
            "(OpenAI-compatible, POSTs to <base_url>/chat/completions). "
            "Use https://api.anthropic.com/v1 for Anthropic. Defaults to "
            "--llm-base-url (same model as the librarian)."
        ),
    )
    parser.add_argument(
        "--verdict-model",
        default=None,
        help=(
            "Model id for the verdict in --librarian-agent mode, e.g. "
            "claude-sonnet-4-6. Defaults to --llm-model (same as the librarian)."
        ),
    )
    parser.add_argument(
        "--verdict-api-key-env",
        default=None,
        help=(
            "Name of the env var holding the API key for --verdict-base-url "
            "(e.g. ANTHROPIC_API_KEY). Required for remote providers; omit for "
            "self-hosted vLLM."
        ),
    )
    parser.add_argument("--full-text", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar and use per-example log lines instead.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    config = _config_from_args(args)
    examples = load_proclaim_examples(
        args.proclaim_path,
        subset=args.subset,
        max_examples=args.max_examples,
    )
    if not examples:
        raise SystemExit("No ProClaim examples selected.")

    result = run_proclaim_evaluation(
        examples,
        config=config,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    LOGGER.info("Predictions: %s", result["predictions_path"])
    LOGGER.info("Metrics: %s", result["metrics_path"])
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

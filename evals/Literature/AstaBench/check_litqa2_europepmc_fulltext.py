"""Build LitQA2 subsets whose source papers have EuropePMC full text.

The AstaBench LitQA2 dev/test split is filtered against AI2's snippet-search
fulltext index. This utility checks the original LAB-bench LitQA2 source papers
against EuropePMC instead, then emits JSON subsets in the original row format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "agents").is_dir()),
    Path(__file__).resolve().parents[3],
)  # repo root = first ancestor containing agents/ (move-proof)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "evals" / "AstaBench" / "data" / "litqa2_europepmc_fulltext"
)
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ASTABENCH_DATASET_REPO = "allenai/asta-bench"
ASTABENCH_DATASET_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _hf_token() -> str | None:
    return (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
        or os.getenv("HF_TOKEN")
    )


def _normalize_source(source: str) -> str:
    return str(source or "").strip()


def _doi_from_source(source: str) -> str | None:
    source = _normalize_source(source)
    if not source:
        return None

    lower_source = source.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/"):
        if lower_source.startswith(prefix):
            return unquote(source[len(prefix) :]).strip()

    parsed = urlparse(source)
    if parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
        doi = unquote(parsed.path.lstrip("/")).strip()
        return doi or None

    match = re.search(r"\b10\.\d{4,9}/\S+\b", source)
    if match:
        return match.group(0).rstrip(").,;")
    return None


def _load_litqa2_rows() -> list[dict[str, Any]]:
    dataset = load_dataset("futurehouse/lab-bench", "LitQA2")
    assert isinstance(dataset, Dataset | DatasetDict | IterableDatasetDict)
    return [dict(row) for row in dataset["train"]]


def _load_astabench_split_lookup() -> dict[str, dict[str, Any]]:
    token = _hf_token()
    mapping = load_dataset(
        ASTABENCH_DATASET_REPO,
        data_files="tasks/labbench/litqa2_mapping.json",
        revision=ASTABENCH_DATASET_REVISION,
        token=token,
    )
    assert isinstance(mapping, Dataset | DatasetDict | IterableDatasetDict)
    return {str(row["id"]): dict(row) for row in mapping["train"]}


def _search_europepmc_by_doi(session: requests.Session, doi: str) -> dict[str, Any]:
    query = f'DOI:"{doi}"'
    fulltext_query = f"{query} AND HAS_FT:Y"
    response = session.get(
        EUROPEPMC_SEARCH_URL,
        params={
            "query": fulltext_query,
            "format": "json",
            "resultType": "core",
            "pageSize": 5,
        },
        timeout=30,
    )
    response.raise_for_status()
    fulltext_payload = response.json()
    fulltext_results = fulltext_payload.get("resultList", {}).get("result", []) or []

    fallback_hit_count = 0
    fallback_results: list[dict[str, Any]] = []
    if not fulltext_results:
        fallback_response = session.get(
            EUROPEPMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": 5,
            },
            timeout=30,
        )
        fallback_response.raise_for_status()
        fallback_payload = fallback_response.json()
        fallback_hit_count = int(fallback_payload.get("hitCount") or 0)
        fallback_results = (
            fallback_payload.get("resultList", {}).get("result", []) or []
        )

    results = fulltext_results or fallback_results
    best = results[0] if results else {}
    return {
        "doi": doi,
        "query": query,
        "fulltext_query": fulltext_query,
        "fulltext_hit_count": int(fulltext_payload.get("hitCount") or 0),
        "any_hit_count": int(fulltext_payload.get("hitCount") or 0)
        if fulltext_results
        else fallback_hit_count,
        "matched": bool(results),
        "has_europepmc_fulltext": bool(fulltext_results),
        "pmid": str(best.get("pmid") or ""),
        "pmcid": str(best.get("pmcid") or ""),
        "source": str(best.get("source") or ""),
        "title": str(best.get("title") or ""),
        "returned_doi": str(best.get("doi") or ""),
        "in_epmc": str(best.get("inEPMC") or "").upper() == "Y",
        "is_open_access": str(best.get("isOpenAccess") or "").upper() == "Y",
        "has_pdf": str(best.get("hasPDF") or "").upper() == "Y",
    }


def _check_fulltext_xml(session: requests.Session, pmcid: str) -> dict[str, Any]:
    if not pmcid:
        return {"fulltext_xml_available": False, "fulltext_xml_status": None}

    response = session.get(f"{EUROPEPMC_REST_URL}/{pmcid}/fullTextXML", timeout=30)
    if response.status_code != 200:
        return {
            "fulltext_xml_available": False,
            "fulltext_xml_status": response.status_code,
        }

    text = response.text.strip()
    return {
        "fulltext_xml_available": bool(text and "<" in text),
        "fulltext_xml_status": response.status_code,
        "fulltext_xml_chars": len(text),
    }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _source_available(source_status: dict[str, Any], require_xml: bool) -> bool:
    if require_xml:
        return bool(source_status.get("fulltext_xml_available"))
    return bool(source_status.get("has_europepmc_fulltext"))


def _row_available(row_status: dict[str, Any], require_xml: bool) -> bool:
    return any(
        _source_available(source_status, require_xml)
        for source_status in row_status["source_statuses"]
    )


def _summary_for_rows(
    rows: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    split_lookup: dict[str, dict[str, Any]],
    require_xml: bool,
) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        row_id = str(row["id"])
        split = split_lookup.get(row_id, {}).get("split")
        split_name = "not_in_dev_or_test" if split is None else str(split)
        by_split.setdefault(split_name, {"total": 0, "available": 0})
        by_split[split_name]["total"] += 1
        if _row_available(statuses[row_id], require_xml):
            by_split[split_name]["available"] += 1

    all_available = sum(
        1 for row in rows if _row_available(statuses[str(row["id"])], require_xml)
    )
    return {
        "availability_definition": "EuropePMC HAS_FT:Y plus fetchable fullTextXML"
        if require_xml
        else "EuropePMC HAS_FT:Y",
        "total_litqa2_rows": len(rows),
        "available_litqa2_rows": all_available,
        "unavailable_litqa2_rows": len(rows) - all_available,
        "by_astabench_split": by_split,
    }


def build_subsets(
    output_dir: Path,
    sleep_seconds: float,
    refresh: bool,
    require_xml: bool,
) -> dict[str, Any]:
    _load_dotenv()
    token = _hf_token()
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token

    rows = _load_litqa2_rows()
    split_lookup = _load_astabench_split_lookup()
    cache_path = output_dir / "source_availability_cache.json"
    cache = {} if refresh else _load_cache(cache_path)

    session = requests.Session()
    source_status_by_source: dict[str, dict[str, Any]] = dict(cache)
    unique_sources = sorted(
        {_normalize_source(source) for row in rows for source in row.get("sources", [])}
    )
    for index, source in enumerate(unique_sources, start=1):
        cached_status = source_status_by_source.get(source)
        if cached_status is not None:
            if require_xml and "fulltext_xml_available" not in cached_status:
                xml_status = _check_fulltext_xml(
                    session, str(cached_status.get("pmcid") or "")
                )
                source_status_by_source[source] = {**cached_status, **xml_status}
                continue
            else:
                continue

        doi = _doi_from_source(source)
        if not doi:
            source_status_by_source[source] = {
                "source_url": source,
                "doi": "",
                "matched": False,
                "has_europepmc_fulltext": False,
                "error": "source_doi_not_found",
            }
            continue

        try:
            status = _search_europepmc_by_doi(session, doi)
            if require_xml:
                status.update(_check_fulltext_xml(session, status.get("pmcid", "")))
            source_status_by_source[source] = {"source_url": source, **status}
        except requests.RequestException as exc:
            source_status_by_source[source] = {
                "source_url": source,
                "doi": doi,
                "matched": False,
                "has_europepmc_fulltext": False,
                "fulltext_xml_available": False,
                "error": str(exc),
            }
        if sleep_seconds:
            time.sleep(sleep_seconds)

        if index % 25 == 0:
            print(f"checked {index}/{len(unique_sources)} unique sources", flush=True)

    _write_json(cache_path, source_status_by_source)

    row_statuses: dict[str, dict[str, Any]] = {}
    row_status_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        mapping = split_lookup.get(row_id, {})
        source_statuses = [
            source_status_by_source[_normalize_source(source)]
            for source in row.get("sources", [])
        ]
        row_status = {
            "id": row_id,
            "question": row.get("question", ""),
            "split": mapping.get("split"),
            "corpus_ids": mapping.get("corpus_ids", []),
            "sources": row.get("sources", []),
            "source_statuses": source_statuses,
        }
        row_status["europepmc_fulltext_available"] = _row_available(
            row_status, require_xml
        )
        row_statuses[row_id] = row_status
        row_status_rows.append(row_status)

    available_rows = [
        row for row in rows if _row_available(row_statuses[str(row["id"])], require_xml)
    ]
    test_rows = [
        row
        for row in available_rows
        if split_lookup.get(str(row["id"]), {}).get("split") == "test"
    ]
    dev_rows = [
        row
        for row in available_rows
        if split_lookup.get(str(row["id"]), {}).get("split") == "dev"
    ]

    _write_json(output_dir / "litqa2_full_europepmc_fulltext.json", available_rows)
    _write_json(output_dir / "litqa2_test_europepmc_fulltext.json", test_rows)
    _write_json(output_dir / "litqa2_dev_europepmc_fulltext.json", dev_rows)
    _write_jsonl(
        output_dir / "litqa2_europepmc_fulltext_availability.jsonl", row_status_rows
    )

    summary = _summary_for_rows(rows, row_statuses, split_lookup, require_xml)
    summary["unique_source_count"] = len(unique_sources)
    summary["output_files"] = {
        "full_subset": str(output_dir / "litqa2_full_europepmc_fulltext.json"),
        "test_subset": str(output_dir / "litqa2_test_europepmc_fulltext.json"),
        "dev_subset": str(output_dir / "litqa2_dev_europepmc_fulltext.json"),
        "availability_jsonl": str(
            output_dir / "litqa2_europepmc_fulltext_availability.jsonl"
        ),
        "source_cache": str(cache_path),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--no-xml-check",
        action="store_true",
        help="Use only EuropePMC HAS_FT:Y without checking the fullTextXML endpoint.",
    )
    args = parser.parse_args()

    summary = build_subsets(
        output_dir=args.output_dir,
        sleep_seconds=args.sleep,
        refresh=args.refresh,
        require_xml=not args.no_xml_check,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())

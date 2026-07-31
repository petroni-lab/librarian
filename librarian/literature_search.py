"""Europe PMC literature search.

``search_scientific_literature_structured`` runs one Europe PMC REST query and
returns a list of paper dicts (title, abstract, authors, ids, full-text
availability, ...). The full agent wrapped this in a LangChain retriever and a
Redis/diskcache look-aside cache; neither is needed to run the librarian, so
this version is a single plain ``requests`` call with a small retry loop.
"""

import time
from typing import Any, Dict, List
from urllib.parse import quote

import requests

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Europe PMC ``result`` record into the agent's paper dict."""
    epmc_id = result.get("id", "")
    epmc_source = result.get("source", "")

    # Full author names and per-author affiliations from the core authorList.
    author_full_names: List[str] = []
    authors_with_affiliations: List[Dict[str, str]] = []
    for author in result.get("authorList", {}).get("author", []):
        first = author.get("firstName", "").strip()
        last = author.get("lastName", "").strip()
        if first and last:
            full_name = f"{first} {last}"
        elif last:
            full_name = last
        else:
            full_name = author.get("fullName", "").strip()
        author_full_names.append(full_name)

        affil_list = author.get("authorAffiliationDetailsList", {}).get(
            "authorAffiliation", []
        )
        affiliation = affil_list[0].get("affiliation", "") if affil_list else ""
        authors_with_affiliations.append(
            {"name": full_name, "affiliation": affiliation}
        )

    # Full-text availability: Europe PMC exposes it via id/url lists, not a scalar.
    full_text_ids = result.get("fullTextIdList", {}).get("fullTextId", [])
    if isinstance(full_text_ids, str):
        full_text_ids = [full_text_ids]
    if not isinstance(full_text_ids, list):
        full_text_ids = []

    full_text_urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
    if isinstance(full_text_urls, dict):
        full_text_urls = [full_text_urls]
    if not isinstance(full_text_urls, list):
        full_text_urls = []

    has_fulltext = bool(full_text_ids) or any(
        str(u.get("availabilityCode", "")).upper() == "OA"
        for u in full_text_urls
        if isinstance(u, dict)
    )

    url = ""
    if epmc_source and epmc_id:
        url = (
            "https://europepmc.org/article/"
            f"{quote(epmc_source, safe='')}/{quote(epmc_id, safe='')}"
        )

    return {
        "title": result.get("title", "No title"),
        "authors": result.get("authorString", "Unknown authors"),
        "authorFullNames": author_full_names,
        "authorsWithAffiliations": authors_with_affiliations,
        "epmcId": epmc_id,
        "epmcSource": epmc_source,
        "sourceCode": epmc_source,
        "pmid": result.get("pmid", ""),
        "pmcid": result.get("pmcid", ""),
        "doi": result.get("doi", ""),
        "source": "Europe PMC",
        "url": url,
        "abstract": result.get("abstractText", ""),
        "journal": result.get("journalInfo", {}).get("journal", {}).get("title", ""),
        "year": result.get("pubYear", ""),
        "isOpenAccess": result.get("isOpenAccess", "N") == "Y",
        "hasPDF": result.get("hasPDF", "N") == "Y",
        "inEPMC": result.get("inEPMC", "N") == "Y",
        "hasFreeFullText": has_fulltext,
        "fullTextIds": full_text_ids,
        "fullTextUrls": full_text_urls,
    }


def search_scientific_literature_structured(
    query: str, page_size: int = 100
) -> List[Dict[str, Any]]:
    """Search Europe PMC and return up to ``page_size`` structured paper dicts."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",
    }

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            response = requests.get(_SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in _RETRYABLE_STATUS or isinstance(
                exc, requests.exceptions.ConnectionError
            )
            if attempt < max_attempts - 1 and retryable:
                time.sleep(2**attempt)
                continue
            raise Exception(f"Error performing literature search: {exc}") from exc

    results = response.json().get("resultList", {}).get("result", [])
    return [_parse_result(r) for r in results]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Telomere shortening in aging"
    print(f"Searching Europe PMC for: {q!r}")
    hits = search_scientific_literature_structured(q, page_size=5)
    print(f"Found {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']} ({hit['year']}) - PMID: {hit['pmid']}")

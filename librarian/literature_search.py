"""Europe PMC literature search.

``search_scientific_literature_structured`` runs one Europe PMC REST query and
returns a list of paper dicts (title, abstract, authors, ids, full-text
availability, ...). The full agent wrapped this in a LangChain retriever and a
Redis/diskcache look-aside cache; neither is needed to run the librarian, so
this version is a single plain ``requests`` call with a small retry loop.

``LiteratureSource`` is the port ``LibrarianAgent`` searches and fetches full
texts through; ``EuropePmcSource`` is the default implementation built on the
plain functions in this module. An embedding application can pass its own
source (with a cache, a worker pool, ...) as
``LibrarianAgent(literature_source=...)``.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol
from urllib.parse import quote

import requests

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Europe PMC full-text fetch endpoint (PMC id -> JATS XML).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
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


@dataclass
class Fulltext:
    """One full-text fetch outcome: the JATS XML, or why it is missing."""

    pmcid: str
    xml: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the XML was retrieved (it may still be empty or unparseable)."""
        return not self.error


def normalize_pmcid(pmcid: str) -> str:
    """Upper-case a PMC id and prefix bare digits, so one paper keys one way.

    :param pmcid: A PMC id in any of the forms Europe PMC hands out
        (``PMC123``, ``pmc123``, ``123``).
    :return: The canonical ``PMC123`` form, used for both the dict key and the URL.
    :rtype: str
    """
    normalized = str(pmcid).strip().upper()
    return f"PMC{normalized}" if normalized.isdigit() else normalized


def fetch_fulltext(pmcid: str) -> Fulltext:
    """Fetch the JATS full-text XML for one PMC id.

    A failed request is returned as a ``Fulltext`` with ``error`` set rather
    than raised, so one missing full text only leaves its paper abstract-only.

    :param pmcid: A PMC id in any form (see ``normalize_pmcid``).
    :return: The XML, or the error that replaced it.
    :rtype: Fulltext
    """
    pmcid = normalize_pmcid(pmcid)
    try:
        response = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
        # Without this a 404 page would come back as if it were the XML.
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # Never an empty error: ok is "no error", so a message-less exception
        # would otherwise read as a successful fetch.
        return Fulltext(pmcid=pmcid, error=str(exc) or type(exc).__name__)
    return Fulltext(pmcid=pmcid, xml=response.text)


def fetch_fulltext_many(pmcids: Iterable[str]) -> Dict[str, Fulltext]:
    """Fetch several full texts one after another, keyed by normalised PMC id.

    Blank ids are dropped and repeats are fetched once.

    :param pmcids: PMC ids in any form; blanks are dropped.
    :type pmcids: Iterable[str]
    :return: One ``Fulltext`` per distinct id, successes and failures alike.
    :rtype: Dict[str, Fulltext]
    """
    unique = dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
    return {pmcid: fetch_fulltext(pmcid) for pmcid in unique}


class LiteratureSource(Protocol):
    """What ``LibrarianAgent`` needs from a literature backend.

    ``search`` returns up to ``page_size`` paper dicts in the shape
    ``_parse_result`` builds. The agent reads ``inEPMC``, ``hasFreeFullText``,
    ``fullTextIds``, ``pmcid``, ``pmid``, ``epmcId``,
    ``epmcSource``/``sourceCode``, ``doi``, ``title``, ``authors``,
    ``journal``, ``year``, ``url``, ``abstract`` and
    ``authorsWithAffiliations``. It must raise on failure, never return ``[]``
    for an outage: the agent lets the exception fail the run, so a caller can
    tell "the search broke" from "no literature found".

    ``fetch_fulltext_many`` is called once per sub-query with one id per paper,
    already normalised (``normalize_pmcid``), or ``""`` for a paper that is not
    open access. Drop the blanks and return one entry per distinct remaining
    id, keyed by that id. The agent only reads each value's ``ok``, ``xml``
    and ``error``, so any object with those attributes will do.
    """

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Run one literature query; return its paper dicts."""
        ...

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Fetch the full texts for a batch of PMC ids."""
        ...


class EuropePmcSource:
    """Default source: plain Europe PMC REST calls, no cache."""

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Search Europe PMC (see ``search_scientific_literature_structured``)."""
        return search_scientific_literature_structured(query, page_size=page_size)

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Fetch full texts sequentially (see ``fetch_fulltext_many``)."""
        return fetch_fulltext_many(pmcids)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Telomere shortening in aging"
    print(f"Searching Europe PMC for: {q!r}")
    hits = search_scientific_literature_structured(q, page_size=5)
    print(f"Found {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']} ({hit['year']}) - PMID: {hit['pmid']}")

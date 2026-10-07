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

``CachedEuropePmcSource`` is that source with a look-aside full-text cache in
front of the fetch. It works with any ``CacheBackend``; ``DiskCacheBackend`` is
the one shipped here. ``default_literature_source`` is what ``LibrarianAgent``
uses when given no source: the disk cache under ``~/.cache/librarian``, moved
with ``LIBRARIAN_CACHE_DIR`` and turned off with ``LIBRARIAN_CACHE_DIR=off``.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib.parse import quote

import diskcache
import requests

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Europe PMC full-text fetch endpoint (PMC id -> JATS XML).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Statuses a retry will never fix: the document is absent or the id is malformed.
_PERMANENT_FULLTEXT_STATUSES = frozenset({400, 403, 404, 410})
# Europe PMC intermittently 404s documents that exist (a whole batch was seen
# doing it, every id serving 200 again minutes later), so a permanent status is
# confirmed by a second request before it is trusted. Paid once per absent paper.
_PERMANENT_CONFIRM_DELAY_S = 0.5
# Full-text fetches only wait on the network, so a fixed count independent of
# cores. One module-level pool shared by every sub-query thread: this is the
# total number of concurrent full-text requests Europe PMC sees, not a per-caller
# limit. Same value as bio-agents' libs/literature_search.py.
# This is the supported override (ablation and load-test scripts set it): assign
# librarian.literature_search._FULLTEXT_WORKERS before the first fetch, because
# the pool is built once, on first use, and keeps that size for the process.
_FULLTEXT_WORKERS = 8
_fulltext_pool: "ThreadPoolExecutor | None" = None
_fulltext_pool_lock = threading.Lock()


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
    # Last, so a positional Fulltext(pmcid, xml, error) still lands in error.
    status: "int | None" = None
    # True when CachedEuropePmcSource served this outcome without a request.
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        """True when the XML was retrieved (it may still be empty or unparseable)."""
        return not self.error

    @property
    def permanent(self) -> bool:
        """True when re-requesting this pmcid cannot change the outcome."""
        return self.status in _PERMANENT_FULLTEXT_STATUSES


def normalize_pmcid(pmcid: str) -> str:
    """Upper-case a PMC id and prefix bare digits, so one paper keys one way.

    :param pmcid: A PMC id in any of the forms Europe PMC hands out
        (``PMC123``, ``pmc123``, ``123``).
    :return: The canonical ``PMC123`` form, used for both the dict key and the URL.
    :rtype: str
    """
    normalized = str(pmcid).strip().upper()
    return f"PMC{normalized}" if normalized.isdigit() else normalized


def _attempt_fulltext(pmcid: str) -> Fulltext:
    """One GET for *pmcid*, with no retry.

    Unlike search, 429/5xx are not retried: the paper just stays abstract-only.
    With ``_FULLTEXT_WORKERS`` requests in flight rate limits are likelier, so
    if "full-text unavailable" logs show them, reuse the search retry loop here.
    """
    try:
        response = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
        # Without this a 404 page would come back as if it were the XML.
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Never an empty error: ok is "no error", so a message-less exception
        # would otherwise read as a successful fetch.
        return Fulltext(
            pmcid=pmcid, status=status, error=f"{status or type(exc).__name__}: {exc}"
        )
    return Fulltext(pmcid=pmcid, xml=response.text, status=response.status_code)


def fetch_fulltext(pmcid: str) -> Fulltext:
    """Fetch the JATS full-text XML for one PMC id, confirming a permanent failure.

    A failed request is returned as a ``Fulltext`` with ``error`` set rather
    than raised, so one missing full text only leaves its paper abstract-only.
    A permanent status (404, ...) is re-requested once before it is trusted,
    because Europe PMC sometimes 404s documents that exist.

    :param pmcid: A PMC id in any form (see ``normalize_pmcid``).
    :return: The XML, or the error that replaced it.
    :rtype: Fulltext
    """
    pmcid = normalize_pmcid(pmcid)
    result = _attempt_fulltext(pmcid)
    if not result.ok and result.permanent:
        time.sleep(_PERMANENT_CONFIRM_DELAY_S)
        result = _attempt_fulltext(pmcid)
    return result


def _get_fulltext_pool() -> ThreadPoolExecutor:
    """The shared full-text fetch pool, created on first use."""
    global _fulltext_pool
    if _fulltext_pool is None:
        with _fulltext_pool_lock:
            if _fulltext_pool is None:
                _fulltext_pool = ThreadPoolExecutor(
                    max_workers=_FULLTEXT_WORKERS, thread_name_prefix="epmc-ft"
                )
    return _fulltext_pool


def fetch_fulltext_many(pmcids: Iterable[str]) -> Dict[str, Fulltext]:
    """Fetch several full texts concurrently, keyed by normalised PMC id.

    Blank ids are dropped and repeats are fetched once. All callers share
    ``_FULLTEXT_WORKERS`` slots, so the sub-query fan-out cannot multiply into a
    burst against Europe PMC.

    :param pmcids: PMC ids in any form; blanks are dropped.
    :type pmcids: Iterable[str]
    :return: One ``Fulltext`` per distinct id, successes and failures alike.
    :rtype: Dict[str, Fulltext]
    """
    unique = list(
        dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
    )
    return {r.pmcid: r for r in _get_fulltext_pool().map(fetch_fulltext, unique)}


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
        """Fetch full texts concurrently (see ``fetch_fulltext_many``)."""
        return fetch_fulltext_many(pmcids)


class CacheBackend(Protocol):
    """Key/value store ``CachedEuropePmcSource`` keeps fetch outcomes in.

    Values are plain JSON-able dicts. Both methods must be best-effort: a
    backend that is down returns ``None`` from ``get`` and drops the ``set``,
    so a cache outage degrades to live requests instead of failing the run.
    """

    def get(self, key: str) -> Any:
        """The stored value for *key*, or None on a miss."""
        ...

    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        """Store *value* under *key* for *expire_seconds*."""
        ...


class DiskCacheBackend:
    """``CacheBackend`` on a local diskcache (SQLite) directory.

    diskcache caps the directory at 1 GB by default and evicts the oldest
    entries past that, so the cache never grows without bound.
    """

    def __init__(self, directory: str) -> None:
        """Open (or create) the cache in *directory*.

        :param directory: Where diskcache keeps its SQLite files.
        :type directory: str
        """
        # diskcache's `timeout` kwarg is the SQLite busy-timeout, not a TTL;
        # entry lifetime is the expire_seconds passed to set().
        self._cache = diskcache.Cache(directory)
        # Concurrent SQLite writes from the fetch pool can raise "database is
        # locked"; serialise only the writes, reads and requests stay parallel.
        self._write_lock = threading.Lock()
        self._warned = False

    def _warn_once(self, exc: Exception) -> None:
        # One warning per backend, so a broken cache doesn't spam every fetch.
        if not self._warned:
            logger.warning("full-text cache unavailable (%s); fetching live", exc)
            self._warned = True

    def get(self, key: str) -> Any:
        """The stored value for *key*, or None on a miss or a broken cache."""
        try:
            return self._cache.get(key)
        except Exception as exc:  # corrupt/locked db -> treat as a miss
            self._warn_once(exc)
            return None

    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        """Store *value* under *key*; a failed write is logged once and dropped."""
        try:
            with self._write_lock:
                self._cache.set(key, value, expire=expire_seconds)
        except Exception as exc:  # best-effort write, never break retrieval
            self._warn_once(exc)


def _fulltext_cache_key(pmcid: str) -> str:
    # Literal normalised PMCID: one paper, one key, however its id was spelled.
    return "epmc:ft:" + normalize_pmcid(pmcid)


def _fulltext_from_entry(pmcid: str, entry: Any) -> Optional[Fulltext]:
    # A bare string is the format from before failures were cached: a success.
    if isinstance(entry, str):
        return Fulltext(pmcid=pmcid, xml=entry, status=200, from_cache=True)
    if isinstance(entry, dict):
        return Fulltext(
            pmcid=pmcid,
            xml=str(entry.get("xml") or ""),
            error=str(entry.get("error") or ""),
            status=entry.get("status"),
            from_cache=True,
        )
    return None


class CachedEuropePmcSource(EuropePmcSource):
    """``EuropePmcSource`` with a look-aside cache in front of the full-text fetch.

    Failures are cached too, because ~15% of the papers Europe PMC flags as
    open access have no fetchable full text. On a read:

    - a cached success is returned as-is;
    - a cached permanent failure (400/403/404/410, already confirmed by
      ``fetch_fulltext``) is returned without a request;
    - a cached transient failure (429/5xx, timeout, connection error) is
      fetched again, and the new outcome replaces it.

    Failures expire on a shorter clock than successes, so a paper that becomes
    open access later is not skipped for the full success TTL. Search is not
    cached.
    """

    def __init__(
        self,
        backend: CacheBackend,
        ttl_days: float = 60,
        negative_ttl_days: float = 7,
    ) -> None:
        """Wrap Europe PMC with *backend*.

        :param backend: Where outcomes are stored (e.g. ``DiskCacheBackend``).
        :type backend: CacheBackend
        :param ttl_days: Lifetime of a cached success.
        :type ttl_days: float
        :param negative_ttl_days: Lifetime of a cached failure.
        :type negative_ttl_days: float
        """
        self.backend = backend
        self.ttl_days = ttl_days
        self.negative_ttl_days = negative_ttl_days

    def get_cached_fulltext(self, pmcid: str) -> Optional[Fulltext]:
        """The cached outcome for *pmcid*, or None on a miss.

        :param pmcid: A PMC id in any form.
        :type pmcid: str
        :return: The cached success or failure, with ``from_cache`` set.
        :rtype: Fulltext or None
        """
        normalized = normalize_pmcid(pmcid)
        entry = self.backend.get(_fulltext_cache_key(normalized))
        cached = _fulltext_from_entry(normalized, entry) if entry is not None else None
        logger.debug(
            "full-text cache %s pmcid=%s",
            "miss" if cached is None else ("hit" if cached.ok else cached.status),
            normalized,
        )
        return cached

    def set_cached_fulltext(self, result: Fulltext) -> None:
        """Store one fetch outcome, success or failure, under its own TTL.

        :param result: The outcome to cache.
        :type result: Fulltext
        """
        ttl_days = self.ttl_days if result.ok else self.negative_ttl_days
        entry = {"xml": result.xml, "status": result.status, "error": result.error}
        self.backend.set(
            _fulltext_cache_key(result.pmcid), entry, int(ttl_days * 86400)
        )

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Serve what the cache settles, fetch the rest on the shared pool.

        :param pmcids: PMC ids in any form; blanks are dropped.
        :type pmcids: Iterable[str]
        :return: One ``Fulltext`` per distinct id, successes and failures alike.
        :rtype: Dict[str, Fulltext]
        """
        unique = list(
            dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
        )
        results: Dict[str, Fulltext] = {}
        misses: List[str] = []
        # ponytail: cache reads run serially before the pool; each is a local or
        # one-round-trip lookup. Move them into the pool if they ever show in traces.
        for pmcid in unique:
            cached = self.get_cached_fulltext(pmcid)
            if cached is not None and (cached.ok or cached.permanent):
                results[pmcid] = cached
            else:
                misses.append(pmcid)
        fetched = super().fetch_fulltext_many(misses)
        for pmcid, result in fetched.items():
            self.set_cached_fulltext(result)
            results[pmcid] = result
        return results


def default_literature_source() -> EuropePmcSource:
    """The source ``LibrarianAgent`` uses when the caller passes none.

    Full texts are cached on disk in ``$LIBRARIAN_CACHE_DIR``, by default
    ``$XDG_CACHE_HOME/librarian`` (``~/.cache/librarian``). Set
    ``LIBRARIAN_CACHE_DIR=off`` for plain uncached requests. A cache that
    cannot be opened (read-only home, ...) is logged and skipped, never fatal.

    :return: A ``CachedEuropePmcSource`` on the disk cache, or a plain
        ``EuropePmcSource`` when caching is off or unavailable.
    :rtype: EuropePmcSource
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    default_directory = os.path.join(xdg_cache_home, "librarian")
    directory = os.environ.get("LIBRARIAN_CACHE_DIR", "").strip() or default_directory
    if directory.lower() == "off":
        return EuropePmcSource()
    try:
        backend = DiskCacheBackend(directory)
    except Exception as exc:  # the cache is an optimisation, never a hard dependency
        logger.warning("full-text cache disabled: cannot open %s (%s)", directory, exc)
        return EuropePmcSource()
    return CachedEuropePmcSource(backend)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Telomere shortening in aging"
    print(f"Searching Europe PMC for: {q!r}")
    hits = search_scientific_literature_structured(q, page_size=5)
    print(f"Found {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']} ({hit['year']}) - PMID: {hit['pmid']}")

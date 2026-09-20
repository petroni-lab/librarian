"""
Europe PMC literature search library for scientific papers.
Provides both a LangChain-compatible retriever and a standalone structured search function.

Optional look-aside cache
-------------------------
One backend, selected by environment variable:

  EPMC_CACHE_DIR       — Directory path for a diskcache (SQLite) cache.

If it is unset the cache is disabled — no crash, no warning after the first.
Entry lifetime is EPMC_CACHE_TTL_DAYS (default 60 days).

NB: this is bio-agents' ``libs/literature_search`` with the Redis backend
removed; it is otherwise kept identical, so fixes flow between the two without
translation. bio-agents additionally supports EPMC_CACHE_REDIS_URL for
deployments where several replicas share one cache.

Two cache layers are maintained:
  * Search-result cache  — keyed by SHA-256(query + page_size), stores Document lists.
  * Full-text cache      — keyed by normalised PMCID, stores the XML *or* the
                           failure that fetching it produced (see below).

Full-text failures are cached too, because ~15% of the papers Europe PMC flags as
open access have no fetchable ``fullTextXML``. A permanent failure (404/410/…) is
remembered so later runs skip the request entirely; a transient one (429/5xx,
timeout, connection error) is remembered only as a breadcrumb and retried on the
next read. Negative entries expire on their own, shorter clock
(EPMC_CACHE_NEGATIVE_TTL_DAYS, default 7) so a paper that becomes open access
later is not skipped for the full 60 days.

Set EPMC_CACHE_DEBUG=1 to log cache hits and misses to stderr.
"""

import hashlib
import json
import os
import sys
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
from urllib.parse import quote
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

try:
    import diskcache as _dc

    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False

# diskcache uses SQLite under the hood; concurrent cache.set() calls from
# multiple threads can race and cause "database is locked" errors.
# This lock serializes only the write operations — HTTP requests remain parallel.
_cache_write_lock = threading.Lock()

# Module-level cache backend singleton — created once on first use.
_cache_backend: "object | None" = None
_cache_warned: bool = False
# Warn once (not per-request) when the cache backend is unreachable.
_cache_error_warned: bool = False


def _warn_cache_unavailable(exc: Exception) -> None:
    """Log a single warning when the cache backend errors (e.g. a locked db).

    The EPMC cache is a look-aside optimization, never a hard dependency: if the
    backend is unreachable we fall back to the live Europe PMC API. We warn only
    once so a sustained outage doesn't spam the logs.
    """
    global _cache_error_warned
    if not _cache_error_warned:
        print(
            f"EPMC cache backend unavailable ({exc}); serving from the live API "
            "until it recovers.",
            file=sys.stderr,
        )
        _cache_error_warned = True


def _documents_to_jsonable(docs: List[Document]) -> List[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def _documents_from_jsonable(items) -> List[Document]:
    """Convert a list of dicts (or raw Document objects) back to Document instances.

    Tolerates legacy diskcache entries that stored raw Document objects directly.
    """
    out = []
    for it in items:
        if isinstance(it, Document):
            out.append(it)
        else:
            out.append(
                Document(page_content=it["page_content"], metadata=it["metadata"])
            )
    return out


class _DiskCacheBackend:
    """Thin wrapper around diskcache.Cache that matches the backend interface."""

    def __init__(self, cache: "_dc.Cache") -> None:
        self._cache = cache

    def get(self, key: str):
        try:
            return self._cache.get(key)
        except Exception as exc:  # corrupt/locked db → treat as a miss
            _warn_cache_unavailable(exc)
            return None

    def set(self, key: str, value, expire_seconds: int) -> None:
        try:
            with _cache_write_lock:
                self._cache.set(key, value, expire=expire_seconds)
        except Exception as exc:  # best-effort write, never break retrieval
            _warn_cache_unavailable(exc)


def _get_cache_backend() -> "_DiskCacheBackend | None":
    """Return the active cache backend, creating it on first call.

    Selection order: EPMC_CACHE_DIR → None.
    """
    global _cache_backend, _cache_warned
    if _cache_backend is not None:
        return _cache_backend

    cache_dir = os.environ.get("EPMC_CACHE_DIR", "").strip()
    if cache_dir:
        if not _DISKCACHE_AVAILABLE:
            print(
                "EPMC cache: EPMC_CACHE_DIR is set but diskcache is not installed — cache disabled.",
                file=sys.stderr,
            )
            return None
        # NB: diskcache.Cache's `timeout` kwarg is the SQLite busy-timeout in
        # seconds (how long to retry on "database is locked"), not a cache
        # TTL — leave it at diskcache's own default (60s). Entry lifetime is
        # controlled separately via EPMC_CACHE_TTL_DAYS below.
        _cache_backend = _DiskCacheBackend(_dc.Cache(cache_dir))
        return _cache_backend

    if not _cache_warned:
        print(
            "EPMC cache: EPMC_CACHE_DIR is not set — cache disabled. "
            "Set it to enable persistent caching.",
            file=sys.stderr,
        )
        _cache_warned = True
    return None


def _epmc_synonym_enabled() -> bool:
    """Whether to ask Europe PMC to expand MeSH/UniProt synonyms (default off).

    Europe PMC's REST default is ``synonym=false``. Turning it on broadens
    recall by matching gene/organism aliases (e.g. leukolysin↔MMP25,
    mu-opioid↔OPRM1) — useful since downstream BM25 + LLM filtering re-tightens
    precision. Toggle with ``LITERATURE_EPMC_SYNONYM=true``.
    """
    return os.environ.get("LITERATURE_EPMC_SYNONYM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _cache_key(query: str, page_size: int, synonym: bool = False) -> str:
    """Stable cache key: SHA-256 of query + page_size (+ synonym when enabled).

    ``synonym=False`` reproduces the historical key exactly, so pre-existing
    cache entries stay valid; ``synonym=True`` gets a distinct key so the two
    settings never collide.
    """
    payload = {"q": query, "n": page_size}
    if synonym:
        payload["syn"] = True
    raw = json.dumps(payload, sort_keys=True)
    return "epmc:" + hashlib.sha256(raw.encode()).hexdigest()


# ── Full text: fetch, negative caching, parallel batch ──────────────────────

# Europe PMC serves open-access full text as JATS XML from a path endpoint (no
# query parameters, unlike /search).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Statuses a retry will never fix: the document is genuinely absent, or the id is
# malformed. Cached so later runs skip the request entirely. Everything else —
# 429, 5xx, timeouts, connection errors — is transient: the entry is written as a
# breadcrumb but re-fetched the next time this pmcid is read.
_PERMANENT_FULLTEXT_STATUSES = frozenset({400, 403, 404, 410})

# Pause before the confirming request that turns a permanent status into a cached
# negative (see _request_fulltext). Long enough to ride out the brief upstream
# episodes that produce false 404s, short enough to stay invisible: it is paid at
# most once per pmcid, and only for the ~15% that really have no full text.
_PERMANENT_CONFIRM_DELAY_S = 0.5

# Full-text fetches only WAIT on Europe PMC (network I/O, no CPU), so this pool is
# a fixed count independent of cores — the same reasoning as the librarian's
# _JUDGE_WORKERS. It is module-level on purpose: every sub-query thread shares it,
# so this is the total number of concurrent full-text requests EBI ever sees, not
# a per-caller limit that multiplies by the sub-query fan-out.
_FULLTEXT_WORKERS = 8
_fulltext_pool: "ThreadPoolExecutor | None" = None
_fulltext_pool_lock = threading.Lock()


@dataclass(frozen=True)
class Fulltext:
    """One full-text fetch outcome: the JATS XML, or the failure that replaced it.

    A failure is a first-class result rather than an empty string, so a caller can
    tell "this paper has no full text" from "the fetch broke" — in production the
    two used to be indistinguishable and both silently degraded the paper to
    abstract-only.
    """

    pmcid: str
    xml: str = ""
    status: "int | None" = None
    error: str = ""
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
    :return: The canonical ``PMC123`` form, used for both the cache key and the URL.
    :rtype: str
    """
    normalized = str(pmcid).strip().upper()
    return f"PMC{normalized}" if normalized.isdigit() else normalized


def _fulltext_cache_key(pmcid: str) -> str:
    """Stable cache key for a full-text entry: literal normalised PMCID."""
    return "epmc:ft:" + normalize_pmcid(pmcid)


def _fulltext_from_entry(pmcid: str, entry: Any) -> "Fulltext | None":
    """Rebuild a ``Fulltext`` from a cache entry, or None if it is unusable.

    Tolerates the legacy format (a bare XML string), which pre-dates negative
    caching and is still sitting in production caches under a 60-day TTL.
    """
    if isinstance(entry, str):
        return Fulltext(pmcid=pmcid, xml=entry, status=200, from_cache=True)
    if isinstance(entry, dict):
        return Fulltext(
            pmcid=pmcid,
            xml=str(entry.get("xml") or ""),
            status=entry.get("status"),
            error=str(entry.get("error") or ""),
            from_cache=True,
        )
    return None


def _get_cached_fulltext(pmcid: str) -> "Fulltext | None":
    """Look up a cached full-text outcome for *pmcid* (None on miss/disabled)."""
    backend = _get_cache_backend()
    if backend is None:
        return None
    hit = backend.get(_fulltext_cache_key(pmcid))
    cached = _fulltext_from_entry(pmcid, hit) if hit is not None else None
    if os.environ.get("EPMC_CACHE_DEBUG") == "1":
        state = "miss"
        if cached is not None:
            state = "hit " if cached.ok else f"neg({cached.status})"
        print(f"[epmc-cache] ft {state} pmcid={pmcid}", file=sys.stderr)
    return cached


def _set_cached_fulltext(result: Fulltext) -> None:
    """Persist a fetch outcome (success or failure); no-op if the cache is off.

    Failures expire on their own, shorter clock: a paper under embargo becomes
    open access later, and a 60-day negative entry would keep it invisible for
    two months after it became fetchable.
    """
    backend = _get_cache_backend()
    if backend is None:
        return
    ttl_env = "EPMC_CACHE_TTL_DAYS" if result.ok else "EPMC_CACHE_NEGATIVE_TTL_DAYS"
    ttl_days = float(os.environ.get(ttl_env, "60" if result.ok else "7"))
    entry = {"xml": result.xml, "status": result.status, "error": result.error}
    backend.set(_fulltext_cache_key(result.pmcid), entry, int(ttl_days * 86400))


def _attempt_fulltext(pmcid: str) -> Fulltext:
    """One GET for *pmcid*, with no caching and no retry."""
    try:
        response = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return Fulltext(
            pmcid=pmcid,
            status=status,
            error=f"{status or type(exc).__name__}: {exc}",
        )
    return Fulltext(pmcid=pmcid, xml=response.text, status=response.status_code)


def _request_fulltext(pmcid: str) -> Fulltext:
    """Fetch *pmcid*, confirm any permanent verdict, and cache the outcome.

    Europe PMC intermittently answers 404 for documents that do exist: a whole
    79-fetch batch was observed 404ing, with every one of those ids serving a
    200 and full JATS again minutes later. A permanent verdict is cached for days
    and silently degrades the paper to abstract-only for that whole window, so it
    has to be confirmed by a second request before it is trusted. A transient
    failure needs no confirmation — it is already retried on the next read.
    """
    result = _attempt_fulltext(pmcid)
    if not result.ok and result.permanent:
        time.sleep(_PERMANENT_CONFIRM_DELAY_S)
        result = _attempt_fulltext(pmcid)
    _set_cached_fulltext(result)
    return result


def fetch_fulltext(pmcid: str) -> Fulltext:
    """Full-text JATS XML for a PMC id, through the negative-aware cache.

    Cache decision, in order: a cached success is returned as-is; a cached
    *permanent* failure (404/410/…) is returned without touching the network; a
    cached *transient* failure (429/5xx, timeout, connection error) is retried
    now, and whatever comes back replaces the entry. A miss is fetched and cached
    either way.

    There is no in-request retry loop — the transient entry IS the retry, taken
    on the next read. Within one librarian run that read comes for free whenever a
    paper is found by more than one sub-query (papers are deliberately not
    deduplicated across sub-queries).

    :param pmcid: A PMC id in any form; normalised internally.
    :type pmcid: str
    :return: The fetch outcome — never raises, failures come back as a
        ``Fulltext`` with ``ok`` False.
    :rtype: Fulltext
    """
    normalized = normalize_pmcid(pmcid)
    cached = _get_cached_fulltext(normalized)
    if cached is not None and (cached.ok or cached.permanent):
        return cached
    return _request_fulltext(normalized)


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

    Ids are deduplicated first, so a pmcid repeated inside one batch costs one
    request. All callers share ``_FULLTEXT_WORKERS`` slots, so the sub-query
    fan-out above cannot multiply into a burst against Europe PMC.

    :param pmcids: PMC ids in any form; blanks are dropped.
    :type pmcids: Iterable[str]
    :return: One ``Fulltext`` per distinct id, successes and failures alike.
    :rtype: Dict[str, Fulltext]
    """
    # ponytail: two sub-query threads asking for the same pmcid at the same moment
    # both fetch it (dedup is per batch, not global). Worst case is one duplicate
    # GET; add a per-pmcid in-flight lock only if that shows up in the numbers.
    unique = list(
        dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
    )
    if not unique:
        return {}
    return {
        result.pmcid: result
        for result in _get_fulltext_pool().map(fetch_fulltext, unique)
    }


def get_cached_fulltext_xml(pmcid: str) -> "str | None":
    """Cached full-text XML for *pmcid*, or None on miss / failure / cache disabled.

    Thin compatibility wrapper over ``fetch_fulltext``'s cache: it reports only
    successes, so a cached failure reads as a miss. New code should call
    ``fetch_fulltext`` and inspect the ``Fulltext`` instead.
    """
    cached = _get_cached_fulltext(normalize_pmcid(pmcid))
    return cached.xml if cached is not None and cached.ok else None


def set_cached_fulltext_xml(pmcid: str, xml_text: str) -> None:
    """Persist *xml_text* for *pmcid* as a successful entry (no-op if disabled)."""
    _set_cached_fulltext(
        Fulltext(pmcid=normalize_pmcid(pmcid), xml=xml_text, status=200)
    )


class LiteratureSearchError(RuntimeError):
    """Raised when Europe PMC could not complete a search.

    Distinct from every other failure in the retrieval pipeline: callers catch it
    to tell the user the search itself broke, instead of reporting a search
    outage as an absence of literature.
    """


class EuropePMCRetriever(BaseRetriever):
    """Retriever that searches Europe PMC database for scientific literature.

    This retriever queries the Europe PMC API to find relevant scientific papers
    and returns them as Document objects.
    """

    page_size: int = 10
    base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Internal method for LangChain to retrieve documents.
        Queries Europe PMC API and wraps results in LangChain Document objects.
        """
        # Search parameters for the REST API
        synonym_on = _epmc_synonym_enabled()
        params = {
            "query": query,
            "format": "json",
            "pageSize": self.page_size,
            "resultType": "core",
        }
        if synonym_on:
            # Europe PMC expects a lowercase string literal here.
            params["synonym"] = "true"

        # Check cache before hitting the network.
        backend = _get_cache_backend()
        key = _cache_key(query, self.page_size, synonym=synonym_on)
        if backend is not None:
            hit = backend.get(key)
            if hit is not None:
                if os.environ.get("EPMC_CACHE_DEBUG") == "1":
                    print(
                        f"[epmc-cache] search hit  key={key[:16]} q_preview={query[:80]}",
                        file=sys.stderr,
                    )
                return _documents_from_jsonable(hit)

        if os.environ.get("EPMC_CACHE_DEBUG") == "1":
            print(
                f"[epmc-cache] search miss key={key[:16]} q_preview={query[:80]}",
                file=sys.stderr,
            )

        _max_attempts = 4
        for _attempt in range(_max_attempts):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if _attempt < _max_attempts - 1 and (
                    getattr(getattr(e, "response", None), "status_code", None)
                    in (429, 500, 502, 503, 504)
                    or isinstance(e, requests.exceptions.ConnectionError)
                ):
                    time.sleep(2**_attempt)
                    continue
                # A network/HTTP failure is not "zero results": propagate it so
                # callers can tell an unreachable Europe PMC from an empty search.
                # Returning [] here reached users as "no literature found".
                raise LiteratureSearchError(f"Europe PMC request failed: {e}") from e

        try:
            data = response.json()
            result_list = data.get("resultList", {}).get("result", [])

            documents = []
            for result in result_list:
                epmc_id = result.get("id", "")
                epmc_source = result.get("source", "")
                title = result.get("title", "No title")
                abstract = result.get("abstractText", "No abstract available")
                authors = result.get("authorString", "Unknown authors")
                pmid = result.get("pmid", "")
                pmcid = result.get("pmcid", "")
                doi = result.get("doi", "")

                journal_info = result.get("journalInfo", {})
                journal = journal_info.get("journal", {}).get("title", "")
                year = result.get("pubYear", "")

                # Extract full author names and per-author affiliations from authorList (core result type)
                author_full_names = []
                authors_with_affiliations = []
                author_list = result.get("authorList", {}).get("author", [])
                for author_entry in author_list:
                    first = author_entry.get("firstName", "").strip()
                    last = author_entry.get("lastName", "").strip()
                    if first and last:
                        full_name = f"{first} {last}"
                    elif last:
                        full_name = last
                    else:
                        full_name = author_entry.get("fullName", "").strip()

                    author_full_names.append(full_name)

                    # Pull the first affiliation string if present
                    affil_details = author_entry.get("authorAffiliationDetailsList", {})
                    affil_list = affil_details.get("authorAffiliation", [])
                    affiliation = (
                        affil_list[0].get("affiliation", "") if affil_list else ""
                    )

                    authors_with_affiliations.append(
                        {"name": full_name, "affiliation": affiliation}
                    )

                # Full text availability indicators
                is_oa = result.get("isOpenAccess", "N") == "Y"
                has_pdf = result.get("hasPDF", "N") == "Y"
                in_epmc = result.get("inEPMC", "N") == "Y"
                full_text_id_list = result.get("fullTextIdList", {}).get(
                    "fullTextId", []
                )
                if isinstance(full_text_id_list, str):
                    full_text_id_list = [full_text_id_list]
                if not isinstance(full_text_id_list, list):
                    full_text_id_list = []
                full_text_url_list = result.get("fullTextUrlList", {}).get(
                    "fullTextUrl", []
                )
                if isinstance(full_text_url_list, dict):
                    full_text_url_list = [full_text_url_list]
                if not isinstance(full_text_url_list, list):
                    full_text_url_list = []
                # Europe PMC search responses expose full-text availability via
                # fullTextIdList/fullTextUrlList rather than a stable
                # hasFreeFullText scalar field.
                has_fulltext = bool(full_text_id_list) or any(
                    str(url_entry.get("availabilityCode", "")).upper() == "OA"
                    for url_entry in full_text_url_list
                    if isinstance(url_entry, dict)
                )
                epmc_url = ""
                if epmc_source and epmc_id:
                    epmc_url = (
                        "https://europepmc.org/article/"
                        f"{quote(epmc_source, safe='')}/{quote(epmc_id, safe='')}"
                    )

                content = f"Title: {title}\n\nAbstract: {abstract}"
                metadata = {
                    "epmcId": epmc_id,
                    "epmcSource": epmc_source,
                    "sourceCode": epmc_source,
                    "title": title,
                    "authors": authors,
                    "authorFullNames": author_full_names,
                    "authorsWithAffiliations": authors_with_affiliations,
                    "journal": journal,
                    "year": year,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "doi": doi,
                    "source": "Europe PMC",
                    "url": epmc_url,
                    "isOpenAccess": is_oa,
                    "hasPDF": has_pdf,
                    "inEPMC": in_epmc,
                    "hasFreeFullText": has_fulltext,
                    "fullTextIds": full_text_id_list,
                    "fullTextUrls": full_text_url_list,
                }

                documents.append(Document(page_content=content, metadata=metadata))

        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # A malformed response is an upstream failure too — don't disguise it
            # as an empty result set.
            raise LiteratureSearchError(
                f"Europe PMC response could not be parsed: {e}"
            ) from e

        # Persist to cache for future runs. Outside the guard above: a bad
        # EPMC_CACHE_TTL_DAYS or a broken cache backend is our failure, not
        # Europe PMC's, and must not be reported as a search outage.
        if backend is not None:
            ttl_days = float(os.environ.get("EPMC_CACHE_TTL_DAYS", "60"))
            backend.set(key, _documents_to_jsonable(documents), int(ttl_days * 86400))

        return documents


def search_scientific_literature_structured(
    query: str, page_size: int = 100
) -> List[Dict[str, Any]]:
    """Search scientific literature and return structured results.

    Args:
        query: The search query
        page_size: Number of results to return

    Returns:
        List of dictionaries containing paper information.
    """
    try:
        base_retriever = EuropePMCRetriever(page_size=page_size)
        documents = base_retriever.invoke(query)

        # Convert to structured format
        results = []
        for doc in documents:
            metadata = doc.metadata
            abstract = ""
            if "Abstract:" in doc.page_content:
                abstract = doc.page_content.split("Abstract:", 1)[1].strip()

            result = {
                "title": metadata.get("title", "No title"),
                "authors": metadata.get("authors", "Unknown authors"),
                "authorFullNames": metadata.get("authorFullNames", []),
                "authorsWithAffiliations": metadata.get("authorsWithAffiliations", []),
                "epmcId": metadata.get("epmcId", ""),
                "epmcSource": metadata.get("epmcSource", ""),
                "sourceCode": metadata.get("sourceCode", ""),
                "pmid": metadata.get("pmid", ""),
                "pmcid": metadata.get("pmcid", ""),
                "doi": metadata.get("doi", ""),
                "source": metadata.get("source", "Europe PMC"),
                "url": metadata.get("url", ""),
                "pageContent": doc.page_content,
                "abstract": abstract,
                "journal": metadata.get("journal", ""),
                "year": metadata.get("year", ""),
                "isOpenAccess": metadata.get("isOpenAccess", False),
                "hasPDF": metadata.get("hasPDF", False),
                "inEPMC": metadata.get("inEPMC", False),
                "hasFreeFullText": metadata.get("hasFreeFullText", False),
                "fullTextIds": metadata.get("fullTextIds", []),
                "fullTextUrls": metadata.get("fullTextUrls", []),
            }
            results.append(result)

        return results

    except LiteratureSearchError:
        raise  # already the specific "the search broke" signal
    except Exception as e:
        detail = str(e) or repr(e)
        raise Exception(f"Error performing literature search: {detail}") from e


if __name__ == "__main__":
    # Test block
    import sys

    test_query = "Telomere shortening in aging"
    if len(sys.argv) > 1:
        test_query = sys.argv[1]

    print(f"Testing EuropePMC search for: '{test_query}'...")
    try:
        hits = search_scientific_literature_structured(test_query, page_size=5)
        print(f"Found {len(hits)} results:")
        for i, hit in enumerate(hits):
            print(f"{i + 1}. {hit['title']} ({hit['year']}) - PMID: {hit['pmid']}")
    except Exception as e:
        print(f"Search failed: {e}")

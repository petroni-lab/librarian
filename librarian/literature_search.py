"""Europe PMC literature search.

``search_scientific_literature_structured`` runs one Europe PMC REST query and
returns a list of paper dicts (title, abstract, authors, ids, full-text
availability, ...): a single plain ``requests`` call with a small retry loop.

``LiteratureSource`` is the port ``LibrarianAgent`` searches and fetches full
texts through; ``EuropePmcSource`` is the default implementation built on the
plain functions in this module. An embedding application can pass its own
source (with a cache, a worker pool, ...) as
``LibrarianAgent(literature_source=...)``.

Supplementary material: ``fetch_supplementary`` downloads a paper's
supplementary-files ZIP and keeps only the document files whose text the
agent can read (``SUPPLEMENTARY_EXTENSIONS``); figures, videos and data files
are dropped. ``fetch_fulltext_many`` attaches them to each ``Fulltext`` when
asked to, and only for papers whose JATS declares supplementary material.

Optional look-aside cache
-------------------------
Ported from bio-agents' ``libs/literature_search.py``, same env vars, key
scheme and TTLs, so the two can share one cache. Two backends are supported:

  EPMC_CACHE_REDIS_URL — Redis URL (e.g. redis://redis:6379/0).  Used by k8s
                         deployments where multiple replicas share one Redis pod.
  EPMC_CACHE_DIR       — Directory path for a diskcache (SQLite) cache.  Used
                         for local runs and evals.

Selection order: EPMC_CACHE_REDIS_URL takes priority over EPMC_CACHE_DIR.
If neither is set the cache is disabled — no crash, no warning after the first.
``diskcache`` is a dependency; ``redis`` is not, install it to use that backend.

Three cache layers share EPMC_CACHE_TTL_DAYS (default 60 days):
  * Search-result cache  — keyed by SHA-256(query + page_size).
  * Full-text cache      — keyed by normalised PMCID, stores the XML *or* the
                           failure that fetching it produced (see below).
  * Supplementary cache  — keyed by normalised PMCID, stores the document files
                           kept from the paper's supplementary ZIP, or the
                           failure that downloading it produced.

Failures are cached too, because ~15% of the papers Europe PMC flags as open
access have no fetchable ``fullTextXML``. A permanent failure (404/410/…) is
remembered so later runs skip the request entirely; a transient one (429/5xx,
timeout, connection error) is remembered only as a breadcrumb and retried on the
next read. Negative entries expire on their own, shorter clock
(EPMC_CACHE_NEGATIVE_TTL_DAYS, default 7) so a paper that becomes open access
later is not skipped for the full 60 days.

Set EPMC_CACHE_DEBUG=1 to log cache hits and misses to stderr.
"""

import base64
import hashlib
import io
import json
import os
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib.parse import quote

import requests

try:
    import diskcache as _dc

    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False

try:
    import redis as _redis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Europe PMC full-text fetch endpoint (PMC id -> JATS XML).
_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
# Europe PMC supplementary-files endpoint (PMC id -> ZIP of every supplementary
# file, plus the article's own figure images).
_SUPPLEMENTARY_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
)
# Supplementary files kept from the ZIP: documents whose text can be extracted.
SUPPLEMENTARY_EXTENSIONS = (".pdf", ".docx", ".doc")
# Concurrent supplementary ZIP downloads per batch: network-bound, and kept
# small so a batch stays polite to Europe PMC.
_SUPPLEMENTARY_WORKERS = 4
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

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


# ── Look-aside cache ─────────────────────────────────────────────────────────

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
    """Log a single warning when the cache backend errors (e.g. redis is down).

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


class _RedisBackend:
    """Thin wrapper around redis.Redis that stores values as zlib-deflated JSON.

    Full-text bodies dominate this cache and are highly compressible (~5.5x
    measured on a real Europe PMC full text), so deflating them multiplies
    effective capacity by the same factor for one CPU-cheap call per access.
    The client is created without ``decode_responses``, so values round-trip
    as bytes.
    """

    def __init__(self, client) -> None:
        self._client = client

    def get(self, key: str):
        # A cache outage must degrade to a miss, never break retrieval: any
        # backend error (redis down/unreachable/timeout) is treated as "no hit"
        # so the caller falls through to the live Europe PMC API.
        try:
            raw = self._client.get(key)
        except Exception as exc:
            _warn_cache_unavailable(exc)
            return None
        if raw is None:
            return None
        # Entries written before compression landed are plain JSON: zlib.error
        # identifies one, and it is read as-is rather than discarded.
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def set(self, key: str, value, expire_seconds: int) -> None:
        # Best-effort write: if the backend is down, skip caching rather than
        # propagate the error up into the search path.
        encoded_json = json.dumps(value).encode("utf-8")
        try:
            self._client.set(key, zlib.compress(encoded_json), ex=expire_seconds)
        except Exception as exc:
            _warn_cache_unavailable(exc)


def _get_cache_backend() -> "_DiskCacheBackend | _RedisBackend | None":
    """Return the active cache backend, creating it on first call.

    Selection order: EPMC_CACHE_REDIS_URL → EPMC_CACHE_DIR → None.
    """
    global _cache_backend, _cache_warned
    if _cache_backend is not None:
        return _cache_backend

    redis_url = os.environ.get("EPMC_CACHE_REDIS_URL", "").strip()
    if redis_url:
        if not _REDIS_AVAILABLE:
            if not _cache_warned:
                print(
                    "EPMC cache: EPMC_CACHE_REDIS_URL is set but redis-py is not installed — cache disabled.",
                    file=sys.stderr,
                )
                _cache_warned = True
            return None
        client = _redis.Redis.from_url(redis_url)
        _cache_backend = _RedisBackend(client)
        return _cache_backend

    cache_dir = os.environ.get("EPMC_CACHE_DIR", "").strip()
    if cache_dir:
        if not _DISKCACHE_AVAILABLE:
            if not _cache_warned:
                print(
                    "EPMC cache: EPMC_CACHE_DIR is set but diskcache is not installed — cache disabled.",
                    file=sys.stderr,
                )
                _cache_warned = True
            return None
        # NB: diskcache.Cache's `timeout` kwarg is the SQLite busy-timeout in
        # seconds (how long to retry on "database is locked"), not a cache
        # TTL — leave it at diskcache's own default (60s). Entry lifetime is
        # controlled separately via EPMC_CACHE_TTL_DAYS below.
        _cache_backend = _DiskCacheBackend(_dc.Cache(cache_dir))
        return _cache_backend

    if not _cache_warned:
        print(
            "EPMC cache: neither EPMC_CACHE_REDIS_URL nor EPMC_CACHE_DIR is set — cache disabled. "
            "Set one of these env vars to enable persistent caching.",
            file=sys.stderr,
        )
        _cache_warned = True
    return None


def _cache_debug(message: str) -> None:
    if os.environ.get("EPMC_CACHE_DEBUG") == "1":
        print(f"[epmc-cache] {message}", file=sys.stderr)


def _ttl_seconds(ok: bool) -> int:
    """Entry lifetime: successes on EPMC_CACHE_TTL_DAYS, failures on the shorter
    EPMC_CACHE_NEGATIVE_TTL_DAYS, so a paper under embargo is not hidden for two
    months after it became fetchable."""
    ttl_env = "EPMC_CACHE_TTL_DAYS" if ok else "EPMC_CACHE_NEGATIVE_TTL_DAYS"
    return int(float(os.environ.get(ttl_env, "60" if ok else "7")) * 86400)


def _cache_key(query: str, page_size: int) -> str:
    """Stable cache key: SHA-256 of query + page_size (bio-agents' key scheme)."""
    raw = json.dumps({"q": query, "n": page_size}, sort_keys=True)
    return "epmc:" + hashlib.sha256(raw.encode()).hexdigest()


def _paper_to_cache_entry(paper: Dict[str, Any]) -> Dict[str, Any]:
    """A paper dict as bio-agents stores a search hit: page_content + metadata."""
    metadata = {k: v for k, v in paper.items() if k != "abstract"}
    content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    return {"page_content": content, "metadata": metadata}


def _paper_from_cache_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of ``_paper_to_cache_entry``."""
    content = str(entry.get("page_content") or "")
    abstract = content.split("Abstract:", 1)[1].strip() if "Abstract:" in content else ""
    return {**entry["metadata"], "abstract": abstract}


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
    """Search Europe PMC and return up to ``page_size`` structured paper dicts.

    Served from the look-aside cache when the same query and page size were
    searched before (see the module docstring).
    """
    backend = _get_cache_backend()
    key = _cache_key(query, page_size)
    if backend is not None:
        hit = backend.get(key)
        if hit is not None:
            _cache_debug(f"search hit  key={key[:16]} q_preview={query[:80]}")
            return [_paper_from_cache_entry(entry) for entry in hit]
        _cache_debug(f"search miss key={key[:16]} q_preview={query[:80]}")

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
    papers = [_parse_result(r) for r in results]
    if backend is not None:
        backend.set(key, [_paper_to_cache_entry(p) for p in papers], _ttl_seconds(True))
    return papers


@dataclass
class Fulltext:
    """One full-text fetch outcome: the JATS XML, or why it is missing.

    ``supplementary`` holds the paper's supplementary document files
    (filename -> bytes) when they were requested and found;
    ``supplementary_error`` says why they are missing when their fetch failed.
    Neither affects ``ok``: a paper whose supplements fail keeps its body text.
    """

    pmcid: str
    xml: str = ""
    error: str = ""
    status: Optional[int] = None
    from_cache: bool = False
    supplementary: Dict[str, bytes] = field(default_factory=dict)
    supplementary_error: str = ""

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


def _fulltext_cache_key(pmcid: str) -> str:
    """Stable cache key for a full-text entry: literal normalised PMCID."""
    return "epmc:ft:" + normalize_pmcid(pmcid)


def _fulltext_from_entry(pmcid: str, entry: Any) -> Optional[Fulltext]:
    """Rebuild a ``Fulltext`` from a cache entry, or None if it is unusable.

    Tolerates the legacy format (a bare XML string), which pre-dates negative
    caching and is still sitting in bio-agents' caches under a 60-day TTL.
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


def _get_cached_fulltext(pmcid: str) -> Optional[Fulltext]:
    """Look up a cached full-text outcome for *pmcid* (None on miss/disabled)."""
    backend = _get_cache_backend()
    if backend is None:
        return None
    hit = backend.get(_fulltext_cache_key(pmcid))
    cached = _fulltext_from_entry(pmcid, hit) if hit is not None else None
    state = "miss"
    if cached is not None:
        state = "hit " if cached.ok else f"neg({cached.status})"
    _cache_debug(f"ft {state} pmcid={pmcid}")
    return cached


def _set_cached_fulltext(result: Fulltext) -> None:
    """Persist a fetch outcome (success or failure); no-op if the cache is off."""
    backend = _get_cache_backend()
    if backend is None:
        return
    entry = {"xml": result.xml, "status": result.status, "error": result.error}
    backend.set(_fulltext_cache_key(result.pmcid), entry, _ttl_seconds(result.ok))


def _attempt_fulltext(pmcid: str) -> Fulltext:
    """One GET for *pmcid*, with no caching and no retry."""
    try:
        response = requests.get(_FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
        # Without this a 404 page would come back as if it were the XML.
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
    either way. With the cache off, every call is one live request.

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


def declares_supplementary(xml: str) -> bool:
    """True when a JATS document lists any supplementary material.

    A cheap text check, used to skip the ZIP request for the (many) papers
    that have nothing to download.
    """
    return "<supplementary-material" in xml or "<inline-supplementary-material" in xml


class SupplementaryTooLarge(ValueError):
    """A supplementary ZIP over the size cap: re-downloading cannot change that."""


def fetch_supplementary(pmcid: str, max_bytes: int) -> Dict[str, bytes]:
    """Download one paper's supplementary ZIP; return its document files.

    Only entries ending in ``SUPPLEMENTARY_EXTENSIONS`` are kept, keyed by
    their base filename. A ZIP larger than ``max_bytes`` is abandoned mid-
    download, and an entry whose uncompressed size exceeds ``max_bytes`` is
    skipped, so one huge supplement cannot stall or exhaust a run.

    :param pmcid: A PMC id in any form (see ``normalize_pmcid``).
    :param max_bytes: Size cap for the ZIP and for each file inside it.
    :return: Document files by filename (empty when there are none).
    :rtype: Dict[str, bytes]
    :raises requests.exceptions.RequestException: on a failed request.
    :raises SupplementaryTooLarge: when the ZIP exceeds ``max_bytes``.
    :raises ValueError: when the download is not a ZIP.
    """
    pmcid = normalize_pmcid(pmcid)
    with requests.get(
        _SUPPLEMENTARY_URL.format(pmcid=pmcid), timeout=120, stream=True
    ) as response:
        response.raise_for_status()
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > max_bytes:
            raise SupplementaryTooLarge(
                f"supplementary ZIP is {declared} bytes (cap {max_bytes})"
            )
        buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1 << 16):
            buffer.write(chunk)
            if buffer.tell() > max_bytes:
                raise SupplementaryTooLarge(
                    f"supplementary ZIP exceeds cap of {max_bytes} bytes"
                )
    try:
        archive = zipfile.ZipFile(buffer)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"supplementary download is not a ZIP: {exc}") from exc

    files: Dict[str, bytes] = {}
    with archive:
        for info in archive.infolist():
            name = info.filename.rsplit("/", 1)[-1]
            if info.is_dir() or not name.lower().endswith(SUPPLEMENTARY_EXTENSIONS):
                continue
            if info.file_size > max_bytes:
                continue
            files[name] = archive.read(info)
    return files


def _supplementary_cache_key(pmcid: str) -> str:
    """Stable cache key for a supplementary entry: literal normalised PMCID."""
    return "epmc:supp:" + normalize_pmcid(pmcid)


def _get_cached_supplementary(pmcid: str, max_bytes: int) -> Optional[Dict[str, Any]]:
    """Cached supplementary outcome for *pmcid*, or None on miss/disabled.

    An entry written under a different size cap is a miss: the cap decides both
    which files were kept and whether the ZIP counted as too large.
    """
    backend = _get_cache_backend()
    if backend is None:
        return None
    hit = backend.get(_supplementary_cache_key(pmcid))
    if not isinstance(hit, dict) or hit.get("max_bytes") != max_bytes:
        hit = None
    state = "miss"
    if hit is not None:
        state = "hit " if not hit["error"] else f"neg({hit['status']})"
    _cache_debug(f"supp {state} pmcid={pmcid}")
    return hit


def _attempt_supplementary(pmcid: str, max_bytes: int) -> Dict[str, Any]:
    """One supplementary download for *pmcid*, as a cache entry; never raises."""
    entry: Dict[str, Any] = {"files": {}, "status": 200, "error": "", "permanent": False}
    try:
        files = fetch_supplementary(pmcid, max_bytes)
    except SupplementaryTooLarge as exc:
        entry.update(status=None, error=str(exc), permanent=True)
    except (requests.exceptions.RequestException, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        entry.update(
            status=status,
            error=f"{status or type(exc).__name__}: {exc}",
            permanent=status in _PERMANENT_FULLTEXT_STATUSES,
        )
    else:
        # base64 so the Redis backend, which stores JSON, can hold the bytes.
        entry["files"] = {
            name: base64.b64encode(data).decode("ascii") for name, data in files.items()
        }
    return entry


def _attach_supplementary(fulltext: Fulltext, max_bytes: int) -> None:
    """Fill ``fulltext.supplementary`` in place through the cache.

    The same decision as ``fetch_fulltext``: a cached success or permanent
    failure (404/410/…, or a ZIP over the cap) is used as-is; a cached transient
    failure (timeout, 5xx, a download that is not a ZIP) is retried now; a
    permanent HTTP verdict is confirmed by a second request before it is cached.
    A failure is recorded on ``supplementary_error`` instead of raised.
    """
    pmcid = fulltext.pmcid
    entry = _get_cached_supplementary(pmcid, max_bytes)
    if entry is None or (entry["error"] and not entry["permanent"]):
        entry = _attempt_supplementary(pmcid, max_bytes)
        if entry["error"] and entry["permanent"] and entry["status"] is not None:
            time.sleep(_PERMANENT_CONFIRM_DELAY_S)
            entry = _attempt_supplementary(pmcid, max_bytes)
        backend = _get_cache_backend()
        if backend is not None:
            backend.set(
                _supplementary_cache_key(pmcid),
                {**entry, "max_bytes": max_bytes},
                _ttl_seconds(not entry["error"]),
            )
    fulltext.supplementary = {
        name: base64.b64decode(data) for name, data in entry["files"].items()
    }
    fulltext.supplementary_error = entry["error"]


def fetch_fulltext_many(
    pmcids: Iterable[str],
    include_supplementary: bool = False,
    max_supplementary_bytes: int = 50_000_000,
) -> Dict[str, Fulltext]:
    """Fetch several full texts one after another, keyed by normalised PMC id.

    Blank ids are dropped and repeats are fetched once. With
    ``include_supplementary``, every fetched paper whose JATS declares
    supplementary material then has its document files downloaded (a few
    papers at a time) onto ``Fulltext.supplementary``.

    :param pmcids: PMC ids in any form; blanks are dropped.
    :type pmcids: Iterable[str]
    :param include_supplementary: Also fetch supplementary document files.
    :type include_supplementary: bool
    :param max_supplementary_bytes: Size cap per ZIP and per file inside it.
    :type max_supplementary_bytes: int
    :return: One ``Fulltext`` per distinct id, successes and failures alike.
    :rtype: Dict[str, Fulltext]
    """
    unique = dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
    fulltexts = {pmcid: fetch_fulltext(pmcid) for pmcid in unique}
    if include_supplementary:
        wanted = [
            f for f in fulltexts.values() if f.ok and declares_supplementary(f.xml)
        ]
        if wanted:
            with ThreadPoolExecutor(max_workers=_SUPPLEMENTARY_WORKERS) as pool:
                list(
                    pool.map(
                        lambda f: _attach_supplementary(f, max_supplementary_bytes),
                        wanted,
                    )
                )
    return fulltexts


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
    and ``error``, so any object with those attributes will do. It also reads
    ``supplementary`` and ``supplementary_error`` when they exist; a source
    without them simply contributes no supplementary text.
    """

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Run one literature query; return its paper dicts."""
        ...

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Fetch the full texts for a batch of PMC ids."""
        ...


class EuropePmcSource:
    """Default source: plain Europe PMC REST calls, through the optional
    look-aside cache (off unless EPMC_CACHE_REDIS_URL or EPMC_CACHE_DIR is set).

    :param include_supplementary: Also download each paper's supplementary
        document files alongside its full text.
    :param max_supplementary_bytes: Size cap per supplementary ZIP and file.
    """

    def __init__(
        self,
        include_supplementary: bool = False,
        max_supplementary_bytes: int = 50_000_000,
    ):
        self.include_supplementary = include_supplementary
        self.max_supplementary_bytes = max_supplementary_bytes

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Search Europe PMC (see ``search_scientific_literature_structured``)."""
        return search_scientific_literature_structured(query, page_size=page_size)

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Fetch full texts sequentially (see ``fetch_fulltext_many``)."""
        return fetch_fulltext_many(
            pmcids,
            include_supplementary=self.include_supplementary,
            max_supplementary_bytes=self.max_supplementary_bytes,
        )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Telomere shortening in aging"
    print(f"Searching Europe PMC for: {q!r}")
    hits = search_scientific_literature_structured(q, page_size=5)
    print(f"Found {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']} ({hit['year']}) - PMID: {hit['pmid']}")

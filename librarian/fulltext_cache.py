"""Opt-in look-aside full-text cache in front of any ``LiteratureSource``.

``CachedSource`` wraps a source and serves cached full-text outcomes before
asking it for the rest; supplementary-file downloads (only made when
``supplementary_enrichment`` is on) are cached the same way, under their own key. It stores them in a ``CacheBackend``; ``DiskCacheBackend``
is the one shipped here, behind the ``cache`` extra (``uv sync --extra cache`` in
a checkout, or ``pip install "librarian[cache] @ <commit tarball URL>"``).
``default_literature_source`` is what ``LibrarianAgent`` uses when given no
source: plain Europe PMC, or Europe PMC behind the disk cache when
``LIBRARIAN_CACHE_DIR`` names a directory.
"""

import dataclasses
import functools
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Protocol

from librarian.literature_search import (
    EuropePmcSource,
    Fulltext,
    LiteratureSource,
    Supplementary,
    SupplementaryOptions,
    normalize_pmcid,
)
from librarian.supplementary import EXTRACTOR_VERSION

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    """Key/value store ``CachedSource`` keeps fetch outcomes in.

    Values are plain JSON-able dicts. A backend may raise; ``CachedSource``
    treats a failed ``get`` as a miss and drops a failed ``set``, so a cache
    outage degrades to live requests instead of failing the run.
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
        import diskcache  # optional extra: only a disk cache needs it

        # diskcache's `timeout` kwarg is the SQLite busy-timeout, not a TTL;
        # entry lifetime is the expire_seconds passed to set().
        self._cache = diskcache.Cache(directory)
        # Sub-query threads write concurrently once their fetches return, and
        # parallel SQLite writes can raise "database is locked"; serialise only
        # the writes, reads and requests stay parallel.
        self._write_lock = threading.Lock()

    def get(self, key: str) -> Any:
        """The stored value for *key*, or None on a miss."""
        return self._cache.get(key)

    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        """Store *value* under *key* for *expire_seconds*."""
        with self._write_lock:
            self._cache.set(key, value, expire=expire_seconds)


def _fulltext_cache_key(pmcid: str) -> str:
    # Literal normalised PMCID: one paper, one key, however its id was spelled.
    return "epmc:ft:" + normalize_pmcid(pmcid)


def _made_under(options: SupplementaryOptions) -> Dict[str, Any]:
    # What a cached supplementary entry is valid for.
    return {**dataclasses.asdict(options), "extractor_version": EXTRACTOR_VERSION}


def _supplementary_cache_key(pmcid: str) -> str:
    # Same scheme as full texts, separate namespace.
    return "epmc:supp:" + normalize_pmcid(pmcid)


class CachedSource:
    """A ``LiteratureSource`` that puts a look-aside cache in front of another.

    ``search`` goes straight to the wrapped source. For full texts, successes
    and permanent failures (400/403/404/410, already confirmed by
    ``fetch_fulltext``) are cached; permanent failures too, because ~15% of the
    papers Europe PMC flags as open access have no fetchable full text.
    Transient failures (429/5xx, timeout, connection error) and empty 200
    bodies are not cached, so the next read fetches them again; if the backend
    holds one anyway (written by another program), it is read as a miss.

    Failures expire on a shorter clock than successes, so a paper that becomes
    open access later is not skipped for the full success TTL.

    Supplementary outcomes follow the same rules: a success (even one with no
    readable documents) and a permanent failure (400/403/404/410, or a ZIP over
    the size cap) are cached; transient ones are not. What is cached is the
    extracted records, not the files: about 1.5% of their size, and no
    re-parsing on a hit. Each entry records the ``SupplementaryOptions`` and
    ``supplementary.EXTRACTOR_VERSION`` it was made under, and is a miss under
    any other, so changing a setting or the extractors never serves stale records.
    """

    def __init__(
        self,
        inner: LiteratureSource,
        backend: CacheBackend,
        ttl_days: float = 60,
        negative_ttl_days: float = 7,
    ) -> None:
        """Wrap *inner* with *backend*.

        :param inner: The source cache misses are fetched from.
        :type inner: LiteratureSource
        :param backend: Where outcomes are stored (e.g. ``DiskCacheBackend``).
        :type backend: CacheBackend
        :param ttl_days: Lifetime of a cached success.
        :type ttl_days: float
        :param negative_ttl_days: Lifetime of a cached permanent failure.
        :type negative_ttl_days: float
        """
        self.inner = inner
        self.backend = backend
        self.ttl_days = ttl_days
        self.negative_ttl_days = negative_ttl_days
        self._warned = False

    def _warn_once(self, exc: Exception) -> None:
        # One warning per source, so a broken cache doesn't spam every fetch.
        if not self._warned:
            logger.warning("full-text cache unavailable (%s); fetching live", exc)
            self._warned = True

    def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
        """Search through the wrapped source; search is not cached."""
        return self.inner.search(query, page_size)

    def _get(self, pmcid: str) -> Optional[Fulltext]:
        # The cached outcome if it can answer without a request, else None.
        try:
            entry = self.backend.get(_fulltext_cache_key(pmcid))
        except Exception as exc:  # down/corrupt backend -> treat as a miss
            self._warn_once(exc)
            return None
        if not isinstance(entry, dict):
            logger.debug("full-text cache miss pmcid=%s", pmcid)
            return None
        cached = Fulltext(
            pmcid=pmcid,
            xml=str(entry.get("xml") or ""),
            error=str(entry.get("error") or ""),
            status=entry.get("status"),
            from_cache=True,
        )
        # Only what _set would store is a hit. Another writer sharing the
        # backend may have stored a transient failure or an empty 200; those
        # are fetched again rather than served until they expire.
        usable = (cached.ok and bool(cached.xml)) or cached.permanent
        logger.debug("full-text cache %s pmcid=%s", "hit" if usable else "stale", pmcid)
        return cached if usable else None

    def _set(self, result: Fulltext) -> None:
        # Best-effort write under the outcome's TTL; a failed write is dropped.
        ttl_days = self.ttl_days if result.ok else self.negative_ttl_days
        entry = {"xml": result.xml, "status": result.status, "error": result.error}
        try:
            self.backend.set(
                _fulltext_cache_key(result.pmcid), entry, int(ttl_days * 86400)
            )
        except Exception as exc:  # never break retrieval over a cache write
            self._warn_once(exc)

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> Dict[str, Fulltext]:
        """Serve what the cache holds, fetch the rest from the wrapped source.

        :param pmcids: PMC ids in any form; blanks are dropped.
        :type pmcids: Iterable[str]
        :return: One ``Fulltext`` per distinct id, successes and failures
            alike, in first-seen order; cached ones have ``from_cache`` set.
        :rtype: Dict[str, Fulltext]
        """
        unique = list(
            dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
        )
        results: Dict[str, Fulltext] = {}
        misses: List[str] = []
        # Cache reads run serially before the fetch; each is a local or
        # one-round-trip lookup. Move them into the pool if they show in traces.
        for pmcid in unique:
            cached = self._get(pmcid)
            if cached is not None:
                results[pmcid] = cached
            else:
                misses.append(pmcid)
        if misses:
            for pmcid, result in self.inner.fetch_fulltext_many(misses).items():
                # An empty 200 is not worth 60 days; let the next read retry it.
                if (result.ok and result.xml) or result.permanent:
                    self._set(result)
                results[pmcid] = result
        # First-seen input order, cached and fetched ids interleaved as given.
        return {pmcid: results[pmcid] for pmcid in unique}

    def _get_supplementary(
        self, pmcid: str, options: SupplementaryOptions
    ) -> Optional[Supplementary]:
        # The cached outcome if it can answer without a request, else None.
        try:
            entry = self.backend.get(_supplementary_cache_key(pmcid))
        except Exception as exc:  # down/corrupt backend -> treat as a miss
            self._warn_once(exc)
            return None
        if not isinstance(entry, dict) or entry.get("made_under") != _made_under(
            options
        ):
            logger.debug("supplementary cache miss pmcid=%s", pmcid)
            return None
        cached = Supplementary(
            pmcid=pmcid,
            records=list(entry.get("records") or []),
            error=str(entry.get("error") or ""),
            status=entry.get("status"),
            too_large=bool(entry.get("too_large")),
            from_cache=True,
        )
        # As for full texts: only what _set_supplementary would store is a hit.
        usable = cached.ok or cached.permanent
        logger.debug(
            "supplementary cache %s pmcid=%s", "hit" if usable else "stale", pmcid
        )
        return cached if usable else None

    def _set_supplementary(
        self, result: Supplementary, options: SupplementaryOptions
    ) -> None:
        # Best-effort write under the outcome's TTL; a failed write is dropped.
        ttl_days = self.ttl_days if result.ok else self.negative_ttl_days
        entry = {
            "records": result.records,
            "status": result.status,
            "error": result.error,
            "too_large": result.too_large,
            "made_under": _made_under(options),
        }
        try:
            self.backend.set(
                _supplementary_cache_key(result.pmcid), entry, int(ttl_days * 86400)
            )
        except Exception as exc:  # never break retrieval over a cache write
            self._warn_once(exc)

    def fetch_supplementary_many(
        self, pmcids: Iterable[str], options: SupplementaryOptions
    ) -> Dict[str, Supplementary]:
        """Serve cached supplementary outcomes, fetch the rest from the wrapped source.

        A wrapped source without ``fetch_supplementary_many`` contributes
        nothing, so its misses are simply left out of the result.

        :param pmcids: PMC ids in any form; blanks are dropped.
        :type pmcids: Iterable[str]
        :param options: Size budget, record cap and PDF extractor, passed to
            the source and checked against cached entries.
        :type options: SupplementaryOptions
        :return: One ``Supplementary`` per distinct id the cache or source
            answered, in first-seen order; cached ones have ``from_cache`` set.
        :rtype: Dict[str, Supplementary]
        """
        unique = list(
            dict.fromkeys(normalize_pmcid(p) for p in pmcids if str(p or "").strip())
        )
        results: Dict[str, Supplementary] = {}
        misses: List[str] = []
        for pmcid in unique:
            cached = self._get_supplementary(pmcid, options)
            if cached is not None:
                results[pmcid] = cached
            else:
                misses.append(pmcid)
        fetch = getattr(self.inner, "fetch_supplementary_many", None)
        if misses and fetch is not None:
            for pmcid, result in fetch(misses, options).items():
                if result.ok or result.permanent:
                    self._set_supplementary(result, options)
                results[pmcid] = result
        return {pmcid: results[pmcid] for pmcid in unique if pmcid in results}


@functools.lru_cache(maxsize=None)
def _source_for(directory: str) -> LiteratureSource:
    # Memoised per directory, fallback included: agents built per request (the
    # orchestrator) share one write lock and SQLite connection, and a broken
    # cache warns once per process instead of once per request.
    try:
        backend = DiskCacheBackend(directory)
    except Exception as exc:  # the cache is an optimisation, never a hard dependency
        logger.warning("full-text cache disabled: cannot open %s (%s)", directory, exc)
        return EuropePmcSource()
    return CachedSource(EuropePmcSource(), backend)


def default_literature_source() -> LiteratureSource:
    """The source ``LibrarianAgent`` uses when the caller passes none.

    Plain Europe PMC requests, unless ``LIBRARIAN_CACHE_DIR`` names a
    directory: then full texts (and supplementary files, when requested) are
    cached on disk there. A cache that cannot
    be opened (read-only directory, ``diskcache`` not installed, ...) is
    logged and skipped, never fatal.

    :return: ``EuropePmcSource``, behind a ``CachedSource`` when the disk
        cache is on.
    :rtype: LiteratureSource
    """
    directory = os.environ.get("LIBRARIAN_CACHE_DIR", "").strip()
    if not directory:
        return EuropePmcSource()
    # Normalised so /x and /x/ share one backend.
    return _source_for(os.path.normpath(os.path.expanduser(directory)))

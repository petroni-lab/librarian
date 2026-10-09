"""Opt-in look-aside full-text cache in front of any ``LiteratureSource``.

``CachedSource`` wraps a source and serves cached full-text outcomes before
asking it for the rest. It stores them in a ``CacheBackend``; ``DiskCacheBackend``
is the one shipped here (``pip install librarian[cache]``).
``default_literature_source`` is what ``LibrarianAgent`` uses when given no
source: plain Europe PMC, or Europe PMC behind the disk cache when
``LIBRARIAN_CACHE_DIR`` names a directory.
"""

import functools
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Protocol

from librarian.literature_search import (
    EuropePmcSource,
    Fulltext,
    LiteratureSource,
    normalize_pmcid,
)

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
        # Concurrent SQLite writes from the fetch pool can raise "database is
        # locked"; serialise only the writes, reads and requests stay parallel.
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


class CachedSource:
    """A ``LiteratureSource`` that puts a look-aside cache in front of another.

    ``search`` goes straight to the wrapped source. For full texts, successes
    and permanent failures (400/403/404/410, already confirmed by
    ``fetch_fulltext``) are cached; permanent failures too, because ~15% of the
    papers Europe PMC flags as open access have no fetchable full text.
    Transient failures (429/5xx, timeout, connection error) are not cached, so
    the next read fetches them again.

    Failures expire on a shorter clock than successes, so a paper that becomes
    open access later is not skipped for the full success TTL.
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
        # The cached outcome, or None on a miss or a broken backend.
        try:
            entry = self.backend.get(_fulltext_cache_key(pmcid))
        except Exception as exc:  # down/corrupt backend -> treat as a miss
            self._warn_once(exc)
            return None
        logger.debug("full-text cache %s pmcid=%s", "hit" if entry else "miss", pmcid)
        if not isinstance(entry, dict):
            return None
        return Fulltext(
            pmcid=pmcid,
            xml=str(entry.get("xml") or ""),
            error=str(entry.get("error") or ""),
            status=entry.get("status"),
        )

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
        :return: One ``Fulltext`` per distinct id, successes and failures alike.
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
                if result.ok or result.permanent:
                    self._set(result)
                results[pmcid] = result
        return results


@functools.lru_cache(maxsize=None)
def _disk_backend(directory: str) -> DiskCacheBackend:
    # One backend per directory per process: agents built per request (the
    # orchestrator) share its write lock and SQLite connection.
    return DiskCacheBackend(directory)


def default_literature_source() -> LiteratureSource:
    """The source ``LibrarianAgent`` uses when the caller passes none.

    Plain Europe PMC requests, unless ``LIBRARIAN_CACHE_DIR`` names a
    directory: then full texts are cached on disk there. A cache that cannot
    be opened (read-only directory, ``diskcache`` not installed, ...) is
    logged and skipped, never fatal.

    :return: ``EuropePmcSource``, behind a ``CachedSource`` when the disk
        cache is on.
    :rtype: LiteratureSource
    """
    directory = os.path.expanduser(os.environ.get("LIBRARIAN_CACHE_DIR", "").strip())
    if not directory:
        return EuropePmcSource()
    try:
        backend = _disk_backend(directory)
    except Exception as exc:  # the cache is an optimisation, never a hard dependency
        logger.warning("full-text cache disabled: cannot open %s (%s)", directory, exc)
        return EuropePmcSource()
    return CachedSource(EuropePmcSource(), backend)

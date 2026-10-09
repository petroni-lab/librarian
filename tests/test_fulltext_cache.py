"""Checks for ``CachedSource``'s cache decision and the default source.

Which outcomes are cached and which are fetched again. Every test runs against
an in-memory backend and a stubbed ``requests.get``, so nothing here touches
Europe PMC.
"""

from typing import Any, Dict, List

import pytest
import requests

from librarian import LibrarianAgent, fulltext_cache, literature_search
from librarian.config import load_runtime_config
from librarian.fulltext_cache import CachedSource, DiskCacheBackend
from librarian.literature_search import EuropePmcSource, Fulltext


class _FakeBackend:
    """In-memory ``CacheBackend`` that also records each entry's TTL."""

    def __init__(self) -> None:
        self.store: Dict[str, Any] = {}
        self.expiries: Dict[str, int] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        self.store[key] = value
        self.expiries[key] = expire_seconds


class _BrokenBackend:
    """A backend whose store is down: every call raises."""

    def get(self, key: str) -> Any:
        raise ConnectionError("cache down")

    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        raise ConnectionError("cache down")


class _FakeResponse:
    """Minimal requests.Response stand-in for the success path."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


def _http_error(status: int) -> requests.exceptions.HTTPError:
    # Shaped like the HTTPError raise_for_status raises.
    response = _FakeResponse("", status_code=status)
    return requests.exceptions.HTTPError(f"{status} Client Error", response=response)


@pytest.fixture(autouse=True)
def no_confirm_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the 404 confirmation pause so the suite stays fast."""
    monkeypatch.setattr(literature_search, "_PERMANENT_CONFIRM_DELAY_S", 0)


@pytest.fixture(autouse=True)
def fresh_default_source() -> None:
    """Forget the per-directory sources memoised by earlier tests."""
    fulltext_cache._source_for.cache_clear()


@pytest.fixture
def backend() -> _FakeBackend:
    """A fresh in-memory backend per test."""
    return _FakeBackend()


@pytest.fixture
def source(backend: _FakeBackend) -> CachedSource:
    """The source under test: Europe PMC behind the in-memory backend."""
    return CachedSource(EuropePmcSource(), backend)


def _stub_get(monkeypatch: pytest.MonkeyPatch, outcomes: List[Any]) -> List[str]:
    """Answer successive GETs from *outcomes* (an exception or XML); the last repeats."""
    requested: List[str] = []

    def fake_get(url: str, timeout: int = 0) -> _FakeResponse:
        requested.append(url)
        outcome = outcomes[min(len(requested), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(literature_search.requests, "get", fake_get)
    return requested


def test_miss_fetches_and_caches(source, backend, monkeypatch):
    """A miss hits the network once and stores the XML under the normalised id."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])

    result = source.fetch_fulltext_many(["pmc123"])["PMC123"]

    assert result.ok
    assert calls == [
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML"
    ]
    assert backend.store["epmc:ft:PMC123"]["xml"] == "<article>body</article>"


def test_cached_success_skips_the_network(source, monkeypatch):
    """A second read of a cached success makes no request."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])
    source.fetch_fulltext_many(["PMC123"])

    result = source.fetch_fulltext_many(["PMC123"])["PMC123"]

    assert result.ok and result.xml == "<article>body</article>"
    assert len(calls) == 1


def test_cached_404_is_skipped_not_retried(source, monkeypatch):
    """A confirmed permanent failure is remembered; later reads never request it."""
    calls = _stub_get(monkeypatch, [_http_error(404)])

    first = source.fetch_fulltext_many(["PMC404"])["PMC404"]
    assert not first.ok and first.permanent
    assert len(calls) == 2, "a 404 is confirmed by a second request before caching"

    second = source.fetch_fulltext_many(["PMC404"])["PMC404"]

    assert not second.ok and second.permanent
    assert len(calls) == 2, "a cached 404 must not be re-requested"


def test_transient_failure_is_not_cached(source, backend, monkeypatch):
    """A 504 leaves no entry, so the next read fetches and caches the success."""
    calls = _stub_get(monkeypatch, [_http_error(504), "<article>body</article>"])

    first = source.fetch_fulltext_many(["PMC504"])["PMC504"]
    assert not first.ok and not first.permanent
    assert "epmc:ft:PMC504" not in backend.store

    second = source.fetch_fulltext_many(["PMC504"])["PMC504"]

    assert second.ok and len(calls) == 2
    assert backend.store["epmc:ft:PMC504"]["xml"] == "<article>body</article>"


def test_negative_entries_expire_sooner(source, backend, monkeypatch):
    """Failures use the shorter TTL so a paper turning open access recovers."""
    _stub_get(monkeypatch, [_http_error(404)])
    source.fetch_fulltext_many(["PMC404"])
    _stub_get(monkeypatch, ["<a/>"])
    source.fetch_fulltext_many(["PMC200"])

    assert backend.expiries["epmc:ft:PMC404"] == 7 * 86400
    assert backend.expiries["epmc:ft:PMC200"] == 60 * 86400


def test_mixed_batch_dedupes_and_only_fetches_misses(source, backend, monkeypatch):
    """Hits come from the cache, each distinct miss costs one request."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])
    backend.store["epmc:ft:PMC1"] = {"xml": "<article>cached</article>", "status": 200}

    results = source.fetch_fulltext_many(["PMC1", "pmc1", "2", "PMC2", "", None])

    assert sorted(results) == ["PMC1", "PMC2"]
    assert results["PMC1"].xml == "<article>cached</article>"
    assert len(calls) == 1


def test_empty_success_is_not_cached(source, backend, monkeypatch):
    """A 200 with an empty body is returned but not stored."""
    _stub_get(monkeypatch, [""])

    assert source.fetch_fulltext_many(["PMC9"])["PMC9"].ok
    assert "epmc:ft:PMC9" not in backend.store


def test_raising_backend_degrades_to_live_fetch(monkeypatch):
    """A backend that raises is a miss and a dropped write, never a failed run."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])
    source = CachedSource(EuropePmcSource(), _BrokenBackend())

    result = source.fetch_fulltext_many(["PMC123"])["PMC123"]

    assert result.ok and result.xml == "<article>body</article>"
    assert len(calls) == 1


def test_wraps_any_source():
    """Misses go to whatever source is wrapped, not to Europe PMC."""

    class _StubSource:
        def __init__(self) -> None:
            self.asked: List[List[str]] = []

        def search(self, query: str, page_size: int) -> List[Dict[str, Any]]:
            return [{"title": query}]

        def fetch_fulltext_many(self, pmcids):
            ids = list(pmcids)
            self.asked.append(ids)
            return {p: Fulltext(pmcid=p, xml="<a/>") for p in ids}

    inner = _StubSource()
    source = CachedSource(inner, _FakeBackend())

    source.fetch_fulltext_many(["PMC1"])
    source.fetch_fulltext_many(["PMC1"])

    assert inner.asked == [["PMC1"]]
    assert source.search("q", 5) == [{"title": "q"}]


def test_disk_backend_round_trips(tmp_path):
    """DiskCacheBackend stores and returns a cache entry unchanged."""
    pytest.importorskip("diskcache")
    disk = DiskCacheBackend(str(tmp_path))
    entry = {"xml": "<a/>", "status": 200, "error": ""}

    disk.set("epmc:ft:PMC1", entry, 60)

    assert disk.get("epmc:ft:PMC1") == entry
    assert disk.get("epmc:ft:PMC2") is None


def test_default_source_is_uncached(monkeypatch):
    """With LIBRARIAN_CACHE_DIR unset the default is plain Europe PMC."""
    monkeypatch.delenv("LIBRARIAN_CACHE_DIR", raising=False)

    assert type(fulltext_cache.default_literature_source()) is EuropePmcSource


def test_cache_dir_turns_on_the_disk_cache(tmp_path, monkeypatch):
    """LIBRARIAN_CACHE_DIR gives Europe PMC behind one shared disk backend."""
    pytest.importorskip("diskcache")
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(tmp_path))
    first = fulltext_cache.default_literature_source()
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(tmp_path) + "/")
    second = fulltext_cache.default_literature_source()

    assert isinstance(first, CachedSource)
    assert isinstance(first.inner, EuropePmcSource)
    assert isinstance(first.backend, DiskCacheBackend)
    assert first.backend is second.backend, "one backend per directory per process"


def test_unopenable_cache_dir_falls_back(tmp_path, monkeypatch):
    """A cache that cannot be created falls back to uncached and warns once."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(blocker / "cache"))
    warnings: List[Any] = []
    monkeypatch.setattr(fulltext_cache.logger, "warning", lambda *a: warnings.append(a))

    for _ in range(3):
        assert type(fulltext_cache.default_literature_source()) is EuropePmcSource
    assert len(warnings) == 1


def test_agent_uses_the_default_source(tmp_path, monkeypatch):
    """LibrarianAgent with no source gets default_literature_source()."""
    pytest.importorskip("diskcache")
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(tmp_path))

    agent = LibrarianAgent(load_runtime_config(), llm_client=object())

    assert isinstance(agent._source, CachedSource)

"""Checks for ``CachedEuropePmcSource``'s cache decision.

Which cached outcomes short-circuit the network and which are retried. Every
test runs against an in-memory backend and a stubbed ``requests.get``, so
nothing here touches Europe PMC.
"""

from typing import Any, Dict, List

import pytest
import requests

from librarian import literature_search
from librarian.literature_search import CachedEuropePmcSource, DiskCacheBackend


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


@pytest.fixture
def backend() -> _FakeBackend:
    """A fresh in-memory backend per test."""
    return _FakeBackend()


@pytest.fixture
def source(backend: _FakeBackend) -> CachedEuropePmcSource:
    """The source under test, on the in-memory backend."""
    return CachedEuropePmcSource(backend)


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

    assert result.ok and not result.from_cache
    assert calls == [
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML"
    ]
    assert backend.store["epmc:ft:PMC123"]["xml"] == "<article>body</article>"


def test_cached_success_skips_the_network(source, monkeypatch):
    """A second read of a cached success makes no request."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])
    source.fetch_fulltext_many(["PMC123"])

    result = source.fetch_fulltext_many(["PMC123"])["PMC123"]

    assert result.ok and result.from_cache and result.xml == "<article>body</article>"
    assert len(calls) == 1


def test_cached_404_is_skipped_not_retried(source, monkeypatch):
    """A confirmed permanent failure is remembered; later reads never request it."""
    calls = _stub_get(monkeypatch, [_http_error(404)])

    first = source.fetch_fulltext_many(["PMC404"])["PMC404"]
    assert not first.ok and first.permanent
    assert len(calls) == 2, "a 404 is confirmed by a second request before caching"

    second = source.fetch_fulltext_many(["PMC404"])["PMC404"]

    assert not second.ok and second.from_cache
    assert len(calls) == 2, "a cached 404 must not be re-requested"


def test_cached_transient_failure_is_retried(source, monkeypatch):
    """A 504 is cached only as a breadcrumb: the next read tries again."""
    calls = _stub_get(monkeypatch, [_http_error(504), "<article>body</article>"])

    first = source.fetch_fulltext_many(["PMC504"])["PMC504"]
    assert not first.ok and not first.permanent

    second = source.fetch_fulltext_many(["PMC504"])["PMC504"]

    assert second.ok and second.xml == "<article>body</article>"
    assert len(calls) == 2


def test_negative_entries_expire_sooner(source, backend, monkeypatch):
    """Failures use the shorter TTL so a paper turning open access recovers."""
    _stub_get(monkeypatch, [_http_error(404)])
    source.fetch_fulltext_many(["PMC404"])
    source.set_cached_fulltext(literature_search.Fulltext(pmcid="PMC200", xml="<a/>"))

    assert backend.expiries["epmc:ft:PMC404"] == 7 * 86400
    assert backend.expiries["epmc:ft:PMC200"] == 60 * 86400


def test_legacy_string_entry_reads_as_a_hit(source, backend, monkeypatch):
    """Entries from before failures were cached are bare XML strings."""
    calls = _stub_get(monkeypatch, ["<article>live</article>"])
    backend.store["epmc:ft:PMC777"] = "<article>legacy</article>"

    result = source.fetch_fulltext_many(["PMC777"])["PMC777"]

    assert result.ok and result.xml == "<article>legacy</article>"
    assert calls == []


def test_mixed_batch_dedupes_and_only_fetches_misses(source, backend, monkeypatch):
    """Hits come from the cache, each distinct miss costs one request."""
    calls = _stub_get(monkeypatch, ["<article>body</article>"])
    backend.store["epmc:ft:PMC1"] = "<article>cached</article>"

    results = source.fetch_fulltext_many(["PMC1", "pmc1", "2", "PMC2", "", None])

    assert sorted(results) == ["PMC1", "PMC2"]
    assert results["PMC1"].from_cache and not results["PMC2"].from_cache
    assert len(calls) == 1


def test_disk_backend_round_trips(tmp_path):
    """DiskCacheBackend stores and returns a cache entry unchanged."""
    disk = DiskCacheBackend(str(tmp_path))
    entry = {"xml": "<a/>", "status": 200, "error": ""}

    disk.set("epmc:ft:PMC1", entry, 60)

    assert disk.get("epmc:ft:PMC1") == entry
    assert disk.get("epmc:ft:PMC2") is None


def test_default_source_caches_on_disk(tmp_path, monkeypatch):
    """With no configuration the agent's default source is the disk cache."""
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(tmp_path))

    source = literature_search.default_literature_source()

    assert isinstance(source, CachedEuropePmcSource)
    assert isinstance(source.backend, DiskCacheBackend)


def test_default_source_off_switch(monkeypatch):
    """LIBRARIAN_CACHE_DIR=off gives plain, uncached Europe PMC requests."""
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", "off")

    source = literature_search.default_literature_source()

    assert not isinstance(source, CachedEuropePmcSource)


def test_default_source_survives_an_unopenable_directory(tmp_path, monkeypatch):
    """A cache that cannot be created falls back to uncached, never raises."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("LIBRARIAN_CACHE_DIR", str(blocker / "cache"))

    source = literature_search.default_literature_source()

    assert not isinstance(source, CachedEuropePmcSource)

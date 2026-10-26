"""CachedSource caches extracted supplementary records (no network).

uv run --extra supplementary pytest tests
"""

from unittest import mock

from librarian.fulltext_cache import CachedSource
from librarian.literature_search import Supplementary, SupplementaryOptions

RECORD = {
    "text": "gene | score ; SOX2 | 3",
    "section_title": "Supplementary: s1.csv",
    "section_type": "supplementary",
    "supplementary_file": "s1.csv",
}


class DictBackend:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, expire_seconds):
        self.store[key] = value


class Inner:
    def __init__(self):
        self.calls = 0

    def search(self, query, page_size):
        return []

    def fetch_fulltext_many(self, pmcids):
        return {}

    def fetch_supplementary_many(self, pmcids, options):
        self.calls += 1
        return {p: Supplementary(p, records=[RECORD], status=200) for p in pmcids}


def test_records_cached_and_invalidated_by_options_and_version():
    inner = Inner()
    source = CachedSource(inner, DictBackend())
    options = SupplementaryOptions(max_bytes=1000, max_records=40)

    source.fetch_supplementary_many(["PMC1"], options)
    hit = source.fetch_supplementary_many(["pmc1"], options)["PMC1"]
    assert hit.from_cache
    assert hit.records == [RECORD]
    assert inner.calls == 1

    source.fetch_supplementary_many(["PMC1"], SupplementaryOptions(1000, 10))
    assert inner.calls == 2  # another record cap: a miss

    with mock.patch("librarian.fulltext_cache.EXTRACTOR_VERSION", 99):
        source.fetch_supplementary_many(["PMC1"], options)
    assert inner.calls == 3  # extractor changed: a miss


def test_an_entry_from_the_previous_extractor_version_is_refetched():
    # Written exactly as the previous revision did: an empty 200 (e.g. a
    # .doc read without a converter) under extractor version 1.
    import dataclasses

    from librarian.fulltext_cache import _supplementary_cache_key

    options = SupplementaryOptions(max_bytes=1000, max_records=40)
    backend = DictBackend()
    backend.store[_supplementary_cache_key("PMC1")] = {
        "records": [],
        "status": 200,
        "error": "",
        "too_large": False,
        "made_under": {**dataclasses.asdict(options), "extractor_version": 1},
    }
    inner = Inner()
    result = CachedSource(inner, backend).fetch_supplementary_many(["PMC1"], options)
    assert not result["PMC1"].from_cache
    assert inner.calls == 1


def test_unreadable_file_is_reported_and_the_outcome_not_cached():
    from librarian.literature_search import EuropePmcSource

    listing = (
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<Contents><Key>PMC1.1/bad.pdf</Key><Size>9</Size></Contents>"
        b"<Contents><Key>PMC1.1/good.txt</Key><Size>9</Size></Contents>"
        b"</ListBucketResult>"
    )

    def get(url, **kwargs):
        response = mock.MagicMock()
        if url.endswith("/"):
            response.content = listing
        else:
            response.content = b"not a pdf" if url.endswith(".pdf") else b"good text"
        return response

    backend = DictBackend()
    source = CachedSource(EuropePmcSource(), backend)
    options = SupplementaryOptions(max_bytes=1000, max_records=40)
    with mock.patch("requests.get", side_effect=get):
        first = source.fetch_supplementary_many(["PMC1"], options)["PMC1"]
        second = source.fetch_supplementary_many(["PMC1"], options)["PMC1"]
    assert [r["text"] for r in first.records] == ["good text"]
    assert first.failed == ["bad.pdf"]
    assert not second.from_cache
    assert backend.store == {}


def test_doc_without_a_converter_is_a_failure_not_an_empty_file():
    from librarian import supplementary

    failed = []
    with mock.patch.object(supplementary.shutil, "which", return_value=None):
        records = supplementary.extract_supplementary_records(
            {"old.doc": b"binary"}, {}, 40, failed
        )
    assert (records, failed) == ([], ["old.doc"])


def test_a_file_skipped_by_the_size_policy_is_not_a_failure():
    from librarian import supplementary

    failed = []
    with mock.patch.object(
        supplementary,
        "_check_package_size",
        side_effect=supplementary.PackageTooLarge("too big"),
    ):
        supplementary.extract_supplementary_records({"t.docx": b"x"}, {}, 40, failed)
    assert failed == []

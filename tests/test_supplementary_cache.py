"""CachedSource caches extracted supplementary records (no network).

uv run python -m unittest discover tests
"""

import unittest
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


class SupplementaryRecordCacheTest(unittest.TestCase):
    def test_records_cached_and_invalidated_by_options_and_version(self):
        inner = Inner()
        source = CachedSource(inner, DictBackend())
        options = SupplementaryOptions(max_bytes=1000, max_records=40)

        source.fetch_supplementary_many(["PMC1"], options)
        hit = source.fetch_supplementary_many(["pmc1"], options)["PMC1"]
        self.assertTrue(hit.from_cache)
        self.assertEqual(hit.records, [RECORD])
        self.assertEqual(inner.calls, 1)

        source.fetch_supplementary_many(["PMC1"], SupplementaryOptions(1000, 10))
        self.assertEqual(inner.calls, 2)  # another record cap: a miss

        with mock.patch("librarian.fulltext_cache.EXTRACTOR_VERSION", 99):
            source.fetch_supplementary_many(["PMC1"], options)
        self.assertEqual(inner.calls, 3)  # extractor changed: a miss

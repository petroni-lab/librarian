"""CachedSource serves a full text as it was fetched, its PDF origin included (no network).

uv run --extra supplementary pytest tests
"""

import pytest

from librarian.fulltext_cache import CachedSource
from librarian.literature_search import Fulltext


class DictBackend:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, expire_seconds):
        self.store[key] = value


class Inner:
    def __init__(self, from_pdf):
        self.from_pdf = from_pdf

    def search(self, query, page_size):
        return []

    def fetch_fulltext_many(self, pmcids):
        return {
            p: Fulltext(p, xml="<article/>", status=200, from_pdf=self.from_pdf)
            for p in pmcids
        }


@pytest.mark.parametrize("from_pdf", [True, False])
def test_a_cache_hit_keeps_whether_the_text_was_built_from_a_pdf(from_pdf):
    source = CachedSource(Inner(from_pdf), DictBackend())
    source.fetch_fulltext_many(["PMC1"])

    cached = source.fetch_fulltext_many(["PMC1"])["PMC1"]

    assert cached.from_cache is True
    assert cached.from_pdf is from_pdf

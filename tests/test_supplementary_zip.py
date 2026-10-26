"""Regression tests for supplementary downloads (no network).

uv run --extra supplementary pytest tests
"""

import io
import zipfile
from unittest import mock

import docx
import requests

from librarian import supplementary
from librarian.fulltext_cache import CachedSource
from librarian.literature_search import (
    EuropePmcSource,
    SupplementaryOptions,
    _fetch_supplementary_zip,
    fetch_supplementary,
    fetch_supplementary_outcome,
)


def _zip(files, compression=zipfile.ZIP_DEFLATED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _response(body):
    """A stand-in for a ``requests.get`` response, streamed or not."""
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Length": str(len(body))}
    response.iter_content.return_value = [body]
    response.content = body
    return response


def _listing(sizes, next_token=None):
    """An S3 ListObjectsV2 reply for ``{key: size}``, truncated if *next_token*."""
    items = "".join(
        f"<Contents><Key>{key}</Key><Size>{size}</Size></Contents>"
        for key, size in sizes.items()
    )
    if next_token:
        items += (
            "<IsTruncated>true</IsTruncated>"
            f"<NextContinuationToken>{next_token}</NextContinuationToken>"
        )
    return _response(
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + items.encode()
        + b"</ListBucketResult>"
    )


def _not_found():
    response = _response(b"")
    response.status_code = 404
    response.raise_for_status.side_effect = requests.HTTPError(
        "404 Not Found", response=response
    )
    return response


class DictBackend:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, expire_seconds):
        self.store[key] = value


def test_extracted_size_is_capped_across_files():
    # 20 highly compressible 900 KB files: a tiny ZIP, 18 MB uncompressed.
    body = _zip({f"s{i}.pdf": b"0" * 900_000 for i in range(20)})
    assert len(body) < 1_000_000
    with mock.patch("requests.get", return_value=_response(body)):
        files = _fetch_supplementary_zip("PMC1", max_bytes=1_000_000)
    assert sum(map(len, files.values())) <= 1_000_000


def test_corrupt_member_fails_only_the_supplement():
    body = bytearray(_zip({"s1.pdf": b"A" * 1000}, zipfile.ZIP_STORED))
    body[body.index(b"A" * 1000)] ^= 0xFF  # break the member's CRC
    listing = _listing({})  # not in the PMC bucket: falls back to the ZIP
    with mock.patch("requests.get", side_effect=[listing, _response(bytes(body))]):
        result = fetch_supplementary_outcome(
            "PMC1", SupplementaryOptions(max_bytes=1_000_000, max_records=40)
        )
    assert not result.ok
    assert result.records == []


def test_latest_version_documents_within_budget():
    listing = _listing(
        {
            "PMC1.1/old.pdf": 10,
            "PMC1.2/PMC1.2.pdf": 10,  # PMC's own rendition of the article
            "PMC1.2/fig1.jpg": 10,
            "PMC1.2/a.docx": 600,
            "PMC1.2/b.pdf": 600,  # would exceed the 1000-byte budget
            "PMC1.2/c.xlsx": 300,
        }
    )
    responses = {"a.docx": b"a" * 600, "c.xlsx": b"c" * 300}

    def get(url, **kwargs):
        if url.endswith("/"):
            return listing
        return _response(responses[url.rsplit("/", 1)[1]])

    with mock.patch("requests.get", side_effect=get) as calls:
        files = fetch_supplementary("PMC1", max_bytes=1000)
    assert sorted(files) == ["a.docx", "c.xlsx"]
    assert calls.call_count == 3  # listing + two downloads


def test_every_listing_page_is_read_before_choosing_the_version():
    # The latest version only appears on the second page of the listing.
    first = _listing({"PMC1.1/old.txt": 5}, next_token="page2")
    second = _listing({"PMC1.2/new.txt": 5})

    def get(url, params=None, **kwargs):
        if url.endswith("/"):
            return second if params.get("continuation-token") == "page2" else first
        return _response(b"new")

    with mock.patch("requests.get", side_effect=get):
        files = fetch_supplementary("PMC1", max_bytes=1000)
    assert files == {"new.txt": b"new"}


def test_one_failed_file_keeps_the_others_and_is_not_cached():
    listing = _listing({"PMC1.1/a.txt": 5, "PMC1.1/b.txt": 5})

    def get(url, **kwargs):
        if url.endswith("/"):
            return listing
        return _not_found() if url.endswith("b.txt") else _response(b"alpha")

    source = CachedSource(EuropePmcSource(), DictBackend())
    options = SupplementaryOptions(max_bytes=1000, max_records=40)
    with mock.patch("requests.get", side_effect=get) as calls:
        first = source.fetch_supplementary_many(["PMC1"], options)["PMC1"]
        second = source.fetch_supplementary_many(["PMC1"], options)["PMC1"]
    assert first.ok
    assert [r["supplementary_file"] for r in first.records] == ["a.txt"]
    assert first.failed == ["b.txt"]
    assert not second.from_cache  # incomplete: fetched again, not cached
    assert source.backend.store == {}
    assert calls.call_count == 6  # (listing + 2 files) x 2


def test_docx_unpacking_past_the_budget_is_not_parsed():
    # A .docx is a ZIP too: ~2 MB of text compresses to a few KB, so the
    # outer download budget alone would let it through.
    document = docx.Document()
    document.add_paragraph("gene " * 400_000)
    buffer = io.BytesIO()
    document.save(buffer)
    outer = _zip({"s1.docx": buffer.getvalue()})
    assert len(outer) < 100_000

    with mock.patch("requests.get", return_value=_response(outer)):
        files = _fetch_supplementary_zip("PMC1", max_bytes=1_000_000)
    assert "s1.docx" in files
    with (
        mock.patch.object(supplementary, "_MAX_PACKAGE_BYTES", 1_000_000),
        mock.patch("docx.Document", side_effect=AssertionError("parsed")),
    ):
        records = supplementary.extract_supplementary_records(files, {}, 40)
    assert records == []

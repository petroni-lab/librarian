"""Regression tests for supplementary downloads (no network).

uv run python -m unittest discover tests
"""

import io
import unittest
import zipfile
from unittest import mock

from librarian.literature_search import (
    _fetch_supplementary_zip,
    fetch_supplementary,
    SupplementaryOptions,
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


def _listing(sizes):
    """An S3 ListObjectsV2 reply for ``{key: size}``."""
    items = "".join(
        f"<Contents><Key>{key}</Key><Size>{size}</Size></Contents>"
        for key, size in sizes.items()
    )
    return _response(
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + items.encode()
        + b"</ListBucketResult>"
    )


class SupplementaryZipTest(unittest.TestCase):
    def test_extracted_size_is_capped_across_files(self):
        # 20 highly compressible 900 KB files: a tiny ZIP, 18 MB uncompressed.
        body = _zip({f"s{i}.pdf": b"0" * 900_000 for i in range(20)})
        self.assertLess(len(body), 1_000_000)
        with mock.patch("requests.get", return_value=_response(body)):
            files = _fetch_supplementary_zip("PMC1", max_bytes=1_000_000)
        self.assertLessEqual(sum(map(len, files.values())), 1_000_000)

    def test_corrupt_member_fails_only_the_supplement(self):
        body = bytearray(_zip({"s1.pdf": b"A" * 1000}, zipfile.ZIP_STORED))
        body[body.index(b"A" * 1000)] ^= 0xFF  # break the member's CRC
        listing = _listing({})  # not in the PMC bucket: falls back to the ZIP
        with mock.patch("requests.get", side_effect=[listing, _response(bytes(body))]):
            result = fetch_supplementary_outcome(
                "PMC1", SupplementaryOptions(max_bytes=1_000_000, max_records=40)
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.records, [])


class SupplementaryAwsTest(unittest.TestCase):
    def test_latest_version_documents_within_budget(self):
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
        self.assertEqual(sorted(files), ["a.docx", "c.xlsx"])
        self.assertEqual(calls.call_count, 3)  # listing + two downloads

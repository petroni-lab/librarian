"""Regression tests for supplementary ZIP handling (no network).

uv run python -m unittest discover tests
"""

import io
import unittest
import zipfile
from unittest import mock

import docx

from librarian import supplementary
from librarian.literature_search import fetch_supplementary, fetch_supplementary_outcome


def _zip(files, compression=zipfile.ZIP_DEFLATED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _response(body):
    """A stand-in for the streamed ``requests.get`` response."""
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Length": str(len(body))}
    response.iter_content.return_value = [body]
    return response


class SupplementaryZipTest(unittest.TestCase):
    def test_extracted_size_is_capped_across_files(self):
        # 20 highly compressible 900 KB files: a tiny ZIP, 18 MB uncompressed.
        body = _zip({f"s{i}.pdf": b"0" * 900_000 for i in range(20)})
        self.assertLess(len(body), 1_000_000)
        with mock.patch("requests.get", return_value=_response(body)):
            files = fetch_supplementary("PMC1", max_bytes=1_000_000)
        self.assertLessEqual(sum(map(len, files.values())), 1_000_000)

    def test_corrupt_member_fails_only_the_supplement(self):
        body = bytearray(_zip({"s1.pdf": b"A" * 1000}, zipfile.ZIP_STORED))
        body[body.index(b"A" * 1000)] ^= 0xFF  # break the member's CRC
        with mock.patch("requests.get", return_value=_response(bytes(body))):
            result = fetch_supplementary_outcome("PMC1", max_bytes=1_000_000)
        self.assertFalse(result.ok)
        self.assertEqual(result.files, {})


class NestedArchiveTest(unittest.TestCase):
    def test_docx_unpacking_past_the_budget_is_not_parsed(self):
        # A .docx is a ZIP too: ~2 MB of text compresses to a few KB, so the
        # outer download budget alone would let it through.
        document = docx.Document()
        document.add_paragraph("gene " * 400_000)
        buffer = io.BytesIO()
        document.save(buffer)
        outer = _zip({"s1.docx": buffer.getvalue()})
        self.assertLess(len(outer), 100_000)

        with mock.patch("requests.get", return_value=_response(outer)):
            files = fetch_supplementary("PMC1", max_bytes=1_000_000)
        self.assertIn("s1.docx", files)
        with (
            mock.patch.object(supplementary, "_MAX_PACKAGE_BYTES", 1_000_000),
            mock.patch("docx.Document", side_effect=AssertionError("parsed")),
        ):
            records = supplementary.extract_supplementary_records(files, {}, 40)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()

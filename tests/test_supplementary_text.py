"""Plain-text supplementary files → records (no network).

uv run python -m unittest discover tests
"""

import unittest

from librarian.supplementary import extract_supplementary_records


class SupplementaryTextTest(unittest.TestCase):
    def test_txt_and_md_split_on_blank_lines_and_merge_short_paragraphs(self):
        long = " ".join(["word"] * 90)
        text = f"# Methods\r\n\r\nShort line.\n\n{long}\n\n\nTail paragraph.".encode()
        records = extract_supplementary_records(
            {"notes.md": text, "readme.txt": b"\xef\xbb\xbfplain text"}, {}, 40
        )
        by_file = {}
        for record in records:
            by_file.setdefault(record["supplementary_file"], []).append(record["text"])
        # Short blocks merge up to _MIN_BLOCK_WORDS; the BOM is dropped.
        self.assertEqual(by_file["notes.md"][0], f"# Methods Short line. {long}")
        self.assertEqual(by_file["notes.md"][1], "Tail paragraph.")
        self.assertEqual(by_file["readme.txt"], ["plain text"])

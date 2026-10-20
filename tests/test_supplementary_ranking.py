"""A paper's supplementary records are ranked per sub-query before the cap.

uv run python -m unittest discover tests
"""

import unittest

from test_supplementary_tables import _stage2_pool

from librarian import agent


def _paragraph(i, extra=""):
    # 85 words, so every paragraph is its own record (blocks merge below 80).
    return " ".join([f"filler{i}"] * 85) + extra


class RankBeforeCapTest(unittest.TestCase):
    def test_a_late_relevant_record_survives_the_per_paper_cap(self):
        paragraphs = [_paragraph(i) for i in range(60)]
        paragraphs[55] = _paragraph(55, " zebrafish knockout GATTACA")
        files = {"s1.txt": "\n\n".join(paragraphs).encode()}
        _, chunks = _stage2_pool(files, subquery="zebrafish knockout")
        self.assertLessEqual(len(chunks), 40)
        self.assertTrue(any("GATTACA" in c["text"] for c in chunks))

    def test_records_are_cached_with_the_safety_limit_not_the_cap(self):
        from librarian import LibrarianAgent, load_runtime_config

        librarian = LibrarianAgent(load_runtime_config(), llm_client=object())
        self.assertEqual(
            librarian._supplementary_options.max_records,
            agent._SUPPLEMENTARY_RECORD_SAFETY_LIMIT,
        )

"""BM25 keywords keep a query's content words and drop negations and filters."""

import unittest

from librarian.agent import _strip_epmc_operators


class StripEpmcOperatorsTest(unittest.TestCase):
    def test_plain_query_unchanged(self) -> None:
        self.assertEqual(
            _strip_epmc_operators("CRISPR cancer therapy"), "CRISPR cancer therapy"
        )

    def test_fields_quotes_and_operators_dropped(self) -> None:
        query = '(TITLE_ABS:"gene editing" OR TITLE_ABS:CRISPR) AND tumor'
        self.assertEqual(_strip_epmc_operators(query), "gene editing CRISPR tumor")

    def test_negated_terms_dropped(self) -> None:
        cases = {
            "cancer NOT p53": "cancer",
            "cancer AND NOT TITLE:mouse": "cancer",
            'cancer NOT "cell line"': "cancer",
            "cancer NOT (mouse OR rat) therapy": "cancer therapy",
            "-mouse cancer -toxicity": "cancer",
            "covid-19 vaccine": "covid-19 vaccine",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(_strip_epmc_operators(query), expected)

    def test_filter_values_dropped(self) -> None:
        query = (
            "(aspirin AND PUB_YEAR:[2020 TO 2024]) HAS_FT:y SRC:MED OPEN_ACCESS:y "
            'LICENSE:"CC BY" LANG:eng FIRST_PDATE:[2024-01-01 TO *] PUB_YEAR:2021'
        )
        self.assertEqual(_strip_epmc_operators(query), "aspirin")


if __name__ == "__main__":
    unittest.main()

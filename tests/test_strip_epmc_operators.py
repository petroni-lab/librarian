"""BM25 keywords keep a query's content words and drop negations and filters."""

import pytest

from librarian.agent import _strip_epmc_operators


def test_plain_query_unchanged() -> None:
    assert _strip_epmc_operators("CRISPR cancer therapy") == "CRISPR cancer therapy"


def test_fields_quotes_and_operators_dropped() -> None:
    query = '(TITLE_ABS:"gene editing" OR TITLE_ABS:CRISPR) AND tumor'
    assert _strip_epmc_operators(query) == "gene editing CRISPR tumor"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cancer NOT p53", "cancer"),
        ("cancer AND NOT TITLE:mouse", "cancer"),
        ('cancer NOT "cell line"', "cancer"),
        ("cancer NOT (mouse OR rat) therapy", "cancer therapy"),
        ("-mouse cancer -toxicity", "cancer"),
        ("covid-19 vaccine", "covid-19 vaccine"),
    ],
)
def test_negated_terms_dropped(query: str, expected: str) -> None:
    assert _strip_epmc_operators(query) == expected


def test_filter_values_dropped() -> None:
    query = (
        "(aspirin AND PUB_YEAR:[2020 TO 2024]) HAS_FT:y SRC:MED OPEN_ACCESS:y "
        'LICENSE:"CC BY" LANG:eng FIRST_PDATE:[2024-01-01 TO *] PUB_YEAR:2021'
    )
    assert _strip_epmc_operators(query) == "aspirin"

"""A selected paper's supplementary records are ranked by the question before the cap.

uv run --extra supplementary pytest tests
"""

from test_supplementary_tables import _supplementary_pool

from librarian import agent


def _paragraph(i, extra=""):
    # 85 words, so every paragraph is its own record (blocks merge below 80).
    return " ".join([f"filler{i}"] * 85) + extra


def test_a_late_relevant_record_survives_the_per_paper_cap():
    paragraphs = [_paragraph(i) for i in range(60)]
    paragraphs[55] = _paragraph(55, " zebrafish knockout GATTACA")
    files = {"s1.txt": "\n\n".join(paragraphs).encode()}
    _, chunks = _supplementary_pool(files, question="zebrafish knockout")
    assert len(chunks) <= 40
    assert any("GATTACA" in c["text"] for c in chunks)


def test_records_are_cached_with_the_safety_limit_not_the_cap():
    from librarian import LibrarianAgent, load_runtime_config

    librarian = LibrarianAgent(load_runtime_config(), llm_client=object())
    assert (
        librarian._supplementary_options.max_records
        == agent._SUPPLEMENTARY_RECORD_SAFETY_LIMIT
    )


def test_records_are_not_modified_when_none_are_dropped():
    import copy

    from librarian.literature_search import Supplementary

    records = [
        {
            "text": "only finding.",
            "section_type": "supplementary",
            "section_title": "Supplementary: s.txt",
            "supplementary_file": "s.txt",
        }
    ]
    outcome = Supplementary("PMC1", records=records, status=200)
    cached = copy.deepcopy(records)
    top = agent._top_supplementary("finding", outcome, 40)
    top.records[0]["source_paragraph_index"] = 7
    assert outcome.records == cached


# bm25s returns tied scores in reverse index order; ties must keep document order.
def _records(n, extra=None):
    extra = extra or {}
    return [
        {
            "text": f"Table S1 row {i}: cohort {i} | value {i * 3}{extra.get(i, '')}",
            "section_title": "Supplementary: s.xlsx",
            "section_type": "supplementary",
            "supplementary_file": "s.xlsx",
        }
        for i in range(n)
    ]


def test_no_matching_record_keeps_the_first_k_not_the_last():
    from librarian.literature_search import Supplementary

    outcome = Supplementary("PMC1", records=_records(100), status=200)
    top = agent._top_supplementary("never smokers lung adenocarcinoma", outcome, 40)
    assert [r["supplementary_index"] for r in top.records] == list(range(40))


def test_matching_records_come_first_then_the_rest_in_document_order():
    from librarian.literature_search import Supplementary

    records = _records(100, extra={90: " never smokers"})
    outcome = Supplementary("PMC1", records=records, status=200)
    top = agent._top_supplementary("never smokers", outcome, 5)
    assert [r["supplementary_index"] for r in top.records] == [0, 1, 2, 3, 90]


def test_body_fallback_returns_the_leading_paragraphs():
    records = [
        {"text": f"paragraph {i} about cells", "section_type": "results"}
        for i in range(10)
    ]
    top = agent._rank_paragraphs_for_subquery("never smokers", records, 3)
    assert [r["text"] for r in top] == [f"paragraph {i} about cells" for i in range(3)]

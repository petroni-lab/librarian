"""A paper's supplementary records are ranked per sub-query before the cap.

uv run --extra supplementary pytest tests
"""

from test_supplementary_tables import _stage2_pool

from librarian import agent


def _paragraph(i, extra=""):
    # 85 words, so every paragraph is its own record (blocks merge below 80).
    return " ".join([f"filler{i}"] * 85) + extra


def test_a_late_relevant_record_survives_the_per_paper_cap():
    paragraphs = [_paragraph(i) for i in range(60)]
    paragraphs[55] = _paragraph(55, " zebrafish knockout GATTACA")
    files = {"s1.txt": "\n\n".join(paragraphs).encode()}
    _, chunks = _stage2_pool(files, subquery="zebrafish knockout")
    assert len(chunks) <= 40
    assert any("GATTACA" in c["text"] for c in chunks)


def test_records_are_cached_with_the_safety_limit_not_the_cap():
    from librarian import LibrarianAgent, load_runtime_config

    librarian = LibrarianAgent(load_runtime_config(), llm_client=object())
    assert (
        librarian._supplementary_options.max_records
        == agent._SUPPLEMENTARY_RECORD_SAFETY_LIMIT
    )


# Different sub-queries keep different subsets of the same paper's records.
def test_cited_findings_survive_different_subquery_subsets():
    import copy
    from dataclasses import replace

    from librarian import LibrarianAgent, load_runtime_config
    from librarian.literature_search import Fulltext, Supplementary

    texts = ["alpha finding.", "shared alpha beta finding.", "beta finding."]
    records = [
        {
            "text": text,
            "section_title": "Supplementary: s.txt",
            "section_type": "supplementary",
            "supplementary_file": "s.txt",
        }
        for text in texts
    ]
    outcome = Supplementary("PMC1", records=records, status=200)
    cached = copy.deepcopy(outcome.records)
    config = replace(
        load_runtime_config(),
        supplementary_enrichment=True,
        max_supplementary_records_per_paper=2,
    )
    librarian = LibrarianAgent(config, llm_client=object())
    paper = {"pmid": "1", "pmcid": "PMC1", "isOpenAccess": True}
    fulltexts = {
        "PMC1": Fulltext("PMC1", xml="<article><body><p>body</p></body></article>")
    }
    selected = []
    for query in ("alpha", "beta"):
        top = agent._top_supplementary(query, outcome, 2)
        chunks = librarian._paragraph_records_for_paper(paper, fulltexts, {"PMC1": top})
        selected.extend(c for c in chunks if c["section_type"] == "supplementary")
    merged = agent._merge_selected_paragraphs(selected)
    registry, ids = {}, []
    for i, chunk in enumerate(merged):
        items = agent._build_sentence_items(
            str(i),
            "1",
            chunk["source_paragraph_index"],
            chunk["word_start"],
            chunk["text"],
            registry,
            section_title=chunk["section_title"],
            section_type=chunk["section_type"],
        )
        ids.extend(item["id"] for item in items)
    # Simulate a judge citing every supplied sentence, without any LLM call.
    result = librarian._papers_from_cited_sentences(ids, registry, {"1": dict(paper)})
    assert set(result[0]["evidence_sentences_full_text"]) == set(texts)
    assert len(merged) == 3, "the shared paragraph must be merged once"
    assert outcome.records == cached, "the cached outcome must not change"


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

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


class StableCoordinatesTest(unittest.TestCase):
    """Different sub-queries keep different subsets of the same paper's records."""

    def test_cited_findings_survive_different_subquery_subsets(self):
        import copy
        from dataclasses import replace

        from librarian import LibrarianAgent, load_runtime_config
        from librarian.literature_search import Fulltext, Supplementary

        texts = ["alpha finding.", "shared alpha beta finding.", "beta finding."]
        records = [
            dict(
                text=text,
                section_title="Supplementary: s.txt",
                section_type="supplementary",
                supplementary_file="s.txt",
            )
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
        paper = dict(pmid="1", pmcid="PMC1", isOpenAccess=True)
        fulltexts = {
            "PMC1": Fulltext("PMC1", xml="<article><body><p>body</p></body></article>")
        }
        selected = []
        for query in ("alpha", "beta"):
            top = agent._top_supplementary(query, outcome, 2)
            chunks = librarian._paragraph_records_for_paper(
                paper, fulltexts, {"PMC1": top}
            )
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
        result = librarian._papers_from_cited_sentences(
            ids, registry, {"1": dict(paper)}
        )
        self.assertEqual(set(result[0]["evidence_sentences_full_text"]), set(texts))
        self.assertEqual(len(merged), 3, "the shared paragraph must be merged once")
        self.assertEqual(outcome.records, cached, "the cached outcome must not change")

    def test_records_are_not_modified_when_none_are_dropped(self):
        import copy

        from librarian.literature_search import Supplementary

        records = [
            dict(
                text="only finding.",
                section_type="supplementary",
                section_title="Supplementary: s.txt",
                supplementary_file="s.txt",
            )
        ]
        outcome = Supplementary("PMC1", records=records, status=200)
        cached = copy.deepcopy(records)
        top = agent._top_supplementary("finding", outcome, 40)
        top.records[0]["source_paragraph_index"] = 7
        self.assertEqual(outcome.records, cached)

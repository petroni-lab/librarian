"""Docling PDF extraction, with a stand-in for Docling (no models, no network).

uv run python -m unittest discover tests
"""

import sys
import types
import unittest
from unittest import mock

from librarian import supplementary
from librarian.supplementary import extract_supplementary_records


def _item(page, text):
    return types.SimpleNamespace(text=text, prov=[types.SimpleNamespace(page_no=page)])


class _Frame:
    def __init__(self, rows):
        self.columns, *body = rows
        self.values = types.SimpleNamespace(tolist=lambda: body)


class _Table:
    def __init__(self, page, rows):
        self.prov = [types.SimpleNamespace(page_no=page)]
        self._frame = _Frame(rows)

    def export_to_dataframe(self, doc=None):
        return self._frame


class FakeConverter:
    def __init__(self, document=None, error=None):
        self.document, self.error, self.calls = document, error, []

    def convert(self, source, page_range):
        self.calls.append(page_range)
        if self.error:
            raise self.error
        return types.SimpleNamespace(document=self.document)


def _run(converter, max_records=40, failed=None, late_pages=()):
    fake_docling = types.ModuleType("docling.datamodel.base_models")
    fake_docling.DocumentStream = lambda name, stream: name
    with (
        mock.patch.dict(sys.modules, {"docling.datamodel.base_models": fake_docling}),
        mock.patch.object(supplementary, "_docling_converter", return_value=converter),
        mock.patch.object(supplementary, "_pdf_texts", return_value=iter(late_pages)),
    ):
        return extract_supplementary_records(
            {"s.pdf": b"%PDF"}, {}, max_records, failed, pdf_extractor="docling"
        )


class DoclingTest(unittest.TestCase):
    def test_tables_come_in_page_order_so_the_cap_reaches_them(self):
        texts = [_item(p, f"page {p} text " * 20) for p in range(1, 51)]
        table = _Table(2, [["gene", "log2FC", "padj"], ["A", "", "0.01"]])
        converter = FakeConverter(types.SimpleNamespace(texts=texts, tables=[table]))
        records = _run(converter, max_records=5)
        titles = [r["section_title"] for r in records]
        self.assertEqual(
            titles,
            [f"Supplementary: s.pdf (p. {p})" for p in (1, 2)]
            + ["Supplementary: s.pdf (p. 2, table)"]
            + [f"Supplementary: s.pdf (p. {p})" for p in (3, 4)],
        )
        # Empty cells keep their column in Docling tables too.
        self.assertEqual(records[2]["text"], "gene | log2FC | padj ; A |  | 0.01")

    def test_only_the_first_pages_are_converted_and_later_ones_read_as_text(self):
        converter = FakeConverter(
            types.SimpleNamespace(texts=[_item(1, "p1")], tables=[])
        )
        records = _run(converter, late_pages=[(21, "late page text")])
        self.assertEqual(converter.calls, [(1, supplementary._DOCLING_MAX_PAGES)])
        self.assertEqual(
            [r["section_title"] for r in records],
            ["Supplementary: s.pdf (p. 1)", "Supplementary: s.pdf (p. 21)"],
        )

    def test_a_failed_conversion_is_reported_as_failed(self):
        failed = []
        records = _run(
            FakeConverter(error=RuntimeError("model download failed")), failed=failed
        )
        self.assertEqual((records, failed), ([], ["s.pdf"]))


class DoclingStage2Test(unittest.TestCase):
    """A Docling table's long cell through the agent's Stage-2 pool, every chunk."""

    def test_a_long_cell_stays_in_its_column_in_every_chunk(self):
        from test_supplementary_tables import _stage2_pool

        description = " ".join(f"word{i}" for i in range(600))
        table = _Table(
            1, [["gene", "score", "description"], ["SOX2", "3", description]]
        )
        converter = FakeConverter(types.SimpleNamespace(texts=[], tables=[table]))
        fake_docling = types.ModuleType("docling.datamodel.base_models")
        fake_docling.DocumentStream = lambda name, stream: name
        modules = {
            "docling": types.ModuleType("docling"),
            "docling.datamodel.base_models": fake_docling,
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                supplementary, "_docling_converter", return_value=converter
            ),
            mock.patch.object(supplementary, "_pdf_texts", return_value=iter(())),
        ):
            max_words, chunks = _stage2_pool(
                {"s.pdf": b"%PDF"}, supplementary_pdf_extractor="docling"
            )
        self.assertGreater(len(chunks), 1)
        continued = []
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"].split()), max_words)
            self.assertTrue(chunk["section_title"].endswith("(p. 1, table)"))
            header, row = chunk["text"].split(" ; ", 1)
            self.assertEqual(header, "gene | score | description")
            cells = row.split(" | ")
            self.assertEqual(len(cells), 3, chunk["text"][:180])
            self.assertEqual(cells[0], "SOX2")
            self.assertIn(cells[1], ("3", ""))
            continued += cells[2].split()
        self.assertEqual(continued, description.split())

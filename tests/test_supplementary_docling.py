"""Docling PDF extraction in an isolated worker process (no Docling models needed).

    uv run python -m unittest discover tests

The worker is replaced by tiny stand-in scripts run as real child processes, so
a crash (SIGSEGV), a hang and missing output are exercised the way a native
Docling failure would happen.
"""

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from librarian import _docling_worker, supplementary
from librarian.supplementary import extract_supplementary_records

CRASH = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
SILENT = "pass"
ECHO = """
import json, os, sys
open(os.environ["WORKER_ARGS"], "w").write(" ".join(sys.argv[1:]))
open(sys.argv[3], "w").write(os.environ["WORKER_PAYLOAD"])
"""
HANG = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(os.environ["WORKER_PIDS"], "w").write(f"{os.getpid()} {child.pid}")
time.sleep(120)
"""


def _worker(code):
    return mock.patch.object(
        supplementary, "_DOCLING_WORKER", (sys.executable, "-c", code)
    )


def _echo(payload, folder):
    args = os.path.join(folder, "args")
    env = {"WORKER_PAYLOAD": json.dumps(payload), "WORKER_ARGS": args}
    return _worker(ECHO), mock.patch.dict(os.environ, env), args


def _extract(files, max_records=40, failed=None):
    return extract_supplementary_records(
        files, {}, max_records, failed, pdf_extractor="docling"
    )


def _gone(pid, within=5.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


class DoclingRecordsTest(unittest.TestCase):
    def test_tables_come_in_page_order_so_the_cap_reaches_them(self):
        payload = {
            "texts": [[p, f"page {p} text " * 20] for p in range(1, 51)],
            "tables": [[2, [["gene", "log2FC", "padj"], ["A", "", "0.01"]]]],
        }
        with tempfile.TemporaryDirectory() as folder:
            worker, env, _ = _echo(payload, folder)
            with (
                worker,
                env,
                mock.patch.object(supplementary, "_pdf_texts", return_value=iter(())),
            ):
                records = _extract({"s.pdf": b"%PDF"}, max_records=5)
        self.assertEqual(
            [r["section_title"] for r in records],
            [f"Supplementary: s.pdf (p. {p})" for p in (1, 2)]
            + ["Supplementary: s.pdf (p. 2, table)"]
            + [f"Supplementary: s.pdf (p. {p})" for p in (3, 4)],
        )
        self.assertEqual(records[2]["text"], "gene | log2FC | padj ; A |  | 0.01")

    def test_only_the_first_pages_are_converted_and_later_ones_read_as_text(self):
        payload = {"texts": [[1, "p1"]], "tables": []}
        with tempfile.TemporaryDirectory() as folder:
            worker, env, args = _echo(payload, folder)
            late = mock.patch.object(
                supplementary, "_pdf_texts", return_value=iter([(21, "late page text")])
            )
            with worker, env, late:
                records = _extract({"s.pdf": b"%PDF"})
            source, max_pages, _ = Path(args).read_text().split(" ")
        self.assertEqual(max_pages, str(supplementary._DOCLING_MAX_PAGES))
        self.assertTrue(source.endswith("input.pdf"))
        self.assertEqual(
            [r["section_title"] for r in records],
            ["Supplementary: s.pdf (p. 1)", "Supplementary: s.pdf (p. 21)"],
        )

    def test_the_worker_maps_a_docling_document_to_texts_and_table_rows(self):
        def item(page, text):
            return types.SimpleNamespace(
                text=text, prov=[types.SimpleNamespace(page_no=page)]
            )

        frame = types.SimpleNamespace(
            columns=["gene", "score"],
            values=types.SimpleNamespace(
                tolist=lambda: [["A", None], ["B", float("nan")]]
            ),
        )
        table = types.SimpleNamespace(
            prov=[types.SimpleNamespace(page_no=3)],
            export_to_dataframe=lambda doc: frame,
        )
        document = types.SimpleNamespace(
            texts=[item(1, "intro"), types.SimpleNamespace(text="no page", prov=[])],
            tables=[table],
        )
        self.assertEqual(
            _docling_worker.payload(document),
            {
                "texts": [[1, "intro"]],
                "tables": [[3, [["gene", "score"], ["A", ""], ["B", ""]]]],
            },
        )


class DoclingIsolationTest(unittest.TestCase):
    def test_a_crashing_worker_fails_only_its_file(self):
        failed = []
        with (
            _worker(CRASH),
            self.assertLogs("librarian.supplementary", "WARNING") as logs,
        ):
            records = _extract(
                {"s.pdf": b"%PDF", "notes.txt": b"good finding"}, failed=failed
            )
        self.assertEqual([r["text"] for r in records], ["good finding"])
        self.assertEqual(failed, ["s.pdf"])
        self.assertIn("s.pdf exited with status -11", " ".join(logs.output))

    def test_a_crash_keeps_the_rest_of_the_paper_in_stage2(self):
        from test_supplementary_tables import _stage2_pool

        with (
            _worker(CRASH),
            mock.patch.dict(sys.modules, {"docling": types.ModuleType("docling")}),
        ):
            _, pool = _stage2_pool(
                {"s.pdf": b"%PDF", "notes.txt": b"good finding"},
                everything=True,
                supplementary_pdf_extractor="docling",
            )
        texts = [r["text"] for r in pool]
        self.assertIn("Unrelated body text.", texts)
        self.assertIn("good finding", texts)

    def test_an_incomplete_outcome_is_not_cached(self):
        from librarian.fulltext_cache import CachedSource
        from librarian.literature_search import EuropePmcSource, SupplementaryOptions

        files = {"s.pdf": b"%PDF", "notes.txt": b"good finding"}
        listing = (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + "".join(
                f"<Contents><Key>PMC1.1/{n}</Key><Size>{len(d)}</Size></Contents>"
                for n, d in files.items()
            )
            + "</ListBucketResult>"
        ).encode()

        def get(url, **kwargs):
            response = mock.MagicMock()
            response.content = (
                listing if url.endswith("/") else files[url.rsplit("/", 1)[1]]
            )
            return response

        class Backend(dict):
            def set(self, key, value, expire_seconds):
                self[key] = value

        backend = Backend()
        options = SupplementaryOptions(
            max_bytes=10_000, max_records=40, pdf_extractor="docling"
        )
        with _worker(CRASH), mock.patch("requests.get", side_effect=get):
            outcome = CachedSource(EuropePmcSource(), backend).fetch_supplementary_many(
                ["PMC1"], options
            )["PMC1"]
        self.assertEqual(outcome.failed, ["s.pdf"])
        self.assertEqual([r["text"] for r in outcome.records], ["good finding"])
        self.assertEqual(dict(backend), {})

    def test_a_hung_worker_and_its_children_are_killed(self):
        with tempfile.TemporaryDirectory() as folder:
            pids = os.path.join(folder, "pids")
            failed = []
            started = time.monotonic()
            with (
                _worker(HANG),
                mock.patch.dict(os.environ, {"WORKER_PIDS": pids}),
                mock.patch.object(supplementary, "_DOCLING_TIMEOUT_S", 2),
                self.assertLogs("librarian.supplementary", "WARNING") as logs,
            ):
                _extract({"s.pdf": b"%PDF"}, failed=failed)
            worker, grandchild = map(int, Path(pids).read_text().split())
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(failed, ["s.pdf"])
        self.assertIn("timed out after 2s on s.pdf", " ".join(logs.output))
        self.assertTrue(_gone(worker) and _gone(grandchild))

    def test_a_worker_without_output_is_a_failure(self):
        failed = []
        with _worker(SILENT):
            records = _extract({"s.pdf": b"%PDF"}, failed=failed)
        self.assertEqual((records, failed), ([], ["s.pdf"]))


class DoclingStage2Test(unittest.TestCase):
    """A Docling table's long cell through the agent's Stage-2 pool, every chunk."""

    def test_a_long_cell_stays_in_its_column_in_every_chunk(self):
        from test_supplementary_tables import _stage2_pool

        description = " ".join(f"word{i}" for i in range(600))
        payload = {
            "texts": [],
            "tables": [
                [1, [["gene", "score", "description"], ["SOX2", "3", description]]]
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            worker, env, _ = _echo(payload, folder)
            with (
                worker,
                env,
                mock.patch.dict(sys.modules, {"docling": types.ModuleType("docling")}),
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

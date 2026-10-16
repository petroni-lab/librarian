"""Supplementary document files → paragraph records.

``extract_supplementary_records`` turns the files ``literature_search``
downloaded for one paper (``.pdf``, ``.docx``, ``.doc``, and ``.csv``,
``.tsv``, ``.xlsx``, ``.xls`` tables) into the same
``{text, section_title, section_type}`` records ``jats.extract_body_paragraphs``
produces for the body, so they join the paper's paragraph pool and go through
chunking, BM25 and the Stage-3 judge unchanged.

Every record has ``section_type = "supplementary"`` and a ``section_title``
naming the file (``"Supplementary: <caption or file name>"``, plus the page for
a PDF or the sheet for a workbook), which is how a cited snippet is traced back
to its supplement. Table rows are rendered like ``jats`` renders article tables
(``a | b ; c | d``), with the header row repeated in every record so a matched
record is readable on its own.
"""

from __future__ import annotations

import csv
import functools
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)
# pypdf warns on every slightly malformed PDF; those are routine here.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPLEMENTARY_SECTION_TYPE = "supplementary"
# Part of every cached supplementary entry's validity: bump it whenever the
# records these extractors produce change, so cached records are re-extracted.
EXTRACTOR_VERSION = 1
# "pypdf" (default: text only, one record per page) or "docling" (layout model:
# tables come out as rows with their header, like spreadsheets; slow, optional).
PDF_EXTRACTORS = ("pypdf", "docling")

# Consecutive short paragraphs (headings, list items, one-line captions) are
# merged until a block reaches this many words, so BM25 does not score a bare
# heading as if it were a paragraph. Longer blocks are split later by the
# agent's own chunker.
_MIN_BLOCK_WORDS = 80


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _blocks(paragraphs: List[str]) -> Iterator[str]:
    """Merge short consecutive paragraphs into blocks of at least ``_MIN_BLOCK_WORDS``."""
    pending: List[str] = []
    words = 0
    for paragraph in paragraphs:
        paragraph = _norm(paragraph)
        if not paragraph:
            continue
        pending.append(paragraph)
        words += len(paragraph.split())
        if words >= _MIN_BLOCK_WORDS:
            yield " ".join(pending)
            pending, words = [], 0
    if pending:
        yield " ".join(pending)


def _pdf_texts(data: bytes) -> Iterator[Tuple[int, str]]:
    """``(page_number, text)`` for every PDF page that has extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    for number, page in enumerate(reader.pages, 1):
        text = _norm(page.extract_text() or "")
        if text:
            yield number, text


@functools.lru_cache(maxsize=1)
def _docling_converter() -> Any:
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


# One Docling conversion at a time: it is CPU/GPU-bound and its thread safety
# is not documented.
_DOCLING_LOCK = threading.Lock()


def _docling_items(name: str, data: bytes) -> Iterator[Tuple[int, str, str]]:
    """``(page, kind, text)`` from Docling: each page's prose, then each table.

    Tables are rendered by ``_sheet_blocks`` (header repeated in every block),
    so a row keeps its column names instead of being flattened into the page.
    """
    from docling.datamodel.base_models import DocumentStream

    with _DOCLING_LOCK:
        document = (
            _docling_converter()
            .convert(DocumentStream(name=name, stream=io.BytesIO(data)))
            .document
        )
    pages: Dict[int, List[str]] = defaultdict(list)
    for item in document.texts:
        if item.prov:
            pages[item.prov[0].page_no].append(item.text)
    for page in sorted(pages):
        text = _norm(" ".join(pages[page]))
        if text:
            yield page, "text", text
    for table in document.tables:
        frame = table.export_to_dataframe(doc=document)
        page = table.prov[0].page_no if table.prov else 0
        for block in _sheet_blocks([list(frame.columns)] + frame.values.tolist()):
            yield page, "table", block


def _docx_paragraphs(data: bytes) -> List[str]:
    """Paragraph texts of a .docx, then its tables rendered ``a | b ; c | d``."""
    import docx

    document = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in document.paragraphs]
    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [_norm(cell.text) for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            paragraphs.append(" ; ".join(rows))
    return paragraphs


def _doc_paragraphs(data: bytes) -> List[str]:
    """Paragraph texts of a legacy binary .doc, via ``antiword`` or macOS ``textutil``.

    Neither is a Python dependency: with neither installed, .doc files are
    skipped (returns ``[]``).
    """
    if shutil.which("antiword"):
        command = ["antiword"]
    elif shutil.which("textutil"):
        command = ["textutil", "-convert", "txt", "-stdout"]
    else:
        logger.debug("no antiword/textutil on PATH; skipping .doc supplement")
        return []
    fd, path = tempfile.mkstemp(suffix=".doc")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        result = subprocess.run(
            command + [path], capture_output=True, timeout=60, check=True
        )
    finally:
        os.unlink(path)
    text = result.stdout.decode("utf-8", errors="replace")
    return re.split(r"\n\s*\n|\r\n\s*\r\n", text)


def _cell(value: Any) -> str:
    """One spreadsheet cell as text: empty for None, 3 not 3.0 for whole numbers."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _norm(str(value))


def _delimited_sheets(data: bytes, delimiter: str) -> List[Tuple[str, List[List[Any]]]]:
    """The one unnamed sheet of a CSV/TSV file."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return [("", list(csv.reader(io.StringIO(text), delimiter=delimiter)))]


def _xlsx_sheets(data: bytes) -> List[Tuple[str, List[List[Any]]]]:
    """Every worksheet of an .xlsx, as (name, rows) with computed cell values."""
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        return [
            (sheet.title, list(sheet.iter_rows(values_only=True))) for sheet in workbook
        ]
    finally:
        workbook.close()


def _xls_sheets(data: bytes) -> List[Tuple[str, List[List[Any]]]]:
    """Every worksheet of a legacy .xls, as (name, rows)."""
    import xlrd

    workbook = xlrd.open_workbook(file_contents=data)
    return [
        (sheet.name, [sheet.row_values(i) for i in range(sheet.nrows)])
        for sheet in workbook.sheets()
    ]


_SHEET_READERS: Dict[str, Callable[[bytes], List[Tuple[str, List[List[Any]]]]]] = {
    ".csv": lambda data: _delimited_sheets(data, ","),
    ".tsv": lambda data: _delimited_sheets(data, "\t"),
    ".xlsx": _xlsx_sheets,
    ".xls": _xls_sheets,
}


def _sheet_blocks(rows: List[List[Any]]) -> Iterator[str]:
    """Rows as ``header ; row ; row`` blocks of about ``_MIN_BLOCK_WORDS`` words.

    The first non-empty row is taken as the header and starts every block, so a
    block BM25 or the judge picks out still says what its columns are.
    """
    header: Optional[str] = None
    pending: List[str] = []
    words = 0
    for row in rows:
        cells = [c for c in (_cell(v) for v in row) if c]
        if not cells:
            continue
        line = " | ".join(cells)
        if header is None:
            header = line
            continue
        pending.append(line)
        words += len(line.split())
        if words >= _MIN_BLOCK_WORDS:
            yield " ; ".join([header] + pending)
            pending, words = [], 0
    if pending:
        yield " ; ".join([header] + pending)


_PARAGRAPH_EXTRACTORS: Dict[str, Callable[[bytes], List[str]]] = {
    ".docx": _docx_paragraphs,
    ".doc": _doc_paragraphs,
}


def _file_records(
    name: str, data: bytes, label: str, pdf_extractor: str
) -> Iterator[Dict[str, str]]:
    """Records for one supplementary file; nothing for an unsupported type."""
    extension = os.path.splitext(name)[1].lower()
    if extension == ".pdf" and pdf_extractor == "docling":
        for page, kind, text in _docling_items(name, data):
            suffix = f"p. {page}, table" if kind == "table" else f"p. {page}"
            yield {
                "text": text,
                "section_title": f"{label} ({suffix})",
                "section_type": SUPPLEMENTARY_SECTION_TYPE,
                "supplementary_file": name,
            }
        return
    if extension == ".pdf":
        for page, text in _pdf_texts(data):
            yield {
                "text": text,
                "section_title": f"{label} (p. {page})",
                "section_type": SUPPLEMENTARY_SECTION_TYPE,
                "supplementary_file": name,
            }
        return
    reader = _SHEET_READERS.get(extension)
    if reader is not None:
        for sheet, rows in reader(data):
            title = f'{label}, sheet "{sheet}"' if sheet else label
            for block in _sheet_blocks(rows):
                yield {
                    "text": block,
                    "section_title": title,
                    "section_type": SUPPLEMENTARY_SECTION_TYPE,
                    "supplementary_file": name,
                }
        return
    extractor = _PARAGRAPH_EXTRACTORS.get(extension)
    if extractor is None:
        return
    for block in _blocks(extractor(data)):
        yield {
            "text": block,
            "section_title": label,
            "section_type": SUPPLEMENTARY_SECTION_TYPE,
            "supplementary_file": name,
        }


def require_extraction_dependencies(pdf_extractor: str = "pypdf") -> None:
    """Raise a clear ``ImportError`` when the ``supplementary`` extra is missing.

    The PDF/Word/spreadsheet readers are imported lazily inside a per-file ``except
    Exception``, which would otherwise turn a missing package into silently
    empty supplements.
    """
    try:
        import docx  # noqa: F401
        import openpyxl  # noqa: F401
        import pypdf  # noqa: F401
        import xlrd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "supplementary_enrichment needs the `supplementary` extra: "
            "`uv sync --extra supplementary` or "
            '`pip install "librarian[supplementary]"`'
        ) from exc
    if pdf_extractor == "docling":
        try:
            import docling  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'supplementary_pdf_extractor = "docling" needs the `docling` '
                "package: `uv sync --extra docling` or `pip install docling`"
            ) from exc


def apply_captions(
    records: List[Dict[str, str]], captions: Dict[str, str]
) -> List[Dict[str, str]]:
    """Records relabelled ``Supplementary: <caption> [<file>]`` where the paper's
    JATS captions their file; extraction labels them by file name alone, so
    records can be cached independently of the article XML.
    """
    labelled = []
    for record in records:
        name = record["supplementary_file"]
        caption = captions.get(name)
        if caption:
            title = record["section_title"].replace(
                f"Supplementary: {name}", f"Supplementary: {caption} [{name}]", 1
            )
            record = {**record, "section_title": title}
        labelled.append(record)
    return labelled


def extract_supplementary_records(
    files: Dict[str, bytes],
    captions: Dict[str, str],
    max_records: int,
    pdf_extractor: str = "pypdf",
) -> List[Dict[str, str]]:
    """All supplementary files of one paper → paragraph records, capped.

    The cap is shared round-robin across files (first record of every file,
    then the second, ...), so one long PDF cannot use it all up and leave the
    paper's other supplements unread.

    :param files: File name → bytes, as ``literature_search.fetch_supplementary``
        returns them.
    :param captions: File name → label from the paper's JATS
        (``jats.extract_supplementary_captions``); the file name is always
        appended, since one generic caption ("Supplementary Data") often
        covers every file.
    :param max_records: Most records returned for the paper, so one long
        supplement (a 200-page PDF) cannot crowd the body out of the BM25 pool.
    :param pdf_extractor: One of ``PDF_EXTRACTORS``.
    :return: Records, interleaved across files, in document order within a file.
    :rtype: List[Dict[str, str]]
    """
    per_file: List[List[Dict[str, str]]] = []
    seen: set = set()
    for name in sorted(files):
        records: List[Dict[str, str]] = []
        try:
            for record in _file_records(
                name, files[name], f"Supplementary: {name}", pdf_extractor
            ):
                if record["text"] in seen:
                    continue
                seen.add(record["text"])
                records.append(record)
                if len(records) >= max_records:
                    break
        except Exception as exc:  # a corrupt file must not cost the paper its body
            logger.debug("could not read supplementary file %s: %s", name, exc)
        if records:
            per_file.append(records)

    interleaved: List[Dict[str, str]] = []
    for position in range(max_records):
        for records in per_file:
            if position < len(records):
                interleaved.append(records[position])
    # Sorted back into file then document order: the agent numbers records by
    # position, and reading order keeps adjacent cited pages adjacent.
    kept = interleaved[:max_records]
    order = {id(r): i for i, r in enumerate(r for records in per_file for r in records)}
    return apply_captions(sorted(kept, key=lambda r: order[id(r)]), captions)

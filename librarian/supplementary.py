"""Supplementary document files → paragraph records.

``extract_supplementary_records`` turns the files ``literature_search``
downloaded for one paper (``.pdf``, ``.docx``, ``.doc``, ``.txt``, ``.md``, and
``.csv``, ``.tsv``, ``.xlsx``, ``.xls`` tables) into the same
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
import io
import itertools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)
# pypdf warns on every slightly malformed PDF; those are routine here.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPLEMENTARY_SECTION_TYPE = "supplementary"
# Part of every cached supplementary entry's validity: bump it whenever the
# records these extractors produce change, so cached records are re-extracted.
# 2: failed extractions are no longer cached; table records carry their header
# (table_header/table_rows), keep empty cells, and read every .xls sheet.
EXTRACTOR_VERSION = 2
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


def _pdf_texts(data: bytes, skip: int = 0) -> Iterator[Tuple[int, str]]:
    """``(page_number, text)`` for every PDF page after the first *skip* that has
    extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    for number, page in enumerate(reader.pages[skip:], skip + 1):
        text = _norm(page.extract_text() or "")
        if text:
            yield number, text


# Most bytes an Office document (.docx, .xlsx) may unpack to. These
# files are ZIPs themselves, so the download budget only bounds their compressed
# size: a 40 KB .docx can hold millions of characters, all loaded at once.
_MAX_PACKAGE_BYTES = 25_000_000


class PackageTooLarge(ValueError):
    """An Office document that would unpack past ``_MAX_PACKAGE_BYTES``."""


def _check_package_size(data: bytes) -> None:
    """Refuse an Office document whose parts would unpack past the budget.

    Reads only the ZIP directory (declared sizes), never the parts themselves;
    a part larger than declared fails its CRC check when it is read later.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        size = sum(info.file_size for info in package.infolist())
    if size > _MAX_PACKAGE_BYTES:
        raise PackageTooLarge(f"unpacks to {size} bytes (budget {_MAX_PACKAGE_BYTES})")


# One Docling conversion at a time: it is CPU/GPU-bound.
_DOCLING_LOCK = threading.Lock()


# Pages converted with Docling (~6 s each on CPU, under _DOCLING_LOCK); later
# pages are read as plain text with pypdf, so a 200-page supplement is still
# covered without blocking every supplementary worker for minutes.
_DOCLING_MAX_PAGES = 20
# Docling runs in its own process (_docling_worker.py), so a native crash or a
# hang cannot end the librarian run; a conversion still going after this many
# seconds (model loading included) is killed and its file reported as failed.
_DOCLING_WORKER = (sys.executable, str(Path(__file__).with_name("_docling_worker.py")))
_DOCLING_TIMEOUT_S = 300


class DoclingFailed(RuntimeError):
    """The Docling worker crashed, timed out, or wrote no valid output."""


def _end_process_group(process: subprocess.Popen) -> None:
    """Kill a worker and anything it started (it leads its own process group)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.communicate()  # reap it


def _docling_payload(name: str, data: bytes) -> Dict[str, Any]:
    """Convert one PDF with Docling in a worker process; its texts and table rows.

    :raises DoclingFailed: on an abnormal exit (e.g. a segmentation fault), a
        timeout, or missing/invalid output. The file then counts as failed, and
        the rest of the paper's evidence is unaffected.
    """
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "input.pdf")
        output = os.path.join(folder, "output.json")
        with open(source, "wb") as handle:
            handle.write(data)
        command = [*_DOCLING_WORKER, source, str(_DOCLING_MAX_PAGES), output]
        with _DOCLING_LOCK:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                _, stderr = process.communicate(timeout=_DOCLING_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _end_process_group(process)
                logger.warning(
                    "Docling timed out after %ss on %s", _DOCLING_TIMEOUT_S, name
                )
                raise DoclingFailed(f"timed out after {_DOCLING_TIMEOUT_S}s") from None
        if process.returncode != 0:
            last = stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
            logger.warning(
                "Docling worker for %s exited with status %s", name, process.returncode
            )
            raise DoclingFailed(
                f"worker exited with status {process.returncode} {last}"
            )
        try:
            with open(output, encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload["texts"], list) or not isinstance(
                payload["tables"], list
            ):
                raise ValueError("texts and tables must be lists")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Docling worker for %s wrote no valid output: %s", name, exc)
            raise DoclingFailed(f"no valid output: {exc}") from exc
    return payload


def _docling_records(name: str, data: bytes, label: str) -> Iterator[Dict[str, Any]]:
    """Records from Docling, in page order: each page's prose, then its tables.

    Tables become header-repeated row blocks (``_table_record``), so a row keeps
    its column names instead of being flattened into the page. Interleaving them
    with the text by page means the per-file record cap reaches the tables of
    early pages instead of being used up by the text of every page first.
    """
    payload = _docling_payload(name, data)
    texts: Dict[int, List[str]] = defaultdict(list)
    tables: Dict[int, List[Tuple[str, List[str]]]] = defaultdict(list)
    for page, text in payload["texts"]:
        texts[page].append(text)
    for page, rows in payload["tables"]:
        tables[page].extend(_sheet_blocks(rows))
    for page in sorted(set(texts) | set(tables)):
        text = _norm(" ".join(texts[page]))
        if text:
            yield _page_record(text, f"{label} (p. {page})", name)
        for header, block in tables[page]:
            yield _table_record(header, block, f"{label} (p. {page}, table)", name)
    for page, text in _pdf_texts(data, skip=_DOCLING_MAX_PAGES):
        yield _page_record(text, f"{label} (p. {page})", name)


def _page_record(text: str, title: str, name: str) -> Dict[str, Any]:
    return {
        "text": text,
        "section_title": title,
        "section_type": SUPPLEMENTARY_SECTION_TYPE,
        "supplementary_file": name,
    }


def _docx_records(name: str, data: bytes, label: str) -> Iterator[Dict[str, Any]]:
    """A .docx's paragraph blocks, then its tables as table records.

    Tables go through ``_sheet_blocks`` like spreadsheets: empty cells keep
    their column, the header row is detected and starts every block.
    """
    import docx

    _check_package_size(data)
    document = docx.Document(io.BytesIO(data))
    for block in _blocks([p.text for p in document.paragraphs]):
        yield {
            "text": block,
            "section_title": label,
            "section_type": SUPPLEMENTARY_SECTION_TYPE,
            "supplementary_file": name,
        }
    for number, table in enumerate(document.tables, 1):
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        for header, block in _sheet_blocks(rows):
            yield _table_record(header, block, f"{label}, table {number}", name)


class ConverterMissing(RuntimeError):
    """The external program a file type needs is not installed here."""


def _doc_paragraphs(data: bytes) -> List[str]:
    """Paragraph texts of a legacy binary .doc, via ``antiword`` or macOS ``textutil``.

    Neither is a Python dependency: with neither installed this raises
    ``ConverterMissing``, so the file counts as failed (and the outcome is not
    cached) rather than as a document with no text.
    """
    if shutil.which("antiword"):
        command = ["antiword"]
    elif shutil.which("textutil"):
        command = ["textutil", "-convert", "txt", "-stdout"]
    else:
        raise ConverterMissing("no antiword or textutil on PATH for .doc")
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


def _text_paragraphs(data: bytes) -> List[str]:
    """Paragraph texts of a plain-text or Markdown file, split on blank lines."""
    return re.split(r"\n\s*\n", data.decode("utf-8-sig", errors="replace"))


def _cell(value: Any) -> str:
    """One spreadsheet cell as text: empty for None, 3 not 3.0 for whole numbers."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _norm(str(value))


# (sheet name, rows): rows are an iterator, read only as far as records are needed.
_Sheets = List[Tuple[str, Iterable[Sequence[Any]]]]
# Rows scanned for a table header (the first row with two or more filled cells)
# before a sheet is taken to be a one-column list.
_HEADER_SCAN_ROWS = 10


@contextmanager
def _delimited_sheets(data: bytes, delimiter: str) -> Iterator[_Sheets]:
    """The one unnamed sheet of a CSV/TSV file."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    yield [("", csv.reader(io.StringIO(text), delimiter=delimiter))]


@contextmanager
def _xlsx_sheets(data: bytes) -> Iterator[_Sheets]:
    """Every worksheet of an .xlsx, rows streamed with computed cell values.

    Read-only mode parses rows as they are asked for: an .xlsx is itself a ZIP,
    so loading it whole could expand far past the download budget.
    """
    import openpyxl

    _check_package_size(data)
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        yield [(sheet.title, sheet.iter_rows(values_only=True)) for sheet in workbook]
    finally:
        workbook.close()


@contextmanager
def _xls_sheets(data: bytes) -> Iterator[_Sheets]:
    """Every worksheet of a legacy .xls (not compressed, so bounded by its size)."""
    import xlrd

    workbook = xlrd.open_workbook(file_contents=data)
    yield [(sheet.name, _xls_rows(sheet)) for sheet in workbook.sheets()]


def _xls_rows(sheet: Any) -> Iterator[List[Any]]:
    """One worksheet's rows. A function of its own, so each generator keeps its
    sheet (a generator expression in the loop would read the last one)."""
    for i in range(sheet.nrows):
        yield sheet.row_values(i)


_SHEET_READERS: Dict[str, Callable[[bytes], ContextManager[_Sheets]]] = {
    ".csv": lambda data: _delimited_sheets(data, ","),
    ".tsv": lambda data: _delimited_sheets(data, "\t"),
    ".xlsx": _xlsx_sheets,
    ".xls": _xls_sheets,
}


def _row(row: Sequence[Any]) -> Tuple[str, int]:
    """A row as ``a | b | c`` and its count of filled cells.

    Empty cells keep their position (``A |  | 0.01``), so every value stays under
    its column; only trailing empty cells (sheet padding) are dropped.
    """
    cells = [_cell(v) for v in row]
    while cells and not cells[-1]:
        cells.pop()
    return " | ".join(cells), sum(1 for c in cells if c)


def _sheet_blocks(rows: Iterable[Sequence[Any]]) -> Iterator[Tuple[str, List[str]]]:
    """Rows as ``(header, [row, row, ...])`` blocks of about ``_MIN_BLOCK_WORDS`` words.

    The header is the first row with two or more filled cells, so a title row
    above the table ("Table S1. DE genes") is not taken for it; such rows are
    kept as a caption. Caption and header start every block, so a block BM25
    or the judge picks out still says what its columns are. Entirely empty rows
    are skipped, and rows are read only as far as blocks are taken.
    """
    rows = iter(rows)
    caption: List[str] = []
    header: Optional[str] = None
    for row in rows:
        line, filled = _row(row)
        if not filled:
            continue
        if filled >= 2:
            header = line
            break
        caption.append(line)
        if len(caption) == _HEADER_SCAN_ROWS:
            break
    data: List[str] = []
    if header is None:  # no multi-column row near the top: a one-column list
        if not caption:
            return
        header, data, caption = caption[0], caption[1:], []
    prefix = " ; ".join(caption + [header])
    pending: List[str] = []
    words = 0
    rest = (line for line, filled in map(_row, rows) if filled)
    for line in itertools.chain(data, rest):
        pending.append(line)
        words += len(line.split())
        if words >= _MIN_BLOCK_WORDS:
            yield prefix, pending
            pending, words = [], 0
    if pending:
        yield prefix, pending


def _table_record(
    header: str, rows: List[str], title: str, name: str
) -> Dict[str, Any]:
    """One table block as a record: ``header ; row ; row``, rendered like
    ``jats`` tables. ``table_header``/``table_rows`` let the agent re-cut a block
    that is still too long by rows, keeping the header in every piece.
    """
    return {
        "text": " ; ".join([header] + rows),
        "section_title": title,
        "section_type": SUPPLEMENTARY_SECTION_TYPE,
        "supplementary_file": name,
        "table_header": header,
        "table_rows": rows,
    }


_PARAGRAPH_EXTRACTORS: Dict[str, Callable[[bytes], List[str]]] = {
    ".doc": _doc_paragraphs,
    ".txt": _text_paragraphs,
    ".md": _text_paragraphs,
}


def _file_records(
    name: str, data: bytes, label: str, pdf_extractor: str
) -> Iterator[Dict[str, str]]:
    """Records for one supplementary file; nothing for an unsupported type."""
    extension = os.path.splitext(name)[1].lower()
    if extension == ".pdf" and pdf_extractor == "docling":
        yield from _docling_records(name, data, label)
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
    if extension == ".docx":
        yield from _docx_records(name, data, label)
        return
    reader = _SHEET_READERS.get(extension)
    if reader is not None:
        with reader(data) as sheets:
            streams = [
                (f'{label}, sheet "{sheet}"' if sheet else label, _sheet_blocks(rows))
                for sheet, rows in sheets
            ]
            # One block from each sheet in turn, so the file's record cap reaches
            # every sheet instead of being used up by the first.
            while streams:
                for stream in list(streams):
                    title, blocks = stream
                    block = next(blocks, None)
                    if block is None:
                        streams.remove(stream)
                        continue
                    yield _table_record(*block, title, name)
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
    failed: Optional[List[str]] = None,
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
    :param failed: If given, receives the names of files that could not be
        read (corrupt, parser error, missing converter). A file skipped by
        policy (``PackageTooLarge``) is not a failure: that outcome is the same
        on every run.
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
        except PackageTooLarge as exc:
            logger.debug("skipped supplementary file %s: %s", name, exc)
        except Exception as exc:  # a corrupt file must not cost the paper its body
            logger.debug("could not read supplementary file %s: %s", name, exc)
            if failed is not None:
                failed.append(name)
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

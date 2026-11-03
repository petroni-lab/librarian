"""PDF → minimal JATS, so a PDF-only article body flows through ``extract_body_paragraphs``.

Some open-access articles have a PDF but no usable XML body (old scanned issues,
records whose XML is not served). A source that can fetch such a PDF converts it
here and returns the result as ordinary JATS: ``<article><body><sec><title>…
</title><p>…</p></sec>…</body></article>``. Everything downstream (paragraph
extraction, section metadata, caching) then works unchanged.

Text comes from the PDF's own text layer (pypdf, the ``supplementary`` extra);
there is no OCR here. ``text_layer_is_readable`` lets a caller tell a usable
layer from one that lost its word spaces or is missing. Only the body is built:
paper metadata comes from the search record, references are cut, and repeated
page headers/footers are dropped.
"""

from __future__ import annotations

import io
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter

# Headings recognised without numbering, compared lower-case without trailing
# punctuation. "abstract" is recognised so its section can be dropped: the
# search record already carries the abstract.
_KNOWN_HEADINGS = {
    "abstract",
    "summary",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "material and methods",
    "methods and materials",
    "patients and methods",
    "experimental procedures",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "case report",
    "case presentation",
}
# Everything from one of these headings on is back matter, not body.
_END_HEADINGS = {"references", "bibliography", "literature cited", "reference"}
# "2. Methods", "3.1 Cell culture", "II. RESULTS".
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+[A-Z][^.]{1,78}$")
_PAGE_NUMBER_RE = re.compile(r"^\W*\d{1,4}\W*$")
# XML 1.0 forbids these; pypdf output can contain them.
_INVALID_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_HEADING_LEN = 80
# A line repeated in the first/last lines of this share of pages is a header/footer.
_REPEATED_LINE_SHARE = 0.5
_EDGE_LINES = 2
# A layer is readable when prose-like: English is ~15-18% spaces, ~5-char words.
_MIN_SPACE_SHARE = 0.08
_MAX_MEAN_WORD_LEN = 12.0
_MIN_CHARS_PER_PAGE = 200


def pdf_page_texts(data: bytes) -> list[str]:
    """The text layer of each page of a PDF, in page order ("" for image-only pages).

    :param data: The PDF bytes.
    :return: One string per page.
    """
    from pypdf import PdfReader  # optional extra: only PDF conversion needs it

    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]


def text_layer_is_readable(page_texts: list[str]) -> bool:
    """True when the text layer reads as prose, so it is usable without OCR.

    False for image-only PDFs and for OCR layers whose words run together
    ("PlacentalTransferofCadmium").

    :param page_texts: Per-page text, as from ``pdf_page_texts``.
    """
    text_pages = [t for t in page_texts if len(t.strip()) >= _MIN_CHARS_PER_PAGE]
    if not text_pages:
        return False
    joined = " ".join(text_pages)
    words = joined.split()
    space_share = joined.count(" ") / len(joined)
    mean_word_len = statistics.fmean(len(word) for word in words)
    return space_share >= _MIN_SPACE_SHARE and mean_word_len <= _MAX_MEAN_WORD_LEN


def _repeated_edge_lines(pages: list[list[str]]) -> set:
    # Lines that open or close most pages: running headers, footers, journal names.
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages:
        edges = set(lines[:_EDGE_LINES] + lines[-_EDGE_LINES:])
        counts.update(edges)
    threshold = _REPEATED_LINE_SHARE * len(pages)
    return {line for line, count in counts.items() if count >= threshold}


def _heading_key(line: str) -> str:
    # Lower-case, no trailing punctuation, no leading numbering.
    stripped = re.sub(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", "", line.strip())
    return stripped.rstrip(" .:").lower()


def _is_heading(line: str) -> bool:
    # Short line that is a known section name or a numbered/upper-case title.
    text = line.strip()
    if not text or len(text) > _MAX_HEADING_LEN:
        return False
    key = _heading_key(text)
    if key in _KNOWN_HEADINGS or key in _END_HEADINGS:
        return True
    if _NUMBERED_HEADING_RE.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and text.isupper() and not text.endswith(".")


def _sections(page_texts: list[str]) -> list[tuple[str, list[str]]]:
    # (title, paragraphs) in reading order, up to the first end heading.
    pages = [[line.strip() for line in t.splitlines()] for t in page_texts]
    repeated = _repeated_edge_lines(pages)
    lines = [
        line
        for page in pages
        for line in page
        if line not in repeated and not _PAGE_NUMBER_RE.match(line)
    ]
    widths = [len(line) for line in lines if line]
    full_width = statistics.median(widths) if widths else 0
    sections: list[tuple[str, list[str]]] = [("", [])]
    current: list[str] = []

    def flush() -> None:
        # Close the paragraph being built, joining hyphenated line breaks.
        if current:
            text = " ".join(current)
            sections[-1][1].append(re.sub(r"(\w)- (\w)", r"\1\2", text))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            if _heading_key(line) in _END_HEADINGS:
                break
            sections.append((line.rstrip(" :"), []))
            continue
        current.append(line)
        # A short line ending a sentence ends its paragraph.
        if line.endswith((".", "?", "!", ":")) and len(line) < 0.8 * full_width:
            flush()
    flush()
    # Once headings exist, the untitled text before the first one is the title
    # block (title, authors, affiliations), not body. Without any heading (many
    # scans) that untitled text is the whole article and is kept.
    if len(sections) > 1:
        sections = sections[1:]
    return [
        (title, paragraphs)
        for title, paragraphs in sections
        if paragraphs and _heading_key(title) != "abstract"
    ]


def page_texts_to_jats(page_texts: list[str]) -> str | None:
    """Build minimal JATS from per-page PDF text, or None when no body remains.

    :param page_texts: Per-page text, as from ``pdf_page_texts``.
    :return: A JATS ``<article>`` string with one ``<sec>`` per detected section.
    """
    sections = _sections(page_texts)
    if not sections:
        return None
    article = ET.Element("article")
    body = ET.SubElement(article, "body")
    for title, paragraphs in sections:
        sec = ET.SubElement(body, "sec")
        ET.SubElement(sec, "title").text = _INVALID_XML_RE.sub("", title)
        for paragraph in paragraphs:
            ET.SubElement(sec, "p").text = _INVALID_XML_RE.sub("", paragraph)
    return ET.tostring(article, encoding="unicode")


def pdf_to_jats(data: bytes) -> str | None:
    """Convert a PDF with a readable text layer to minimal JATS.

    :param data: The PDF bytes.
    :return: JATS, or None when the text layer is unreadable (needs OCR) or
        yields no body.
    """
    page_texts = pdf_page_texts(data)
    if not text_layer_is_readable(page_texts):
        return None
    return page_texts_to_jats(page_texts)

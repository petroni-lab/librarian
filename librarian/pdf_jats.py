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
from collections.abc import Iterator

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
# Headings that open an article's text: what comes before the first of them is the
# title block (title, authors, affiliations). Only these, so a stray heading late
# in a scan never discards the body above it.
_OPENING_HEADINGS = {"abstract", "summary", "introduction", "background"}
# Sections that are front or back matter: their text is left out, the rest goes on.
_SKIPPED_HEADINGS = {
    "keywords",
    "key words",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "funding",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "author contributions",
    "correspondence",
    "citation",
    "copyright",
    "data availability",
    "edited and reviewed by",
    "abbreviations",
}
# A keyword list is a single line, and the body can follow it with no heading or
# paragraph break (Frontiers editorials), so these sections end after one line.
_ONE_LINE_HEADINGS = {"keywords", "key words"}
# Front-matter boxes printed beside the first page's text (Frontiers' sidebar).
# The text layer puts them after that page's body, so the next page's body follows
# with no heading: these sections end at a page break. Back-matter sections
# (acknowledgements, funding, ...) do run across pages and keep running.
_SIDEBAR_HEADINGS = {
    "keywords",
    "key words",
    "edited and reviewed by",
    "correspondence",
    "citation",
    "copyright",
}
# Everything from one of these headings on is back matter, not body.
_END_HEADINGS = {"references", "bibliography", "literature cited", "reference"}
# "2. Methods", "3.1 Cell culture", "II. RESULTS": a 1-2 digit number followed by
# a full stop or a sub-number (or a Roman numeral), then a title with no digits,
# commas or full stops. That rules out table rows ("12 Controls 45 23", "12 GP
# surgeries expressed"), lines opening with a year ("2019 Guidelines recommend")
# and numbered affiliations ("1 Institute of Examples, Heidelberg").
_NUMBERED_HEADING_RE = re.compile(
    r"^(\d{1,2}\.(\d{1,2}\.?)*|\d{1,2}(\.\d{1,2})+|[IVX]{1,4}\.)\s+[A-Z][^.,\d]{1,78}$"
)
# Longer numbered lines are list items ("1. That the hereditary ... of cancer").
_MAX_HEADING_WORDS = 8
# Figure and table captions are upper-case but not headings.
_CAPTION_RE = re.compile(r"^(fig(ure)?|table|plate|chart)\b", re.IGNORECASE)
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


def pdf_page_texts(data: bytes, skip: int = 0) -> Iterator[str]:
    """The text layer of each page of a PDF, in page order ("" for image-only pages).

    Pages are read lazily, so a caller that stops early never reads (or fails on)
    later pages. Line breaks are kept: paragraph and heading detection need them.

    :param data: The PDF bytes.
    :param skip: Number of leading pages to leave out.
    :return: One string per page after the first *skip*.
    """
    from pypdf import PdfReader  # optional extra: only PDF conversion needs it

    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages[skip:]:
        yield page.extract_text() or ""


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
        # Blank lines are paragraph breaks, never headers: skip them here.
        text_lines = [line for line in lines if line]
        edges = set(text_lines[:_EDGE_LINES] + text_lines[-_EDGE_LINES:])
        counts.update(edges)
    threshold = _REPEATED_LINE_SHARE * len(pages)
    return {line for line, count in counts.items() if count >= threshold}


def _heading_key(line: str) -> str:
    # Lower-case, no leading numbering or markers, no trailing punctuation.
    stripped = re.sub(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", "", line.strip())
    return stripped.strip(" *").rstrip(" .:").lower()


def _is_heading(line: str) -> bool:
    # Short line that is a known section name or a numbered/upper-case title.
    text = line.strip()
    if not text or len(text) > _MAX_HEADING_LEN:
        return False
    key = _heading_key(text)
    if key in _KNOWN_HEADINGS or key in _END_HEADINGS or key in _SKIPPED_HEADINGS:
        return True
    if _NUMBERED_HEADING_RE.match(text):
        return len(text.split()) <= _MAX_HEADING_WORDS
    # An upper-case line, but not a caption, not gene names or identifiers
    # ("BRCA1 AND TP53", "REACTOME_INNATE_IMMUNE_SYSTEM") and not an author line
    # ending in a comma ("B. P. WATSON, M.D., F.R.C.S.,").
    if _CAPTION_RE.match(text) or any(c.isdigit() or c == "_" for c in text):
        return False
    if text.endswith(","):
        return False
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and text.isupper() and not text.endswith(".")


def _sections(page_texts: list[str]) -> list[tuple[str, list[str]]]:
    # (title, paragraphs) in reading order, up to the first end heading.
    pages = [[line.strip() for line in t.splitlines()] for t in page_texts]
    repeated = _repeated_edge_lines(pages)
    lines = [
        (page_index, line)
        for page_index, page in enumerate(pages)
        for line in page
        if line not in repeated and not _PAGE_NUMBER_RE.match(line)
    ]
    widths = [len(line) for _, line in lines if line]
    full_width = statistics.median(widths) if widths else 0
    sections: list[tuple[str, list[str]]] = [("", [])]
    current: list[str] = []

    def flush() -> None:
        # Close the paragraph being built.
        if current:
            sections[-1][1].append(" ".join(current))
            current.clear()

    def end_skipped_section() -> None:
        # Close the skipped section: the text after it is body again, untitled.
        flush()
        sections.append(("", []))

    previous_page = 0
    for page_index, line in lines:
        title_key = _heading_key(sections[-1][0])
        if page_index != previous_page and title_key in _SIDEBAR_HEADINGS:
            end_skipped_section()
        previous_page = page_index
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            if _heading_key(line) in _END_HEADINGS:
                break
            sections.append((line.rstrip(" :"), []))
            continue
        # A word hyphenated across a line break ("micro-" / "scope") is rejoined;
        # hyphens inside a line ("3- and 6-month") are left alone.
        if current and current[-1].endswith("-") and line[:1].islower():
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)
        # A short line ending a sentence ends its paragraph.
        if line.endswith((".", "?", "!", ":")) and len(line) < 0.8 * full_width:
            flush()
        if _heading_key(sections[-1][0]) in _ONE_LINE_HEADINGS:
            end_skipped_section()
    flush()
    # Text before the first heading is the title block, dropped only when that
    # heading opens an article ("Abstract", "Introduction", …). A late "SUMMARY"
    # or a "Background" subsection never discards the body above it; without an
    # opening heading first (many scans) everything is kept.
    # An all-caps paper title counts as the first heading, so its title block is
    # kept as body; tighten if that shows up in answers.
    first_heading_opens = (
        len(sections) > 1 and _heading_key(sections[1][0]) in _OPENING_HEADINGS
    )
    if first_heading_opens:
        sections = sections[1:]
    sections = _without_abstract(sections)
    return [
        (title, paragraphs)
        for title, paragraphs in sections
        if paragraphs and _heading_key(title) not in _SKIPPED_HEADINGS
    ]


def _without_abstract(
    sections: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    # Drop the abstract (the search record carries it). A structured abstract's
    # subsections (Background, Methods, Results, ...) go with it, up to the
    # "Introduction" or first numbered heading; without one only the "Abstract"
    # section itself goes, so a short paper's body is never dropped.
    titles = [_heading_key(title) for title, _ in sections]
    if "abstract" not in titles:
        return sections
    start = titles.index("abstract")
    body_starts = [
        index
        for index in range(start + 1, len(sections))
        if titles[index] == "introduction"
        or _NUMBERED_HEADING_RE.match(sections[index][0])
    ]
    end = body_starts[0] if body_starts else start + 1
    return sections[:start] + sections[end:]


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
    page_texts = list(pdf_page_texts(data))
    if not text_layer_is_readable(page_texts):
        return None
    return page_texts_to_jats(page_texts)

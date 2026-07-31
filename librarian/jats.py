"""JATS full-text XML → body paragraph records.

Europe PMC serves open-access full text as JATS (Journal Article Tag Suite) XML.
``extract_body_paragraphs`` walks that tree and returns one record per body
paragraph or table, in document order, skipping front/back matter, references,
and boilerplate sections (funding, supplementary material, …).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# Ancestor tags whose entire subtree is never body content.
_EXCLUDED_ANCESTOR_NAMES = {
    "abstract",
    "app",
    "app-group",
    "article-meta",
    "back",
    "bio",
    "fn-group",
    "front",
    "journal-meta",
    "ref",
    "ref-list",
    "table-wrap-foot",
}
_EXCLUDED_SEC_TYPES = {
    "abbreviations",
    "acknowledgements",
    "acknowledgments",
    "author-contributions",
    "competing-interests",
    "conflict",
    "conflict-of-interest",
    "data-availability",
    "ethics",
    "funding",
    "references",
    "supplementary-material",
}


def local_name(tag: str) -> str:
    """Strip the XML namespace from an element tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_excluded(element: ET.Element, parent: Dict[ET.Element, ET.Element]) -> bool:
    """True if ``element`` or any ancestor is boilerplate (front/back/refs/…).

    Table tags are deliberately not in ``_EXCLUDED_ANCESTOR_NAMES``: a
    ``table-wrap`` is rendered by ``_render_table``, not skipped, so checking
    this on the table-wrap element itself (not its parent) still does the
    right thing.
    """
    current: Optional[ET.Element] = element
    while current is not None:
        local = local_name(current.tag)
        if local in _EXCLUDED_ANCESTOR_NAMES:
            return True
        if local == "sec":
            sec_type = str(current.attrib.get("sec-type", "")).strip().lower()
            if sec_type in _EXCLUDED_SEC_TYPES or "reference" in sec_type:
                return True
        current = parent.get(current)
    return False


def _nearest_section(
    element: ET.Element, parent: Dict[ET.Element, ET.Element]
) -> Tuple[str, str]:
    current = parent.get(element)
    while current is not None:
        if local_name(current.tag) == "sec":
            title = ""
            for child in list(current):
                if local_name(child.tag) == "title":
                    title = _norm("".join(child.itertext()))
                    break
            return title, str(current.attrib.get("sec-type", "")).strip()
        current = parent.get(current)
    return "", ""


def _prose_text(element: ET.Element) -> str:
    """itertext of a paragraph, but skipping any embedded table subtree.

    Tables are rendered separately by ``_render_table``; pulling their cells in
    here would mash them into the prose as an unsplittable blob.
    """
    parts: List[str] = []

    def walk(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if local_name(child.tag) not in {"table", "table-wrap"}:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return _norm("".join(parts))


def _render_table(table_wrap: ET.Element) -> str:
    """Render a <table-wrap> as readable text: label + caption, then each row
    with its cells separated, so the table survives as interpretable evidence
    instead of a run-together blob with no cell boundaries."""
    header: List[str] = []
    for node in table_wrap.iter():
        if local_name(node.tag) in {"label", "caption"}:
            part = _norm("".join(node.itertext()))
            if part:
                header.append(part)
    rows: List[str] = []
    for row in table_wrap.iter():
        if local_name(row.tag) != "tr":
            continue
        cells = [
            _norm("".join(cell.itertext()))
            for cell in row
            if local_name(cell.tag) in {"td", "th"}
        ]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    pieces = header + ([" ; ".join(rows)] if rows else [])
    return _norm(". ".join(pieces))


def extract_body_paragraphs(xml_text: str) -> List[Dict[str, str]]:
    """Body paragraph/table records: ``{text, section_title, section_type}``.

    Skips front/back matter, references, and excluded sec-types
    (acknowledgements, funding, supplementary, …); carries section metadata so
    the BM25 doc can include section title/type. Tables are kept but rendered
    readably (label + caption + rows with ``|`` between cells, ``;`` between
    rows) instead of mashed into one blob; a table embedded inside a ``<p>`` is
    stripped out of that paragraph's prose and emitted as its own record.
    """
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    parent = {child: node for node in root.iter() for child in node}
    body_roots = [e for e in root.iter() if local_name(e.tag) == "body"]
    containers = body_roots or [root]

    records: List[Dict[str, str]] = []
    seen: set = set()
    for container in containers:
        # Single document-order pass over prose paragraphs and tables, so a
        # cited table keeps its position relative to the surrounding text.
        for element in container.iter():
            local = local_name(element.tag)
            if local == "p" and not _is_excluded(element, parent):
                text = _prose_text(element)
            elif local == "table-wrap" and not _is_excluded(element, parent):
                text = _render_table(element)
            else:
                continue
            if not text or text in seen:
                continue
            seen.add(text)
            title, sec_type = _nearest_section(element, parent)
            records.append(
                {"text": text, "section_title": title, "section_type": sec_type}
            )
    return records

"""PDF page text → minimal JATS that ``extract_body_paragraphs`` reads as a body.

uv run --extra supplementary pytest tests
"""

from librarian.jats import extract_body_paragraphs
from librarian.pdf_jats import (
    _is_heading,
    page_texts_to_jats,
    pdf_page_texts,
    text_layer_is_readable,
)

_HEADER = "Journal of Examples 12 (2023)"
_PROSE = "Cells were cultured for three days in standard medium before imaging. "


def _page(*lines: str) -> str:
    # One PDF page: running header, body lines, page number.
    return "\n".join((_HEADER,) + lines + ("7",))


def test_sections_and_paragraphs_survive_headers_and_references_do_not():
    pages = [
        _page(
            "A study of cells",
            "1 Institute of Examples, Heidelberg",
            "Abstract",
            "We summarise the study here.",
            "1. Introduction",
            _PROSE * 3,
        ),
        _page("2. Methods", _PROSE * 2, "", "Imaging used a confocal micro-", "scope."),
        _page("Results", _PROSE, "References", "1. Smith J. A paper. 2020."),
    ]
    records = extract_body_paragraphs(page_texts_to_jats(pages))
    titles = [r["section_title"] for r in records]
    texts = " ".join(r["text"] for r in records)
    assert titles == ["1. Introduction", "2. Methods", "2. Methods", "Results"]
    assert "Imaging used a confocal microscope." in texts
    assert _HEADER not in texts and "Smith J" not in texts
    assert "Institute of Examples" not in texts  # title block before the first heading
    assert "summarise" not in texts  # the abstract comes from the search record


def test_a_layer_without_word_spaces_needs_ocr():
    readable = [_PROSE * 10]
    run_together = ["PlacentalTransferofCadmiuminRats:InfluenceofDoseand" * 10]
    assert text_layer_is_readable(readable)
    assert not text_layer_is_readable(run_together)
    assert not text_layer_is_readable(["", ""])  # image-only pages


def test_an_article_without_headings_keeps_all_its_text():
    records = extract_body_paragraphs(page_texts_to_jats([_PROSE * 3]))
    assert len(records) == 1 and records[0]["section_title"] == ""


def test_back_matter_sections_are_left_out():
    pages = [
        f"Introduction\n{_PROSE}\nKEYWORDS\ncells; imaging\n*CORRESPONDENCE\nA. Author"
    ]
    texts = " ".join(
        r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))
    )
    assert (
        "imaging." in texts
        and "cells; imaging" not in texts
        and "A. Author" not in texts
    )


def test_body_after_a_keyword_line_or_a_sidebar_box_is_kept():
    # Frontiers editorial layout (PMC11922827): the body follows the keyword line
    # with no break, and page 1 ends with a sidebar box that page 2's body follows.
    pages = [
        (
            f"KEYWORDS\ncells, imaging\nThe editorial opens here. {_PROSE * 2}\n"
            "COPYRIGHT\n© 2025 Author. This is an open-access article"
        ),
        f"continued on the next page. {_PROSE * 2}",
    ]
    texts = " ".join(
        r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))
    )
    assert "The editorial opens here." in texts
    assert "continued on the next page." in texts
    assert "cells, imaging" not in texts and "open-access article" not in texts


def test_back_matter_running_onto_the_next_page_stays_out():
    pages = [
        f"Introduction\n{_PROSE * 2}\nFunding\nThis work was funded by the",
        "Example Foundation under grant 42",
    ]
    texts = " ".join(
        r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))
    )
    assert "imaging." in texts and "Example Foundation" not in texts


def test_no_body_gives_none():
    assert page_texts_to_jats(["", "  "]) is None


def test_lines_that_only_look_like_headings_stay_body():
    lines = [
        "12 Controls 45 23",
        "2019 Guidelines recommend early screening",
        "1 Institute of Examples, Heidelberg",
        "FIGURE 2",
        "TABLE 1",
        "BRCA1 AND TP53",
        "12 GP surgeries expressed",
        "REACTOME_INNATE_IMMUNE_SYSTEM",
        "B. P. WATSON, M.D., F.R.C.S.,",
        "1. That the hereditary and congenital acquirement of cancer",
    ]
    assert not any(_is_heading(line) for line in lines)
    assert _is_heading("2. Methods") and _is_heading("II. RESULTS")


def test_blank_lines_split_paragraphs_on_every_page():
    # Distinct text per page: lines repeated on most pages are running headers.
    pages = [
        f"Page {n} opens with {_PROSE}\n\nPage {n} goes on with {_PROSE}" for n in "ABC"
    ]
    records = [r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))]
    # The blank line splits each page; a paragraph running to the bottom of a page
    # continues on the next one, so 3 pages x 2 paragraphs make 4 records.
    assert len(records) == 4
    assert records[0].startswith("Page A opens") and records[1].startswith(
        "Page A goes"
    )


def test_a_stray_heading_in_a_scan_keeps_the_text_above_it():
    pages = [_PROSE * 4, "TABLE I\nResults of the survey were mixed."]
    texts = " ".join(
        r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))
    )
    assert _PROSE.strip() in texts


def test_a_late_opening_heading_keeps_the_text_above_it():
    late_summary = [f"RESULTS\n{_PROSE * 3}", "SUMMARY\nCells grew."]
    background_subsection = [f"METHODS\n{_PROSE * 3}", "Background\nCells grew."]
    for pages in (late_summary, background_subsection):
        texts = " ".join(
            r["text"] for r in extract_body_paragraphs(page_texts_to_jats(pages))
        )
        assert _PROSE.strip() in texts and "Cells grew." in texts


def test_a_structured_abstract_is_left_out_whole():
    pages = [
        "Abstract\nBACKGROUND\nWhy.\nMETHODS\nHow.\nRESULTS\nWhat.",
        f"Introduction\n{_PROSE * 3}\nMethods\n{_PROSE}",
    ]
    records = extract_body_paragraphs(page_texts_to_jats(pages))
    titles = [r["section_title"] for r in records]
    assert titles == ["Introduction", "Methods"]


def test_without_an_introduction_only_the_abstract_goes():
    pages = [f"Abstract\nShort summary.\nMETHODS\n{_PROSE * 3}"]
    records = extract_body_paragraphs(page_texts_to_jats(pages))
    assert [r["section_title"] for r in records] == ["METHODS"]


def test_hyphens_inside_a_line_are_kept():
    page = f"{_PROSE}\nPatients were seen at 3- and 6-month visits."
    texts = " ".join(
        r["text"] for r in extract_body_paragraphs(page_texts_to_jats([page]))
    )
    assert "3- and 6-month" in texts


def test_pages_are_read_lazily(monkeypatch):
    import pypdf

    class _Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            if self.text is None:
                raise ValueError("broken page")
            return self.text

    class _Reader:
        def __init__(self, stream):
            self.pages = [_Page("Readable page."), _Page(None)]

    monkeypatch.setattr(pypdf, "PdfReader", _Reader)
    pages = pdf_page_texts(b"%PDF")
    assert next(pages) == "Readable page."  # page 2 not read yet

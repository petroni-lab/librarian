"""PDF page text → minimal JATS that ``extract_body_paragraphs`` reads as a body.

uv run --extra supplementary pytest tests
"""

from librarian.jats import extract_body_paragraphs
from librarian.pdf_jats import page_texts_to_jats, text_layer_is_readable

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


def test_no_body_gives_none():
    assert page_texts_to_jats(["", "  "]) is None

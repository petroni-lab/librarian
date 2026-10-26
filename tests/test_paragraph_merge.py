"""Unit tests for ``_merge_selected_paragraphs``, Stage 2's overlap union.

Two sub-queries often select the same source paragraph, sometimes at overlapping
word offsets. Merging collapses those into one selection per contiguous interval,
so the Stage 3 judge pays for a paragraph once however many sub-queries found it.

These are hand-written rather than fixture-driven on purpose: a recorded run only
exercises merging if its sub-queries happen to overlap, and when they don't,
merging is an identity transform and every branch below goes untested.
"""

from __future__ import annotations

from typing import Any

from librarian.agent import _merge_selected_paragraphs

# One shared source paragraph, long enough to slice several distinct intervals out.
_SOURCE_TEXT = " ".join(f"w{index}" for index in range(20))
_PAPER = {"pmid": "111", "title": "A paper"}
_OTHER_PAPER = {"pmid": "222", "title": "Another paper"}


def _selection(
    word_start: int,
    word_end: int,
    *,
    paper: dict[str, Any] = _PAPER,
    source_paragraph_index: int = 0,
    source_text: str = _SOURCE_TEXT,
) -> dict[str, Any]:
    """One Stage 2 paragraph record selecting ``[word_start, word_end)`` of its source."""
    words = source_text.split()
    return {
        "paper": paper,
        "text": " ".join(words[word_start:word_end]),
        "source_text": source_text,
        "source_paragraph_index": source_paragraph_index,
        "word_start": word_start,
        "word_end": word_end,
    }


def _intervals(merged: list[dict[str, Any]]) -> list[tuple[str, int, int, int]]:
    """The merged selections as ``(pmid, source paragraph, start, end)`` tuples."""
    return [
        (
            paragraph["paper"]["pmid"],
            paragraph["source_paragraph_index"],
            paragraph["word_start"],
            paragraph["word_end"],
        )
        for paragraph in merged
    ]


def test_disjoint_selections_are_left_alone() -> None:
    """Selections with a gap between them stay separate, in document order."""
    merged = _merge_selected_paragraphs([_selection(0, 5), _selection(10, 15)])
    assert _intervals(merged) == [("111", 0, 0, 5), ("111", 0, 10, 15)]


def test_overlapping_selections_become_one_interval() -> None:
    """Two sub-queries selecting overlapping windows yield a single union."""
    merged = _merge_selected_paragraphs([_selection(0, 8), _selection(5, 12)])
    assert _intervals(merged) == [("111", 0, 0, 12)]
    assert merged[0]["text"] == " ".join(f"w{index}" for index in range(12))


def test_touching_selections_become_one_interval() -> None:
    """Selections that meet exactly at a boundary merge, leaving no one-word seam."""
    merged = _merge_selected_paragraphs([_selection(0, 6), _selection(6, 11)])
    assert _intervals(merged) == [("111", 0, 0, 11)]


def test_identical_selections_collapse_to_one() -> None:
    """The same paragraph found by two sub-queries costs the judge one item."""
    merged = _merge_selected_paragraphs([_selection(2, 9), _selection(2, 9)])
    assert _intervals(merged) == [("111", 0, 2, 9)]


def test_intervals_are_ordered_within_a_source_paragraph() -> None:
    """Selections are rebuilt in document order even when pooled out of order."""
    merged = _merge_selected_paragraphs([_selection(12, 16), _selection(1, 4)])
    assert _intervals(merged) == [("111", 0, 1, 4), ("111", 0, 12, 16)]


def test_groups_follow_first_selection_order() -> None:
    """Different papers and source paragraphs stay separate, ordered by first sight."""
    merged = _merge_selected_paragraphs(
        [
            _selection(0, 3, paper=_OTHER_PAPER),
            _selection(0, 3, source_paragraph_index=1),
            _selection(0, 3),
        ]
    )
    assert _intervals(merged) == [("222", 0, 0, 3), ("111", 1, 0, 3), ("111", 0, 0, 3)]


def test_same_offsets_in_different_source_text_do_not_merge() -> None:
    """A paragraph is identified by its text too, so re-extracted sources stay apart."""
    merged = _merge_selected_paragraphs(
        [
            _selection(0, 3),
            _selection(0, 3, source_text="different words entirely here"),
        ]
    )
    assert len(merged) == 2

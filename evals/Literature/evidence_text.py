"""Retrieved evidence, rendered as prompt text.

Every bench that asks a model to answer *from* the librarian's evidence has to
flatten ``LibrarianAgent.run``'s records into a block of text first: pull the
snippets out of nested payloads, normalise whitespace, drop repeats, label each
paper, and cut it to a budget. This module holds those steps, so a bench writes
only the layout it actually needs.

The evidence payload is not one shape. A snippet may arrive as a string, a list
of strings, or a dict of either, under ``evidence`` or under the older
``evidence_abstract`` / ``evidence_fulltext`` pair. :func:`collect_texts`
flattens all of them. A bare ``"|"`` is the agent's marker for a gap between
non-contiguous spans, not a snippet, and is dropped.

Nothing here decides layout: a bench renders its own blocks from these pieces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The keys an evidence entry may carry its snippets under.
EVIDENCE_FIELDS = ("evidence", "evidence_abstract", "evidence_fulltext")

#: ``(prefix, key)`` for the identifiers a paper or evidence entry may carry,
#: most specific first. Two keys share the ``CorpusId`` prefix because the
#: agent and the Asta corpus tools spell it differently.
IDENTIFIER_FIELDS = (
    ("PMID", "pmid"),
    ("PMCID", "pmcid"),
    ("DOI", "doi"),
    ("CorpusId", "corpus_id"),
    ("CorpusId", "corpusId"),
)

#: What the agent puts between two non-contiguous spans of one paper.
GAP_MARKER = "|"


def normalize(text: Any) -> str:
    """Collapse all whitespace in *text* to single spaces."""
    return " ".join(str(text).split())


def collect_texts(value: Any) -> list[str]:
    """Flatten a snippet payload into normalised strings.

    :param value: A string, or any nesting of lists and dicts holding strings.
        ``None`` yields nothing.
    :returns: The strings found, in order, with whitespace collapsed and the
        empties and :data:`GAP_MARKER` left out.
    """
    texts: list[str] = []
    _extend(texts, value)
    return texts


def _extend(target: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = normalize(value)
        if text and text != GAP_MARKER:
            target.append(text)
    elif isinstance(value, list):
        for item in value:
            _extend(target, item)
    elif isinstance(value, dict):
        for item in value.values():
            _extend(target, item)


def evidence_texts(
    entry: dict[str, Any], fields: Sequence[str] = EVIDENCE_FIELDS
) -> list[str]:
    """Flatten one evidence entry's snippets.

    :param entry: An entry from a retrieval result's ``evidence`` list.
    :param fields: Which keys to read; defaults to :data:`EVIDENCE_FIELDS`.
    :returns: The entry's snippets, as :func:`collect_texts` returns them.
    """
    texts: list[str] = []
    for field in fields:
        texts.extend(collect_texts(entry.get(field)))
    return texts


def dedupe(texts: Sequence[str]) -> list[str]:
    """Drop repeats, ignoring case and whitespace, keeping first appearance."""
    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = normalize(text)
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        deduped.append(normalized)
    return deduped


def truncate(text: Any, limit: int) -> str:
    """Normalise *text* and cut it to *limit* characters, ellipsis included."""
    compact = normalize(text)
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def identifier_keys(obj: dict[str, Any]) -> list[str]:
    """Every identifier *obj* carries, for matching an entry to a paper.

    :param obj: A paper or an evidence entry.
    :returns: The values present, most specific first. ``_evidence_id`` leads:
        it is the agent's own key, and it matches exactly when a paper carries
        no public identifier at all.
    """
    keys = []
    for key in ("_evidence_id", *(field for _, field in IDENTIFIER_FIELDS)):
        value = str(obj.get(key) or "").strip()
        if value:
            keys.append(value)
    return keys


def first_identifier(obj: dict[str, Any]) -> str:
    """The most specific identifier *obj* carries, or ``""``."""
    keys = identifier_keys(obj)
    return keys[0] if keys else ""


def identifier_label(obj: dict[str, Any]) -> str:
    """Render *obj*'s identifiers for a citation line.

    :param obj: A paper or an evidence entry.
    :returns: e.g. ``"PMID: 12345678; DOI: 10.1/x"``, or ``""`` if it has none.
    """
    labels: list[str] = []
    for prefix, key in IDENTIFIER_FIELDS:
        value = str(obj.get(key) or "").strip()
        if value and f"{prefix}: {value}" not in labels:
            labels.append(f"{prefix}: {value}")
    return "; ".join(labels)


def _self_check() -> None:
    # Every payload shape the benches see flattens the same way, and the gap
    # marker between two spans of one paper is not a snippet.
    entry = {
        "evidence": ["  one\n  span ", GAP_MARKER, {"nested": "two"}],
        "evidence_fulltext": "three",
        "evidence_abstract": None,
    }
    assert evidence_texts(entry) == ["one span", "two", "three"]
    assert evidence_texts(entry, fields=("evidence",)) == ["one span", "two"]
    assert collect_texts(None) == [] and collect_texts(GAP_MARKER) == []

    assert dedupe(["A b", "a  B", "", "c"]) == ["A b", "c"]
    assert truncate("a b", 10) == "a b" and truncate("abcdefgh", 6) == "abc..."

    paper = {"pmid": "123", "corpusId": "7", "title": "T"}
    assert identifier_label(paper) == "PMID: 123; CorpusId: 7"
    assert identifier_keys(paper) == ["123", "7"]
    assert first_identifier(paper) == "123"
    # The agent's own key wins, so an entry with no public identifier still
    # matches the paper it came from.
    assert first_identifier({"_evidence_id": "e1", "pmid": "123"}) == "e1"
    assert identifier_label({}) == "" and first_identifier({}) == ""
    print("evidence_text self-check ok")


if __name__ == "__main__":
    _self_check()

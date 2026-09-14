"""Author-year citation keys and the paper blocks the summarizer cites from.

Two consumers share this module, which is why it lives in the package rather
than next to either of them:

  ``skills/librarian/scripts/step4_finalize.py``  writes ``04_report.md``
  ``librarian/synthesis.py``                      builds the summarizer's papers text

Both need the same thing: one block per paper carrying a ready-made ``Cite as:``
markdown link, so the model copies a citation instead of assembling one out of
metadata. Deriving a surname, noticing that two papers share an author and year,
and picking the a/b suffix are all things Python does exactly and a model does
approximately.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def _short_authors(authors: str, keep: int = 3) -> str:
    """First ``keep`` author names, then ``et al.`` — the report stays scannable."""
    names = [name.strip() for name in str(authors).split(",") if name.strip()]
    if len(names) <= keep:
        return ", ".join(names)
    return ", ".join(names[:keep]) + ", et al."


def _first_author_surname(authors: str) -> str:
    """Surname of the first author, initials dropped.

    Europe PMC writes each author as ``Surname II``, so the trailing all-caps
    initials block is what gets cut; multi-word surnames ("van der Meer") stay.
    """
    first_author = str(authors).split(",")[0].strip()
    parts = first_author.split()
    initials_trail = len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3
    if initials_trail:
        parts = parts[:-1]
    return " ".join(parts)


def citation_keys(evidence: List[Dict[str, Any]]) -> List[str]:
    """The author-year citation key for each paper, in report order.

    The summarizer cites papers by these keys rather than by list position, so
    they are computed here once: deriving a surname, spotting that two papers
    share an author and year, and picking the a/b suffix are all things Python
    can do exactly and a model cannot.

    :param evidence: The evidence records, in the order the report prints them.
    :type evidence: list[dict[str, Any]]
    :return: One key per record, e.g. ``["Chen 2023a", "Chen 2023b", "Kuo 2012"]``.
    :rtype: list[str]
    """
    bases: List[str] = []
    for record in evidence:
        surname = _first_author_surname(record["authors"])
        year = str(record["year"]).strip()
        identifier = record["pmid"] or record["doi"]
        # Author-year when we have it; an identifier is the fallback so a key is
        # never empty and never two papers' key at once.
        bases.append(
            " ".join(part for part in (surname, year) if part)
            or (f"PMID {identifier}" if record["pmid"] else f"doi:{identifier}")
            or "unattributed source"
        )

    shared = Counter(bases)
    used: Counter = Counter()
    keys: List[str] = []
    for base in bases:
        if shared[base] == 1:
            keys.append(base)
            continue
        # ponytail: 26 same-author-same-year papers in one run would run past 'z';
        # switch to a numeric suffix if that ever shows up.
        keys.append(f"{base}{chr(ord('a') + used[base])}")
        used[base] += 1
    return keys


def render_papers(evidence: List[Dict[str, Any]]) -> str:
    """One compact block per paper: citation, link, and the judge-cited spans.

    The heading is the ready-made markdown citation for the paper, so the
    summarizer copies it inline instead of building one from the metadata.

    The spans are the whole point of the run, so they are printed in full and
    uncapped — the judge already decided which sentences answer the question.

    :param evidence: The evidence records, in the order they should be cited.
    :type evidence: list[dict[str, Any]]
    :return: The rendered blocks, or the no-papers line when ``evidence`` is empty.
    :rtype: str
    """
    if not evidence:
        return "No papers survived the relevance judge."

    lines: List[str] = []
    for key, record in zip(citation_keys(evidence), evidence):
        identifiers = []
        if record["pmid"]:
            identifiers.append(f"PMID {record['pmid']}")
        if record["doi"]:
            identifiers.append(f"doi:{record['doi']}")
        venue = " ".join(part for part in (record["journal"], record["year"]) if part)
        source = "full text available" if record["has_fulltext"] else "abstract only"
        citation = f"[{key}]({record['url']})" if record["url"] else f"[{key}]"
        lines.append(f"## {record['title']}")
        lines.append(f"- Cite as: {citation}")
        lines.append(f"- {_short_authors(record['authors'])}")
        lines.append(
            "- "
            + " · ".join(
                part for part in (venue, " · ".join(identifiers), source) if part
            )
        )
        for span in record["evidence_snippets"]:
            lines.append(f"- Evidence: {span}")
        lines.append("")
    return "\n".join(lines)


def render_report(query: str, evidence: List[Dict[str, Any]], summary: str) -> str:
    """The step-4 report: a titled header, the run's stats line, then the papers.

    :param query: The user's original question.
    :type query: str
    :param evidence: The evidence records, in citation order.
    :type evidence: list[dict[str, Any]]
    :param summary: The one-line pipeline stats string shown under the title.
    :type summary: str
    :return: The full markdown report, newline-terminated.
    :rtype: str
    """
    header = f"# Librarian results — {query}\n\n{summary}\n"
    if not evidence:
        return f"{header}\n{render_papers(evidence)}\n"
    return f"{header}\n{render_papers(evidence)}".rstrip() + "\n"

"""Docling conversion of one PDF, run as a separate process.

    python _docling_worker.py <input.pdf> <max_pages> <output.json>

``supplementary._docling_payload`` starts this for each supplementary PDF, so a
native crash, hang or exception inside Docling ends only this process, never the
librarian run. It writes ``{"texts": [[page, text], ...], "tables": [[page,
rows], ...]}`` (``rows`` includes the header row) and needs nothing but Docling.
"""

import json
import sys


def _value(value):
    # Empty cells come out of the table frame as None or NaN.
    return "" if value is None or value != value else value


def payload(document) -> dict:
    """The page texts and table rows of a converted Docling document."""
    texts = [[item.prov[0].page_no, item.text] for item in document.texts if item.prov]
    tables = []
    for table in document.tables:
        frame = table.export_to_dataframe(doc=document)
        rows = [list(frame.columns)] + [
            [_value(v) for v in row] for row in frame.values.tolist()
        ]
        tables.append([table.prov[0].page_no if table.prov else 0, rows])
    return {"texts": texts, "tables": tables}


def main(argv) -> None:
    source, max_pages, output = argv[1], int(argv[2]), argv[3]
    from docling.document_converter import DocumentConverter

    document = DocumentConverter().convert(source, page_range=(1, max_pages)).document
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload(document), handle)


if __name__ == "__main__":
    main(sys.argv)

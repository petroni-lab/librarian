"""Full texts come from the PMC bucket on AWS first: its JATS, else its PDF (no network).

uv run pytest tests
"""

from unittest import mock

import requests

from librarian.jats import extract_body_paragraphs
from librarian.literature_search import EuropePmcSource

_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
_PROSE = "Cells were cultured for three days in standard medium before imaging. "
_BODY_XML = (
    "<article><body><sec><title>Results</title>"
    + "".join(f"<p>{_PROSE} Paragraph {i}.</p>" for i in range(4))
    + "</sec></body></article>"
)
# What PMC holds for an old scanned issue: metadata, no body.
_BODYLESS_XML = "<article><front><article-meta/></front></article>"


def _pdf(lines):
    """A one-page PDF whose text layer is *lines*, one per line."""
    text = "".join(f"({line}) Tj T* " for line in lines)
    stream = f"BT /F1 9 Tf 12 TL 40 800 Td {text}ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    return pdf + b"startxref\n%d\n%%%%EOF\n" % xref


def _response(content, status=200):
    response = mock.MagicMock(status_code=status, content=content)
    response.text = content.decode("utf-8", errors="replace")
    response.raise_for_status.return_value = None
    return response


def _listing(keys, sizes=None):
    sizes = sizes or {}
    items = "".join(
        f"<Contents><Key>{key}</Key><Size>{sizes.get(key, len(body))}</Size></Contents>"
        for key, body in keys.items()
    )
    return _response(
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + items.encode()
        + b"</ListBucketResult>"
    )


def _fetch(bucket, epmc_xml=_BODY_XML, pmcid="PMC1", sizes=None):
    """Fetch *pmcid* with *bucket* ``{key: bytes}`` on AWS; return it and the URLs asked."""
    asked = []

    def get(url, params=None, **kwargs):
        asked.append(url)
        if url == _BUCKET:
            return _listing(bucket, sizes=sizes)
        if url.startswith(_BUCKET):
            return _response(bucket[url[len(_BUCKET) :]])
        return _response(epmc_xml.encode())

    with mock.patch("librarian.literature_search.requests.get", side_effect=get):
        fulltext = EuropePmcSource().fetch_fulltext_many([pmcid])[pmcid]
    return fulltext, asked


def _europe_pmc_asked(asked):
    return [url for url in asked if not url.startswith(_BUCKET)]


def test_the_bucket_jats_is_read_without_asking_europe_pmc():
    fulltext, asked = _fetch(
        {"PMC1.1/PMC1.1.xml": b"old", "PMC1.2/PMC1.2.xml": _BODY_XML.encode()}
    )
    assert fulltext.ok and fulltext.xml == _BODY_XML  # the latest version
    assert fulltext.from_pdf is False
    assert not _europe_pmc_asked(asked)


def test_a_bodyless_jats_is_replaced_by_the_bucket_pdf():
    lines = []
    for heading in ("Introduction", "Methods", "Results"):
        lines += [heading, f"{_PROSE}In {heading.lower()}.", ""]
    fulltext, asked = _fetch(
        {
            "PMC1.1/PMC1.1.xml": _BODYLESS_XML.encode(),
            "PMC1.1/PMC1.1.pdf": _pdf(lines),
        }
    )
    records = extract_body_paragraphs(fulltext.xml)
    assert fulltext.ok and len(records) >= 3
    assert [r["section_title"] for r in records] == ["Introduction", "Methods", "Results"]
    # Evidence read from it can say it came from the PDF, not from JATS.
    assert fulltext.from_pdf is True
    assert not _europe_pmc_asked(asked)


def test_a_paper_missing_from_the_bucket_comes_from_europe_pmc():
    fulltext, asked = _fetch({})
    assert fulltext.ok and fulltext.xml == _BODY_XML
    assert len(_europe_pmc_asked(asked)) == 1


def test_a_broken_bucket_pdf_falls_back_to_europe_pmc():
    # A PDF whose stream decoding raises a non-PyPdfError (TypeError in pypdf)
    # must fall back to Europe PMC rather than crashing the batch.
    broken_pdf = _pdf(["placeholder"]).replace(
        b"<< /Length ", b"<< /Filter /CCITTFaxDecode /Length ", 1
    )
    fulltext, asked = _fetch(
        {
            "PMC1.1/PMC1.1.xml": _BODYLESS_XML.encode(),
            "PMC1.1/PMC1.1.pdf": broken_pdf,
        }
    )
    assert fulltext.ok and fulltext.xml == _BODY_XML
    assert len(_europe_pmc_asked(asked)) == 1


def test_a_preprint_id_skips_the_pmc_bucket():
    fulltext, asked = _fetch({}, pmcid="PPR123")
    assert fulltext.ok and fulltext.xml == _BODY_XML
    assert not any(url.startswith(_BUCKET) for url in asked)
    assert len(_europe_pmc_asked(asked)) == 1


def test_an_oversized_bucket_pdf_is_skipped():
    pdf_key = "PMC1.1/PMC1.1.pdf"
    fulltext, asked = _fetch(
        {"PMC1.1/PMC1.1.xml": _BODYLESS_XML.encode(), pdf_key: b"unused"},
        sizes={pdf_key: 100 * 1024 * 1024},
    )
    assert fulltext.ok and fulltext.xml == _BODY_XML
    assert _BUCKET + pdf_key not in asked
    assert len(_europe_pmc_asked(asked)) == 1


def test_an_aws_error_falls_back_to_europe_pmc():
    asked = []

    def get(url, params=None, **kwargs):
        asked.append(url)
        if url.startswith(_BUCKET):
            raise requests.exceptions.ConnectionError("S3 unreachable")
        return _response(_BODY_XML.encode())

    with mock.patch("librarian.literature_search.requests.get", side_effect=get):
        fulltext = EuropePmcSource().fetch_fulltext_many(["PMC1"])["PMC1"]
    assert fulltext.ok and fulltext.xml == _BODY_XML
    assert len(_europe_pmc_asked(asked)) == 1

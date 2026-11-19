"""``extract_body_paragraphs`` drops boilerplate sections marked only by title (#101)."""

from __future__ import annotations

from librarian.jats import extract_body_paragraphs

# Structure of PMC12275167: declarations as plain <sec>s, no sec-type.
_XML = """<article><body>
<sec><title>Results</title><p>Drug X reduced tumour volume.</p></sec>
<sec><title>Funding of public health programmes</title><p>Budgets rose in 2020.</p></sec>
<sec id="sec10"><title>Declaration</title>
  <sec><title>Ethics approval and consent to participate:</title><p>Approved by board.</p></sec>
  <sec><title>Consent for publication:</title><p>Not applicable.</p></sec>
  <sec><title>Conflict of Interest:</title><p>None declared.</p></sec>
  <sec><title>Funding:</title><p>No funding.</p></sec>
  <sec><title>Authors’ Contribution:</title><p>All authors wrote it.</p></sec>
  <sec><title>Acknowledgments:</title><p>We thank the lab.</p></sec>
</sec>
<sec><title>Competing interests</title><p>None.</p></sec>
<sec><title>Data Availability Statement.</title><p>On request.</p></sec>
</body></article>"""


def test_title_only_boilerplate_sections_are_dropped() -> None:
    titles = [r["section_title"] for r in extract_body_paragraphs(_XML)]
    assert titles == ["Results", "Funding of public health programmes"]


def test_supplementary_sec_type_is_dropped() -> None:
    # "supplementary" is the section_type reserved for supplementary-file
    # records; a body section carrying it crashed the run (#106).
    xml = """<article><body>
<sec><title>Results</title><p>Drug X reduced tumour volume.</p></sec>
<sec sec-type="supplementary"><title>Supplementary Information</title>
<p>Table S1 lists primers.</p></sec>
</body></article>"""
    titles = [r["section_title"] for r in extract_body_paragraphs(xml)]
    assert titles == ["Results"]

"""Every evidence record says whether its full-text body was read, and why not.

uv run --extra supplementary pytest tests
"""

import pytest

from librarian import LibrarianAgent, load_runtime_config
from librarian.literature_search import Fulltext

_JATS = (
    "<article><body><sec><title>Results</title>"
    "<p>One.</p><p>Two.</p><p>Three.</p></sec></body></article>"
)
_FRAGMENT_JATS = "<article><body><sec><p>Lone caption.</p></sec></body></article>"
_EMPTY_JATS = "<article><body></body></article>"

_OA = {"inEPMC": True, "isOpenAccess": True, "pmcid": "PMC1"}


@pytest.mark.parametrize(
    ("paper", "fulltexts", "status"),
    [
        (_OA, {"PMC1": Fulltext("PMC1", xml=_JATS)}, "extracted"),
        (_OA, {"PMC1": Fulltext("PMC1", xml=_EMPTY_JATS)}, "no_body"),
        (_OA, {"PMC1": Fulltext("PMC1", xml=_FRAGMENT_JATS)}, "short_body"),
        (_OA, {"PMC1": Fulltext("PMC1", error="HTTP 500", status=500)}, "no_body"),
        (_OA, {"PMC1": Fulltext("PMC1", error="timeout")}, "fetch_failed"),
        (_OA, {"PMC1": Fulltext("PMC1", error="HTTP 503", status=503)}, "fetch_failed"),
        (_OA, {}, "fetch_failed"),
        ({"inEPMC": True, "isOpenAccess": True}, {}, "no_body"),
        (
            {"inEPMC": True, "isOpenAccess": False, "pmcid": "PMC1"},
            {},
            "not_open_access",
        ),
        ({"inEPMC": False, "isOpenAccess": False}, {}, "not_in_epmc"),
    ],
)
def test_status_and_has_fulltext_follow_what_stage_2_read(paper, fulltexts, status):
    agent = LibrarianAgent(load_runtime_config(), llm_client=object())
    paper = {"abstract": "Abstract text.", **paper}
    agent._paragraph_records_for_paper(paper, fulltexts)
    record = agent._passage_from_paper(paper)
    assert record["fulltext_status"] == status
    assert record["has_fulltext"] is (status in {"extracted", "short_body"})
    assert bool(record["fulltext_note"]) is (status != "extracted")


def test_enrichment_off_is_its_own_status():
    agent = LibrarianAgent(
        load_runtime_config(), llm_client=object(), full_text_enrichment=False
    )
    paper = dict(_OA, abstract="Abstract text.")
    agent._paragraph_records_for_paper(paper, {})
    assert agent._passage_from_paper(paper)["fulltext_status"] == "enrichment_disabled"

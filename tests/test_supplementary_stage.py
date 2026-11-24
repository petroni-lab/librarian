"""Stage 3b/3c: the supplementary files of the papers the judge kept, through ``run``.

uv run --extra supplementary pytest tests
"""

import json
import re
import threading
from dataclasses import replace

from librarian import LibrarianAgent, load_runtime_config
from librarian.literature_search import EuropePmcSource, Fulltext, Supplementary

_SENTENCE_RE = re.compile(r'"id": "(paragraph_\d+_[A-Z]+)",\s*"text": "((?:[^"\\]|\\.)*)"')


class _LLM:
    """Plans one query; as the judge, cites every sentence containing ``CITE``."""

    def __init__(self, fail_on_supplementary=False):
        self.fail_on_supplementary = fail_on_supplementary

    def chat_completion(self, messages, **kwargs):
        prompt = " ".join(m["content"] for m in messages)
        if '"sentences"' not in prompt:
            return json.dumps({"queries": ["zebrafish fin regeneration"]})
        if self.fail_on_supplementary and "Table S2" in prompt:
            raise TimeoutError("judge timed out")
        cited = [sid for sid, text in _SENTENCE_RE.findall(prompt) if "CITE" in text]
        return json.dumps({"relevant_ids": cited})


def _paper(n, body):
    pmcid = f"PMC{n}"
    paper = {
        "pmid": str(n),
        "pmcid": pmcid,
        "isOpenAccess": True,
        "inEPMC": True,
        "fullTextIds": [pmcid],
        "title": f"Paper {n}",
        "abstract": "",
        "year": "2024",
    }
    xml = (
        f"<article><body><p>{body}</p></body><back>"
        "<supplementary-material/></back></article>"
    )
    return paper, Fulltext(pmcid, xml=xml, status=200)


def _record(text, name="s1.txt"):
    return {
        "text": text,
        "section_title": f"Supplementary: {name}",
        "section_type": "supplementary",
        "supplementary_file": name,
    }


class _Source(EuropePmcSource):
    """Two papers; the judge keeps only paper 1, whose body sentence says CITE."""

    def __init__(self, supplementary_records):
        pairs = [
            _paper(1, "Zebrafish fins regenerate within two weeks CITE."),
            _paper(2, "Mice do not regenerate digits after amputation."),
        ]
        self.papers = [paper for paper, _ in pairs]
        self.fulltexts = {fulltext.pmcid: fulltext for _, fulltext in pairs}
        self.supplementary_records = supplementary_records
        self.fetched = []

    def search(self, query, page_size):
        return [dict(p) for p in self.papers]

    def fetch_fulltext_many(self, pmcids):
        return {p: self.fulltexts[p] for p in pmcids if p}

    def fetch_supplementary_many(self, pmcids, options):
        self.fetched.extend(pmcids)
        return {
            p: Supplementary(p, records=self.supplementary_records, status=200)
            for p in pmcids
        }


def _run(source, llm=None, **config_overrides):
    overrides = {"supplementary_stage": True, **config_overrides}
    config = replace(load_runtime_config(), **overrides)
    agent = LibrarianAgent(config, llm_client=llm or _LLM(), literature_source=source)
    return agent.run("How fast do zebrafish fins regenerate?")


def test_a_cited_supplementary_span_is_appended_to_the_kept_papers_evidence():
    source = _Source([_record("Table S2: fins regrew 95% by day 14 CITE.")])
    result = _run(source)
    assert source.fetched == ["PMC1"], "only the paper the judge kept is read"
    [passage] = result.papers
    assert passage["evidence_snippets"] == [
        "Zebrafish fins regenerate within two weeks CITE.",
        "Table S2: fins regrew 95% by day 14 CITE.",
    ]
    assert passage["evidence_sections"][1] == "Supplementary: s1.txt"
    assert result.debug["supplementary_evidence_count"] == 1
    assert result.debug["supplementary_stage"]["records"] == 1


def test_a_failed_supplementary_judge_keeps_stage_3s_evidence():
    source = _Source([_record("Table S2: fins regrew 95% by day 14 CITE.")])
    result = _run(source, llm=_LLM(fail_on_supplementary=True))
    [passage] = result.papers
    assert passage["evidence_snippets"] == [
        "Zebrafish fins regenerate within two weeks CITE."
    ]
    assert result.debug["supplementary_evidence_count"] == 0


def test_with_the_stage_off_no_supplementary_file_is_fetched():
    source = _Source([_record("Table S2: fins regrew 95% by day 14 CITE.")])
    _run(source, supplementary_stage=False)
    assert source.fetched == []


def test_papers_past_the_deadline_lose_only_their_supplementary_records():
    release = threading.Event()

    class SlowSource(_Source):
        def fetch_supplementary_many(self, pmcids, options):
            release.wait(5)
            return super().fetch_supplementary_many(pmcids, options)

    source = SlowSource([_record("Table S2: fins regrew 95% by day 14 CITE.")])
    try:
        result = _run(source, supplementary_deadline_seconds=0.1)
    finally:
        release.set()
    [passage] = result.papers
    assert passage["evidence_snippets"] == [
        "Zebrafish fins regenerate within two weeks CITE."
    ]
    assert result.debug["supplementary_stage"]["papers_with_records"] == 0

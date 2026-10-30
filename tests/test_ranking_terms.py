"""Stage-1 ranking terms reach both Stage-2 cuts (#91).

uv run --extra supplementary pytest tests
"""

import json

from librarian import LibrarianAgent, SearchPlan, agent, load_runtime_config


class _PlannerLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat_completion(self, messages, **kwargs):
        self.calls += 1
        return self.reply


def test_the_planner_reply_carries_the_ranking_terms_without_another_call():
    llm = _PlannerLLM(
        json.dumps(
            {
                "queries": ["lung adenocarcinoma never smokers"],
                "ranking_terms": ["smoking status", " Non-smoker", "non-smoker", 7, ""],
            }
        )
    )
    librarian = LibrarianAgent(load_runtime_config(), llm_client=llm)
    plan = librarian._generate_queries("How many never smokers?")
    assert plan == SearchPlan(
        ["lung adenocarcinoma never smokers"], ["smoking status", "Non-smoker", "7"]
    )
    assert llm.calls == 1


def test_a_reply_without_ranking_terms_ranks_by_the_subquery_alone():
    llm = _PlannerLLM(json.dumps({"queries": ["q"], "ranking_terms": "smoke"}))
    librarian = LibrarianAgent(load_runtime_config(), llm_client=llm)
    assert librarian._generate_queries("q").ranking_terms == []


def test_ranking_terms_are_capped():
    terms = [f"term{i}" for i in range(30)]
    assert agent._parse_ranking_terms({"ranking_terms": terms}) == terms[
        : agent._MAX_RANKING_TERMS
    ]


def test_without_terms_the_fused_ranking_is_the_subquery_ranking():
    docs = ["smoke no yes", "never smokers cohort", "tumor immune", "smoking status"]
    for terms in ((), [], ["", "  "]):
        assert agent._fused_rank("never smokers", terms, docs) == agent._bm25_rank(
            "never smokers", docs
        )


def test_a_document_matching_neither_channel_scores_zero():
    docs = ["smoke no yes", "never smokers cohort", "tumor immune"]
    fused = dict(agent._fused_rank("never smokers", ["smoke"], docs))
    assert fused[0] > 0 and fused[1] > 0
    assert fused[2] == 0


# ── q23 regression: "never smokers" vs Table S1's "Smoke | No" ──────────────
#
# SuppQA q23 asks how many never smokers PMC10960411's datasets hold (179). The
# answer is a Docling record of Table S1 whose only smoking word is "Smoke":
# Snowball stems the sub-query's "smokers" to "smoker" and "Smoke" to "smoke",
# so BM25 against the sub-query scores it 0. The four records below are the
# paper's real Docling records (indices 57, 59, 61 and 68 of 103), verbatim;
# the rest of the paper is replaced by tables that do match the sub-query, as
# its real lncRNA tables do.

_ROW = (
    "Accession | TCGA-LUAD | GSE30219 | GSE31210 | GSE3141 | GSE50081 | GSE72094 ; "
    "Smoke ; No | -- | -- | 123 (50.0%) | -- | 23 (18.1%) | 33 (7.5%) ; "
    "Yes | -- | -- | 123 (50.0%) | -- | 92 (72.4%) | 335 (75.8%) ; "
    "Not available | -- | -- | -- | -- | 12 (9.4%) | 74 (16.7%) ; "
    "EGFR ; Mut | -- | -- | 127 (51.6%) | -- | -- | 47 (10.6%) ; "
    "WT | -- | -- | 99 (40.2%) | -- | -- | 395 (89.4%)"
)
_OTHER_SMOKE_RECORDS = {
    57: "Cohort | Clinical Characteristics | HR | HR.95L | HR.95H | P-value ; "
    "GSE31210 | TRLS (Risk score) | 9.051775 | 2.508588 | 32.66165 | 0.000766 ; "
    "GSE31210 | Age (>65 vs. <=65) | 4.2514 | 2.005999 | 9.010178 | 0.000159 ; "
    "GSE31210 | Gender (Male vs. Female) | 0.937433 | 0.339494 | 2.5885 | 0.900779 ; "
    "GSE31210 | Stage (II vs. I) | 4.152957 | 2.012823 | 8.568588 | 0.000117 ; "
    "GSE31210 | Smoke (Yes vs. No) | 1.057295 | 0.380189 | 2.940307 | 0.914977 ; "
    "GSE31210 | EGFR (Yes vs. No) | 0.762918 | 0.35264 | 1.650535 | 0.491908",
    59: "0 | 1 | 2 | 3 | 4 | 5 ; "
    "GSE50081 | Stage (II vs. I) | 2.0483 | 1.068326 | 3.927206 | 0.030852 ; "
    "GSE50081 | T stage (T3-T4 vs. T1-T2) | 3.326755 | 0.669704 | 16.52566 | 0.141632 ; "
    "GSE50081 | N Stage (N1 vs. N0) | 1.558513 | 0.84828 | 2.863397 | 0.152781 ; "
    "GSE50081 | Smoke (Yes vs. No) | 0.948476 | 0.39599 | 2.271788 | 0.905513 ; "
    "GSE72094 | TRLS (Risk score) | 7.92721363 | 3.230854782 | 19.45018275 | 6.16E-06 ; "
    "GSE72094 | Age (>65 vs. <=65) | 1.135667755 | 0.692246387 | 1.863124567 | 0.614472912",
    61: "0 | 1 | 2 | 3 | 4 | 5 ; "
    "GSE72094 | Smoke (Yes vs. No) | 1.038127287 | 0.427795763 | 2.519212102 | 0.934069567",
}
_ROW_INDEX = 68
_SUBQUERY = 'TITLE:"lncRNA signature" AND TITLE:"lung adenocarcinoma"'
# What the live planner (glm-5.3-flash) returned for q23 with this prompt.
_TERMS = ["never smokers", "smoking status", "non-smoker", "smoker", "tobacco"]


def _paper_records():
    records = []
    for i in range(103):
        if i == _ROW_INDEX:
            text, name = _ROW, "12915_2024_1866_MOESM7_ESM.pdf"
        elif i in _OTHER_SMOKE_RECORDS:
            text, name = _OTHER_SMOKE_RECORDS[i], "12915_2024_1866_MOESM14_ESM.pdf"
        else:
            text = (
                "lncRNA | log2FC | P-value | signature coefficient ; "
                f"LINC{i:05d} | {i % 7}.2 | 0.00{i % 9 + 1} | 0.{i:03d} ; "
                f"AC{i:06d}.1 | -{i % 5}.4 | 0.0{i % 3 + 1} | -0.{i:03d} ; "
                "lung adenocarcinoma TME-related lncRNA signature genes"
            )
            name = f"12915_2024_1866_MOESM{8 + i % 6}_ESM.pdf"
        records.append(
            {
                "text": text,
                "section_title": f"Supplementary: {name} (p. 1, table)",
                "section_type": "supplementary",
                "supplementary_file": name,
            }
        )
    return records


# Competing papers a broad lncRNA/LUAD sub-query returns: every one matches the
# sub-query, and some discuss smoking in their body text, as PMC9169227 does.
def _competing_paper(n):
    smoking = (
        " Smoking status was recorded for each patient: never smokers, former "
        "smokers and current smokers were compared, and the risk score did not "
        "differ by smoking history or tobacco exposure."
        if n % 10 == 0
        else ""
    )
    abstract = (
        f"We built a {n}-lncRNA signature for lung adenocarcinoma from TCGA and "
        "validated its prognostic value; high-risk patients had shorter survival."
    )
    body = [
        f"The lncRNA signature stratified lung adenocarcinoma patients in cohort {n} "
        "into high- and low-risk groups with distinct immune infiltration." + smoking,
        "Functional enrichment linked the signature lncRNAs to cell cycle and "
        f"immune pathways in lung adenocarcinoma samples (set {n}).",
    ]
    return abstract, body


def _stage2(ranking_terms):
    """The real Stage 2 for one sub-query: (cut-1 indices kept, cut-2 texts kept)."""
    from dataclasses import replace

    from librarian.literature_search import EuropePmcSource, Fulltext, Supplementary

    papers, fulltexts = [], {}
    target = {
        "pmid": "38515127",
        "pmcid": "PMC10960411",
        "isOpenAccess": True,
        "fullTextIds": ["PMC10960411"],
        "title": "Single-cell RNA sequencing integrated with bulk RNA sequencing "
        "analysis identifies a tumor immune microenvironment-related lncRNA "
        "signature in lung adenocarcinoma",
        "abstract": "We identified a TME-related lncRNA signature (TRLS) in lung "
        "adenocarcinoma from single-cell and bulk RNA sequencing.",
        "year": "2024",
    }
    papers.append(target)
    fulltexts["PMC10960411"] = Fulltext(
        "PMC10960411",
        xml="<article><body><p>The TRLS lncRNA signature predicted survival in lung "
        "adenocarcinoma cohorts.</p></body><back><supplementary-material/></back>"
        "</article>",
        status=200,
    )
    for n in range(30):
        pmcid = f"PMC9{n:06d}"
        abstract, body = _competing_paper(n)
        papers.append(
            {
                "pmid": str(n),
                "pmcid": pmcid,
                "isOpenAccess": True,
                "fullTextIds": [pmcid],
                "title": f"A lncRNA signature in lung adenocarcinoma {n}",
                "abstract": abstract,
                "year": "2023",
            }
        )
        fulltexts[pmcid] = Fulltext(
            pmcid,
            xml="<article><body>"
            + "".join(f"<p>{p}</p>" for p in body)
            + "</body></article>",
            status=200,
        )
    outcome = Supplementary("PMC10960411", records=_paper_records(), status=200)

    class Source(EuropePmcSource):
        def search(self, query, page_size):
            return [dict(p) for p in papers]

        def fetch_fulltext_many(self, pmcids):
            return {p: fulltexts[p] for p in pmcids}

        def fetch_supplementary_many(self, pmcids, options):
            return {p: outcome for p in pmcids if p == "PMC10960411"}

    config = replace(load_runtime_config(), supplementary_enrichment=True)
    librarian = LibrarianAgent(config, llm_client=object(), literature_source=Source())
    cut1 = []
    original = agent._top_supplementary

    def capture(subquery, supplementary, k, terms=()):
        kept = original(subquery, supplementary, k, terms)
        cut1.extend(r["supplementary_index"] for r in kept.records)
        return kept

    from unittest import mock

    with mock.patch("librarian.agent._top_supplementary", side_effect=capture):
        top = librarian._paragraphs_for_subquery(_SUBQUERY, ranking_terms)
    assert config.max_supplementary_records_per_paper == 40
    assert config.paragraphs_per_subquery == 16
    return cut1, [r["text"] for r in top]


def test_q23_row_is_lost_at_both_cuts_without_ranking_terms():
    cut1, cut2 = _stage2([])
    assert _ROW_INDEX not in cut1
    assert not any("123 (50.0%)" in text for text in cut2)


def test_q23_row_survives_both_cuts_with_ranking_terms_unchanged():
    cut1, cut2 = _stage2(_TERMS)
    assert _ROW_INDEX in cut1
    kept = [text for text in cut2 if "123 (50.0%)" in text]
    assert kept, "the Table S1 row must reach Stage 3"
    # The judge is shown the record as extracted: the terms only rank it.
    assert all(cell in kept[0] for cell in ("23 (18.1%)", "33 (7.5%)", "Smoke ; No"))

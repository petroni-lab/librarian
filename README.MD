<div align="center">

<img src="assets/ai-librarian.png" alt="EMBL AI Librarian logo" width="140">

<h1><span style="color:#18974C">EMBL AI Librarian</span></h1>

<h3>A life-sciences knowledge layer for AI agents</h3>

<!-- TODO: project page, once one exists
<h4><a href="https://your-project-page.example">Project Page 🚀</a></h4>
-->

<!--
  TODO: capture a real run and drop it here as assets/demo.gif or assets/teaser.png
  (e.g. a terminal recording of `uv run main.py "<question>"` showing the spinner
  stages and the final ranked evidence passages).
-->
<!-- <img src="assets/PLACEHOLDER-demo.gif" alt="Demo: a natural-language question going in, ranked evidence passages coming out" width="90%"> -->

[![Paper](https://img.shields.io/badge/arXiv-2607.28229-b31b1b.svg)](https://arxiv.org/abs/2607.28229)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=petroni-lab.librarian&format=true)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Europe PMC](https://img.shields.io/badge/data-Europe%20PMC%20(40M%2B%20records)-0A5032.svg)](https://europepmc.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)
[![Stars](https://img.shields.io/github/stars/petroni-lab/librarian?style=social)](https://github.com/petroni-lab/librarian/stargazers)



<h5>If EMBL AI Librarian is useful to you, please consider giving it a ⭐</h5>

</div>

---

## 🛠️ Installation

**Prerequisites:** Python 3.10+

The project is managed with [uv](https://docs.astral.sh/uv/). First install uv itself:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repo. A single `uv sync` creates the virtual environment (fetching a
compatible Python if needed) and installs every dependency:
```bash
git clone https://github.com/petroni-lab/librarian.git
cd librarian
uv sync
```

## ▶️ Quickstart

The agent needs an OpenAI-compatible LLM endpoint (a self-hosted vLLM server or a remote
provider). Configure it in a `.env` file — copy the template and edit:

```bash
cp .env.example .env
# then edit .env:
#   LLM_BASE_URL=http://localhost:8000/v1
#   LLM_MODEL=your-model-name
#   LLM_API_KEY=EMPTY   # or your provider key

uv run main.py "does metformin extend lifespan in mammals?"
```

`main.py` loads `.env` automatically. You can also just export the same three variables
in your shell.

> **Reproducing the paper's numbers.** All reported results use **GLM-5**
> (`glm-5-fp8`), self-hosted on an NVIDIA DGX B200 and served through
> [vLLM](https://github.com/vllm-project/vllm), for every LLM stage: subquery planning,
> relevance filtering, and synthesis. Librarian is model-agnostic and runs with any
> OpenAI-compatible endpoint, but smaller or weaker models will produce noticeably
> different evidence quality — the planner in particular has to emit valid Europe PMC
> fielded syntax. See [⚙️ Configuration](#️-configuration) for the exact retrieval
> settings used in all experiments.

On a terminal you'll see a live spinner reporting the current stage (`Searching Europe
PMC`, `Fetching full text · 42/210`, `Judging paper relevance · 6/28 batches`, ...) so
you can tell how far along a run is. When the output is piped/redirected the spinner is
skipped and the agent prints plain `[Librarian]` log lines instead.
`LibrarianAgent.run()` accepts an `on_progress=callable` hook if you want to drive your
own UI.

---

## 📰 News

* **[2026.07.31]** Preprint available on arXiv.
* **[2026.07.30]** Initial release of the EMBL AI Librarian agent and retrieval pipeline.

---

## 😮 Highlights

### 🗣️ Natural language in, ranked evidence out

Europe PMC and similar resources were not built for AI agents: they take keywords and
complex query syntax and return whole papers, so every agent has to learn the syntax,
issue several searches, and read full papers just to find the evidence it needs.
**EMBL AI Librarian** upgrades that interface — an agent asks a question the way a person
would, and gets back the passages that answer it, each paired with citation metadata and
the Europe PMC query that retrieved it.

### 🧭 No index to maintain: one planning call, deterministic retrieval, one judging pass

Librarian keeps no dense vector index of its own — it queries Europe PMC's live search
directly. One LLM call plans complementary sub-queries; everything in between is
deterministic and cheap (query validation, search, full-text fetch, BM25 paragraph
ranking, hierarchical dedup). A single batched LLM pass then filters, re-ranks, and
extracts the final evidence sentences.

### 📈 Consistent gains across four benchmark suites

Equipping life-science agents with the Librarian knowledge layer helps on all four
suites we evaluate:

| Benchmark | Metric | Baseline | + Librarian |
|---|---|---|---|
| ScholarQA-Bench (Bio) | Citation F1 | 56.7 (PaperQA2) / 67.0 (same agent, BM25 on OSDS) | **73.8** |
| ScholarQA-Bench (Neu) | Citation F1 | 63.1 (OpenScholar-70B) / 68.5 (same agent, BM25 on OSDS) | **79.4** |
| ProClaim-eval | Agreement w/ expert consensus | 0.75 | **0.80** |
| Open-form LitQA2 | Accuracy | 70.3 (GPT-5.4 + web search) | **78.9** |
| LAB-Bench | Macro accuracy (GPT-5.4) | 50.5 | **54.6** (+11.3 on SeqQA) |

Two comparisons matter and they answer different questions. Against **strong published
baselines**, Librarian is more than **+16 Citation F1** on ScholarQA-Bench — but part of
that gap comes from our synthesis agent's larger backbone (GLM-5, ~700B, vs. ~70B),
not from retrieval. The controlled comparison — same agent, same prompt, swapping a BM25
index over the OpenScholar Data Store for Librarian — isolates the knowledge layer and
gives **+6.8 Bio / +10.9 Neu**. See the paper for the full evaluation and ablations.

---

## 🚀 Main Results

<!--
  TODO: drop in a real benchmark chart, e.g. assets/results.png —
  bar chart of Citation F1 / LitQA2 accuracy vs. baselines.
-->
<div align="center"><img src="assets/benchmark_summary.png" alt="Benchmark results: Librarian vs. baselines on ScholarQA-Bench, ProClaim-eval, open-form LitQA2, and LAB-Bench" width="80%"></div>

---

## 🔄 Pipeline Overview

`LibrarianAgent.run(query)` is a single pass (`librarian/agent.py`):

<div align="center"><img src="assets/PMC_Librarian_v5-1.png" alt="Pipeline diagram: natural-language query to sub-queries, Europe PMC retrieval with BM25 paragraph ranking, then filtering and evidence extraction" width="80%"></div>


<details>
<summary><b>Stage 1: Subqueries Generation</b></summary>

The LLM turns the question into up to N complementary Europe PMC sub-queries
(`prompts/europepmc_claude_librarian.md`). Each sub-query is checked by a deterministic
syntax validator (`query_validation/`, pure parsing — no LLM call); invalid field names
and malformed Boolean expressions are rejected.

Two conservative fallbacks guard the degenerate cases:

- **Planner output unparseable** — retried once at temperature 0 with a strict-JSON
  instruction. If that also fails, the raw user query is used as the single search query.
- **Every sub-query rejected by the validator** — the planned sub-queries are searched
  unmodified, so a validator false positive can never empty the search plan. (The raw
  user query is *not* substituted in this case.)

</details>

<details>
<summary><b>Stage 2: Paper-level Retrieval + Paragraph BM25 Ranking</b></summary>

Validated sub-queries run in parallel against the live Europe PMC search service, each
returning up to P records (pool of at most M = N × P).

Because complementary sub-queries often return the same paper, candidates are
**deduplicated here, before any full text is fetched**, so a paper found by several
sub-queries is fetched exactly once. Deduplication is hierarchical: PMID first, then DOI
when a PMID is absent, then title — PMID alone is insufficient because Europe PMC also
indexes preprints and some PMC content without one.

For each surviving open-access paper, Librarian fetches the JATS XML (cached by PMCID),
extracts the body paragraphs and discards references and other front/back matter
(`literature_search.py`, `jats.py`), then BM25-ranks those paragraphs against the
original question — each paragraph concatenated with its section title and type. Every
paper passes k paragraphs onward: always its abstract, plus the highest-ranked
paragraphs when full text is available, keeping the top ~2300 words as a
`full_text_excerpt`. Papers without full text contribute their abstract only.

</details>

<details>
<summary><b>Stage 3: Filter, Re-ranking & Evidence Extraction</b></summary>

Paragraphs are segmented into sentences with pySBD and each sentence is given a unique
identifier. Two judges run concurrently: a full-text judge over papers that carry an
excerpt (`prompts/relevance_filter_fulltext.md`) and an abstract triage over
abstract-only candidates (`prompts/relevance_filter.md`). Each call returns an ordered
list of sentence identifiers, which simultaneously filters irrelevant paragraphs,
re-ranks the relevant ones, and extracts the supporting sentences that become each
paper's `evidence_snippets`. Because the model emits short identifiers rather than
regenerating text, it decodes far fewer tokens.

One last fallback: if both relevance filters reject every candidate, the unfiltered
candidate pool is returned instead of an empty evidence set — trading precision for
non-empty output.

</details>

The result is a list of passage dicts:

```python
{
  "evidence_snippets": ["...cited sentence span...", ...],
  "paper_id": "12345678",
  "title": "...", "authors": "...", "journal": "...", "year": "2023",
  "pmid": "12345678", "doi": "10.xxxx/...", "has_fulltext": True,
}
```

---

## ⚙️ Configuration

Defaults live in `librarian/config.py`, and are identical to the settings used in every
experiment reported in the paper. A few knobs can be overridden via env vars without
editing code:

| Env var | Default | Meaning |
|---|---|---|
| `LITERATURE_MAX_QUERY_COUNT` | 7 | max sub-queries generated |
| `LITERATURE_MAX_FULLTEXT_PAPERS` | 30 | full-text papers in the answer |
| `LITERATURE_MAX_ABSTRACT_ONLY_PAPERS` | 30 | abstract-only papers in the answer |
| `LITERATURE_FULL_TEXT_TARGET_TOKENS_PER_PAPER` | 3072 | per-paper excerpt budget |
| `LITERATURE_EVIDENCE_SNIPPET_MAX_WORDS` | 250 | evidence words per paper |
| `LITERATURE_FILTER_TEMPERATURE` | 0.1 | relevance-judge temperature |

---

## 📁 Layout

```
pyproject.toml                 project metadata + dependencies (managed by uv)
main.py                        CLI entrypoint
librarian/
  agent.py                     the retrieval pipeline (start here)
  config.py                    tuning knobs (env-var overridable)
  llm_client.py                minimal OpenAI-compatible client
  literature_search.py         Europe PMC search
  jats.py                      JATS full-text XML -> body paragraphs
  progress.py                  stdlib terminal spinner for live progress
  prompts/                     query planner + relevance-judge prompts
  query_validation/            deterministic Europe PMC query validation
```

---

## ⚠️ Scope & Limitations

Worth knowing before you plug Librarian into a pipeline:

- **Coverage is bounded by Europe PMC.** The full text of paywalled papers is not
  available; those records contribute their abstract only. Extending coverage to other
  EMBL resources (OpenTargets, UniProt, ChEMBL) is planned.
- **Single retrieval round.** Librarian issues multiple complementary sub-queries to
  improve recall, but it does not loop. Multi-hop questions are left to the downstream
  agent. Letting Librarian decide whether to keep searching is future work.
- **Text only.** Figures and supplementary material are ignored, and tables are handled
  as text, so evidence living in those places is currently inaccessible. This is why our
  LAB-Bench evaluation excludes FigQA, TableQA, and SuppQA.
---

## 🤝 Contributing

Issues and pull requests are welcome.

---

## 📄 Citation

If you use EMBL AI Librarian in your research, please cite:

```bibtex
@misc{sigillo2026emblailibrarianlifesciences,
      title={EMBL AI Librarian: Life-Sciences Knowledge Layer for AI Agents}, 
      author={Luigi Sigillo and Matteo Silvestri and Francesco Tabaro and Rajat Bhatnagar and Syed Irtaza Mubashar and Matt Jeffryes and Daljit Nijjer and Vittorio Perera and Ola Spjuth and Julio Saez-Rodriguez and Melissa Harrison and Fabio Petroni},
      year={2026},
      eprint={2607.28229},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.28229}, 
}
```

---

## 🙏 Acknowledgements

We thank the maintainers of:

- **[Europe PMC](https://europepmc.org/)** — the literature search and full-text API this project builds on
- **[uv](https://docs.astral.sh/uv/)** — Python packaging and environment management
- **[bm25s](https://github.com/xhluca/bm25s)** — fast BM25 paragraph ranking
- **[pysbd](https://github.com/nipunsadvilkar/pySBD)** — sentence boundary detection for evidence extraction

---

## 🌟 Star History

<!-- NOTE: the sealed_token below is embedded in a public file — confirm with
     star-history what it grants before leaving it in the repo. -->

<a href="https://www.star-history.com/?repos=petroni-lab%2Flibrarian&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=petroni-lab/librarian&type=date&theme=dark&legend=top-left&sealed_token=7YgSzSG-SjWbF5xWHHCFQtFqKLapQcVuOL42QMnicXID2d2AbFdS_vqV8pxk-39o8e-i4oeUzGYgEYi2iYQJHVxVVCKGohjFjutFhp25g23bUxe7gZ9LbhXfaB0_AlK5x3pig7NRcSuM_545_CQCKU4zdLXdbRX8bnLwhTPoNO_gF0wTkS2a9ZgMzpGh" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=petroni-lab/librarian&type=date&legend=top-left&sealed_token=7YgSzSG-SjWbF5xWHHCFQtFqKLapQcVuOL42QMnicXID2d2AbFdS_vqV8pxk-39o8e-i4oeUzGYgEYi2iYQJHVxVVCKGohjFjutFhp25g23bUxe7gZ9LbhXfaB0_AlK5x3pig7NRcSuM_545_CQCKU4zdLXdbRX8bnLwhTPoNO_gF0wTkS2a9ZgMzpGh" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=petroni-lab/librarian&type=date&legend=top-left&sealed_token=7YgSzSG-SjWbF5xWHHCFQtFqKLapQcVuOL42QMnicXID2d2AbFdS_vqV8pxk-39o8e-i4oeUzGYgEYi2iYQJHVxVVCKGohjFjutFhp25g23bUxe7gZ9LbhXfaB0_AlK5x3pig7NRcSuM_545_CQCKU4zdLXdbRX8bnLwhTPoNO_gF0wTkS2a9ZgMzpGh" />
 </picture>
</a>

---

## 📝 License

Released under the MIT License. See [LICENSE](LICENSE) for details.
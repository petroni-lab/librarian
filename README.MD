# Librarian

A minimal, self-contained **literature retrieval agent**. Give it a
natural-language biology question; it returns ranked evidence passages from
[Europe PMC](https://europepmc.org/), each with paper metadata and the exact
sentences that support relevance.

## The flow

`LibrarianAgent.run(query)` is a single pass (`librarian/agent.py`):

1. **Generate queries** — the LLM turns the question into up to 7 complementary
   Europe PMC sub-queries (`prompts/europepmc_claude_librarian.md`).
2. **Validate queries** — drop sub-queries with broken Europe PMC field syntax
   (`query_validation/`, pure deterministic parsing).
3. **Search + enrich** (parallel per sub-query) — search Europe PMC, fetch each
   open-access paper's JATS full text (`literature_search.py`, `jats.py`), and
   BM25-rank its paragraphs against the original question, keeping the top
   ~2300 words as a `full_text_excerpt`.
4. **Dedup** candidate papers across sub-queries.
5. **Two-pass relevance filter** (concurrent) — a full-text judge over papers
   with an excerpt (`prompts/relevance_filter_fulltext.md`) plus an abstract
   triage over all candidates (`prompts/relevance_filter.md`). The judge cites
   the supporting sentences, which become each paper's `evidence_snippets`.

The result is a list of passage dicts:

```python
{
  "evidence_snippets": ["...cited sentence span...", ...],
  "paper_id": "12345678",
  "title": "...", "authors": "...", "journal": "...", "year": "2023",
  "pmid": "12345678", "doi": "10.xxxx/...", "has_fulltext": True,
}
```

## Install

```bash
pip install -r requirements.txt
```

## Run

The agent needs an OpenAI-compatible LLM endpoint (a self-hosted vLLM server or
a remote provider). Configure it in a `.env` file — copy the template and edit:

```bash
cp .env.example .env
# then edit .env:
#   LLM_BASE_URL=http://localhost:8000/v1
#   LLM_MODEL=your-model-name
#   LLM_API_KEY=EMPTY   # or your provider key

python main.py "does metformin extend lifespan in mammals?"
```

`main.py` loads `.env` automatically. You can also just export the same three variables in your shell.

On a terminal you'll see a live spinner reporting the current stage
(`Searching Europe PMC`, `Fetching full text · 42/210`, `Judging paper
relevance · 6/28 batches`, ...) so you can tell how far along a run is. When the
output is piped/redirected the spinner is skipped and the agent prints plain
`[Librarian]` log lines instead. `LibrarianAgent.run()` accepts an
`on_progress=callable` hook if you want to drive your own UI.

## Layout

```
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

## Configuration

Defaults live in `librarian/config.py` (the values that gave the best LitQA2
accuracy). A few knobs can be overridden via env vars without editing code:

| Env var | Default | Meaning |
|---|---|---|
| `LITERATURE_MAX_QUERY_COUNT` | 7 | max sub-queries generated |
| `LITERATURE_MAX_FULLTEXT_PAPERS` | 30 | full-text papers in the answer |
| `LITERATURE_MAX_ABSTRACT_ONLY_PAPERS` | 30 | abstract-only papers in the answer |
| `LITERATURE_FULL_TEXT_TARGET_TOKENS_PER_PAPER` | 3072 | per-paper excerpt budget |
| `LITERATURE_EVIDENCE_SNIPPET_MAX_WORDS` | 250 | evidence words per paper |
| `LITERATURE_FILTER_TEMPERATURE` | 0.1 | relevance-judge temperature |

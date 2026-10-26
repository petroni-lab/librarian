# Librarian run fixtures

One recorded case is a real Librarian run played back offline. The recorder runs
`LibrarianAgent.run` once against an LLM endpoint and Europe PMC, capturing what
the agent's external calls answered; `test_recorded_run.py` hands those answers
back through the agent's `llm_client` and `literature_source` and calls the same
`run` again. Everything in between — sub-query validation, Stage 2's thread
fan-out and BM25 ranking, paragraph merging, Stage 3's sentence registry,
batching and span regrouping, evidence extraction — is the agent's own
deterministic logic and runs for real in both directions.

Neither side re-implements a stage, so a fixture always describes what the agent
actually does. What both sides must agree on lives in one place,
[`fixture_replay.py`](../fixture_replay.py), because a call is only replayable if
the recorder and the tests key it identically.

A replay catches regressions: it says the pipeline still does what it did when
the case was recorded. It does not say that what it did was right.

## Record a case

With the LLM endpoint reachable, run this from the repository root:

```bash
uv run python tests/record_fixture.py \
  --case macrophage_activation \
  --query "How does interferon-gamma activate inflammatory gene expression in human macrophages?"
```

The script uses `http://127.0.0.1:8000/v1`, `glm-5.3-flash`, and the small
[`recording.toml`](recording.toml) configuration (`--config` to pick another),
and enables full-text enrichment. If the endpoint needs a key, set
`LLM_API_KEY`; it is never written to the fixture. Europe PMC is queried
directly, never through `LIBRARIAN_CACHE_DIR`.

Use `--overwrite` to replace an existing case after an intended change to the
pipeline. Nothing is written until the run completes, so a failed recording
leaves the previous case untouched.

Adding a case is a recording — the tests parametrize over every directory under
`integration/`, so no test needs editing.

Before committing a case, check that every `boundaries/fulltext/*.xml` is
non-empty and comes from an article whose licence allows redistribution (CC BY
or similar): the full text is committed verbatim.

## Files

```text
integration/<case>/
├── 00_run.json                    query, date, model, agent construction, config,
│                                  BM25 stemming
├── 01_generated_queries.json      Stage 1: parsed, deduplicated, capped sub-queries
├── 02_validated_queries.json      Stage 1: those that passed EPMC syntax validation
├── 03_subquery_paragraphs.json    Stage 2: each sub-query's top-k selections
├── 04_merged_paragraphs.json      Stage 2: the merged pool Stage 3 judges
├── 05_relevant_papers.json        Stage 3: cited papers, in rank order, with spans
├── 06_evidence.json               RunResult.papers, the agent's public output
└── boundaries/
    ├── llm_calls.json             one entry per LLM call, keyed for replay
    ├── search_results.json        Europe PMC results, keyed by sub-query
    └── fulltext/<PMCID>.xml       JATS full text, one file per paper
```

The `boundaries/` files are what makes an offline run possible; the numbered stage
files exist to localize a failure, so a regression names its stage instead of
producing one diff of the final output.

Stages 2 and 3 record one line per item rather than whole records: a paragraph
record embeds the full Europe PMC paper dict, and pytest diffs a list of strings
line by line while it truncates a list of nested dicts. The text behind those
coordinates stays reproducible from `boundaries/`, so nothing is lost.

`RunResult.debug` is not recorded: it holds per-call timings, which no replay
reproduces.

## When a test fails

- `test_run_reproduces_recorded_evidence` is the contract; the stage tests say
  where the pipeline diverged.
- `MissingRecording` means the run asked for something the fixture never captured
  — usually a changed prompt, a changed Stage 3 batch size, or a different
  sub-query. Re-record the case.
- A skipped case means its directory is missing files (re-record it), or it was
  recorded with BM25 stemming on and this environment has it off, or vice versa
  (install `PyStemmer`, or check `LITERATURE_BM25_STEMMING`).

Re-record only when the change was intended: a fixture is the record of what the
agent used to do.

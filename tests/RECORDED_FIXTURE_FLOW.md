# Recorded Librarian fixture flow

This doc is for whoever reads this next: a PR reviewer, a colleague picking up
work nearby, or someone extending the pipeline who has never touched this
harness before. It assumes you know Python and pytest, not this repo's
specific patterns — those are explained below.

## The problem this solves

`LibrarianAgent.run(query)` calls a real LLM, a real Europe PMC search, and
real JATS full-text fetches. Testing against the live versions of all three is
slow, flaky (network, model sampling), and costs real tokens and quota on
every run. Hand-mocking them risks testing an imagined version of the
pipeline instead of the real one — mocks drift from the code they stand in
for, silently, as the code changes underneath them.

This harness takes a third path: **record** one real run once, then **replay**
it by feeding those same recorded answers back into the exact same,
unmodified `LibrarianAgent.run()`. Every test after that is fast,
deterministic, and offline, while still exercising the agent's real
orchestration and stage logic end to end — not a stand-in for it.

What it buys is regression detection, not correctness. A passing replay says
the pipeline still turns the recorded answers into the recorded output; it
says nothing about whether that output was right in the first place. Judging
that is what the benchmarks under `evals/` are for.

## Vocabulary

- **Boundary** — a call `run()` makes to something outside the process, where
  the agent has no control over the answer. All of them go through the two
  collaborators the agent is constructed with (below).
- **Stage** — one phase of `run()`'s *own* logic, wired between boundaries
  (e.g. Stage 2 turns raw search results into ranked paragraphs). Stages are
  deterministic: same inputs always produce the same outputs, with no I/O of
  their own.
- **Case** — one recorded example: a directory under
  `fixtures/integration/<case_id>/` holding one query, one exact agent
  configuration, what each boundary answered, and what each stage produced
  from those answers.
- **Recording** — running `record_fixture.py` once, live, to create a case.
- **Replay** — running `test_recorded_run.py`, which feeds a case's recorded
  boundary answers into a freshly-built `LibrarianAgent` and checks that
  `run()` still reproduces every recorded stage and the final evidence.
- **Capture-then-patch** — the technique `StageCapture` uses to observe each
  stage without changing it. Explained in detail
  [below](#the-capture-then-patch-pattern).

## The boundaries

`run()` touches the outside world only through the two collaborators
`LibrarianAgent.__init__` accepts:

1. `llm_client.chat_completion(messages, temperature=..., max_tokens=...)` —
   Stage 1 query planning, Stage 3 relevance judging.
2. `literature_source.search(query, page_size)` — Stage 2's Europe PMC search.
3. `literature_source.fetch_fulltext_many(pmcids)` — Stage 2's JATS full-text
   fetch, batched once per sub-query's result set rather than per paper.

(`literature_source.fetch_supplementary_many` is a fourth, called only when
`supplementary_enrichment` is on. It is not recorded yet: `recording.toml` pins
it off, the recorder refuses a config with it on, and `ReplaySource` raises
`MissingRecording` if it is ever called.)

So neither direction patches anything at the boundary. Recording passes in
`RecordingLLM` and `RecordingSource`, which forward to a real
`create_llm_client(...)` and a plain `EuropePmcSource()` and keep what came
back. Replay passes in `ReplayLLM` and `ReplaySource`, which answer from the
case and log what they were asked. `fixture_replay.build_agent()` takes both
collaborators as required arguments, so an agent built for a test can never
fall back to a live endpoint. Everything else `run()` does — every stage,
every prompt substitution, every retry and validation rule — executes for
real, unmodified, in both recording and replay.

## The five files

| File | Owns |
|---|---|
| [`fixture_replay.py`](fixture_replay.py) | What recording and replay must **agree on**: file layout and naming, case loading, `build_agent()`, the keys that identify one call, and `StageCapture` (shared because both directions need "Stage 2's output" to mean the same thing). |
| [`record_fixture.py`](record_fixture.py) | The **recording** half only — runs one live query through `RecordingLLM` and `RecordingSource`, writes a complete case. |
| [`test_recorded_run.py`](test_recorded_run.py) | The **replay** half only — rebuilds the recorded agent around `ReplayLLM` and `ReplaySource`, and asserts the real pipeline reproduces every recorded stage. |
| [`test_paragraph_merge.py`](test_paragraph_merge.py) | Unit tests for one Stage 2 algorithm (interval merging) that a single recorded case may not exercise every branch of. See [why it's separate](#why-test_paragraph_mergepy-is-separate). |
| [`fixtures/recording.toml`](fixtures/recording.toml) | The small runtime config cases are recorded with. |

Boundary recording and replay are each one-directional — the recorder only
ever writes, the replayer only ever reads — so neither lives in the shared
`fixture_replay.py`. A reader of that file sees only the parts both sides
must agree on, nothing that belongs to just one of them.

## A case, on disk

```text
integration/<case>/
├── 00_run.json                 query, agent construction settings, BM25 stemming
├── 01_generated_queries.json   Stage 1 after parsing, deduplication, and cap
├── 02_validated_queries.json   Stage 1 after Europe PMC syntax validation
├── 03_subquery_paragraphs.json Stage 2 top paragraphs per validated sub-query
├── 04_merged_paragraphs.json   Stage 2 pool passed to the relevance judge
├── 05_relevant_papers.json     Stage 3 cited-paper order and evidence spans
├── 06_evidence.json            RunResult.papers, as returned by run()
└── boundaries/
    ├── llm_calls.json          raw LLM responses plus replay keys and parameters
    ├── search_results.json     Europe PMC results keyed by sub-query
    └── fulltext/<PMCID>.xml    JATS XML keyed by normalize_pmcid()
```

The numbered files are **outputs of the real stages** — what to check replay
against. The `boundaries/` files are **answers from outside the process** —
the inputs that make offline replay possible at all. Both halves come from
the same one live run; nothing here is hand-written or synthesized.
`llm_calls.json` holds responses and sampling parameters only, never the
endpoint or a key.

A case's `00_run.json` also pins the agent configuration
(`full_text_enrichment`, `runtime_config`), so `build_agent()` always
reconstructs the *same* agent that produced the recording — never "an" agent
built from whatever today's defaults happen to be. `model_name` and
`reasoning_effort` are kept as a record of what answered; replay never reaches
an LLM, so it does not read them. `00_run.json` also records `bm25_stemming`:
Stage 2's ranking depends on whether `agent._BM25_STEMMER` exists, which is
decided at import by `LITERATURE_BM25_STEMMING` and by whether `PyStemmer` is
installed, not by any config knob. The `case` fixture skips a case recorded
under the other setting rather than failing four stage tests on code that did
not change.

A case is only usable once every file in `fixture_replay.CASE_FILES` and the
`boundaries/fulltext/` directory are present; `missing_files()` treats a
partial directory (interrupted recording, or one predating a fixture-format
change) as *absent* rather than as a failing comparison, so
`test_recorded_run.py` skips it by name with a re-recording instruction
instead of silently comparing against something stale. Cases need no separate
registry: `available_cases()` lists every directory under
`fixtures/integration/`, and each becomes one pytest parameter named after its
directory — dropping in a new case is the only step, no test file needs
editing.

`StageCapture` deliberately serializes Stage 2 and Stage 3 output as
readable lines (`paragraph_line`, `judged_paper_lines`) rather than the raw
nested paper dicts those stages actually return. Full paper metadata stays
reproducible from `boundaries/`; the compact form is what gives a useful
pytest diff — "this paragraph's word range moved" — instead of an elided dump
of a 120KB nested structure.

`RunResult.debug` is not recorded: it carries per-call timings
(`judge_requests[].seconds`), which no replay reproduces.

## The capture-then-patch pattern

Boundaries are injected, but stages are internal methods, so observing them
means patching. Python looks up `agent.some_method` on the instance's own
`__dict__` before falling back to the class. That's the whole trick
`StageCapture.install()` runs on: each of its five wrappers replaces one name,
on one object, with a function that calls through to the original and records
what passed through as a side effect. `run()` itself never finds out.

Worked example, `capture_generated` (Stage 1):

```python
generate_queries = agent._generate_queries  # 1. save the original


def capture_generated(query, *args, **kwargs):  # 2. wrap it
    self.generated_queries = generate_queries(query, *args, **kwargs)
    return self.generated_queries


set_attribute(agent, "_generate_queries", capture_generated)  # 3. patch it in
```

1. **Save the original bound method first, in a closure variable.**
   `agent._generate_queries` is already *bound* — `self` is baked in, so
   calling `generate_queries(query)` works without passing it explicitly.
   This has to happen *before* the patch: step 3 overwrites
   `agent._generate_queries`, so after that point the name no longer refers
   to the real implementation. Skip this and call
   `agent._generate_queries(...)` from inside the wrapper instead, and it
   calls itself — infinite recursion.

2. **Wrap it in a function that calls through, unchanged, and records the
   result as a side effect.** `capture_generated` returns exactly what the
   original returned, so any other code path calling
   `agent._generate_queries` after the patch behaves identically to before.
   The side effect — assigning `self.generated_queries` (`self` here is the
   `StageCapture` instance, not the agent) — is how the value becomes
   observable *after* `agent.run()` returns, without changing what `run()`
   computes or when.

3. **Patch the instance, not the class.** `set_attribute(agent,
   "_generate_queries", capture_generated)` writes into `agent.__dict__`,
   shadowing `LibrarianAgent._generate_queries` for this one object only —
   other agents, including ones built for other parametrized cases, are
   unaffected. Because instance attributes are plain values rather than
   descriptors, Python does not auto-bind `self` to them the way it does for
   a method on the class — which is exactly why `capture_generated`'s
   signature has no `self` parameter. The real `self` is already baked into
   the `generate_queries` closure variable from step 1.

The one exception is `_merge_selected_paragraphs`, a module-level function in
`librarian/agent.py`, so it is patched on that module. Under pytest,
`set_attribute` is `monkeypatch.setattr`, so every patch is undone after the
test.

Two supporting details worth knowing:

- **`llm_call_key()`** identifies an LLM call by its *content*, not its
  order — `query_generation`, `query_generation_retry`, or
  `judge:<paragraph ids>` — because Stage 3's judge batches run concurrently
  and arrive in no fixed order. The recorder can store concurrent responses
  in any arrival order; replay looks up the right one by the batch actually
  being evaluated. Every key needs a positive match on its system prompt
  (the planner's is matched by exact equality, since the judge's starts with
  the same sentence): a request matching none raises `MissingRecording`, so a
  new `chat_completion` call site can never be answered with another site's
  recorded response.
- **`MissingRecording`** is what a replay boundary raises when asked for
  something the case never recorded — an unrecorded sub-query, judge batch,
  full-text id, or any supplementary fetch. It deliberately subclasses
  `BaseException`, not `Exception`: `_evaluate_batch` catches ordinary
  `Exception` to degrade a malformed judge response gracefully, and a stale
  fixture must not be swallowed by that same path. The `replayed` fixture
  turns it into `pytest.fail` once `run()` has raised it, because a bare
  `BaseException` escaping a parametrized fixture leaves pytest's fixture
  cache broken for every later case.

## Recording a fixture

From the repo root, with the LLM endpoint reachable:

```bash
uv run python tests/record_fixture.py \
  --case macrophage_activation \
  --query "How does interferon-gamma activate inflammatory gene expression in human macrophages?"
```

It validates the case name, refuses to overwrite an existing case unless
`--overwrite` is given, refuses a config with `supplementary_enrichment` on,
builds a `LibrarianAgent` with full-text enrichment on around the recording
collaborators, installs `StageCapture`, and calls `agent.run(query)` once for
real. Nothing is written to disk until that run succeeds and returns
non-empty evidence — a failed recording leaves the previous case (or no case)
in place, never a partial one.

Two rules keep a recording replayable. `RecordingSource.search` stores a deep
copy of each search result, taken before the pipeline sees it: Stage 3 writes
the evidence fields onto those same paper dicts, and a snapshot taken after the
run would hand replay the evidence it is meant to rebuild. And a paper fetched
by two sub-queries with different outcomes (one success, one failure) stops the
recording, because replay answers every fetch of a paper from one file.

The script puts the repository root on `sys.path` itself, like the scripts
under `evals/`, so it runs as a plain file and can import `fixture_replay`
from beside it.

## Replaying and diagnosing a failure

`test_recorded_run.py`'s `replayed` fixture, once per case, per test:

1. `load_case()` — reads the query, construction settings, and boundary data.
2. `build_agent()` — rebuilds the same agent around `ReplayLLM` and
   `ReplaySource`.
3. `StageCapture.install()` — observes the real stages again.
4. `agent.run(case.query)` — runs the current implementation for real.

The one test that states the actual contract is
`test_run_reproduces_recorded_evidence`: does `RunResult.papers` still match
`06_evidence.json`? Every other test narrows down *where* a failure lives,
so a regression names its stage instead of dumping a diff of the final
answer:

| Test area | Detects divergence in |
| --- | --- |
| Stage 1 output | LLM response parsing, deduplication/capping, or validation |
| Stage 1 request | planner call count, retry behaviour, prompt substitution |
| LLM sampling | every call's `temperature` and `max_tokens` against the recording |
| Stage 2 boundary use | validated sub-query fan-out to Europe PMC |
| Stage 2 selections | paragraph extraction/chunking and per-sub-query BM25 ranking |
| Stage 2 merged pool | pool composition and merge integration |
| Stage 3 results | sentence IDs, batching, citation regrouping, and paper/span order |

The request-level assertions matter because the replay response is keyed
only by the *current* request's content: without them, a changed prompt
could still get answered by an old recorded response and hide the drift.

If you hit `MissingRecording`, that almost always means the pipeline now
makes a genuinely different call — a new LLM prompt shape, a different
search, a changed full-text batch, a new judge batching layout. That's a
compatibility break in the fixture, not test flakiness — re-record, don't
retry.

## Why not two simpler scripts?

A natural-looking alternative: skip all of the above, and write two plain
scripts that step through the pipeline by hand — one that calls
`agent._generate_queries()`, then `agent._validate_queries()`, then the
search and full-text calls, then `agent._paragraphs_for_subquery()`, and so
on in sequence, saving each result; and a matching replay script that drives
the same sequence, substituting recorded answers at each step. No stage
capture, arguably much less code.

The cost of that shortcut is what it gives up: **`run()`'s own orchestration
never gets exercised**, because a hand-driven script has to already know —
and hard-code — the order stages run in, which ones fan out over a thread
pool, and when full text gets fetched relative to search. That orchestration
is exactly the kind of thing that changes over time. Full text, for example,
is fetched once per sub-query's whole result set before extraction starts; a
hand-driven script would have to notice if that ever moved and update its own
sequence by hand, while replaying through the real `run()` follows it for
free.

Concretely, three things go wrong with the two-script shape that don't
happen here:

- **The reimplementation can drift from the real thing and still pass.** A
  hand-driven sequence encodes a second, forkable copy of what `run()`
  does. If `run()`'s real control flow changes and the script's manual
  sequencing doesn't follow, the script doesn't fail — it just quietly stops
  testing the actual pipeline, while still going green.
- **Two independently-written scripts can silently disagree with each
  other.** Whatever the recorder assumes (a call's identity, a stage's file
  layout, what "recorded but empty" means for a failed fetch) has to be kept
  in sync by hand with what the replay script assumes, instead of living
  once in a shared module both import.
- **Concurrency gets solved twice, or not at all.** Stage 2's sub-query
  fan-out and Stage 3's judge batches genuinely run on thread pools inside
  `run()`. Answering at the boundary lets those real pools run, so their
  actual concurrent behaviour — and any bugs in it — stays part of what's
  under test. A stage-by-stage script has to either serialize what is
  actually concurrent (testing something the pipeline doesn't do) or
  reimplement the same locking a second time in test code.

The unifying reason: answering at the boundaries and calling the real,
unmodified `run()` means every failure here is a genuine regression in the
current implementation. Hand-driving stages one by one means a failure might
just mean the script fell out of sync with `run()` — a much weaker signal,
and one that gets weaker exactly as the pipeline gets more actively
developed.

## Adding a new pipeline stage

Say a new step is added to `LibrarianAgent.run` between two existing stages
— e.g. a reranking pass between Stage 2's merge and Stage 3's relevance
filter. This is a real, expected way the fixture format grows, and the
design is meant to make the follow-up steps hard to skip by accident, not to
need zero changes:

1. **`librarian/agent.py`** — implement the stage, call it from `run()` in
   its place. This is the only change to the pipeline itself; everything
   below is the fixture/test machinery catching up to a new source of truth.

2. **`fixture_replay.py`**:
   - Add a new numbered file constant (e.g. `RERANKED_PAPERS_FILE =
     "04b_reranked_papers.json"`, or renumber upward — numbering is
     `ls`-order readability only, nothing depends on the numeric value) and
     add it to `CASE_FILES`.
   - Add a field to `StageCapture` and a `capture_*` wrapper in
     `StageCapture.install()`, following the [capture-then-patch
     shape](#the-capture-then-patch-pattern) the other five already use.
     Give it the same per-call lock as `capture_subquery` only if the new
     stage also runs on a thread pool.
   - Pick a serialization: compact lines (like `paragraph_line`/
     `judged_paper_lines`) if the output embeds large nested objects and
     per-item diffs matter, or the value as-is (like `01`/`02`) if it's
     already small and flat.
   - If the new stage makes its own external call (a new LLM prompt, a
     different search) rather than being purely deterministic,
     `llm_call_key()` needs a new case for it too.

3. **`record_fixture.py`** — write the new field in `_write_case()`. If a
   new external call was added, give `RecordingLLM` or `RecordingSource` a
   matching method, following the existing ones as a template.

4. **`test_recorded_run.py`** — add one test mirroring the existing stage
   tests (compare `replayed.stages.<new field>` against
   `case.expected(fixture_replay.<NEW_FILE>)`), and add a row to the table
   above. If a new external call was added, give `ReplayLLM` or
   `ReplaySource` the matching method and assert against it the way
   `test_stage1_asks_the_llm_the_way_it_was_recorded` does.

5. **Docs** — update the file-layout diagram here and in
   [`fixtures/README.md`](fixtures/README.md).

6. **Re-record every existing case.** Not a separate step to remember:
   adding the new filename to `CASE_FILES` makes every already-recorded case
   instantly incomplete, so `test_recorded_run.py` skips each one with a
   re-recording instruction instead of silently comparing against a fixture
   that predates the new stage. Re-run `record_fixture.py --case <name>
   --query "<same query>" --overwrite` for each committed case — the
   original query is sitting in that case's own `00_run.json`.

## Why `test_paragraph_merge.py` is separate

After Stage 2 concatenates the top paragraphs from every sub-query,
`_merge_selected_paragraphs()` groups selections by paper identity, source
paragraph index, and source text, unions overlapping or touching half-open
word intervals (`[start, end)`) within each group, and rebuilds each
surviving interval from the source text. Group order follows first
occurrence; intervals within a group are document ordered.

The recorded-run test confirms merging produces the previously recorded pool
for one realistic query, but can't guarantee branch coverage — a given case
may have no overlapping selections at all, making merge a no-op for that
run. The unit tests fill that gap with small synthetic cases: disjoint
intervals stay distinct; overlapping, touching, and identical intervals
collapse; output order is correct even when input order isn't; different
papers or source paragraphs stay separate and keep group order; equal
offsets from different source text don't merge.

Together: the recorded test pins end-to-end integration against fixed
external inputs, and the merge test pins the local overlap-union rules in
isolation.

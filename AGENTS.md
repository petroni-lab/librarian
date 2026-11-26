# AGENTS.md

Working contract for coding agents (Claude Code, Codex, opencode, Antigravity, ...) in
this repository. `CLAUDE.md` imports this file; edit this one.

## What this repo is

EMBL AI Librarian: a natural-language question goes in, ranked Europe PMC evidence
passages with citations come out. It ships as a Python library, a CLI (`main.py`), an
HTTP server (`orchestrator.py`), an agent skill (`skills/librarian/`) and a benchmark
harness (`evals/Literature/`). Usage, configuration and layout are in `README.MD`; read
it before a non-trivial change.

Other projects embed Librarian by installing a commit tarball and plugging their own
pieces into its constructor and `run()`. Treat everything `README.MD` documents for
them as a public contract (see "Public contract").

## Start here by task

- **Retrieval pipeline (stages 1-3)**: `librarian/agent.py`, start at `LibrarianAgent.run()`.
- **Cited answer, routing, conversation history**: `librarian/synthesis.py`, `librarian/citations.py`.
- **Iterative retrieval**: `librarian/research_loop.py`.
- **What an LLM is asked**: `librarian/prompts/`, one file per LLM call.
- **Europe PMC query syntax checks**: `librarian/query_validation/`.
- **Search and full-text fetch**: `librarian/literature_search.py`, `librarian/jats.py`,
  `librarian/fulltext_cache.py`.
- **Supplementary files**: `librarian/supplementary.py`, `librarian/pdf_jats.py`.
- **Tuning knobs**: `librarian/config.toml` (values) and `librarian/config.py` (schema).
- **Extension points**: `librarian/tracing_port.py`, `librarian/scope_port.py`, and the
  `LiteratureSource` protocol in `librarian/literature_search.py`.
- **The agent skill and its launchers**: `skills/librarian/`.
- **Benchmarks**: `evals/Literature/`; each suite has its own README and environment.

## Commands

```bash
uv sync --locked --extra supplementary   # dev environment, as CI builds it
uv run pytest tests                      # the whole suite, offline (matches CI)
uv run main.py "<question>"              # one real run; needs LLM_* in .env
```

Re-recording a fixture needs a live LLM endpoint and Europe PMC; the command is in
`README.MD`, "Testing", and the mechanics in `tests/RECORDED_FIXTURE_FLOW.md`.

Only document commands that a file in this repo supports. Never report a check as run
or passing when it did not run; a skipped test checked nothing.

## Public contract

- The public surface is what `README.MD` documents: the constructor arguments, `run()`
  and its result types, the injectable `llm_client=`, `literature_source=`, `tracer=`
  and `scope_guard=`, the CLI flags, the HTTP routes, the `config.toml` knobs, and the
  skill's behaviour.
- Embedders pin a commit and move forward automatically when their own tests pass. A
  change that breaks the public surface must say so in its PR title or body and add a
  migration note to `README.MD` (see "Migrating from 1.0.0 to 1.0.1"). Prefer an
  additive change (a new optional argument, a new hook) to a breaking one.
- Embedders customise Librarian by injecting dependencies and overriding hooks, never by
  copying `run()`. When someone needs different behaviour, add a no-op hook or an
  injectable dependency upstream rather than a flag that only one caller uses.
- One agent instance serves concurrent calls: keep per-run state in locals and in the
  returned result, never on `self`. Injected dependencies must be thread-safe too.
- `pyproject.toml` sets the minimum Python version; don't use syntax or stdlib APIs newer
  than it.

## Tests

1. **Test behaviour at the public entry point**: `LibrarianAgent.run()`,
   `SynthesisAgent.run()`, the HTTP routes, a module's public functions. Fake only what
   is outside the process (the LLM, Europe PMC, the network), through the injectable
   arguments (`llm_client=`, `literature_source=`). Don't patch our own classes, call
   private methods, read source files as text, or build objects with `__new__`.
2. **No tautological tests.** A test fails when the behaviour it names breaks, and only
   then. Don't re-state the code, copy a value out of `config.toml`, assert what a mock
   was told to return, or guard code that has been deleted.
3. **Every fix comes with a regression test** that reproduces the observed failure
   through a public entry point. Write it first and check that it fails.

- Check answers by shape and invariants (every citation resolves to a returned paper, a
  failed search is reported instead of crashing), not by exact model text.
- Near-duplicate tests become one `@pytest.mark.parametrize` test.
- No helpers in library code that exist only for tests.
- The recorded-run fixtures under `tests/fixtures/integration/` are regression tests, not
  proof of correctness. Re-record them in the same PR as any intended change to prompts,
  sub-query planning, ranking, chunking, merging, the judge or `LibrarianRuntimeConfig`'s
  fields. Never hand-edit a recording to make a test pass.

## Code conventions

- **Be prescriptive.** There is one supported way to do each thing. Code works as
  intended or fails loudly where it is misused: no accepting several input shapes and
  quietly picking one, no fallback that hides a wiring mistake (no
  `config.get(key, default)` for a value that must be set; raise instead).
- **Smallest clear design.** Implement the current requirement; add abstractions and
  configuration only when this task needs them. Preserve supported inputs, error
  handling and existing contracts.
- Type annotations on everything you write or change.
- Public functions and classes get Sphinx-style docstrings (`:param:`, `:returns:`,
  `:raises:`); private ones (`_name`) get one or two comment lines.
- One thing per line; use intermediate variables with specific names.
- Comment non-obvious paths and the reason behind them, so the next reader recovers the
  intent without rereading the repo.
- New runtime dependencies need a reason. Optional features go in an extra in
  `pyproject.toml` (as `cache`, `supplementary` and `docling` do), and the core install
  stays light.

## Config and prompts

- `config.toml` sets every knob and `config.py` holds the schema with no field
  defaults: a missing knob fails at load. No env-var overrides for tuning knobs; a
  one-off change is `dataclasses.replace(load_runtime_config(), ...)` by the caller.
- Model names, batch sizes, temperatures and fan-out widths are config knobs, never
  literals in code or prompts.
- One prompt file per LLM call, named after the stage it drives. A cap a prompt states
  comes from the same knob the code enforces, so the instruction and the truncation
  can't disagree.
- Rules and rubrics live in the prompt. Python renders, parses and validates; it does
  not restate them. Don't fork a prompt per caller or experiment.

## Documentation

- Each fact has one home, the doc nearest the code it describes; other docs link to it.
  Prefer updating an existing doc to adding a new one.
- Don't copy a value that code or config sets (a limit, a default, a model name) into
  prose; name the file and the key.
- A PR that changes documented behaviour (a flag, a route, a knob, an env var, a stage,
  an install step) updates `README.MD`, `skills/librarian/SKILL.md` or the relevant
  README in the same PR.
- Draw diagrams in Mermaid or as a text tree, not as an image, unless the picture is the
  content. Images are files in `assets/`, referenced by relative path.

## This is a public repository

Everything here is public: code, comments, prompts, fixtures, commit messages, PR and
issue text.

- Don't mention private projects, internal systems, hostnames, cluster paths, internal
  issue numbers or colleagues' private work. Describe a change by what it does for
  Librarian's users. Context that only matters to a private consumer belongs in that
  consumer's own repository.
- Write comments and docs so they stand alone: "matches the Europe PMC rate limit", not
  "same value as <private repo>".
- Never commit secrets, `.env` files, API keys or tokens. Recorded fixtures hold
  public Europe PMC data and model replies only; check a new recording before
  committing it.

## Git and pull requests

- Branch: `<user>/<short-description>`.
- PR title: `TAG: Short description`, with `feature`, `fix` or `chore`.
- PR body:

  ```markdown
  <what this PR changes and why, in one paragraph a reader understands on its own>

  ## Changes
  - `<file>`: <what changed>

  ## Verification
  <commands run and their output>

  ## Decisions
  <defaults picked, assumptions made, what was left out; "None" if none>
  ```

- One concern per PR, reviewable in under 45 minutes; split a larger change into a
  stack.
- Docs and re-recorded fixtures go in the same PR as the change that needs them.

## Reviewing PRs

- Review without modifying files. For each finding give the trigger, the affected
  behaviour, the evidence and the smallest fix.
- Block on demonstrated failures of supported behaviour with real impact, including rare
  failures with severe consequences; on an undocumented break of the public contract;
  on a missing doc update; and on anything that breaks "This is a public repository".
- Track minor bugs as issues instead of blocking. Don't block on hypothetical future
  requirements or on equivalent design preferences.

# AstaBench Integration

This directory contains the librarian's integration with the literature subset of
AstaBench.

## What Is Here

- `bio_agent_wrapper.py`: Inspect solver wrappers plus task-specific output
  adapters for:
  - `PaperFindingBench`
  - `LitQA2-FullText`
  - `LitQA2-FullText-OpenJudge`
  - `ScholarQA-CS2`
  - `ArxivDIGESTables-Clean`
- `prompts.py`: The Asta-corpus librarian prompt used by Config B.
- `run_evals.py`: Python runner that launches Config A and/or Config B with
  dedicated Inspect log directories under `results/`.
- `astabench_patch.py`: Two run-time resilience patches for the clone's MCP
  client, applied only for `standard_tooling`. See its docstring.

## Configs

- `standard_tooling` (default): Runs the benchmark the way the AstaBench
  authors intended, routing retrieval through AstaBench's task-provided Asta
  corpus tools instead of EuropePMC. The wrapper also enriches hits with
  `snippet_search(...)` evidence when the task exposes it.
- `custom_tooling`: Runs LitQA2 against Europe PMC through the librarian, which is
  what the paper's +Librarian row measures. Use this to compare the librarian's
  source stack against the benchmark-native one.

## Running

`../setup.sh --bench litqa2` clones AstaBench at a pinned commit into
`vendor/` and leaves it **pristine** — `../setup.sh --check` asserts that. It
then builds this bench's environment from `../envs/litqa2.lock` and installs
the clone into it, unmodified, with `--no-deps`.

Installing rather than putting the clone on `sys.path` is what keeps it
unmodified: upstream's `astabench/__init__.py` calls `get_version("astabench")`,
which fails without install metadata, and eagerly imports the whole eval suite.
Reaching it through `sys.path` means editing the clone; installing it costs a
heavier environment instead.

Two other things the clone would otherwise need changing for, and what replaces
each:

- **`huggingface_hub` 1.0 removed `HfFolder`**, which the LitQA2 task calls.
  The lock pins `huggingface_hub<1.0`, so the task runs as published.
- **The MCP client's 5-second connect timeout and narrow retry set** make long
  unattended `standard_tooling` runs flaky. `astabench_patch.py` replaces the
  two functions at run time, and only for that config — the paper's +Librarian
  row is `custom_tooling`, which never opens the Asta MCP endpoint at all.

Nothing replaces the extra top-level `accuracy` metric an earlier version of
this harness added to the task, because nothing needs to: it was
`sum(is_correct)/n`, which the clone already reports as `is_correct/accuracy`
through Inspect's own `accuracy()`.

```bash
python evals/Literature/AstaBench/run_evals.py --split validation
```

Smoke-test a small slice first:

```bash
python evals/Literature/AstaBench/run_evals.py --split validation --limit 1
```

Run a single dataset:

```bash
python evals/Literature/AstaBench/run_evals.py --split validation --limit 1 --task ScholarQA-CS2
```

Run both source stacks for comparison:

```bash
python evals/Literature/AstaBench/run_evals.py --split validation --config both --limit 1 --task ScholarQA-CS2
```

Run the open-answer LitQA2 diagnostic:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --limit 1 \
  --task LitQA2-FullText-OpenJudge \
  --litqa2-open-judge-model openai/gpt-4o-2024-11-20
```

Full-text enrichment is enabled by default when using Europe PMC. To disable it:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge \
  --solver bio_agent \
  --config custom_tooling \
  --no-bio-agent-full-text
```

Use the simple BM25 librarian for a run:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge-EuropePMCFullText \
  --solver bio_agent \
  --config custom_tooling \
  --bio-agent-query-planner simple_bm25
```

`--bio-agent-query-planner simple_bm25` selects
`agents/deprecated/literature/prompts/simple_bm25_librarian.md`. Use
`--bio-agent-simple-bm25-max-queries N` to change its query cap.

You can also pass the stem of any `.md` librarian prompt in
`agents/deprecated/literature/prompts/`:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge-EuropePMCFullText \
  --solver bio_agent \
  --config custom_tooling \
  --bio-agent-query-planner europepmc_claude_librarian \
  --bio-agent-max-query-count 4
```

`LitQA2-FullText-OpenJudge` is not part of the default task set. It lets
the agent answer the LitQA2 question in natural language, then asks a separate
judge whether that open answer contains the known gold answer. The judge does
not see the distractors, and the evaluated agent receives a question-only input,
so this diagnostic measures whether the user-facing answer contains the correct
conclusion rather than whether a judge or agent can solve a forced-choice
mapping problem.

For agent runs, this OpenJudge path disables the literature-agent summary
step and builds the final judged answer from the retrieved evidence snippets
instead. The evidence formatter accepts both `{pmid, evidence}` and the older
`{pmid, evidence_abstract, evidence_fulltext}` payload shapes. The final answer
step uses a short JSON extraction prompt so the judged text starts with the
specific value, gene, mutation, residue range, or other exact answer when the
evidence supports one.

Run the same open-answer diagnostic on the full original LitQA2 dataset:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge-Full \
  --solver llm_web_search \
  --llm-base-url "https://api.openai.com/v1" \
  --llm-model gpt-5.4
```

`LitQA2-FullText-OpenJudge-Full` uses all rows from `futurehouse/lab-bench`
LitQA2 instead of AstaBench's filtered dev/test subset. Report it separately
from AstaBench numbers because the Asta subset is filtered to target papers
available in their snippet search index.

Build and run the EuropePMC-fulltext-only LitQA2 subset:

```bash
python evals/Literature/AstaBench/check_litqa2_europepmc_fulltext.py

python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge-EuropePMCFullText \
  --solver bio_agent \
  --config custom_tooling
```

The checker writes `data/litqa2_europepmc_fulltext/summary.json`, a per-row
availability JSONL, and full/dev/test JSON subsets. The runner task above uses
the full subset, i.e. LitQA2 rows whose source papers have EuropePMC `HAS_FT:Y`
and a fetchable `fullTextXML` document.

`data/litqa2_europepmc_fulltext/litqa2_public_subset_ids.txt` is the committed
list of the 91 row ids in that subset. Each id is the exact `id` column from the
`futurehouse/lab-bench` `LitQA2` config, so it joins straight back to that
dataset with no separate mapping. `reconstruct_litqa2_from_ids.py` rebuilds the
full rows from it — which is how the subset stays reproducible without this
repository redistributing the dataset's content.

To create the broader EuropePMC `HAS_FT:Y` subset without requiring
`fullTextXML` fetchability:

```bash
python evals/Literature/AstaBench/check_litqa2_europepmc_fulltext.py \
  --no-xml-check \
  --output-dir evals/Literature/AstaBench/data/litqa2_europepmc_has_ft
```

## Using other solvers

The runner now supports choosing a solver implementation with `--solver`.

LLM-only baseline (no retrieval):

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge \
  --solver llm_only \
  --llm-base-url "http://localhost:8000/v1" \
  --llm-model glm-5-fp8
```

OpenScholar API (OpenSciLM demo) for ScholarQA-CS2 or LitQA2-FullText-OpenJudge:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task ScholarQA-CS2 \
  --solver openscholar_api

python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge \
  --solver openscholar_api
```

Notes:
- `--solver llm_only` is a direct LLM baseline; it does not use the librarian
  retrieval flow. This is separate from `--config llm_only`, which only affects
  the open-answer LitQA2 diagnostic.
- `--solver openscholar_api` supports `ScholarQA-CS2` and `LitQA2-FullText-OpenJudge`.

LLM web search (OpenAI Responses API) for LitQA2-FullText-OpenJudge or LitQA2-FullText-Search:

```bash
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge \
  --solver llm_web_search \
  --llm-base-url "https://api.openai.com/v1" \
  --llm-model gpt-5.4 \
  --web-search-tool-choice required \
  --web-search-context-size medium

python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-Search \
  --solver llm_web_search \
  --llm-base-url "https://api.openai.com/v1" \
  --llm-model gpt-5.4 \
  --web-search-tool-choice required \
  --web-search-context-size medium
```

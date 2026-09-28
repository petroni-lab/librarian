# LitQA2, on an unmodified AstaBench

The paper's LitQA2 row: Coverage 95.6 / Precision 82.6 / Accuracy 78.9 over the
91-question `europepmc_fulltext` split, judged by `gpt-4o-2024-11-20`.

Normally you do not call anything here directly:

```bash
./evals/Literature/literature_eval.sh --bench litqa2
```

That runs [`run_litqa2.sh`](run_litqa2.sh); see `run_litqa2.sh -h` for the
environment it reads.

## What is here

- [`bio_agent_wrapper.py`](bio_agent_wrapper.py) — the Inspect solver that runs
  the librarian for a LitQA2 task, plus the output adapters each task shape
  needs.
- [`run_evals.py`](run_evals.py) — the runner, which builds the Inspect task and
  launches it with a log directory under `results/`.
- [`litqa2_open_judge.py`](litqa2_open_judge.py) — the open-answer judge task.
- [`astabench_patch.py`](astabench_patch.py) — two run-time patches to the
  clone's MCP client, applied only for `standard_tooling`. See its docstring.
- [`prompts.py`](prompts.py), [`compat.py`](compat.py) — the Asta-corpus
  librarian prompt, and the inspect_ai glue.

## Configs and solvers

`--config` picks where retrieval comes from:

- **`custom_tooling`** — Europe PMC through the librarian. This is the paper's
  +Librarian row.
- **`standard_tooling`** — AstaBench's own task-provided Asta corpus tools, the
  way the benchmark's authors intended. Needs `ASTA_TOOL_KEY`.
- **`llm_only`** — no retrieval.

`--solver` picks what answers. `bio_agent` is the librarian; the other three are
the paper's comparison rows:

| solver | row |
|---|---|
| `bio_agent` | +Librarian |
| `llm_only` | Parametric |
| `llm_web_search` | Web Search |
| `openscholar_api` | OpenScholar-8B |

## Tasks

`--task` takes any of these; without one, the runner's default set runs.

| task | dataset |
|---|---|
| `LitQA2-FullText` | AstaBench's LitQA2, forced choice |
| `LitQA2-FullText-OpenJudge` | the same questions, open answer + judge |
| `LitQA2-FullText-OpenJudge-Full` | all `futurehouse/lab-bench` LitQA2 rows |
| `LitQA2-FullText-OpenJudge-EuropePMCFullText` | **the paper's 91-question split** |
| `LitQA2-FullText-Search` | LitQA2 scored as a paper-finding task |

The OpenJudge variants let the agent answer in natural language and then ask a
separate judge whether that answer contains the gold answer. The judge does not
see the distractors and the agent gets a question-only input, so it measures
whether the user-facing answer carries the right conclusion rather than whether
a forced-choice mapping can be solved. On that path the librarian's summary step
is disabled and the judged answer is built from the retrieved evidence.

`LitQA2-FullText-OpenJudge-Full` uses every LitQA2 row rather than AstaBench's
filtered subset, so report it separately: the Asta subset is filtered to papers
in their snippet-search index.

Only LitQA2 runs here. AstaBench's other benchmarks come with the clone,
untouched and unused.

## The clone stays unmodified

`../setup.sh --bench litqa2` clones AstaBench at a pinned commit into `vendor/`
and leaves it pristine, which `../setup.sh --check` asserts. It then builds this
bench's environment from `../envs/litqa2.lock` and installs the clone into it
with `--no-deps`.

Installing rather than putting the clone on `sys.path` is what keeps it
unmodified: upstream's `astabench/__init__.py` calls `get_version("astabench")`,
which fails without install metadata. Reaching it through `sys.path` would mean
editing the clone; installing it costs a heavier environment instead.

Two other things the clone would otherwise need changing for:

- **`huggingface_hub` 1.0 removed `HfFolder`**, which the LitQA2 task calls. The
  lock pins `huggingface_hub<1.0`, so the task runs as published.
- **The MCP client's 5-second connect timeout and narrow retry set** make long
  unattended `standard_tooling` runs flaky. `astabench_patch.py` replaces the
  two functions at run time, and only for that config — the paper's row is
  `custom_tooling`, which never opens the Asta MCP endpoint.

## Data

The `europepmc_fulltext` split is built, not redistributed.
`run_litqa2.sh` builds it on first use; to do it yourself:

```bash
python evals/Literature/AstaBench/check_litqa2_europepmc_fulltext.py
```

That queries Europe PMC and writes `data/litqa2_europepmc_fulltext/`: a
per-row availability JSONL, a `summary.json`, and full/dev/test JSON subsets.
The split is the LitQA2 rows whose source papers have `HAS_FT:Y` and a fetchable
`fullTextXML`.

`data/litqa2_europepmc_fulltext/litqa2_public_subset_ids.txt` is the committed
list of the 91 row ids. Each is the `id` column from the `futurehouse/lab-bench`
`LitQA2` config, so it joins straight back to that dataset;
[`reconstruct_litqa2_from_ids.py`](reconstruct_litqa2_from_ids.py) rebuilds the
rows from it, which is how the split stays reproducible without this repository
redistributing the dataset.

For the broader `HAS_FT:Y` subset, without requiring `fullTextXML`:

```bash
python evals/Literature/AstaBench/check_litqa2_europepmc_fulltext.py \
  --no-xml-check \
  --output-dir evals/Literature/AstaBench/data/litqa2_europepmc_has_ft
```

## Ad-hoc runs

```bash
# Smoke-test one question.
python evals/Literature/AstaBench/run_evals.py --split validation --limit 1

# The paper's task, one baseline solver.
python evals/Literature/AstaBench/run_evals.py \
  --split validation \
  --task LitQA2-FullText-OpenJudge-EuropePMCFullText \
  --solver llm_only \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model glm-5-fp8

# Both source stacks, to compare the librarian against the benchmark-native one.
python evals/Literature/AstaBench/run_evals.py \
  --split validation --config both --limit 1 --task LitQA2-FullText
```

`--limit` shrinks the question count; `--max-samples` is Inspect's concurrency,
not a question count. Full-text enrichment is on by default under
`custom_tooling`; `--no-bio-agent-full-text` turns it off.

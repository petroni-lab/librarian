# LabBench (LAB-Bench, text-only MCQ tasks)

One task-agnostic runner for the text-only multiple-choice subtasks of
[LAB-Bench](https://github.com/Future-House/LAB-Bench) (paper:
[arXiv:2407.10362](https://arxiv.org/abs/2407.10362)), selected with `--task`.
The prompt, choice shuffling, answer parser and metrics match the upstream
harness and are reimplemented in `labbench_compat.py`, so no `labbench` or
`chembench` install is needed.

| `--task` | LAB-Bench subtask | Qs |
|---|---|---|
| `DbQA` | Retrieving information from biological databases | 520 |
| `SeqQA` | Manipulating biological sequences | 600 |
| `ProtocolQA` | Troubleshooting biological protocols | 108 |
| `CloningScenarios` | Molecular cloning workflows | 33 |
| `TableQA` | Reading data tables reported in the literature | 244 |

`SeqQA` strips sequences from the search query by default; `ProtocolQA` prepends
the row's `protocol` field to the question, as upstream does; `TableQA` runs
retrieval-only, ignoring the dataset's table image and source DOI.

## Setup

```bash
./evals/Literature/setup.sh --bench labbench
```

That builds the bench's locked environment; the first run builds it anyway if
you skip this. Set `OPENAI_API_KEY` for the answering model, in the repo `.env`
or the environment. The dataset streams from `futurehouse/lab-bench` on the
Hugging Face Hub and is gated, so run `hf auth login` once; `--data-file` takes
a downloaded `<task>-v1-public.jsonl` instead.

## Reproduce the paper's +Librarian row

SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, CloningScenarios 48.5.

```bash
./evals/Literature/literature_eval.sh --bench labbench \
    --librarian-url http://localhost:8000/v1 --librarian-model glm-5-fp8
```

That loops the four tasks x {`gpt-5.4`, `gpt-4o`} and then aggregates every
result directory with `audit_results.py`. `--only` takes a task name or a model
name and narrows to either: `--only DbQA --only gpt-4o`. `--limit N` caps the
questions, and `--dry-run` prints the commands without running them.

The baseline row is the same command with `--mode baseline` on the runner below.

## Run one task directly

```bash
python -m evals.Literature.LabBench.run_labbench_eval \
    --task SeqQA --mode knowledge --model gpt-4o \
    --agent-model <agent-llm> --agent-base-url <agent-llm-url> \
    --out-dir results/seqqa_knowledge
```

`--mode baseline` answers from the model's own knowledge and takes no agent
flags. Run both with the same `--seed` to get identically shuffled options.
`--model` defaults to `gpt-4o-2024-05-13`, the snapshot the paper's floating
`gpt-4o` alias resolved to.

Other flags: `--evidence-char-budget` (raw passage characters per paper),
`--grounding augment|strict`, `--strip-query-sequences` /
`--no-strip-query-sequences`, `--full-text` / `--no-full-text`, `--top-k`,
`--max-workers`, `--resume`, `--max-examples`, `--via-api`. `--mode librarian`
is a deprecated alias of `--mode knowledge`.

## Outputs (in `--out-dir`)

- `predictions.jsonl` — one row per question: task, gold and predicted letter,
  correct, sure, raw output, and in knowledge mode the queries issued and the
  evidence count.
- `metrics.json` — accuracy, precision, coverage, counts.
- `summary.md` — the same as a table.
- `run_config.json` — the run configuration.

accuracy = correct / total; precision = correct / answered, where answered means
the model did not pick the refusal option; coverage = answered / total.

## Files

- `labbench_compat.py` — task registry (`TASKS`), prompt template, choice
  shuffling, answer parsing, metrics, dataset loading.
- `solvers.py` — the baseline and knowledge-layer answering models, the latter
  taking its retrieval agent as an injected `search_fn`.
- `run_labbench_eval.py` — CLI runner, agent wiring, output writer.
- `audit_results.py` — aggregates result directories into the paper table.
- `run_labbench.sh` — the paper sweep, dispatched by `literature_eval.sh`.

# LabBench (LAB-Bench, text-only MCQ tasks)

A single, task-agnostic runner for the text-only multiple-choice subtasks of
[LAB-Bench](https://github.com/Future-House/LAB-Bench) (paper:
[arXiv:2407.10362](https://arxiv.org/abs/2407.10362)) — one prompt, shuffling,
parser, and metrics shared across all tasks, driven by a `--task` flag. The
former standalone `CloningScenarios/` eval has been merged in as one of these
tasks; its historical run outputs are preserved under `results/cloning_*`.

The point is the same as CloningScenarios: use one of our retrieval agents as a
**knowledge layer** for a general LLM. Instead of letting the LLM answer from its
parametric knowledge, we make it gather evidence through our literature retrieval
and answer from what it finds — then compare against the same LLM run as a plain
baseline.

## Tasks

| `--task` | LAB-Bench subtask | Qs | Notes |
|---|---|---|---|
| `DbQA` | Retrieving information from biological databases | 520 | |
| `SeqQA` | Manipulating biological sequences | 600 | sequences stripped from the search query by default |
| `ProtocolQA` | Troubleshooting biological protocols | 108 | the row's `protocol` field is prepended to the question (matches upstream) |
| `CloningScenarios` | Molecular cloning workflows | 33 | merged in from its old standalone folder; historical runs are in `results/cloning_*` |
| `TableQA` | Reading data tables reported in the literature | 244 | retrieval-only: the dataset's table image and source DOI are ignored, so the model must find the right table in the retrieved passages |

### Why these and not the others

LAB-Bench has eight subtasks. The five above are the MCQ tasks the **text**
knowledge layer can serve. TableQA joined once literature retrieval began
carrying table text (rendered from full-text JATS) — a retrieved paper's tables
are now usable evidence, so the task runs through the ordinary retrieval flow
rather than needing the figure/table image. The rest are deliberately excluded:

- **FigQA** (181) — each example is a figure **image** the model must reason
  over. Our literature retrieval layer is text-only, so it cannot ground it; it
  needs a multimodal harness.
- **SuppQA** (82) — answers live in **specific papers' supplementary files**, not
  the open literature the knowledge layer searches, so the retrieval signal is
  weak. Excluded for now.
- **LitQA2** (199) — excluded by request (covered by the AstaBench / other
  literature evals).

## Modes

- `--mode baseline` — the LAB-Bench setup: the model answers each MCQ from its
  own parametric knowledge. Reproduces the paper's baselines.
- `--mode knowledge` — the same model, grounded in the **`LibrarianAgent`**
  (Europe PMC → BM25 → relevance judge) as a knowledge layer (single-shot RAG).
  The librarian is run **once** on the question, its retrieved passages are
  injected into the prompt, and the model answers in **one** API call. The model
  receives the agent's **raw retrieved passages** (`evidence`), not a generated
  summary.

  How the evidence is used is set by `--grounding`:
  - `augment` (default) — evidence is helpful **context**; the model reasons over
    it (plus the question) and commits to the best-supported option.
  - `strict` — the model may use **only** the evidence and is told to pick
    "Insufficient information" when the evidence doesn't support an option.

  (`--mode librarian` is accepted as a deprecated alias of `--mode knowledge`.)

Everything else — the verbatim LAB-Bench MCQ prompt (with the chembench
`"Think step by step."` chain-of-thought prefix), the shuffled options plus the
`"Insufficient information to answer the question"` refusal option, the
`[ANSWER]X[/ANSWER]` parser, and the accuracy / precision / coverage metrics —
matches the upstream harness (reimplemented in `labbench_compat.py`, so no
`labbench`/`chembench` install is needed).

## Files

- `labbench_compat.py` — task registry (`TASKS`), prompt template, deterministic
  choice shuffling, answer parsing, metrics, and dataset loading.
- `solvers.py` — the baseline and knowledge-layer answering models (the latter is
  agent-agnostic via an injected `search_fn`).
- `run_labbench_eval.py` — CLI runner, agent wiring, and output writer.

## Model version

The LAB-Bench paper evaluated the floating `gpt-4o` and `gpt-4-turbo` aliases,
with experiments run around June–July 2024. At that time `gpt-4o` resolved to
**`gpt-4o-2024-05-13`**, so the runner defaults `--model` to that snapshot to
reproduce the paper's numbers.

## Setup

```bash
pip install -r evals/Literature/LabBench/requirements.txt
```

Credentials are read from the repo `.env` (loaded automatically): set
`OPENAI_API_KEY` for the answering model. The **knowledge-layer agent's own LLM**
is separate (it does query planning and relevance filtering) — configure it with
`--agent-model` / `--agent-base-url` (aliases: `--librarian-model` /
`--librarian-base-url`), or the `LLM_*` env vars `LLMClient` reads.

Data loads from HuggingFace (`futurehouse/lab-bench`, config = the `--task` name)
by default; the dataset is gated, so `huggingface-cli login` may be required.
Alternatively download `<task>-v1-public.jsonl` from the LAB-Bench repo and pass
`--data-file`.

## Run

Knowledge layer (GPT-4o grounded in `LibrarianAgent`) on SeqQA:

```bash
python -m evals.Literature.LabBench.run_labbench_eval \
    --task SeqQA --mode knowledge --model gpt-4o-2024-05-13 \
    --agent-model <agent-llm> --agent-base-url <agent-llm-url> \
    --out-dir results/seqqa_knowledge
```

Parametric baseline (same seed → identical shuffled options for a fair compare):

```bash
python -m evals.Literature.LabBench.run_labbench_eval \
    --task SeqQA --mode baseline --model gpt-4o-2024-05-13 \
    --out-dir results/seqqa_baseline
```

Smoke test on a few questions: add `--max-examples 3`.

Useful flags: `--agent-model` / `--agent-base-url` (the librarian's own LLM,
separate from the answering model), `--evidence-char-budget` (raw passage chars
per paper fed to the LLM), `--strip-query-sequences` / `--no-strip-query-sequences`
(per-task default), `--full-text` / `--no-full-text` (on by default), `--top-k`,
`--seed`, `--resume`, `--data-file`.

## Outputs (in `--out-dir`)

- `predictions.jsonl` — one row per question: task, gold/predicted letter,
  correct, sure, raw output, and (knowledge mode) the queries issued and evidence
  count.
- `metrics.json` — accuracy, precision, coverage, counts.
- `summary.md` — human-readable summary table.
- `run_config.json` — the run configuration.

## Metrics

- **accuracy** = correct / total
- **precision** = correct / answered (questions where the model did not pick the
  refusal option)
- **coverage** = answered / total

A higher precision at comparable coverage in `knowledge` mode vs `baseline` is
the signal that the knowledge layer improves answers rather than the model
guessing from memory.

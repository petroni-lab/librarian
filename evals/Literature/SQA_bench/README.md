# ScholarQA-Bench

The paper's ScholarQA-Bench rows: Citation F1 on the **bio** (1451 q) and
**neuro** (1308 q) sets, and Citation F1 plus LLM-judge scores on the
29-question biomedical subset of **ScholarQA-Multi**.

Normally you do not call anything here directly:

```bash
./evals/Literature/literature_eval.sh --bench sqa            # all three
./evals/Literature/literature_eval.sh --bench sqa --only bio # one
```

`--only` takes `bio`, `neu`, `multi`, and repeats. See
[`run_sqa.sh -h`](run_sqa.sh) for the flags it forwards.

## How a run is put together

[`run_sqa_new_stack.py`](run_sqa_new_stack.py) is the prediction step for every
arm. It runs `LibrarianAgent` for retrieval and `SynthesisAgent` over the
passages it returns, and writes one predictions file per run.

- **bio / neu** call it directly. It runs AutoAIS citation scoring itself at the
  end, so no container is involved. `--skip-citation-eval` stops after the
  predictions; a later run with `--resume` picks up the scoring.
- **multi** goes through [`run_sqa_multi.sh`](run_sqa_multi.sh): predictions,
  citation scoring, then each Prometheus judge started under Apptainer around
  its own pass. Both passes merge into one `judge_eval/results.json`.

### Citation format

Citation F1 is scored on the citations in the written answer, and the AutoAIS
scorer only extracts **numeric** ones — `[2]`, `[2, 7]`, `[REF_2]`, `[2.3]`. The
shipped `SynthesisAgent` cites author-year markdown links instead
(`[Chen 2023](https://europepmc.org/article/MED/…)`), which parses here as *no
citations at all*: every sentence scores unsupported and Citation F1 collapses
to roughly zero, with the run completing normally and reporting the number.

[`numbered_citations.py`](numbered_citations.py) is what this benchmark
synthesises through instead — papers rendered as `[REF_n]` blocks with a lookup
table, plus the matching summarizer prompt. Retrieval, ranking and the evidence
are untouched; only how the answer names a paper changes.

The scorers are derived from ScholarQABench (MIT) but live in
[`scorers/`](scorers/NOTICE), not in the clone: `citation_correctness_eval.py`
for AutoAIS, `prometheus_eval.py` for the two judge passes, and `run_utils.py`.
`../setup.sh` clones ScholarQABench at a pinned commit into `code/` and leaves
it **pristine**, which `../setup.sh --check` asserts. The rubric file is read
straight out of the clone.

## What it needs

| | GPU | Notes |
|---|---|---|
| `--only bio` / `neu` | 1 × ≥8 GB | AutoAIS (`attrscore-flan-t5-xl`, ~5.7 GB in bf16) |
| `--only multi` | 4 × H100 | two Prometheus 8x7B judges at tensor-parallel size 4 |

No API keys. The judge models need NVLink — without it NCCL falls back to PCIe
peer-to-peer and they die in `initialize_model_parallel`.

`[sqa] apptainer_image` in `literature_eval.toml` is the container their vLLM
runs in. Apptainer takes either form in that one field — a `docker://` URI,
which it pulls and converts to a SIF on first use (several GB, cached in
`$APPTAINER_CACHEDIR`), or the path of a `.sif` you already have. It defaults to
`docker://vllm/vllm-openai:v0.29.0`, so it works without a prepared image.
**Leave it empty and `multi` is skipped rather than failing** — `--bench sqa`
still produces the bio and neuro rows, which need no container. Asking for
`--only multi` without one is an error.

### Disk

The AutoAIS weights are **11.4 GB** to download — the published checkpoint is
fp32, even though the scorer loads it in bf16 — and they land in the Hugging
Face cache, which defaults to `$HOME/.cache/huggingface`. On a home directory
with a quota that fails partway through, after the answers have been generated,
either as `Disk quota exceeded` or as a `Background writer channel closed` from
the Xet downloader. Point `[paths] hf_home` in `literature_eval.toml` (or
`HF_HOME`) at scratch or node-local storage first. `--only multi` additionally
pulls the two Prometheus judges, which are far larger.

AutoAIS also runs on CPU with identical scores, but roughly **40× slower**
(measured: 1046 s vs 25 s for 3 questions). The generation step is pure network,
so the cheap split is to answer on a CPU box with `--skip-citation-eval` and
score on the GPU one afterwards.

The scoring stack is CUDA-specific and is a second environment of its own
(`../envs/sqa-scoring.lock`, resolved for Linux), built only when a run reaches
a scoring step. To build it ahead of time:

```bash
../setup.sh --bench sqa      # clone, data, and both environments
```

## Data

Not redistributed here. `../setup.sh` takes the bio, neuro and multi files out
of the [ScholarQABench](https://github.com/AkariAsai/ScholarQABench) clone it
makes under `code/`, and rebuilds the biomedical subset by filtering the
gold-reference file through the committed
`data/scholarqa_multi/scholar_multi_biomed_public_subset_ids.txt`. That id list
pins which 29 questions the paper's `multi` row was measured on;
[`reconstruct_scholar_multi_biomed_from_ids.py`](reconstruct_scholar_multi_biomed_from_ids.py)
applies it.

## Baselines

`--no-librarian` answers from the model alone, with no retrieval — the
parametric row the librarian is compared against.

The paper's other comparison row, the OpenScholar-style BM25 baseline, is not
reproducible here: it read chunks from an OpenScholar datastore Elasticsearch
index that is not part of this repository, so no flag for it is offered.

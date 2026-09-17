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

- **bio / neu** call it directly. It then runs AutoAIS citation scoring itself
  at the end, so no container is involved. `--skip-citation-eval` stops after
  the predictions; a later run with `--resume` picks up the scoring.
- **multi** goes through
  [`run_local_multieval_new_stack_apptainer.sh`](run_local_multieval_new_stack_apptainer.sh),
  which runs predictions, then citation scoring, then starts each Prometheus
  judge in turn and scores against it. The judge passes are driven through
  [`run_multi_evals_k8s.sh`](run_multi_evals_k8s.sh), which is given a
  `--pred_file` and so skips straight to scoring.

### Citation format

Citation F1 is scored on the citations in the written answer, and the AutoAIS
scorer only extracts **numeric** ones — `[2]`, `[2, 7]`, `[REF_2]`, `[2.3]`. The
shipped `SynthesisAgent` cites author-year markdown links instead
(`[Chen 2023](https://europepmc.org/article/MED/…)`), which is the better output
for a reader but parses here as *no citations at all*: every sentence scores
unsupported and Citation F1 collapses to roughly zero, with the run completing
normally and reporting the number.

So this benchmark keeps the citation contract its metric is defined on.
[`numbered_citations.py`](numbered_citations.py) is that contract — papers
rendered as `[REF_n]` blocks with a lookup table, plus the matching summarizer
prompt — and `run_sqa_new_stack.py` synthesises through it. Retrieval, ranking
and the evidence are untouched; only how the answer names a paper changes.

The scorers come from ScholarQABench (MIT). `../setup.sh` clones it at a pinned
commit into `code/` and copies our changes — the four files under
[`overlay/`](overlay/NOTICE) — over it: `citation_correctness_eval.py` for
AutoAIS, `prometheus_eval.py` for the two judge passes, `run_utils.py`, and
`requirements.txt`.

## What it needs

| | GPU | Notes |
|---|---|---|
| `--only bio` / `neu` | 1 × ≥8 GB | AutoAIS (`attrscore-flan-t5-xl`, ~6.5 GB) |
| `--only multi` | 4 × H100 | two Prometheus 8x7B judges at tensor-parallel size 4 |

No API keys. The judge models need NVLink — without it NCCL falls back to PCIe
peer-to-peer and they die in `initialize_model_parallel`. `APPTAINER_IMAGE` in
`[sqa] apptainer_image` in `literature_eval.toml` must point at a vLLM image
you can read.

AutoAIS also runs on CPU with identical scores, but roughly **40× slower**
(measured: 1046 s vs 25 s for 3 questions) — fine for a smoke run, not for the
full sets. The generation step is pure network, so the cheap split is to answer
on a CPU box with `--skip-citation-eval` and score on the GPU one afterwards.

The local scoring stack (torch, transformers, vLLM) is not in the repository's
`evals` extra, because it is CUDA-specific:

```bash
uv pip install -r evals/Literature/SQA_bench/code/requirements.txt   # after setup.sh
```

## Data

Not redistributed here — run `../setup.sh`, which takes the bio, neuro and multi
files out of the [ScholarQABench](https://github.com/AkariAsai/ScholarQABench)
clone it makes under `code/` and rebuilds the biomedical subset by filtering the
gold-reference file through the committed
`data/scholarqa_multi/scholar_multi_biomed_public_subset_ids.txt`. That id list
is what pins which 29 questions the paper's `multi` row was measured on;
[`reconstruct_scholar_multi_biomed_from_ids.py`](reconstruct_scholar_multi_biomed_from_ids.py)
is the script that applies it.

## Baselines

`run_sqa_new_stack.py` keeps the flags for the rows the librarian is compared
against, but only the parametric one runs here:

- `--no-librarian` — answer from the model alone, no retrieval. Works.
- `--bm25-retrieval` — the OpenScholar-style BM25 baseline. It read chunks from
  an OpenScholar datastore Elasticsearch index, which is not part of this
  repository, so the flag fails with a message saying so rather than silently
  retrieving something else.

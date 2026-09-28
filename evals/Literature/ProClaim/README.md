# ProClaim-eval

ProClaim-eval measures claim-level biomedical evidence grounding. Each example
is a short biological claim with one gold verdict: `SUPPORT`, `REFUTE`, or
`UNCERTAIN`.

This integration evaluates the librarian as a claim-verification agent, not as a
long-form report generator. The agent receives only the natural-language claim
and verification instructions, runs literature retrieval without the normal
summary step, and the harness builds the verdict from the retrieved evidence.

## Data

The two claim sets are ProClaim's own and are not redistributed here. They
arrive with the clone `../setup.sh --bench proclaim` makes, and are read
straight out of it:

- `ProClaim_src/datasets/signor.csv` — SIGNOR-Fact signed protein-protein
  interaction claims.
- `ProClaim_src/datasets/connectomedb.csv` — ConnectomeDB-Fact ligand-receptor
  interaction claims.

The CSVs carry gold labels and provenance for scoring. The inference adapter
does not pass those fields to the librarian.

## Two arms

`run_proclaim.sh` runs both by default; `--only` picks one.

- **`verifier`** (0.66) — the one-prompt Verifier Agent. Needs the librarian
  endpoint and a verdict model, and nothing else: it ports the retrieval
  primitives it needs rather than importing ProClaim.
- **`proclaim`** (0.80) — the full ProClaim pipeline, cloned by `../setup.sh` to
  `ProClaim_src/` from [saezlab/ProClaim](https://github.com/saezlab/ProClaim)
  (**GPL-3.0**). Our librarian retrieval backend lives in [`backend/`](NOTICE)
  and is grafted on at run time by `direct_entry.py`; the clone itself stays
  pristine, which `../setup.sh --check` asserts.

  This arm has its own environment (`proclaim-pipeline`, resolved for Linux and
  CUDA) and a third endpoint, the evidence subagent, which the runner starts
  from `[proclaim] apptainer_image` and stops on exit. One already answering at
  `[proclaim] subagent_url` is reused and left running.

  `--only verifier` needs neither.

**Licensing.** `backend/`, `direct_entry.py`, `proclaim_librarian.py` and
`configs/` are GPL-3.0, not MIT, and the repository's root LICENSE does not
reach them. See [NOTICE](NOTICE).

## Run

For the paper's two rows use `./run_proclaim.sh` — see
[../README.md](../README.md#what-each-bench-needs). The commands below are for
ad-hoc runs and baselines, and all take exactly one retrieval mode:
`--librarian-agent`, `--pubmed-s2`, `--web-search` or `--no-agent`.

Smoke test:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --librarian-agent \
  --subset all \
  --max-examples 3 \
  --out-dir results/proclaim_eval_smoke
```

A full subset, resumable and cached:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --librarian-agent \
  --subset signor \
  --out-dir results/proclaim_eval_signor \
  --resume \
  --cache
```

`--subset` takes `signor`, `connectomedb` or `all`. `--model-config` reads
agent settings (`llm_base_url`, `llm_model_name`, `thinking`, …) from a JSON or
YAML file; CLI flags override it. A `tqdm` progress bar appears when `tqdm` is
installed — `--no-progress` restores per-example log lines. One librarian agent
is reused for the run, but each claim is a fresh prompt with no history.

### Baselines

OpenAI Responses API web search, which bypasses the librarian entirely:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset all \
  --out-dir results/proclaim_eval_gpt54_web_search \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-5.4 \
  --web-search \
  --web-search-tool-choice required \
  --web-search-context-size medium \
  --resume --cache
```

PubMed + Semantic Scholar abstracts, the retrieval ProClaim's own evidence
programmer uses:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset all \
  --out-dir results/proclaim_eval_pubmed_s2 \
  --pubmed-s2 \
  --pubmed-s2-top-k 5 \
  --resume --cache
```

It fetches the top-k relevance-sorted PubMed abstracts (entity-AND boolean
query, `usehistory=y` + `sort=relevance`) and the top-k Semantic Scholar
abstracts, merges and deduplicates by PMID (PubMed wins ties), then classifies
from those abstracts in one call — no iterative loop, no full text. Only the
retrieval source differs from `--librarian-agent`; verdict synthesis is
identical, so the two are directly comparable. `--pubmed-s2-top-k` defaults to 5
and applies per source. `PUBMED_API_KEY` / `S2_API_KEY` raise the rate limits.

The paper's remaining baseline used an internal literature agent over an
Elasticsearch index that the librarian replaced. That retriever is not part of
this repository, so it is not offered here.

## Outputs

Written to `--out-dir`:

- `predictions.jsonl` — one row per claim: claim, gold label, predicted label,
  reasoning, citations, raw output, run metadata.
- `metrics.json` — SIGNOR-Fact, ConnectomeDB-Fact, and combined.
- `confusion_matrix.csv` — per-subset and combined counts.
- `summary.md`, `latex_table.tex` — readable summary and paper-ready rows.

## Metrics

- `AGR`: prediction-label agreement, equivalent to accuracy.
- `macro_fpr` / `macro_fnr`: average one-vs-rest false positive / negative rate
  over the three labels.

`latex_table.tex` carries both the AGR-only row and the AGR/FPR/FNR row, to
match the ProClaim paper's table format.

## Leakage constraint

During inference the librarian must not receive SIGNOR or ConnectomeDB names,
gold labels, curated evidence or comments, PMIDs, entity columns, or provenance
fields. Only the claim and the fixed verification instructions are model-facing;
gold labels stay in the harness for scoring after prediction.

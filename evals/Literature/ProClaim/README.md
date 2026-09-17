# ProClaim-eval

ProClaim-eval measures claim-level biomedical evidence grounding. Each example is
a short biological claim with one gold verdict: `SUPPORT`, `REFUTE`, or
`UNCERTAIN`.

This integration evaluates the librarian as a claim-verification agent, not as a
long-form report generator. The agent receives only the natural-language claim
and verification instructions, runs literature retrieval without the normal
summary step, and the harness builds the final verdict from the retrieved
evidence snippets.

## Data

The two claim sets are ProClaim's own and are not redistributed here. They
arrive with the clone `../setup.sh --bench proclaim` makes, and the harness
reads them straight out of it — nothing is copied:

- `ProClaim_src/datasets/signor.csv` — SIGNOR-Fact signed protein-protein
  interaction claims.
- `ProClaim_src/datasets/connectomedb.csv` — ConnectomeDB-Fact ligand-receptor
  interaction claims.

The CSVs include gold labels and benchmark provenance fields for scoring and
debugging. The inference adapter does not pass those fields to the librarian.

## Two arms

`run_proclaim.sh` runs both by default; `--only` picks one.

- **`verifier`** (0.66) — the one-prompt Verifier Agent. Needs only the
  librarian endpoint and a verdict model. This arm is self-contained: it ports
  the retrieval primitives it needs rather than importing them.
- **`proclaim`** (0.80) — the full ProClaim pipeline, which runs **ProClaim**,
cloned by `../setup.sh` to `ProClaim_src/` from
  [saezlab/ProClaim](https://github.com/saezlab/ProClaim) (**GPL-3.0**) with our
  librarian retrieval backend laid over it — see [`overlay/`](overlay/NOTICE).
  It needs its own environment (`cd ProClaim_src && uv sync`) and a third
  endpoint, the evidence subagent — which the runner starts for you from
  `[proclaim] apptainer_image`, and stops on exit. A subagent already answering
  at `[proclaim] subagent_url` is reused and left running.

  `--only verifier` needs neither.

## Run

To reproduce the paper's two +Librarian rows use `./run_proclaim.sh` — see
[../README.md](../README.md#what-each-bench-needs). The commands below are for
ad-hoc runs and baselines.

Smoke test:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset all \
  --max-examples 3 \
  --out-dir results/proclaim_eval_smoke
```

Full subsets:

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset signor \
  --out-dir results/proclaim_eval_signor \
  --resume \
  --cache

python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset connectomedb \
  --out-dir results/proclaim_eval_connectomedb \
  --resume \
  --cache
```

Use `--model-config` for a flat JSON/YAML file with existing agent settings
such as `llm_base_url`, `llm_model_name`, `retriever`, `es_url`, or `thinking`.
CLI flags such as `--llm-model`, `--retriever elastic`, and `--thinking` override
the config file.

The runner shows a `tqdm` progress bar when `tqdm` is installed. Use
`--no-progress` to disable the progress bar and restore per-example log lines.
The evaluator reuses one librarian agent for the run; each claim is
still passed as a fresh prompt with no conversation history. Use
`--librarian-agent` for the `LibrarianAgent` retrieval path, which is what
`run_proclaim.sh` passes. The paper's baseline rows used an internal literature
agent over an Elasticsearch index that the librarian replaced; that retriever is
not part of this repository, so without the flag the runner raises rather than
retrieving something else. Only the +Librarian rows are reproducible here.

OpenAI Responses API web-search baseline:

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
  --resume \
  --cache
```

This bypasses the librarian and sends each claim to the OpenAI Responses API with
the `web_search` tool enabled. Use it as an LLM web-search baseline, not as a
a librarian retrieval run.

PubMed + Semantic Scholar abstract baseline (ProClaim evidence-programmer
retrieval):

```bash
python -m evals.Literature.ProClaim.verifier \
  --proclaim-path evals/Literature/ProClaim \
  --subset all \
  --out-dir results/proclaim_eval_pubmed_s2 \
  --pubmed-s2 \
  --pubmed-s2-top-k 5 \
  --resume \
  --cache
```

This replaces the librarian with the ProClaim evidence-programmer retrieval
primitives: it fetches the top-k relevance-sorted PubMed abstracts (entity-AND
boolean query, `usehistory=y` + `sort=relevance`) and the top-k Semantic Scholar
abstracts (free-text query), merges and deduplicates them by PMID (PubMed wins
ties), then asks the configured model to classify `SUPPORT`, `REFUTE`, or
`UNCERTAIN` from those abstracts in a single call — no iterative loop and no full
text. Only the retrieval source changes versus `--librarian-agent`; the verdict
synthesis is identical, so the two are directly comparable. `--pubmed-s2-top-k`
defaults to 5 and applies per source (5 PubMed + 5 S2). Set `PUBMED_API_KEY` /
`S2_API_KEY` env vars for higher search rate limits.

## Outputs

The evaluator writes:

- `predictions.jsonl` — one row per claim with claim, gold label, predicted
  label, reasoning, citations, raw output, and runtime metadata.
- `metrics.json` — metrics for SIGNOR-Fact, ConnectomeDB-Fact, and combined.
- `confusion_matrix.csv` — per-subset and combined confusion counts.
- `summary.md` — human-readable summary.
- `latex_table.tex` — paper-ready table rows.

## Metrics

- `AGR`: prediction-label agreement, equivalent to accuracy.
- `macro_fpr`: average one-vs-rest false positive rate over the three labels.
- `macro_fnr`: average one-vs-rest false negative rate over the three labels.

`latex_table.tex` contains both the AGR-only row and the AGR/FPR/FNR row used to
reproduce the ProClaim paper table format.

## Leakage Constraint

During inference, the librarian must not receive SIGNOR or ConnectomeDB names, gold
labels, curated evidence/comments, PMIDs, entity columns, or provenance fields.
Only the natural-language claim and the fixed verification instructions are
model-facing. Gold labels and provenance stay inside the harness for scoring
after prediction.

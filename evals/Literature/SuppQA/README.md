# SuppQA (LAB-Bench): supplementary-material retrieval

[LAB-Bench SuppQA](https://github.com/Future-House/LAB-Bench/tree/main/SuppQA):
82 public multiple-choice questions, each answerable only from one paper's
supplementary material. The bench measures what `supplementary_enrichment` adds
to accuracy, what it costs in retrieval time, and how supplementary paragraphs
fare against main-text ones at each stage.

Questions whose paper has no PMC id in Europe PMC are skipped. As upstream, each
question is prefixed with the paper's title and DOI; that text is also the
librarian's query. Scoring is the LabBench multiple-choice scoring, so no LLM judge is needed.

## Arms

| `--arm` | What it runs |
|---|---|
| `preflight` | No LLM. Fetches each question's own paper and checks whether its gold `key-passage` appears in the extracted supplement and/or the main text. This sets the ceiling. |
| `baseline` | The answering model alone |
| `main` | Librarian, abstracts + full text, `supplementary_enrichment` off |
| `supp` | The same, plus supplementary PDF/Word files |

For a clean comparison, run `supp` first, then `main` with `--queries-from` that
run. Both arms then search the same sub-queries, and only the supplementary files
differ. The relevance filter runs at temperature 0 in both.

## Run

From the repository root, with the librarian's LLM in `.env` (`LLM_*`):

```bash
R=evals/Literature/results/suppqa
ANSWER=(--model gpt-4o-2024-05-13)   # or a local vLLM: (--model "$LLM_MODEL" --base-url "$LLM_BASE_URL"), with OPENAI_API_KEY=EMPTY
PY=(uv run --with openai python -m evals.Literature.SuppQA.run_suppqa_eval)

"${PY[@]}" --arm preflight --out-dir $R/preflight
"${PY[@]}" --arm baseline --out-dir $R/baseline "${ANSWER[@]}"
"${PY[@]}" --arm supp     --out-dir $R/supp     "${ANSWER[@]}"
"${PY[@]}" --arm main     --out-dir $R/main     "${ANSWER[@]}" --queries-from $R/supp

uv run --with openai python -m evals.Literature.SuppQA.analyze_suppqa \
    $R/preflight $R/baseline $R/main $R/supp > $R/report.md
```

`--resume` keeps finished rows and re-runs errored ones. `--max-examples N` runs
a pilot, and `--max-workers` (default 4) sets how many questions run concurrently.
Keep `--max-workers` the same for both arms, or the timings are not comparable.

## Report

- **Accuracy:** accuracy, precision and coverage per arm, over the questions every arm
  answered. Also main→supp flips with an exact McNemar p-value.
- **Retrieval time:** seconds per question for Stage 2 + 3, with a per-span breakdown
  (supplementary downloads fall under `fulltext`, their parsing under `bm25`) and the
  number of supplementary fetch errors.
- **Evidence by source:** main-text vs supplementary paragraphs in the candidate pool,
  kept by BM25, sent to the judge, cited, and shown to the answering model.
- **Target-paper funnel:** whether each question's own paper and its key passage
  reached each of those stages.

`--evidence-char-budget` defaults to 6000 characters per paper, against LabBench's
1500. Supplementary spans come last in a paper's evidence, so a small budget cuts
them first. The "shown to the answering model" rows measure this.

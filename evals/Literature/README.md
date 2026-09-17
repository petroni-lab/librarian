# Literature Benchmarks

The evaluation harness behind the paper's four literature tables. One command
reproduces the **+Librarian row** of each:

```bash
./evals/Literature/literature_eval.sh --bench all \
    --librarian-url http://localhost:8000/v1 --librarian-model glm-5-fp8
```

`--bench` also takes a subset (`--bench litqa2,labbench`) or a single benchmark.

---

## Start here

**1. Install the harness.**

```bash
uv sync --extra evals
```

That covers driving the benchmarks and the LitQA2 / LAB-Bench scoring, which is
all API-hosted. The local scoring stacks — ScholarQA-Bench's AutoAIS scorer and
Prometheus judges — need torch, transformers and vLLM against your own CUDA, so
they stay in
[`SQA_bench/overlay/requirements.txt`](SQA_bench/overlay/requirements.txt) —
which `setup.sh` puts at `SQA_bench/code/requirements.txt` — to install
separately when you get to `--bench sqa`.

**2. Fetch the benchmarks.**

```bash
./evals/Literature/setup.sh               # --check to see what is missing first
./evals/Literature/setup.sh --bench sqa   # or fetch just one bench's inputs
```

Three of the four suites are built on someone else's benchmark repository, and
none of the data is ours to redistribute. So this repository carries only the
files we changed or added, and `setup.sh` clones each upstream repository at a
pinned commit and copies our changes over it. It takes a few seconds and is
safe to re-run. `--bench` fetches only what those benches need, so running one
suite never requires another's inputs. See
[Data provenance](#data-provenance) for what comes from where.

ProClaim's second arm runs a second project with its own dependencies, so it
needs its own environment — once the clone exists:

```bash
cd evals/Literature/ProClaim/ProClaim_src && uv sync
```

`--bench proclaim --only verifier` does not need it.

**3. Serve the librarian model.** Every bench needs it — the librarian is an
agent, not a model: it plans Europe PMC sub-queries and judges retrieved
paragraphs, so it needs an LLM behind it, and there is no offline fallback.
Any OpenAI-compatible endpoint works:

```bash
vllm serve zai-org/GLM-5-FP8 --served-model-name glm-5-fp8 --port 8000
```

`--librarian-url` is then `http://localhost:8000/v1`. The paper's numbers are
GLM-5 numbers; a different model makes them incomparable rather than wrong, so
say which one you used when you report a result.

**4. Smoke-test it** before committing to a full run. `--limit N` caps the
question count on every bench, and `--dry-run` prints the commands without
calling anything. Work up the ladder — each rung adds one requirement:

```bash
L="--librarian-url http://localhost:8000/v1 --librarian-model glm-5-fp8"

# 0. nothing is called; checks wiring, paths and that setup.sh ran
./evals/Literature/literature_eval.sh --bench all --dry-run $L

# 1. no GPU, no scoring model to download. Needs OPENAI_API_KEY.
./evals/Literature/literature_eval.sh --bench litqa2,labbench --limit 3 $L

# 2. one >=8 GB GPU: downloads AutoAIS (~6.5 GB) on first use
./evals/Literature/literature_eval.sh --bench sqa --only bio --limit 3 $L

# 3. one >=24 GB GPU. Needs ANTHROPIC_API_KEY. `verifier` is the cheap arm —
#    it is the one that does NOT need the evidence subagent.
./evals/Literature/literature_eval.sh --bench proclaim --only verifier --limit 3 $L
```

Rung 0 is worth running on its own first: it is the one that tells you whether
`setup.sh` fetched everything, without spending a token.

Two things deliberately have no smoke rung. `sqa --only multi` wants four H100s
and an `APPTAINER_IMAGE`, and `proclaim --only proclaim` wants a third endpoint
(the evidence subagent) — start those only when the cheaper rungs pass.

---

## Reference results (paper)

The **+Librarian** row each `--bench` reproduces:

| `--bench` | Runner | Reproduces |
|---|---|---|
| `litqa2` | [AstaBench/run_litqa2.sh](AstaBench/run_litqa2.sh) | Cov 95.6 / Prec 82.6 / Acc 78.9 |
| `labbench` | [LabBench/run_labbench.sh](LabBench/run_labbench.sh) | SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, Cloning 48.5 |
| `proclaim` | [ProClaim/run_proclaim.sh](ProClaim/run_proclaim.sh) | Verifier 0.66, ProClaim 0.80 |
| `sqa` | [SQA_bench/run_sqa.sh](SQA_bench/run_sqa.sh) | Citation F1 (Bio, Neu); Citation F1 + LLM judge (Multi) |

### LitQA2, in full

Open-form LitQA2 with GPT-5.4 as the answering agent, under four retrieval
settings, against OpenScholar-8B:

| Agent | Retriever | Coverage | Precision | Accuracy |
|-------|-----------|---------:|----------:|---------:|
| OpenScholar-8B | Semantic Scholar | 84.6 | 40.3 | 34.1 |
| QA Agent | Parametric | **100.0** | 17.6 | 17.6 |
| QA Agent | Web Search | 92.3 | *76.2* | *70.3* |
| QA Agent | LibShort | *95.6* | **82.6** | **78.9** |

- **Coverage** — fraction of questions answered (i.e. not abstained).
- **Precision** — fraction of *answered* questions that are correct.
- **Accuracy** — fraction of *all* questions answered correctly.

91 questions (the `europepmc_fulltext` split), judge `gpt-4o-2024-11-20`.

**Expect close, not identical.** These scripts pass no knob overrides, so a
rerun measures the librarian as currently configured in
[`librarian/config.toml`](../../librarian/config.toml). Treat a gap over a few
points as worth investigating rather than something to tune the scripts around.

Only the **+Librarian** rows are scripted. The baseline rows each bench compares
against used a pre-librarian retrieval stack that this repository does not ship;
asking for one now fails with a message saying so rather than silently
substituting something else. Each runner's `-h` header names the one-flag change
that produced its baseline.

---

## Configuration

Every endpoint, model and path lives in
[`literature_eval.toml`](literature_eval.toml), grouped by which bench uses it —
there is no need to edit the per-bench runners:

```toml
[librarian]              # the one endpoint every bench needs
url   = ""               # required; or pass --librarian-url
model = "glm-5-fp8"

[sqa]
synthesis_model = ""     # empty means "inherit" — here, librarian.model
apptainer_image = ""     # only --only multi needs this
```

Precedence is **flags > environment variables > the file**, so nothing in it is
sticky:

```bash
# one-off override, no editing
LIBRARIAN_URL=http://myhost:8080/v1 ./evals/Literature/literature_eval.sh --bench litqa2

# a personal profile kept outside the repo
LITERATURE_EVAL_CONFIG=~/my_eval.toml ./evals/Literature/literature_eval.sh --bench litqa2
```

The only required setting is `[librarian] url`. Worth knowing about:
`[paths] results_root` (defaults to `evals/Literature/results/`, git-ignored),
`[paths] python` (defaults to the repo's `.venv`, then `python3`), and — only
for `sqa --only multi` — `[sqa] apptainer_image`.

[`load_config.py`](load_config.py) is what maps the TOML onto the flat
environment variables the runner scripts read. To drive a per-bench runner
directly, apply the same config first:

```bash
eval "$(python evals/Literature/load_config.py)"
bash evals/Literature/SQA_bench/run_sqa.sh --only bio
```

`MAX_SAMPLES` / `MAX_CONNECTIONS` / `MAX_WORKERS` are wall-clock only and never
change a score.

### In-process (default) vs `--via-api`

By default the agents are built inside the eval process and talk to
`LIBRARIAN_URL` directly. Nothing else is needed.

`--via-api` instead routes every question through a running
[`orchestrator.py`](../../orchestrator.py), which builds the same agents inside
the server:

```bash
uv run uvicorn orchestrator:app --port 8080     # in another shell
./evals/Literature/literature_eval.sh --bench sqa --via-api
```

This is worth it when the server sits closer to the LLM endpoint than your eval
box does: the librarian's Stage-2 BM25 runs on the server's CPUs, and a question
costs one round trip instead of one per LLM call. Pointed at localhost it buys
nothing. The trade is that the server runs the configuration it was started
with, so no per-run knob can be honoured and synthesis and retrieval share one
model — `literature_eval.sh` notices an arm that needs otherwise and stands
down on its own:

```
NOTE  --no-librarian has no API equivalent — using in-process agents for this run.
```

See [`orchestrator_client.py`](orchestrator_client.py) for the full rationale.

---

## What each bench needs

Everything below is *in addition* to the librarian endpoint, which they all
need. The GPUs are for **scoring**, not retrieval.

| `--bench` | Local GPU | Other endpoints | API keys |
|---|---|---|---|
| `litqa2` | none | — | `OPENAI_API_KEY` |
| `labbench` | none | — | `OPENAI_API_KEY` |
| `proclaim` | 1 × ≥24 GB | evidence subagent | `ANTHROPIC_API_KEY` |
| `sqa --only bio` / `neu` | 1 × ≥8 GB | — | none |
| `sqa --only multi` | 4 × H100 | 2 × Prometheus judge | none |

Pick the tier you need rather than the largest:

- **`litqa2` and `labbench` need no GPU at all** and are the two to start with;
  their answering and judge models are API-hosted.
- One ≥24 GB card covers both ProClaim's evidence subagent (Qwen3.5-9B, 19.3 GB
  of weights, 23.6 GB measured) and SQA's AutoAIS scorer
  (`attrscore-flan-t5-xl`, ~6.5 GB). They are separate jobs, not concurrent.
- **Only `sqa --only multi` needs four GPUs**, for the two Prometheus 8x7B
  judges at tensor-parallel size 4. It is a 29-question subset and the only
  bench with LLM-judge scores; skip it and `--only bio --only neu` still give
  Citation F1 on the full 1451- and 1308-question sets.

Two things that bite on multi-GPU runs: the Prometheus judges need **NVLink**
(without it NCCL falls back to PCIe peer-to-peer and they die in
`initialize_model_parallel` with `NCCL error: unhandled system error`), and
AutoAIS runs on CPU with identical scores but roughly **40× slower** (measured:
1046 s vs 25 s for 3 questions), which is not viable for the full sets.

**Cheaper split.** Generation is pure network, so answer on a CPU box and score
on the GPU one afterwards — `--resume` skips everything already answered:

```bash
# CPU box: answers only
./evals/Literature/literature_eval.sh --bench sqa --only bio --skip-citation-eval
# GPU box: goes straight to AutoAIS
./evals/Literature/literature_eval.sh --bench sqa --only bio
```

### Missing an API key?

Runs fail fast with a 401 rather than silently degrading:

- **No `OPENAI_API_KEY`** → `litqa2` and `labbench` cannot run; the answering
  model (`gpt-5.4`) and the LitQA2 judge (`gpt-4o-2024-11-20`) are both OpenAI.
  Point `ANSWER_URL` / `ANSWER_MODEL` and `LITQA2_JUDGE_MODEL` at a local
  endpoint to avoid the dependency — the scores are then no longer comparable.
- **No `ANTHROPIC_API_KEY`** → `proclaim` cannot run; both arms use Sonnet 4.6.
  To smoke-test the pipeline without it, point the verdict at any
  OpenAI-compatible endpoint, which skips the Anthropic preflight entirely:
  `PROCLAIM_VERDICT_URL=http://localhost:8000/v1 PROCLAIM_VERDICT_MODEL=my-model
  ... --only verifier` (verdicts parse; the score is no longer the paper's).

Keys are read from the environment or the repository's `.env`.

### `--only`, and the two arms that need more

`--only` narrows within a bench and can be repeated: a LAB-Bench task or model,
`verifier` / `proclaim`, `bio` / `neu` / `multi`. `--only CloningScenarios
--only gpt-5.4` picks one LAB-Bench cell. Each `run_<bench>.sh` also runs
standalone, and `-h` on any of them prints its own header.

**ProClaim's second arm** (the full ProClaim pipeline, 0.80) runs the ProClaim
project, which `setup.sh` clones to `ProClaim/ProClaim_src/` and lays our
librarian backend over — see [`ProClaim/overlay/`](ProClaim/overlay/NOTICE).
Its dependencies are not the harness's, so it gets its own environment:

```bash
cd evals/Literature/ProClaim/ProClaim_src && uv sync
```

It also needs a third endpoint — the evidence subagent — which you start
yourself:

```bash
vllm serve Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b \
    --port 9900 --gpu-memory-utilization 0.55 --max-model-len 32768
```

`--only verifier` (the 0.66 Verifier Agent row) needs neither and runs with just
the librarian and a verdict model.

**`sqa --only multi`** starts its two judge endpoints for you, running
`vllm serve` inside apptainer (`APPTAINER_IMAGE`) for
`prometheus-eval/prometheus-bgb-8x7b-v2.0` (answer quality) and then
`prometheus-eval/prometheus-8x7b-v2.0` (relevance) — one after the other, so
4 GPUs covers both. It fails early if fewer are visible.

---

## What is ours

None of the four upstream benchmarks knows anything about the librarian. Searched
across every branch of each one, the word appears in **zero lines of code** —
0 hits in `asta-bench`, 0 in `saezlab/ProClaim`, and the 3 in `ScholarQABench`
are inside ScholarQA-CS *question data*, not source. The integration in each
suite is this repository's contribution, which is the whole reason these files
are committed here rather than fetched.

| Suite | Ours | From upstream |
|---|---|---|
| shared harness | 8 files — the runner, config, `llm_compat.py`, the `--via-api` transport, `setup.sh` | — |
| LitQA2 | 16 — `bio_agent_wrapper.py` (the solver), the open judges, the split reconstruction | 8 files, as [`AstaBench/overlay/`](AstaBench/overlay/NOTICE) |
| LAB-Bench | 8 — `solvers.py`, `labbench_compat.py`, the runner and audit | — (streamed from the Hub) |
| ScholarQA-Bench | 10 — `run_sqa_new_stack.py`, `numbered_citations.py`, the subset reconstruction | 4 files, as [`SQA_bench/overlay/`](SQA_bench/overlay/NOTICE) |
| ProClaim | 5, plus 10 as [`ProClaim/overlay/`](ProClaim/overlay/NOTICE) — the librarian retrieval backend, which exists in no upstream branch | [ProClaim](https://github.com/saezlab/ProClaim) (GPL-3.0) |

ProClaim is the clearest case: its public release has no librarian backend at
all, so `retrieval_backend: "pubmed" | "librarian"`, the two modules behind it
and the three YAML configs that select it exist nowhere upstream. They are why
that arm cannot be reduced to a clone plus an overlay the way the other two are.

---

## Data provenance

Benchmark data is fetched from its original release rather than redistributed
here. What *is* committed is the code, plus the id lists that pin exactly which
rows each subset contains — so a subset is reproducible without us re-publishing
someone else's dataset.

| Data | Source | How you get it |
|---|---|---|
| ScholarQA-Bench bio / neuro / multi | [ScholarQABench](https://github.com/AkariAsai/ScholarQABench) (MIT) | `setup.sh` clones the repository and copies them out of it |
| ScholarQA-Multi biomedical subset (29 q) | derived | `setup.sh` filters the gold-reference file through the committed `scholar_multi_biomed_public_subset_ids.txt` |
| LitQA2 `europepmc_fulltext` split (91 q) | [`futurehouse/lab-bench`](https://huggingface.co/datasets/futurehouse/lab-bench) (CC-BY-SA-4.0) | `run_litqa2.sh` builds it on first use |
| LAB-Bench | the same Hub dataset | streamed at run time |
| ProClaim claim sets | [ProClaim](https://github.com/saezlab/ProClaim)'s `datasets/`, derived from SIGNOR and ConnectomeDB | `setup.sh --bench proclaim` clones them in; read from the clone |

### A note on citation format

ScholarQA-Bench scores Citation F1 on numeric citations (`[2]`, `[2.3]`), while
the shipped `SynthesisAgent` writes author-year markdown links. `--bench sqa`
therefore synthesises through
[`SQA_bench/numbered_citations.py`](SQA_bench/numbered_citations.py), which keeps
the citation contract the metric is defined on. Nothing else about the pipeline
changes, and no other bench is affected — LitQA2 and ProClaim retrieve with the
summary step off, and LAB-Bench answers with its own model.

Third-party **code** is not copied into this repository either. What is
committed is the changes we made to it, under each suite's `overlay/`, at the
paths the files occupy upstream. `setup.sh` clones the upstream repository at a
pinned commit and copies the overlay over it, which means the clone's own `git
diff` is an exact, always-current statement of what we changed:

| Overlay | Upstream | Pinned at | Lands in |
|---|---|---|---|
| [`AstaBench/overlay/`](AstaBench/overlay/NOTICE) — 8 files | [asta-bench](https://github.com/allenai/asta-bench) (AI2), Apache-2.0 | `a9e3380` | `AstaBench/vendor/` |
| [`SQA_bench/overlay/`](SQA_bench/overlay/NOTICE) — 4 files | [ScholarQABench](https://github.com/AkariAsai/ScholarQABench), MIT | `95e6fc5` | `SQA_bench/code/` |
| [`ProClaim/overlay/`](ProClaim/overlay/NOTICE) — 10 files | [ProClaim](https://github.com/saezlab/ProClaim) (Saez-Rodriguez lab), GPL-3.0 | `6316258` | `ProClaim/ProClaim_src/` |

```bash
git -C evals/Literature/AstaBench/vendor    diff   # everything we changed, line by line
git -C evals/Literature/SQA_bench/code      diff
git -C evals/Literature/ProClaim/ProClaim_src diff
```

All three landing directories are git-ignored and disposable: re-running
`setup.sh` resets them to the pinned commit and re-applies the overlay, so edit
`overlay/` rather than the clone.

**ProClaim's overlay is GPL-3.0, not MIT.** Its files are modified versions of
GPL-3.0 originals, so the repository's root LICENSE does not reach them; the
licence text arrives with the clone. The other two overlays are Apache-2.0 and
MIT respectively, as their NOTICEs record.

---

## Layout

```
literature_eval.sh        one entry point for all four benches (start here)
literature_eval.toml      every endpoint, model and path, in one file
load_config.py            maps that TOML onto the runners' environment variables
setup.sh                  clone the upstream benchmarks, apply overlay/, fetch data
orchestrator_client.py    the --via-api transport

AstaBench/                LitQA2 — run_litqa2.sh -> run_evals.py -> bio_agent_wrapper.py
  overlay/                  our changes to asta-bench; setup.sh lays them over vendor/
LabBench/                 LAB-Bench — run_labbench.sh -> run_labbench_eval.py
ProClaim/                 ProClaim-eval — run_proclaim.sh -> verifier.py (arm 1)
                                                         -> proclaim_librarian.py (arm 2)
  overlay/                  the librarian backend; setup.sh lays it over ProClaim_src/
                            GPL-3.0, not MIT
SQA_bench/                ScholarQA-Bench — run_sqa.sh -> run_sqa_new_stack.py
  overlay/                  our changes to ScholarQABench; setup.sh lays them over code/
```

`vendor/`, `code/`, `ProClaim_src/` and everything under `data/` are created by
`setup.sh` and git-ignored; nothing in them is committed.

Each bench directory has its own README with the ad-hoc commands and baselines
that the top-level runner does not cover.

# Literature Benchmarks

The harness behind the paper's literature tables, reproducing the **+Librarian**
row of each. Every bench runs from a plain shell against an OpenAI-compatible
endpoint you are already serving; nothing here starts a model server, and
nothing needs a scheduler.

Four benches: **LitQA2** and **LAB-Bench**, **ProClaim-eval**, and
**ScholarQA-Bench**.

## Start here

The benches need Python 3.11: every lock under `envs/` is compiled for it, and
`setup.sh` refuses to build from anything else. `uv` finds or installs one;
`[paths] python` in `literature_eval.toml` points at a specific interpreter.

```bash
uv python install 3.11          # if you have no 3.11
./evals/Literature/setup.sh     # clones each upstream, prepares data, builds envs
```

`--fetch-only` clones and prepares data without building any environment, which
is enough for a dry run; `--check` reports what is present, missing or drifted
without writing anything.

## Smoke-test it

Each rung adds one requirement. Stop at the last one your machine can do.

| rung | command | needs |
|---|---|---|
| 0 | `--bench all --dry-run` | `setup.sh --fetch-only`; no endpoint |
| 1 | `--bench litqa2,labbench --limit 3` | the librarian endpoint, `OPENAI_API_KEY` |
| 2 | `--bench sqa --only bio --limit 3` | 1 GPU and ~12 GB of disk for AutoAIS |
| 3 | `--bench proclaim --only verifier --limit 3` | no GPU of its own; keyless against a local verdict endpoint |
| 4 | `--bench proclaim --only proclaim --limit 3` | a free ≥24 GB GPU for the evidence subagent |
| 5 | `--bench sqa --only multi` | 4 NVLink GPUs and apptainer |

```bash
L="--librarian-url http://localhost:8000/v1 --librarian-model my-model"

# 0 — prints what each bench would run, calls nothing.
./evals/Literature/literature_eval.sh --bench all --dry-run $L

# 2 — generation is pure network; AutoAIS then scores on the GPU.
./evals/Literature/literature_eval.sh --bench sqa --only bio --limit 3 $L

# 3 — no Anthropic key needed when the verdict model is served locally.
PROCLAIM_VERDICT_URL=http://localhost:8000/v1 PROCLAIM_VERDICT_MODEL=my-model \
    ./evals/Literature/literature_eval.sh --bench proclaim --only verifier --limit 3 $L
```

Rung 2 downloads 11.4 GB into the Hugging Face cache; see `[paths] hf_home` if
`$HOME` has a quota. Rung 4's `[proclaim] apptainer_image` and rung 5's
`[sqa] apptainer_image` are the containers their vLLMs run in.

The first run of a bench builds its environment and says so. To pay that cost
at a moment you chose instead:

```bash
./evals/Literature/setup.sh --bench labbench
```

## How a bench is put together

**No upstream repository is modified.** A bench that needs one clones it at a
pinned commit and leaves it unchanged; what we add lives beside the clone and is
applied at run time. `setup.sh --check` asserts a clone is byte-for-byte its
pinned commit and reports it as *drifted* otherwise.

**Every bench has its own locked environment**, under `.envs/`, installed from a
committed, fully-hashed lock in [`envs/`](envs/) and never resolved at build
time. The repository's own `.venv` is not involved: `uv sync` at the root pulls
no benchmark dependency. One command changes what an environment resolves to,
and it produces a reviewable diff:

```bash
./evals/Literature/setup.sh --relock labbench   # envs/labbench.in -> .lock
```

**Environments build on demand.** The first run of a bench builds its
environment and stamps it with a hash of the lock, the pinned commit and the
Python version; a mismatch rebuilds. `setup.sh --bench <name>` does the same
eagerly, and `--dry-run` builds nothing.

### What a bench declares

Each bench directory carries a `bench.manifest`, so neither `setup.sh` nor
`literature_eval.sh` holds a list of benches and adding one touches only its own
directory:

```
bench=labbench          # the name --bench takes
name=LAB-Bench          # what to call it in output
run=run_labbench.sh     # the runner literature_eval.sh dispatches to
order=20                # position within --bench all
url=…  commit=…  clone=…    # omitted here: LAB-Bench clones nothing
post_fetch=…  post_install=…  # optional hooks
envs=sqa,sqa-scoring    # optional; defaults to the bench name
envs_optional=sqa-scoring   # of those, ones eager setup may skip here
```

Each environment is `envs/<name>.lock`, compiled from `envs/<name>.in`. A bench
declares more than one when its parts run on different machines, such as
generating answers on a laptop and scoring them on a GPU box. An `.in` file can
say how it must be resolved:

```
# uv-compile-args: --python-platform x86_64-unknown-linux-gnu
```

which is how a lock for a GPU box is regenerated on a laptop. An environment
listed in `envs_optional` may fail to build during eager setup without failing
the setup; a run that needs it tries again and fails there.

`envs/_librarian.in` is the shared base every bench includes — the librarian's
own runtime dependencies, since every bench runs the agent in-process.

The repository is not a package (`[tool.uv] package = false`), so its code
reaches a bench environment on `PYTHONPATH` rather than through an install. Only
its dependencies are in the lock; keep `envs/_librarian.in` in step with
`pyproject.toml` by hand.

## Configuration

Every endpoint, model and path lives in
[`literature_eval.toml`](literature_eval.toml).
`load_config.py` maps it onto the flat environment variables the runners read,
and leaves any variable that is already set alone — so a real environment
variable, and the command-line flags, always win:

```bash
# one-off override, no editing
LIBRARIAN_URL=http://myhost:8000/v1 ./evals/Literature/literature_eval.sh --bench labbench

# a personal profile kept outside the repository
LITERATURE_EVAL_CONFIG=~/my_eval.toml ./evals/Literature/literature_eval.sh --bench labbench
```

`paths.python` names the interpreter the per-bench environments are *built
from*, not one the runners share. Changing it rebuilds all of them.

## The four benches

| bench | paper row | local GPU | other endpoints | API keys |
|---|---|---|---|---|
| `litqa2` | Cov 95.6 / Prec 82.6 / Acc 78.9 | none | — | `OPENAI_API_KEY` |
| `labbench` | SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, Cloning 48.5 | none | — | `OPENAI_API_KEY` |
| `proclaim` | Verifier 0.66, ProClaim 0.80 | none for `--only verifier`; 1 × ≥24 GB for `--only proclaim` | evidence subagent (arm 2 only) | `ANTHROPIC_API_KEY`, unless the verdict model is served locally |
| `sqa` | Citation F1 (Bio, Neu) + Citation F1 & LLM (Multi) | 1 × ≥8 GB, or 4 × H100 for `--only multi` | 2 × Prometheus judge for `multi` | none |

Every bench needs the librarian endpoint on top of that. The GPUs are for
*scoring*, not retrieval, so `litqa2` and `labbench` are the two to start with.
`--only verifier` (ProClaim) and `--skip-citation-eval` (SQA) are the two flags
that drop the heavy half of a bench, and neither builds the environment it
would have needed.

## LAB-Bench

Paper row: SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, CloningScenarios 48.5.

No GPU and no clone. The dataset streams from the gated
[`futurehouse/lab-bench`](https://huggingface.co/datasets/futurehouse/lab-bench)
dataset on the Hugging Face Hub at run time, which *is* the paper's ~80% public
subset, so `hf auth login` is the only data prerequisite. Needs `OPENAI_API_KEY`
for the answering models.

The runner loops the paper's four tasks × {`gpt-5.4`, `gpt-4o`}; `--only` takes
a task name or a model name and narrows to either axis. TableQA exists in the
runner but is not in the paper table, so it is excluded unless asked for. See
[`LabBench/README.md`](LabBench/README.md) and `run_labbench.sh -h`.

## Reference results

The paper's numbers are GLM-5 (`zai-org/GLM-5-FP8`, served as `glm-5-fp8`) with
the retrieval knobs now in `librarian/config.toml`. These scripts pass no knob
overrides, so a rerun measures the agent as currently configured: expect close,
not identical. Say which librarian model you used — a different one makes a
result incomparable rather than wrong.

## Troubleshooting

Things that cost time on a shared cluster, none of them defects in the harness.

**`vllm serve` dies in `ssl.create_default_context`.** RHEL-family hosts export
`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` pointing at
`/etc/pki/tls/certs/ca-bundle.crt`, which does not exist inside the
Ubuntu-based vLLM image. The launchers here pass `apptainer exec --cleanenv`
and are unaffected; starting the librarian model by hand needs the same, or
`env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE`.

**A `docker://` image is converted on every start.** `[sqa]` and `[proclaim]
apptainer_image` default to `docker://vllm/vllm-openai:v0.29.0`, which pulls
about 8 GB and builds a SIF each time. Build it once and point the setting at
the file:

```bash
APPTAINER_TMPDIR=/tmp/$USER/apptmp \
    apptainer build ~/containers/vllm-openai_v0.29.0.sif docker://vllm/vllm-openai:v0.29.0
```

Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` away from a quota-limited home
while doing it.

**Job-scoped scratch disappears with the job**, and takes anything running in
the background with it. Keep the SIF and the Hugging Face cache outside
`/scratch/jobs/<id>`.

**Disk.** The SQA scorer downloads 11.4 GB and the Prometheus judges far more.
`[paths] hf_home` moves the Hugging Face cache; `[sqa] scratch_dir` moves the
judge container's pip cache and `TMPDIR`.

## Layout

```
evals/Literature/
  bench_env.sh          per-bench locked environments, built on demand
  setup.sh              clone pinned upstreams, prepare data, build environments
  literature_eval.sh    one entry point; dispatches via each bench.manifest
  literature_eval.toml  every endpoint, model and path
  load_config.py        the TOML -> environment-variable mapping
  llm_compat.py         one LLM client surface across the benches
  evidence_text.py      retrieved evidence, flattened for a prompt
  orchestrator_client.py  the --via-api transport
  envs/                 _librarian.in + one .in/.lock pair per bench
  .envs/                built virtualenvs (git-ignored, disposable)
  AstaBench/            LitQA2, over a pristine AstaBench clone
  LabBench/             LAB-Bench; no clone at all
  ProClaim/             both ProClaim arms, and the librarian backend (GPL-3.0)
  SQA_bench/            ScholarQA-Bench, and the scorers derived from it
```

Each bench directory holds its own `bench.manifest`, `README.md`, and — where
anything is derived from upstream — the `NOTICE` that says what and under which
licence. `ProClaim/backend/` is **GPL-3.0**; the root LICENSE does not reach it.

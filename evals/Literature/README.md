# Literature Benchmarks

The harness behind the paper's literature tables, reproducing the **+Librarian**
row of each. Every bench runs from a plain shell against an OpenAI-compatible
endpoint you are already serving; nothing here starts a model server, and
nothing needs a scheduler.

This layer is the machinery plus **LAB-Bench**, the one bench that needs no
upstream repository. The other three arrive in their own layers.

## Start here

```bash
# Nothing is called: checks wiring and paths, builds no environment.
./evals/Literature/literature_eval.sh --bench labbench --dry-run \
    --librarian-url http://localhost:8000/v1

# Three questions of one task. Needs OPENAI_API_KEY.
./evals/Literature/literature_eval.sh --bench labbench --only DbQA --limit 3 \
    --librarian-url http://localhost:8000/v1 --librarian-model my-model
```

The first real run builds LAB-Bench's environment and says so. To pay that cost
at a moment you chose instead:

```bash
./evals/Literature/setup.sh --bench labbench
```

## How a bench is put together

Three rules, and everything else follows from them.

**Nothing upstream is modified.** This repository contains no copy of, and no
patch to, any benchmark repository. A bench that needs one clones it at a pinned
commit and leaves it exactly as it found it; the changes we need are applied at
run time, from our own tree. `setup.sh --check` asserts that a clone is
byte-for-byte its pinned commit, and reports it as *drifted* if not — a modified
clone means the numbers came from something other than the benchmark it claims
to be.

**Every bench has its own locked environment.** The benches do not agree with
each other about torch, transformers, pydantic or inspect_ai, and resolving them
together either fails or silently degrades whichever one loses the tie-break.
Each therefore gets a virtualenv of its own under `.envs/`, built from a
committed, fully-hashed lock in [`envs/`](envs/). The repository's own `.venv`
stays out of it: `uv sync` at the root pulls no benchmark dependency, and a
contributor who never runs an eval never pays for one.

An environment that resolves fresh on each machine is not an eval environment —
two people get different transitive versions and their numbers quietly stop
being comparable. Changing what an environment resolves to is therefore a
deliberate act with a reviewable diff:

```bash
./evals/Literature/setup.sh --relock labbench   # envs/labbench.in -> .lock
```

**Environments build on demand.** Installing every bench up front costs several
GB for someone who wants one row of one table, so the first run of a bench
builds its environment and stamps it with a hash of the lock, the pinned commit
and the Python version. A mismatch rebuilds — which is what stops a stale
environment surviving a lock bump. `setup.sh --bench <name>` does the same thing
eagerly. `--dry-run` deliberately builds nothing.

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
declares more than one when its parts run on different machines — generating
answers on a laptop and scoring them on a GPU box — because forcing a CUDA
scoring stack into the generation environment would stop the generation half
installing at all. An `.in` file can say how it must be resolved:

```
# uv-compile-args: --python-platform x86_64-unknown-linux-gnu
```

which is how a lock for a GPU box stays correct when it is regenerated on a
laptop. An environment listed in `envs_optional` may fail to build during eager
setup without failing the setup — a laptop can still prepare a bench whose
scoring half is CUDA-only. A *run* that reaches that half and cannot build it
still fails, which is the right moment to find out.

`envs/_librarian.in` is the shared base every bench includes — the librarian's
own runtime dependencies, since every bench runs the agent in-process.

The repository is not a package (`[tool.uv] package = false`), so its code
reaches a bench environment on `PYTHONPATH` rather than through an install. Only
its dependencies are in the lock, and `envs/_librarian.in` has to be kept in
step with `pyproject.toml` by hand.

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
from*, not one the runners share — changing it rebuilds all of them, because
wheels are not portable across a Python minor version.

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

## Reference results, and what a delta means

The paper's numbers are GLM-5 (`zai-org/GLM-5-FP8`, served as `glm-5-fp8`) with
the retrieval knobs now in `librarian/config.toml`. These scripts pass no knob
overrides, so a rerun measures the agent as currently configured: expect close,
not identical. A different librarian model makes a result **incomparable rather
than wrong** — say which you used. A delta is not automatically a bug; report it
rather than tuning the scripts until it goes away.

## Layout

```
evals/Literature/
  bench_env.sh          per-bench locked environments, built on demand
  setup.sh              clone pinned upstreams, prepare data, build environments
  literature_eval.sh    one entry point; dispatches via each bench.manifest
  literature_eval.toml  every endpoint, model and path
  load_config.py        the TOML -> environment-variable mapping
  llm_compat.py         one LLM client surface across the benches
  orchestrator_client.py  the --via-api transport
  envs/                 _librarian.in + one .in/.lock pair per bench
  .envs/                built virtualenvs (git-ignored, disposable)
  LabBench/             the bench itself, plus its bench.manifest
```

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

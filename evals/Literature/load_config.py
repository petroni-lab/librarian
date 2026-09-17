"""Turn ``literature_eval.toml`` into the environment the benchmark runners read.

The runners are shell scripts that read plain environment variables
(``LIBRARIAN_URL``, ``ANSWER_MODEL``, ...). This is the one place that maps the
TOML onto those names, so the config file can be grouped and commented for a
reader while the scripts keep their flat, greppable variables.

Prints ``export`` lines for a shell to evaluate::

    eval "$(python evals/Literature/load_config.py)"

A variable already set and non-empty in the environment is left alone, so a
real environment variable always beats the file — that is what makes

    LIBRARIAN_URL=http://myhost:8000/v1 ./evals/Literature/literature_eval.sh ...

work without editing anything.

``LITERATURE_EVAL_CONFIG`` selects a different file, for a personal profile
kept outside the repository.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10, via mashumaro[toml]
    import tomli as tomllib  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "literature_eval.toml"

#: ``(toml path, env var, env var to inherit from when blank)``.
#: The inherit column is why ``sqa.synthesis_model = ""`` means "same model as
#: the librarian" rather than "empty".
SPEC: list[tuple[str, str, str | None]] = [
    ("librarian.url", "LIBRARIAN_URL", None),
    ("librarian.model", "LIBRARIAN_MODEL", None),
    ("litqa2.answer_model", "ANSWER_MODEL", None),
    ("litqa2.answer_url", "ANSWER_URL", None),
    ("litqa2.judge_model", "LITQA2_JUDGE_MODEL", None),
    ("sqa.synthesis_model", "SYNTHESIS_MODEL", "LIBRARIAN_MODEL"),
    ("sqa.judge_gpus", "JUDGE_GPUS", None),
    ("sqa.apptainer_image", "APPTAINER_IMAGE", None),
    ("proclaim.verdict_model", "PROCLAIM_VERDICT_MODEL", None),
    ("proclaim.verdict_url", "PROCLAIM_VERDICT_URL", None),
    ("proclaim.librarian_url", "PROCLAIM_LIBRARIAN_URL", "LIBRARIAN_URL"),
    ("proclaim.librarian_model", "PROCLAIM_LIBRARIAN_MODEL", "LIBRARIAN_MODEL"),
    ("proclaim.subagent_url", "PROCLAIM_SUBAGENT_URL", None),
    ("proclaim.subagent_model", "PROCLAIM_SUBAGENT_MODEL", None),
    ("api.url", "LIBRARIAN_API_URL", None),
    ("api.model", "LIBRARIAN_API_MODEL", "LIBRARIAN_MODEL"),
    ("paths.results_root", "RESULTS_ROOT", None),
    ("paths.hf_home", "HF_HOME", None),
    ("paths.python", "PYTHON", None),
    ("concurrency.max_samples", "MAX_SAMPLES", None),
    ("concurrency.max_connections", "MAX_CONNECTIONS", None),
]


def _dig(data: dict[str, Any], dotted: str) -> Any:
    """Return ``data["a"]["b"]`` for ``"a.b"``, or ``None`` if absent."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve(config_path: Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """Map the TOML onto environment variables.

    :param config_path: The TOML file to read.
    :param env: The environment to treat as already-set; defaults to the real one.
    :returns: Only the variables that need exporting, in dependency order.
    :raises FileNotFoundError: if ``config_path`` does not exist.
    """
    env = dict(os.environ if env is None else env)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}

    for dotted, var, inherit in SPEC:
        if env.get(var):  # a real environment variable wins
            continue
        value = _dig(data, dotted)
        text = "" if value is None else str(value)
        if not text and inherit:
            text = env.get(inherit, "") or out.get(inherit, "")
        if not text:
            continue
        out[var] = text
        env[var] = text

    # Two paths have a computed default rather than a literal one.
    if not env.get("RESULTS_ROOT"):
        out["RESULTS_ROOT"] = str(HERE / "results")
    if not env.get("HF_HOME"):
        out["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    return out


def main() -> int:
    path = Path(os.environ.get("LITERATURE_EVAL_CONFIG") or DEFAULT_CONFIG)
    if not path.is_file():
        print(f"ERROR: no config at {path}", file=sys.stderr)
        return 1
    try:
        resolved = resolve(path)
    except Exception as exc:  # a broken TOML should name itself, not traceback
        print(f"ERROR: could not read {path}: {exc}", file=sys.stderr)
        return 1
    for var, value in resolved.items():
        print(f"export {var}={shlex.quote(value)}")
    return 0


def _self_check() -> None:
    """Assert the mapping's three rules without touching the real environment."""
    import tempfile

    toml = """
[librarian]
url = "http://x:8000/v1"
model = "m1"
[sqa]
synthesis_model = ""
[proclaim]
librarian_url = "http://other:9000/v1"
[concurrency]
max_samples = 16
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.toml"
        p.write_text(toml)

        got = resolve(p, env={})
        assert got["LIBRARIAN_URL"] == "http://x:8000/v1", got
        # blank + inherit -> the inherited value, not empty
        assert got["SYNTHESIS_MODEL"] == "m1", got
        # a set value is not overridden by its inherit source
        assert got["PROCLAIM_LIBRARIAN_URL"] == "http://other:9000/v1", got
        # numbers become strings, as the shell needs
        assert got["MAX_SAMPLES"] == "16", got
        # computed defaults appear
        assert got["RESULTS_ROOT"].endswith("/results"), got

        # a real environment variable wins, and inheritance follows it
        got = resolve(p, env={"LIBRARIAN_MODEL": "override"})
        assert "LIBRARIAN_MODEL" not in got, got
        assert got["SYNTHESIS_MODEL"] == "override", got

        # values with spaces survive the round trip through the shell
        p.write_text('[paths]\nresults_root = "/tmp/a b"\n')
        line = f"export RESULTS_ROOT={shlex.quote(resolve(p, env={})['RESULTS_ROOT'])}"
        assert line == "export RESULTS_ROOT='/tmp/a b'", line
    print("load_config self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())

#!/usr/bin/env bash
# Run a Librarian script with the project's Python environment.
#
#   run.sh search --provider claude "<question>"
#
# The project root is derived from this script's location, so the skill works
# from a plugin cache, a clone, or a vendored copy — no cwd or git root needed.
# LIBRARIAN_PYTHON overrides the interpreter; otherwise uv provisions the
# environment on first call, so a fresh install needs no separate setup step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/../../.."
step="${1:?usage: run.sh search --provider <claude|codex|antigravity> \"<question>\"}"
shift
# pyproject sets [tool.uv] package = false, so neither path below installs the
# librarian package itself; the scripts import it from the checkout.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${LIBRARIAN_PYTHON:-}" ]; then
    exec "$LIBRARIAN_PYTHON" "$HERE/$step.py" "$@"
fi
# The uv installer writes to ~/.local/bin, which a host that was already running
# (a desktop app, an agent session) may not have on PATH yet.
PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin"
command -v uv >/dev/null || {
    echo "run.sh: need uv (https://astral.sh/uv) or LIBRARIAN_PYTHON set to a Python with Librarian's dependencies." >&2
    exit 127
}
exec uv run --project "$ROOT" python "$HERE/$step.py" "$@"

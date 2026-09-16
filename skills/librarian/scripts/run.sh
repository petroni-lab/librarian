#!/usr/bin/env bash
# Run one Librarian pipeline step with the project's Python environment.
#
#   run.sh step1_query_prompt --provider claude "<question>"
#   run.sh step2_retrieve --run "<RUN_DIR>"
#
# The project root is derived from this script's location, so the skill works
# from a plugin cache, a clone, or a vendored copy — no cwd or git root needed.
# LIBRARIAN_PYTHON overrides the interpreter; otherwise uv provisions the
# environment on first call, so a fresh install needs no separate setup step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
step="${1:?usage: run.sh <step_name> [args...]}"
shift

if [ -n "${LIBRARIAN_PYTHON:-}" ]; then
    exec "$LIBRARIAN_PYTHON" "$HERE/$step.py" "$@"
fi
command -v uv >/dev/null || {
    echo "run.sh: need uv (https://astral.sh/uv) or LIBRARIAN_PYTHON set to a Python with Librarian's dependencies." >&2
    exit 127
}
exec uv run --project "$HERE/../../.." python "$HERE/$step.py" "$@"

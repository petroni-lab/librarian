#!/usr/bin/env bash
# post_install.sh — install the pristine AstaBench clone into this bench's env.
#
# Run by ../bench_env.sh after the lock has been installed, with
# BENCH_ENV_PYTHON and BENCH_CLONE_DIR set.
#
# WHY INSTALL RATHER THAN PUT IT ON sys.path
#
# Upstream's astabench/__init__.py calls get_version("astabench"), which raises
# PackageNotFoundError when there is no install metadata, and eagerly imports
# the whole eval suite. Reaching the clone through sys.path therefore means
# editing two of its __init__.py files — which is exactly what this layout
# exists to avoid. Installing it costs a heavier environment and keeps the clone
# untouched.
#
# --no-deps on purpose: ../envs/litqa2.lock transcribes AstaBench's own
# dependency list at the pinned commit and is the single authority on what is
# in here. Letting pip resolve them again would silently float versions the
# lock pins.
set -euo pipefail

PY="${BENCH_ENV_PYTHON:?post_install.sh needs BENCH_ENV_PYTHON}"
CLONE="${BENCH_CLONE_DIR:?post_install.sh needs BENCH_CLONE_DIR}"

[ -d "$CLONE" ] || { echo "ERROR: no AstaBench clone at $CLONE" >&2; exit 1; }

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" --no-deps --editable "$CLONE"
else
    "$PY" -m pip install --quiet --no-deps --editable "$CLONE"
fi

# An editable install that cannot be imported is the failure this whole step
# exists to prevent; say so here rather than at the first eval.
"$PY" - <<'EOF'
import astabench
print(f"astabench {astabench.__version__} installed from the clone")
EOF

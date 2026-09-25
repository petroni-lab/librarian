#!/usr/bin/env bash
# post_fetch.sh — after cloning ProClaim, check that it still fits backend/.
#
# Run by ../setup.sh as this bench's post_fetch hook. `--check` behaves the
# same; the check is read-only either way.
#
# backend/ stands beside ProClaim rather than inside it, so the join is made of
# imports and attribute names rather than of a patch that would refuse to apply.
# check_seam.py is what replaces that refusal, and it parses rather than imports,
# so it runs here — right after the clone, on any machine, without ProClaim's
# dependency stack and long before the pipeline environment exists.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${BENCH_BASE_PYTHON:-python3}"

"$PY" "$BENCH_DIR/check_seam.py" --clone "$BENCH_DIR/ProClaim_src"

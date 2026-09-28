#!/usr/bin/env bash
# post_install.sh — fetch the NLTK data the AutoAIS scorer needs.
#
# Run by ../bench_env.sh after an environment is installed, with BENCH_ENV_NAME,
# BENCH_ENV_PYTHON and BENCH_ENV_DIR set.
#
# `nltk` is in the lock, but its tokenizer data is not: it is downloaded at run
# time, and without it citation_correctness_eval.py raises LookupError partway
# through scoring, after the answers have been generated. The data goes inside
# the environment, which NLTK already searches, so it is stamped and removed
# with it and needs no NLTK_DATA.
#
# Only the scoring environment imports the scorer; `sqa` just generates answers.
set -euo pipefail

PY="${BENCH_ENV_PYTHON:?post_install.sh needs BENCH_ENV_PYTHON}"
ENV_DIR="${BENCH_ENV_DIR:?post_install.sh needs BENCH_ENV_DIR}"

[ "${BENCH_ENV_NAME:-}" = "sqa-scoring" ] || exit 0

echo "fetching NLTK punkt_tab into ${ENV_DIR##*/}/nltk_data"
# Via -c rather than `-m nltk.downloader`, which warns about its own import.
"$PY" -c "import nltk, sys; sys.exit(0 if nltk.download('punkt_tab', download_dir='$ENV_DIR/nltk_data', quiet=True) else 1)"

# A download that lands somewhere NLTK does not search is the failure this step
# exists to prevent; prove it resolves from inside the environment.
"$PY" - <<'PYEOF'
from nltk import sent_tokenize
assert sent_tokenize("One. Two.") == ["One.", "Two."]
print("punkt_tab resolves from the scoring environment")
PYEOF

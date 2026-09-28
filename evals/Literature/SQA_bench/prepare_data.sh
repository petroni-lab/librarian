#!/usr/bin/env bash
# prepare_data.sh — put ScholarQA-Bench's data where the harness reads it.
#
# Run by ../setup.sh as this bench's post_fetch hook, after the clone and before
# the environments. `--check` reports without writing.
#
# Nothing is downloaded: the three gold files are copied out of the clone, and
# the 29-question biomedical subset is rebuilt from the committed id list.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA="$BENCH_DIR/data"
CLONE="$BENCH_DIR/code"

CHECK_ONLY=false
[ "${1:-}" = "--check" ] && CHECK_ONLY=true

# Any Python will do: the reconstruction script is standard library only, and
# runs before the bench environments exist.
PY="${BENCH_BASE_PYTHON:-python3}"

missing=0
FILES=(
    "scholarqa_bio/scholarqabench_bio.jsonl"
    "scholarqa_neuro/scholarqabench_neuro.jsonl"
    "scholarqa_multi/human_answers.json"
)
for rel in "${FILES[@]}"; do
    dest="$DATA/$rel"
    if [ -s "$dest" ]; then
        echo "OK      data/$rel"
        continue
    fi
    if [ "$CHECK_ONLY" = true ]; then
        echo "MISSING data/$rel"
        missing=$((missing + 1))
        continue
    fi
    src="$CLONE/data/$rel"
    [ -s "$src" ] || { echo "ERROR: $src not in the ScholarQABench clone" >&2; exit 1; }
    echo "COPY    data/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -p "$src" "$dest"
done

SUBSET="$DATA/scholarqa_multi/scholar_multi_biomed_eval.json"
if [ -s "$SUBSET" ]; then
    echo "OK      data/scholarqa_multi/scholar_multi_biomed_eval.json"
elif [ "$CHECK_ONLY" = true ]; then
    echo "MISSING data/scholarqa_multi/scholar_multi_biomed_eval.json"
    missing=$((missing + 1))
else
    echo "BUILD   data/scholarqa_multi/scholar_multi_biomed_eval.json"
    "$PY" "$BENCH_DIR/reconstruct_scholar_multi_biomed_from_ids.py"
fi

[ "$missing" -eq 0 ]

#!/usr/bin/env bash
# setup.sh — fetch the third-party benchmark code and data, then apply our changes.
#
#   ./evals/Literature/setup.sh [--bench litqa2|labbench|proclaim|sqa|all[,...]] [--check]
#
# Three of the four suites are built on someone else's benchmark repository.
# This repository carries no copy of any of them: it carries only the files we
# changed or added, under each suite's `overlay/`. This script clones the
# upstream repository at the exact commit the paper's numbers were produced
# against, then copies our overlay over it. Nothing it writes is committed.
#
#   --bench   fetch only what those benches need (default: all). Running one
#             bench does not require the others' data.
#   --check   report what is present, missing or drifted, and exit.
#
# WHAT EACH BENCH NEEDS
#
#   litqa2    AstaBench/vendor/ — allenai/asta-bench (Apache-2.0).
#             8 files replaced from AstaBench/overlay/. LitQA2 is the only
#             AstaBench task run here; its other benchmarks are untouched.
#   sqa       SQA_bench/code/ — AkariAsai/ScholarQABench (MIT), the AutoAIS and
#             Prometheus scorers. 4 files replaced from SQA_bench/overlay/.
#             Plus the bio/neuro/multi data, copied out of that clone, and the
#             29-question biomedical subset rebuilt from the committed id list.
#   proclaim  ProClaim/ProClaim_src/ — saezlab/ProClaim (GPL-3.0), the
#             claim-verification pipeline. 10 files from ProClaim/overlay/ add
#             the librarian retrieval backend, which exists nowhere upstream.
#             Its two claim sets arrive with the clone, in ProClaim_src/datasets/,
#             and are read from there.
#   labbench  nothing — it streams its dataset from the Hub at run time.
#
# Each repository is pinned by commit, not by branch: the overlay was written
# against that tree, so a moving `main` would eventually mix our files with an
# upstream they no longer match.
#
# WHAT IT NEVER DOWNLOADS
#
#   LitQA2 — `run_litqa2.sh` builds its 91-question europepmc_fulltext split on
#   first use from `futurehouse/lab-bench` on the Hugging Face Hub (CC-BY-SA-4.0).
#   The committed id list under AstaBench/data/ pins exactly which rows that
#   split contains; AstaBench/reconstruct_litqa2_from_ids.py rebuilds it.
#
#   LAB-Bench — streamed from the same Hub dataset at run time.
#
# The ProClaim arm needs its own environment, because its dependencies are not
# the harness's: `cd ProClaim/ProClaim_src && uv sync`. `--only verifier` does
# not.
#
# The clone directories are git-ignored and treated as disposable: a re-run
# resets them to the pinned commit and re-applies the overlay, so local edits
# inside them do not survive. Edit `overlay/` instead — that is what ships.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `uv sync` creates .venv but does not put it on PATH, and plenty of systems
# have no bare `python` at all -- so resolve an interpreter that exists rather
# than dying at the one step that needs one.
resolve_python() {
    if [ -n "${PYTHON:-}" ]; then printf '%s' "$PYTHON"; return; fi
    if [ -x "$SCRIPT_DIR/../../.venv/bin/python" ]; then
        printf '%s' "$SCRIPT_DIR/../../.venv/bin/python"; return
    fi
    if command -v python3 >/dev/null 2>&1; then printf 'python3'; return; fi
    printf 'python'
}
PYTHON_BIN="$(resolve_python)"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [ -x "$PYTHON_BIN" ] || {
    echo "ERROR: no Python interpreter found ($PYTHON_BIN). Run \`uv sync --extra evals\`" >&2
    echo "       at the repository root, or set PYTHON=/path/to/python." >&2
    exit 1
}

CHECK_ONLY=false
WANT="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=true ;;
        --bench) WANT="${2:-}"; shift ;;
        --bench=*) WANT="${1#--bench=}" ;;
        -h|--help) sed -n '1,50p' "$0"; exit 0 ;;
        *) echo "usage: $0 [--bench litqa2|labbench|proclaim|sqa|all[,...]] [--check]" >&2; exit 2 ;;
    esac
    shift
done

# Is a bench in the requested set? `all` matches everything.
wanted() {
    [ "$WANT" = "all" ] && return 0
    case ",$WANT," in *",$1,"*) return 0 ;; esac
    return 1
}

for b in ${WANT//,/ }; do
    case "$b" in
        litqa2|labbench|proclaim|sqa|all) ;;
        *) echo "ERROR: unknown bench '$b'" >&2; exit 2 ;;
    esac
done

# bench <TAB> name <TAB> url <TAB> pinned commit <TAB> clone dir <TAB> overlay dir
# `labbench` needs nothing: it streams its dataset from the Hub at run time.
REPOS=(
"litqa2	asta-bench	https://github.com/allenai/asta-bench.git	a9e338070ffff195cb4cc5ddcbcf9ca8805f141d	AstaBench/vendor	AstaBench/overlay"
"sqa	ScholarQABench	https://github.com/AkariAsai/ScholarQABench.git	95e6fc52b0a8a0ce0a74956029991e3bb00c38b9	SQA_bench/code	SQA_bench/overlay"
"proclaim	ProClaim	https://github.com/saezlab/ProClaim.git	6316258ba31386657389fea5bb2db2c97362e06c	ProClaim/ProClaim_src	ProClaim/overlay"
)

missing=0

# Files that document the overlay rather than belonging to the clone.
overlay_is_metadata() {
    case "$1" in LICENSE|NOTICE) return 0 ;; *) return 1 ;; esac
}

fetch_repo() {
    local name="$1" url="$2" sha="$3" dest="$4"

    if [ -d "$dest/.git" ]; then
        local have
        have="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo unknown)"
        if [ "$have" = "$sha" ]; then
            echo "OK      $name @ ${sha:0:12}"
            return 0
        fi
        echo "RESET   $name ${have:0:12} -> ${sha:0:12}"
    else
        # An empty directory is a leftover (rm -rf on NFS often leaves them);
        # anything else there is someone's, and not ours to overwrite.
        if [ -e "$dest" ]; then
            rmdir "$dest" 2>/dev/null || {
                echo "ERROR: $dest exists but is not a git clone" >&2; exit 1; }
        fi
        echo "CLONE   $name @ ${sha:0:12}"
        mkdir -p "$dest"
        git -C "$dest" init --quiet
        git -C "$dest" remote add origin "$url"
    fi

    # Ask for the one commit rather than the history; fall back to a full fetch
    # for a server that will not serve an arbitrary sha.
    if ! git -C "$dest" fetch --quiet --depth 1 origin "$sha" 2>/dev/null; then
        git -C "$dest" fetch --quiet origin || {
            echo "ERROR: could not fetch $url" >&2; exit 1; }
    fi
    git -C "$dest" reset --quiet --hard "$sha" || {
        echo "ERROR: $url has no commit $sha" >&2; exit 1; }
    git -C "$dest" clean -qfd
}

apply_overlay() {
    local ov="$1" dest="$2" n=0
    while IFS= read -r rel; do
        overlay_is_metadata "$rel" && continue
        mkdir -p "$dest/$(dirname "$rel")"
        cp -p "$ov/$rel" "$dest/$rel"
        n=$((n + 1))
    done < <(cd "$ov" && find . -type f | sed 's|^\./||' | sort)
    echo "OVERLAY $n file(s) -> ${dest#"$SCRIPT_DIR/"}"
}

check_repo() {
    local name="$1" sha="$3" dest="$4" ov="$5"
    if [ ! -d "$dest/.git" ]; then
        echo "MISSING $name (not cloned)"
        return 1
    fi
    local have
    have="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ "$have" != "$sha" ]; then
        echo "DRIFTED $name at ${have:0:12}, pinned ${sha:0:12}"
        return 1
    fi
    local stale=0 rel
    while IFS= read -r rel; do
        overlay_is_metadata "$rel" && continue
        cmp -s "$ov/$rel" "$dest/$rel" || stale=$((stale + 1))
    done < <(cd "$ov" && find . -type f | sed 's|^\./||' | sort)
    if [ "$stale" -ne 0 ]; then
        echo "DRIFTED $name overlay: $stale file(s) differ from overlay/"
        return 1
    fi
    echo "OK      $name @ ${sha:0:12}, overlay applied"
}

for entry in "${REPOS[@]}"; do
    IFS=$'\t' read -r bench name url sha rel_dest rel_ov <<<"$entry"
    wanted "$bench" || continue
    dest="$SCRIPT_DIR/$rel_dest"
    ov="$SCRIPT_DIR/$rel_ov"
    [ -d "$ov" ] || { echo "ERROR: missing overlay $rel_ov" >&2; exit 1; }
    if [ "$CHECK_ONLY" = true ]; then
        check_repo "$name" "$url" "$sha" "$dest" "$ov" || missing=$((missing + 1))
    else
        fetch_repo "$name" "$url" "$sha" "$dest"
        apply_overlay "$ov" "$dest"
    fi
done

# ── Data ─────────────────────────────────────────────────────────────────────
# The harness reads these from SQA_bench/data/; the clone carries them under
# SQA_bench/code/data/. Copied rather than symlinked so a re-clone cannot leave
# dangling links behind.
SQA_DATA="$SCRIPT_DIR/SQA_bench/data"
SQA_CLONE="$SCRIPT_DIR/SQA_bench/code"
DATA_FILES=(
    "scholarqa_bio/scholarqabench_bio.jsonl"
    "scholarqa_neuro/scholarqabench_neuro.jsonl"
    "scholarqa_multi/human_answers.json"
)
for rel in $(wanted sqa && printf '%s\n' "${DATA_FILES[@]}"); do
    dest="$SQA_DATA/$rel"
    if [ -s "$dest" ]; then
        echo "OK      data/$rel"
        continue
    fi
    if [ "$CHECK_ONLY" = true ]; then
        echo "MISSING data/$rel"
        missing=$((missing + 1))
        continue
    fi
    src="$SQA_CLONE/data/$rel"
    [ -s "$src" ] || { echo "ERROR: $src not in the ScholarQABench clone" >&2; exit 1; }
    echo "COPY    data/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -p "$src" "$dest"
done

# The biomedical subset is derived, not downloaded: the id list is committed and
# the content comes from the gold-reference file copied above.
SUBSET="$SQA_DATA/scholarqa_multi/scholar_multi_biomed_eval.json"
if ! wanted sqa; then
    :
elif [ -s "$SUBSET" ]; then
    echo "OK      data/scholarqa_multi/scholar_multi_biomed_eval.json"
elif [ "$CHECK_ONLY" = true ]; then
    echo "MISSING data/scholarqa_multi/scholar_multi_biomed_eval.json"
    missing=$((missing + 1))
else
    echo "BUILD   data/scholarqa_multi/scholar_multi_biomed_eval.json"
    "$PYTHON_BIN" "$SCRIPT_DIR/SQA_bench/reconstruct_scholar_multi_biomed_from_ids.py"
fi

if [ "$CHECK_ONLY" = true ]; then
    if [ "$missing" -eq 0 ]; then
        echo "Everything is in place."
    else
        echo "$missing item(s) missing or drifted — run without --check to fix."
        exit 1
    fi
    exit 0
fi

echo
echo "Ready ($WANT). Whatever was fetched sits in a git-ignored clone directory"
echo "and is rebuilt by re-running this script."
echo "LitQA2 and LAB-Bench pull from the Hugging Face Hub at run time."

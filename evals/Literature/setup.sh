#!/usr/bin/env bash
# setup.sh — fetch each bench's upstream code and build its environment.
#
#   ./evals/Literature/setup.sh [--bench <name>[,<name>...]] [--check] [--relock [name]]
#
# This repository contains no copy of, and no patch to, any upstream benchmark.
# Every change we needed lives in our own tree and is applied at run time; the
# clones this script makes are pinned, read-only inputs. `--check` asserts that:
# a clone that differs from its pinned commit in any way is reported as drifted,
# because a modified clone means the numbers came from something other than the
# benchmark it claims to be.
#
#   --bench    act on these benches only (default: all that are present)
#   --check    report what is present, missing or drifted, and exit non-zero if
#              anything needs doing. Never builds, fetches or writes.
#   --relock   regenerate envs/<bench>.lock from envs/<bench>.in and exit. This
#              is the only thing that changes what an environment resolves to,
#              and it produces a reviewable diff.
#
# WHAT A BENCH DECLARES
#
# Each bench directory carries a `bench.manifest`, so this script needs no list
# of its own and a bench can be added or removed without touching it:
#
#   bench=sqa                                  the name used by --bench
#   name=ScholarQABench                        what to call it in output
#   url=https://github.com/...git              omit if the bench clones nothing
#   commit=95e6fc52b0a8...                     pinned; never a branch
#   clone=code                                 directory, relative to the bench
#   post_fetch=prepare_data.sh                 optional, runs after the clone
#   post_install=post_install.sh               optional, runs after the env build
#   envs=sqa,sqa-scoring                       optional; defaults to the bench name
#   envs_optional=sqa-scoring                  of those, the ones eager setup may
#                                              skip when they will not build here
#
# Each environment is envs/<name>.lock, compiled from envs/<name>.in. A bench
# declares more than one when its parts run on different machines — generating
# answers on a laptop and scoring them on a GPU box.
#
# A repository is pinned by commit rather than by branch because the run-time
# patches are written against one tree, and a moving `main` would eventually
# meet code they no longer fit.
#
# The clone directories are git-ignored and disposable: a re-run resets them to
# the pinned commit, so edits inside them do not survive. There is deliberately
# nowhere in this repository to make one — our code sits beside the clone and
# patches it in memory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIT_ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=evals/Literature/bench_env.sh
. "$SCRIPT_DIR/bench_env.sh"

CHECK_ONLY=false
RELOCK=false
RELOCK_WHAT=""
WANT="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=true ;;
        --bench) WANT="${2:-}"; shift ;;
        --bench=*) WANT="${1#--bench=}" ;;
        --relock)
            RELOCK=true
            case "${2:-}" in -*|"") ;; *) RELOCK_WHAT="$2"; shift ;; esac
            ;;
        --relock=*) RELOCK=true; RELOCK_WHAT="${1#--relock=}" ;;
        -h|--help) sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "usage: $0 [--bench <name>[,...]] [--check] [--relock [name]]" >&2; exit 2 ;;
    esac
    shift
done

# ── Discovery ────────────────────────────────────────────────────────────────

# Every bench that has contributed a manifest, in directory order.
all_benches() {
    local m b
    for m in "$LIT_ROOT"/*/bench.manifest; do
        [ -f "$m" ] || continue
        b="$(sed -n 's/^bench=//p' "$m" | head -1)"
        [ -n "$b" ] && printf '%s\n' "$b"
    done
}

wanted() {
    [ "$WANT" = "all" ] && return 0
    case ",$WANT," in *",$1,"*) return 0 ;; esac
    return 1
}

KNOWN="$(all_benches | tr '\n' ' ')"
if [ "$WANT" != "all" ]; then
    for b in ${WANT//,/ }; do
        case " $KNOWN " in
            *" $b "*) ;;
            *) echo "ERROR: unknown bench '$b' (have: ${KNOWN% })" >&2; exit 2 ;;
        esac
    done
fi

# ── --relock ─────────────────────────────────────────────────────────────────
# Regenerating a lock is not part of setting up: it changes what everyone
# resolves to, so it is an explicit request with its own exit.

relock_one() {
    local bench="$1" spec lock
    spec="$(bench_spec_file "$bench")"
    lock="$(bench_lock_file "$bench")"
    [ -f "$spec" ] || { echo "ERROR: no spec at $spec" >&2; return 1; }
    command -v uv >/dev/null 2>&1 || { echo "ERROR: --relock needs uv" >&2; return 1; }
    echo "RELOCK  $bench: $(basename "$spec") -> $(basename "$lock")"
    # --generate-hashes so a lock pins content, not just a version number; a
    # yanked-and-replaced release should fail the install, not change it.
    # shellcheck disable=SC2046
    uv pip compile --quiet --generate-hashes \
        --python-version "$(_bench_base_python_version)" \
        $(bench_compile_args "$bench") \
        --output-file "$lock" "$spec"
}

if [ "$RELOCK" = true ]; then
    if [ -n "$RELOCK_WHAT" ]; then
        relock_one "$RELOCK_WHAT"
    else
        for b in $(all_benches); do
            for e in $(bench_envs "$b"); do relock_one "$e"; done
        done
    fi
    echo
    echo "Locks regenerated. Review the diff, then re-run setup.sh to rebuild."
    exit 0
fi

# ── Clones ───────────────────────────────────────────────────────────────────

missing=0

fetch_repo() {
    local name="$1" url="$2" sha="$3" dest="$4"

    if [ -d "$dest/.git" ]; then
        local have
        have="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo unknown)"
        if [ "$have" = "$sha" ] && git -C "$dest" diff --quiet 2>/dev/null; then
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

# The invariant this whole layout exists to make checkable: the clone is
# byte-for-byte the pinned upstream commit. Nothing we ship is inside it.
check_repo() {
    local name="$1" sha="$2" dest="$3"
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
    if ! git -C "$dest" diff --quiet 2>/dev/null || [ -n "$(git -C "$dest" status --porcelain 2>/dev/null)" ]; then
        echo "DRIFTED $name @ ${sha:0:12} has local modifications; it must be pristine"
        return 1
    fi
    echo "OK      $name @ ${sha:0:12}, pristine"
}

for bench in $(all_benches); do
    wanted "$bench" || continue
    manifest="$(bench_manifest "$bench")"
    bench_dir="$(dirname "$manifest")"
    name="$(bench_manifest_get "$manifest" name)"
    url="$(bench_manifest_get "$manifest" url)"
    sha="$(bench_manifest_get "$manifest" commit)"
    clone_rel="$(bench_manifest_get "$manifest" clone)"

    if [ -n "$url" ]; then
        [ -n "$sha" ] || { echo "ERROR: $manifest has a url but no commit" >&2; exit 1; }
        dest="$bench_dir/$clone_rel"
        if [ "$CHECK_ONLY" = true ]; then
            check_repo "$name" "$sha" "$dest" || missing=$((missing + 1))
        else
            fetch_repo "$name" "$url" "$sha" "$dest"
        fi
    fi

    # Data preparation, if the bench has any: rebuilding a subset from a
    # committed id list, copying gold files out of the clone, and so on. Never
    # a download of anything we are not allowed to redistribute.
    post_fetch="$(bench_manifest_get "$manifest" post_fetch)"
    if [ -n "$post_fetch" ]; then
        if [ "$CHECK_ONLY" = true ]; then
            BENCH_DIR="$bench_dir" LIT_ROOT="$LIT_ROOT" REPO_ROOT="$REPO_ROOT" \
                bash "$bench_dir/$post_fetch" --check || missing=$((missing + 1))
        else
            BENCH_DIR="$bench_dir" LIT_ROOT="$LIT_ROOT" REPO_ROOT="$REPO_ROOT" \
                bash "$bench_dir/$post_fetch"
        fi
    fi

    # The environment. Eager here, lazy at run time — same builder either way,
    # so a warm setup and a cold first run cannot end up with different trees.
    for env_name in $(bench_envs "$bench"); do
        optional=false
        bench_env_is_optional "$bench" "$env_name" && optional=true
        if [ "$CHECK_ONLY" = true ]; then
            if bench_env_check "$env_name"; then
                :
            elif [ "$optional" = true ]; then
                echo "        (optional here; built on demand by a run that needs it)"
            else
                missing=$((missing + 1))
            fi
        elif [ "$optional" = true ]; then
            # An optional environment that will not build on this machine is not
            # a failed setup: it is a machine that does not run that half.
            bench_python "$env_name" >/dev/null || {
                echo "SKIP    $env_name environment does not build here."
                echo "        This machine cannot host it — see envs/$env_name.in for what"
                echo "        it needs. The rest of $bench is set up, and a run that"
                echo "        reaches this half will try again and fail there instead."
            }
        else
            bench_python "$env_name" >/dev/null
        fi
    done
done

if [ "$CHECK_ONLY" = true ]; then
    if [ "$missing" -eq 0 ]; then
        echo "Everything is in place."
        exit 0
    fi
    echo "$missing item(s) missing or drifted — run without --check to fix."
    exit 1
fi

echo
echo "Ready ($WANT). The clones are pinned and pristine; the environments are"
echo "built from committed locks under evals/Literature/envs/."
echo "Both are git-ignored and rebuilt by re-running this script."

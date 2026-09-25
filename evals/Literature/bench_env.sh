#!/usr/bin/env bash
# bench_env.sh — one locked virtualenv per bench, built the first time it is needed.
#
# Source this; do not execute it. It gives the run scripts three functions:
#
#     bench_python <bench>            print that bench's interpreter, building
#                                     the environment first if it is missing or
#                                     stale
#     bench_run    <bench> <cmd...>   run a command under that interpreter, with
#                                     PYTHONPATH pointing at the repository root
#     bench_env_check <bench>         report the environment's state and return
#                                     non-zero if it would need building
#
# WHY A SEPARATE ENVIRONMENT PER BENCH
#
# The benches do not agree with each other. ScholarQA-Bench's scorers want a
# CUDA torch and a pinned transformers; AstaBench wants a particular inspect_ai;
# ProClaim wants its own pydantic-settings stack. Resolving all of that into one
# lock either fails or silently degrades whichever bench loses the tie-break.
# Resolving each separately costs disk and nothing else.
#
# The repository's own .venv stays out of it entirely, so `uv sync` at the root
# never pulls a benchmark dependency and a contributor who never runs an eval
# never pays for one.
#
# WHY IT IS LOCKED
#
# An environment that resolves fresh on each machine is not an eval environment:
# two people get different transitive versions and their numbers stop being
# comparable, silently. Each bench therefore commits `envs/<bench>.lock`, a
# fully pinned `uv pip compile` output, and the environment is built by
# `uv pip sync` against exactly that. `setup.sh --relock` regenerates it, which
# is a deliberate act with a reviewable diff.
#
# WHY IT IS BUILT ON DEMAND
#
# Cloning and installing every bench up front costs several GB and many minutes
# for someone who wants one row of one table. So the build is lazy: the first
# run of a bench builds its environment, prints what it is doing, and stamps it.
# `setup.sh --bench <name>` does the same thing eagerly, for anyone who would
# rather pay the cost at a moment they chose.
#
# THE STAMP
#
# `.envs/<bench>/.stamp` holds a hash of everything the environment is derived
# from: the lock file, the pinned upstream commit if the bench has a clone, and
# the Python version. A mismatch rebuilds. This is what stops a stale
# environment surviving a lock bump, which is the failure that would quietly
# change results.

# Resolved relative to this file, so a run script can live anywhere below it.
LIT_ROOT="${LIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$LIT_ROOT/../.." && pwd)}"
# Overridable so a cluster can put the environments on fast local disk rather
# than the shared filesystem the checkout lives on.
BENCH_ENV_ROOT="${BENCH_ENV_ROOT:-$LIT_ROOT/.envs}"

# ── Small helpers ────────────────────────────────────────────────────────────

_bench_die() { echo "ERROR: $*" >&2; return 1; }

# sha256 of stdin. macOS ships shasum, most Linux images ship sha256sum, and a
# container with neither still has Python.
_bench_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | cut -d' ' -f1
    else
        "${PYTHON:-python3}" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
    fi
}

bench_env_dir()  { printf '%s/%s' "$BENCH_ENV_ROOT" "$1"; }
# Extra `uv pip compile` arguments an environment needs, taken from a magic
# comment in its .in file:
#
#     # uv-compile-args: --python-platform x86_64-unknown-linux-gnu
#
# A lock for a CUDA scoring stack has to be resolved for the machine that will
# run it, which is not the machine someone is likely to relock on.
bench_compile_args() {
    sed -n 's/^# *uv-compile-args: *//p' "$(bench_spec_file "$1")" 2>/dev/null | head -1
}
bench_lock_file() { printf '%s/envs/%s.lock' "$LIT_ROOT" "$1"; }
bench_spec_file() { printf '%s/envs/%s.in' "$LIT_ROOT" "$1"; }

# The manifest a bench layer contributes; absent for a bench with no upstream
# repository. Parsed by setup.sh too — see its header for the format.
bench_manifest() {
    local m
    for m in "$LIT_ROOT"/*/bench.manifest; do
        [ -f "$m" ] || continue
        # shellcheck disable=SC2143
        if [ "$(sed -n 's/^bench=//p' "$m" | head -1)" = "$1" ]; then
            printf '%s' "$m"
            return 0
        fi
    done
    return 1
}

bench_manifest_get() {
    local manifest="$1" key="$2"
    [ -f "$manifest" ] || return 1
    sed -n "s/^${key}=//p" "$manifest" | head -1
}

# Every environment a bench owns, in build order; the bench's own name when it
# declares none. A bench needs more than one when parts of it run on different
# machines: ScholarQA-Bench generates answers anywhere and scores them on a GPU
# box, and forcing a CUDA scoring stack into the generation environment would
# stop the generation half installing at all on a laptop.
bench_envs() {
    local manifest declared
    if manifest="$(bench_manifest "$1")"; then
        declared="$(bench_manifest_get "$manifest" envs)"
    fi
    if [ -n "${declared:-}" ]; then
        printf '%s\n' "${declared//,/ }" | tr ' ' '\n' | sed '/^$/d'
    else
        printf '%s\n' "$1"
    fi
}

# Environments a bench can do without on this machine. ScholarQA-Bench's scoring
# stack is CUDA-only, so a laptop must still be able to set the bench up and
# generate answers; the environment it cannot build is built later, on the box
# that will actually score. Optional applies to *eager* setup only — a run that
# reaches a scoring step and cannot build it still fails, loudly, which is the
# right moment to find out.
bench_optional_envs() {
    local manifest declared
    if manifest="$(bench_manifest "$1")"; then
        declared="$(bench_manifest_get "$manifest" envs_optional)"
    fi
    printf '%s' "${declared:-}"
}

bench_env_is_optional() {
    case ",$(bench_optional_envs "$1")," in *",$2,"*) return 0 ;; esac
    return 1
}

# ── The stamp ────────────────────────────────────────────────────────────────

# Everything the built environment is a function of. Any change here means the
# environment on disk no longer matches what the lock describes.
_bench_stamp_value() {
    local bench="$1" lock manifest commit=""
    lock="$(bench_lock_file "$bench")"
    if manifest="$(bench_manifest "$bench")"; then
        commit="$(bench_manifest_get "$manifest" commit)"
    fi
    {
        printf 'lock\n'
        cat "$lock" 2>/dev/null
        printf 'commit=%s\n' "$commit"
        printf 'python=%s\n' "$(_bench_base_python_version)"
        printf 'compile_args=%s\n' "$(bench_compile_args "$bench")"
        printf 'schema=2\n'
    } | _bench_sha256
}

# The interpreter the virtualenv is created from, as opposed to the one inside
# it. A minor-version bump upstream must invalidate the stamp, because wheels
# are not portable across it.
_bench_base_python() {
    if [ -n "${BENCH_BASE_PYTHON:-}" ]; then printf '%s' "$BENCH_BASE_PYTHON"; return; fi
    if command -v python3 >/dev/null 2>&1; then printf 'python3'; return; fi
    printf 'python'
}

_bench_base_python_version() {
    "$(_bench_base_python)" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown
}

# ── Building ─────────────────────────────────────────────────────────────────

# Two runs of the same bench started together would otherwise build the same
# environment on top of each other. Serialise on a lock directory: mkdir is
# atomic everywhere, including the NFS mounts these often live on, which
# flock(1) is not.
_bench_with_lock() {
    local bench="$1"; shift
    local lockdir="$BENCH_ENV_ROOT/.$bench.building" waited=0
    mkdir -p "$BENCH_ENV_ROOT"
    while ! mkdir "$lockdir" 2>/dev/null; do
        if [ "$waited" -eq 0 ]; then
            echo "  waiting for another process to finish building the $bench environment..." >&2
        fi
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge 1800 ]; then
            _bench_die "timed out after 30m waiting for $lockdir; remove it if no build is running"
            return 1
        fi
    done
    # shellcheck disable=SC2064
    trap "rmdir '$lockdir' 2>/dev/null || true" RETURN
    "$@"
}

_bench_build() {
    local bench="$1"
    local lock env_dir stamp
    lock="$(bench_lock_file "$bench")"
    env_dir="$(bench_env_dir "$bench")"

    [ -f "$lock" ] || { _bench_die "no lock file for '$bench' at $lock (run setup.sh --relock $bench)"; return 1; }

    echo "BUILD   $bench environment -> ${env_dir#"$REPO_ROOT/"}" >&2
    echo "        from ${lock#"$REPO_ROOT/"}; this is a one-off and can take several minutes." >&2

    rm -rf "$env_dir"
    mkdir -p "$(dirname "$env_dir")"

    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$(_bench_base_python)" "$env_dir" >&2 \
            || { _bench_die "could not create $env_dir"; return 1; }
        VIRTUAL_ENV="$env_dir" uv pip sync --python "$env_dir/bin/python" "$lock" >&2 \
            || { _bench_die "installing $lock into $env_dir failed"; return 1; }
    else
        echo "        (uv not found; falling back to venv + pip, which is slower)" >&2
        "$(_bench_base_python)" -m venv "$env_dir" >&2 \
            || { _bench_die "could not create $env_dir"; return 1; }
        "$env_dir/bin/python" -m pip install --quiet --upgrade pip >&2
        "$env_dir/bin/python" -m pip install --quiet --no-deps -r "$lock" >&2 \
            || { _bench_die "installing $lock into $env_dir failed"; return 1; }
    fi

    # A bench may need something the lock cannot express — an editable install
    # of a pristine upstream clone, most often. The hook runs inside the built
    # environment, with BENCH_ENV_PYTHON and BENCH_CLONE_DIR set.
    local manifest hook
    if manifest="$(bench_manifest "$bench")"; then
        hook="$(bench_manifest_get "$manifest" post_install)"
        if [ -n "$hook" ]; then
            local bench_dir clone_rel
            bench_dir="$(dirname "$manifest")"
            clone_rel="$(bench_manifest_get "$manifest" clone)"
            echo "        post-install: $hook" >&2
            BENCH_ENV_PYTHON="$env_dir/bin/python" \
            BENCH_CLONE_DIR="${clone_rel:+$bench_dir/$clone_rel}" \
            BENCH_DIR="$bench_dir" \
            REPO_ROOT="$REPO_ROOT" \
                bash "$bench_dir/$hook" >&2 \
                || { _bench_die "post-install hook failed for $bench"; return 1; }
        fi
    fi

    stamp="$(_bench_stamp_value "$bench")"
    printf '%s\n' "$stamp" > "$env_dir/.stamp"
    echo "OK      $bench environment ready" >&2
}

# A build that fails part-way leaves a virtualenv with no stamp, which then
# reports as "no stamp" rather than "not built" — needlessly cryptic for an
# optional environment that simply cannot be built on this machine. Only the
# stamped state is meaningful, so an unstamped directory is swept away.
_bench_build_or_clean() {
    local bench="$1" env_dir
    env_dir="$(bench_env_dir "$bench")"
    if _bench_build "$bench"; then
        return 0
    fi
    [ -f "$env_dir/.stamp" ] || rm -rf "$env_dir"
    return 1
}

# ── The public surface ───────────────────────────────────────────────────────

# Print the reason the environment is not usable as-is, or nothing if it is.
_bench_env_staleness() {
    local bench="$1" env_dir
    env_dir="$(bench_env_dir "$bench")"
    [ -x "$env_dir/bin/python" ] || { echo "not built"; return; }
    [ -f "$env_dir/.stamp" ] || { echo "no stamp"; return; }
    local want have
    want="$(_bench_stamp_value "$bench")"
    have="$(cat "$env_dir/.stamp" 2>/dev/null || true)"
    [ "$want" = "$have" ] || echo "stale (lock, pin or Python version changed)"
}

# Build if needed, then print the interpreter path on stdout. Every diagnostic
# goes to stderr so `$(bench_python sqa)` stays usable.
bench_python() {
    local bench="$1" why env_dir
    env_dir="$(bench_env_dir "$bench")"
    why="$(_bench_env_staleness "$bench")"
    if [ -n "$why" ]; then
        if [ "${BENCH_ENV_NO_BUILD:-false}" = true ]; then
            _bench_die "$bench environment $why, and BENCH_ENV_NO_BUILD is set"
            return 1
        fi
        echo "The $bench environment is $why." >&2
        _bench_with_lock "$bench" _bench_build_or_clean "$bench" || return 1
    fi
    printf '%s/bin/python' "$env_dir"
}

# Like bench_python, but a dry run prints where the interpreter WOULD be without
# building anything. A dry run exists to show the commands that would run;
# spending several GB and several minutes to print a line nobody will execute
# defeats the point, and makes `--dry-run` useless as a cheap wiring check.
bench_python_maybe() {
    local bench="$1" dry="${2:-false}"
    if [ "$dry" = true ]; then
        printf '%s/bin/python' "$(bench_env_dir "$bench")"
        return 0
    fi
    bench_python "$bench"
}

# Run a command in the bench's environment. PYTHONPATH carries the repository
# itself, which is not a package (`[tool.uv] package = false`) and so cannot be
# installed into the environment the ordinary way.
bench_run() {
    local bench="$1"; shift
    local py
    py="$(bench_python "$bench")" || return 1
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$py" "$@"
}

# For `setup.sh --check`: say where the environment stands, and return non-zero
# if it is not ready. Never builds.
bench_env_check() {
    local bench="$1" why
    why="$(_bench_env_staleness "$bench")"
    if [ -z "$why" ]; then
        echo "OK      $bench environment"
        return 0
    fi
    echo "MISSING $bench environment ($why)"
    return 1
}

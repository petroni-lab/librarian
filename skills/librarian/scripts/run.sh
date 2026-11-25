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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$HERE/../../.."
step="${1:?usage: run.sh search --provider <claude|codex|antigravity> \"<question>\"}"
shift
# pyproject sets [tool.uv] package = false, so neither path below installs the
# librarian package itself; the scripts import it from the checkout.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

if [ -n "${LIBRARIAN_PYTHON:-}" ]; then
    exec "$LIBRARIAN_PYTHON" "$HERE/$step.py" "$@"
fi
# The uv installer writes to ~/.local/bin, which a host that was already running
# (a desktop app, an agent session) may not have on PATH yet.
PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin"
uv="$(command -v uv || true)"
if [ -z "$uv" ]; then
    uv_cache="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
    uv_bin_dir="$(mkdir -p "$uv_cache/librarian-bin" && cd "$uv_cache/librarian-bin" && pwd -P)"
    uv="$uv_bin_dir/uv"
    if [ ! -x "$uv" ]; then
        echo "[Librarian] Python setup: installing uv from astral.sh." >&2
        (
            installer="$(mktemp)"
            trap 'rm -f "$installer"' EXIT
            curl -fsSL --max-time 60 https://astral.sh/uv/0.12.0/install.sh -o "$installer"
            UV_UNMANAGED_INSTALL="$uv_bin_dir" sh "$installer" >&2
        )
    fi
fi
# The plugin checkout can be read-only. Keep environments with matching locks
# together in uv's writable cache, separate from the checkout's dev environment.
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    cache_dir="$("$uv" cache dir)"
    lock_checksum="$(cksum < "$ROOT/uv.lock")"
    export UV_PROJECT_ENVIRONMENT="$cache_dir/librarian/${lock_checksum// /-}"
fi
if "$uv" sync --project "$ROOT" --locked --no-dev; then
    exec "$uv" run --project "$ROOT" --no-sync python "$HERE/$step.py" "$@"
else
    setup_code=$?
    echo "[Librarian] Python setup failed. Use the uv error above and follow SKILL.md setup recovery." >&2
    exit "$setup_code"
fi

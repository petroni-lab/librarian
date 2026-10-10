---
name: librarian
description: Retrieve Europe PMC evidence for a biomedical research question. Use when the user needs papers, citations, or literature-backed scientific evidence.
license: MIT
compatibility: Requires network access to Europe PMC; installs uv on first run if missing. Runs in any Agent Skills-compatible harness, but the two LLM steps shell out to a child session, so the `claude`, `codex`, or Antigravity `agy` CLI must be installed and signed in once (the skill installs it if missing).
metadata:
  version: "1.0.0"
---

# Librarian

Runs the repo's own `LibrarianAgent` (`librarian/agent.py`) — query planning,
Europe PMC search, full-text chunking, BM25, relevance judge, evidence assembly —
with one change: its LLM client is a fresh Claude Code, Codex, or Antigravity CLI
session instead of a served model. Nothing from the pipeline is reimplemented in
the skill. Step 2 is a response-only synthesis from the instructions the script prints
after retrieval; it is how every run's final answer is presented.

## Failure boundary

Use only this pipeline's Europe PMC evidence. Never fall back to web search,
browser tools, other literature sources, or an answer from memory when the
script fails. On a setup or authentication failure, apply the recovery steps
below before reporting a blocker. On other CLI-session failures, stop and
report the actual CLI error; do not attribute it to Europe PMC. For an empty
retrieval, follow the retry rule below.

The CLI sessions need network access to their model provider, and the script
needs network access to Europe PMC. In a network-restricted Codex task, run it
through `exec_command` with `sandbox_permissions="require_escalated"` and a
justification for model-provider or Europe PMC access **on the first attempt**.
Use the host's approval flow; do not change sandbox environment variables or
bypass workspace policies. If approval is denied, stop and report the denial.
Network access is not permission to use web search.

A failed connectivity check means no child was launched: retry with approved
network access instead of waiting. Each child session is stopped after 180
seconds, and the whole run after 540. A failed or timed-out judge session only
costs that batch (the agent retries it in halves) and prints a `WARNING: ...
judge call(s) failed` line on stderr: tell the user the evidence may be
incomplete. If every judge call fails, or query planning fails, the script
exits non-zero with the CLI error. Repair a recognized setup failure as described
below; otherwise report it rather than retrying in a loop.

## Context boundary

The prompts go straight from the agent to the CLI session over stdin; the root
agent sees only the script's final report. Use the direct CLI launcher rather
than native subagent tools in any harness. The root agent reads no files; the synthesis
instructions arrive in the script's output.

## First run: install what is missing

The calling agent owns setup and recovery. Install missing tools, locate
executables, set per-command overrides, and retry the requested search yourself;
do not turn recoverable diagnostics into a list of chores for the user. Use the
host's normal permission flow for installations or restricted access. Repair each
identified prerequisite once, then rerun; if the same repair fails again, report
the remaining blocker. Never remove Python environments or login files.
Do not check ahead of time — that costs a model round-trip every run. Just run
step 1; the launcher finds `uv` in `~/.local/bin` and installs the Python
dependencies itself on first call. Only if it exits 127 or prints `need uv`, install
`uv` with the official installer through the host's normal approval flow, then
re-run step 1:

- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

The provider CLI is found on PATH, in `~/.local/bin`, or — for Codex — inside
the ChatGPT/Codex desktop app, so it is usually already there. Only if step 1
reports `CLI (...) is not installed or not on PATH`, first locate an existing
executable (including bundled desktop binaries) and set `LIBRARIAN_CLI_PATH`
for the retry. If none exists and the user did not pin a provider, prefer one
installed alternative under the provider recovery rule below. Otherwise install
the selected CLI and retry once:

- claude (macOS/Linux): `curl -fsSL https://claude.ai/install.sh | bash`
- codex: `npm install -g @openai/codex` (or `brew install codex` on macOS)

Set `LIBRARIAN_CLI_PATH` to the full provider executable path for a custom
installation. This does not change the selected provider.

A freshly installed CLI may still need a one-time sign-in. If the user did not
explicitly choose a provider, you may try one other installed provider; announce
that choice and rerun with its `--provider` value. Do not carry a
`LIBRARIAN_CLI_PATH` override from the old provider into that retry. If the user
chose the provider, or the alternative also fails, launch the relevant CLI in a
host terminal when available and ask the user only to complete the interactive
sign-in (`/login` for Claude). Resume the search after sign-in, without requiring
a new request. Desktop app login may not authenticate a separate CLI child.
Binary discovery happens before retrieval; authentication is checked by the
query-planning session, before Europe PMC is contacted.

## Setup

The pipeline runs through `scripts/run.sh`, in the directory containing this
`SKILL.md`. The launcher derives the project root from its own location and
provisions the Python environment with `uv` on first call, so there is nothing
to install and no absolute path to configure. Write the command out in full on
one line, with the real path and provider substituted — shell variables do not
persist between calls, and `VAR=x cmd "$VAR"` on one line leaves `$VAR` empty:

```bash
bash "<directory containing this SKILL.md>/scripts/run.sh" search --provider <claude|codex|antigravity> "<user question>"
```

On Windows use the native PowerShell launcher instead of Bash:

```powershell
& "<directory containing this SKILL.md>/scripts/run.ps1" search --provider <claude|codex|antigravity> "<user question>"
```

Both launchers enable UTF-8 for Python and child sessions. If Windows uv reports
`Missing expected target directory for Python minor version link`, `run.ps1`
checks the reported directory's `python.exe` and retries setup once with its
full path, only if it launches successfully. It never deletes managed Python
files or repeatedly reinstalls them. If recovery fails, find a working Python
interpreter yourself and rerun with its full path in `UV_PYTHON`, or use
`LIBRARIAN_PYTHON` with an already provisioned environment (`<repo>/.venv/Scripts/python.exe`
on Windows, `<repo>/.venv/bin/python` on macOS/Linux). Verify that a reused
interpreter can import `librarian` with the repo on `PYTHONPATH` before retrying.
If no working interpreter exists, use uv to install a supported Python version
once. Report **Python setup** only if this repair also fails.

A run usually takes 1.5-3 min (provider-dependent) and is capped at 540 s, longer than a
default command timeout (Claude Code's Bash tool stops at 120 s). Run it in the foreground with the host's longest timeout — in
Claude Code, pass `timeout: 600000` to the Bash tool — and wait for it to finish.

When installed as a plugin, that directory is
`${CLAUDE_PLUGIN_ROOT}/skills/librarian`. Set `LIBRARIAN_PYTHON` to a Python
interpreter that already has Librarian's dependencies if `uv` is unavailable.

Choose `--provider` from `claude`, `codex`, or `antigravity`. Honor an explicit user
choice; otherwise use the provider matching your host (Antigravity →
`antigravity`). When that CLI is unavailable or unauthenticated and the user
has not pinned a provider, you may choose one installed alternative using the
recovery rule above. The host and child provider can differ (for example,
`--provider codex` in Antigravity). Honor explicit provider choices and announce
any automatic provider change.

The Antigravity provider calls the official `agy` CLI, which must be on PATH
and authenticated once through an interactive `agy` session. It sends one
JSON-wrapped prompt through stdin and extracts the completed response from
the event stream, using the same timeout and output validation as other
providers. It uses the configured permissions and `gemini-3.7-flash-low`. See
the
[Antigravity headless documentation](https://antigravity.google/docs/cli/headless/).

The launcher gives Codex child sessions a temporary writable state directory
automatically, copying authentication and signed workspace-policy caches from
`CODEX_HOME` (or `~/.codex`). Policies remain enforced by Codex. If policy
loading fails, retry once through the host's normal approved-access flow. If
cache refresh requires an interactive CLI session, open it in a host terminal
when available and involve the user only for sign-in or a required approval.

If authentication/policy caches cannot be read or copied, request the necessary
file access through the host's approval flow and retry yourself. Refreshing login
alone will not fix a sandbox access denial. If access is denied, stop and report
that blocker.

Claude uses its configured default model. Codex uses `gpt-5.6-luna` at low
effort because both child tasks are bounded JSON transformations. Override with
`LIBRARIAN_CODEX_MODEL` or `LIBRARIAN_CODEX_EFFORT`; set either to an empty
value to restore the Codex CLI default. These are still Codex CLI sessions using
the user's existing authentication plan, not API calls from `.env`.

The single command above runs the whole pipeline.

Progress goes to stderr; stdout is `[Librarian] queries=N paragraphs=N relevant=N`
followed by the report. Each paper's block carries a `Cite as:` line — the
author-year markdown link Step 2 cites it with.

Claude child sessions run with thinking off
(`LIBRARIAN_CLAUDE_THINKING_TOKENS=0`, passed on as `MAX_THINKING_TOKENS`) and
`LIBRARIAN_CLAUDE_EFFORT=low`; that is what keeps them fast. Raise either only to
test whether a harder question needs more reasoning — the judge ranks paragraphs
it has already been handed, so it normally does not. An empty value restores the
CLI default.

Codex child sessions likewise default to `LIBRARIAN_CODEX_MODEL=gpt-5.6-luna`
and `LIBRARIAN_CODEX_EFFORT=low`. Raise either only when benchmarking shows a
quality gain worth the extra latency and usage.

Antigravity bakes the reasoning tier into the model slug, so it defaults to
`LIBRARIAN_ANTIGRAVITY_MODEL=gemini-3.7-flash-low`: about twice as fast as the
`-high` tier, at roughly two thirds of its judge recall. Set the variable to
`gemini-3.7-flash-high` when a question is worth the fuller evidence sweep, or
to an empty value to use the account default.

## Step 1 — search

The command above. If it reports `paragraphs=0`, or fails with a Europe PMC
network/HTTP/timeout error, the fault is Europe PMC's, not this skill's:

- Retry exactly once: re-run the identical command, unmodified.
- If the retry also comes back empty or fails, **stop**. Reply to the user with
  exactly this message:
  > I'm extremely sorry — this isn't a problem with the librarian pipeline, it's Europe PMC itself: literature retrieval failed twice. I won't guess at an answer without literature evidence. Please try again later.

## Step 2 — synthesize the final answer

After every successful Step 1, the script has already printed the synthesis
instructions (`librarian/prompts/summarizer.md`) after the report. Do not read
that file separately. Follow those instructions — structure, source-fidelity
rules, citation discipline — to turn the report into the final answer. Always
use this step, whether the request was a broad overview or a narrow question,
and answer the user's exact question directly, not a generic overview of the
retrieved papers.

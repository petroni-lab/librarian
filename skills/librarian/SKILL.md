---
name: librarian
description: Retrieve Europe PMC evidence for a biomedical research question. Use when the user needs papers, citations, or literature-backed scientific evidence.
license: MIT
compatibility: Requires Python 3.10+, uv, and network access to Europe PMC. Runs in any Agent Skills-compatible harness, but the two LLM steps shell out to a child session, so the `claude` or `codex` CLI must be installed and authenticated on PATH.
metadata:
  version: "0.2.0"
---

# Librarian

Run the librarian pipeline without a served LLM. The scripts reuse
`librarian/agent.py` for Europe PMC search, full-text chunking, BM25
ranking, and evidence assembly.

Steps 1 and 3 invoke a fresh Claude Code or Codex CLI session for the two LLM
operations. The root agent never reads their large prompts and never creates an
`Agent` subagent. Step 5 is a response-only synthesis mode: it reads the
literature-synthesis prompt after retrieval is complete, not either pipeline
prompt. Step 5 is how every run's final answer is presented to the user.

## Failure boundary

Use only this pipeline's Europe PMC evidence. Never fall back to web search,
browser tools, other literature sources, or an answer from memory when a step
fails. If step 1 or step 3 fails, stop and report the actual CLI error; do not
attribute a model startup failure to Europe PMC. For step 2, follow the retry
rule below.

The CLI sessions need network access to their model provider, and step 2 needs
network access to Europe PMC. In a network-restricted Codex task, run steps 1,
2, and 3 through `exec_command` with `sandbox_permissions="require_escalated"`
and a justification for model-provider or Europe PMC access **on the first
attempt**. Use the host's approval flow; do not change sandbox environment
variables or bypass workspace policies. If approval is denied, stop and report
the denial. Network access is not permission to use web search.

The launcher prints progress before starting the model. A failed connectivity
check means no child was launched: retry with approved network access instead
of waiting. A model timeout means the child was stopped after 180 seconds:
report the timeout rather than continuing to poll or saying it is still running.

## Context boundary

Steps 1 and 3 render a prompt file in the run directory, then stream that file
directly to a fresh model process through stdin. The process returns JSON on
stdout; the local script validates and writes it to the run directory.

The root agent sees only paths, counts, and the step-4 report. It must never
read `01_query_prompt.md`, `02_paragraphs.json`, files under `03_judge/`, or
`04_evidence.json`. The only exception is the local `prompts/summarizer.md`
read in Step 5 below.

Do not use the Claude Code `Agent` tool or a Codex native subagent for this
workflow: either would make the child perform an extra Read and Write tool call.

## Setup

Every step runs through `scripts/run.sh`, resolved relative to this `SKILL.md`.
The launcher derives the project root from its own location and provisions the
Python environment with `uv` on first call, so there is nothing to install and
no absolute path to configure:

```bash
RUN="<directory containing this SKILL.md>/scripts/run.sh"
PROVIDER="claude"  # or "codex"
```

When installed as a plugin, that directory is
`${CLAUDE_PLUGIN_ROOT}/skills/librarian`. Set `LIBRARIAN_PYTHON` to a Python
interpreter that already has Librarian's dependencies if `uv` is unavailable.

Set `PROVIDER` to `claude` or `codex`. If you are running inside one of those
two, use that one. Otherwise use whichever is installed and authenticated on
PATH — those are the only two child providers the launcher implements.

The launcher gives Codex child sessions a temporary writable state directory
automatically, copying authentication and signed workspace-policy caches from
`CODEX_HOME` (or `~/.codex`). Policies remain enforced by Codex. If policy
loading still fails, report it and ask the user to run Codex once from their
normal terminal to refresh authentication and policy caches, then retry the
skill.

Keep the `RUN_DIR` printed by step 1 and pass it explicitly to later steps. The
direct CLI sessions use the provider's configured default model. Never add a
`--model` flag to these steps.

## Step 1 — plan Europe PMC queries

```bash
"$RUN" step1_query_prompt --provider "$PROVIDER" "<user question>"
```

The script creates `RUN_DIR`, renders `01_query_prompt.md`, starts a fresh CLI
session with that file as stdin, and writes the returned `{"queries": [...]}`
to `01_queries.json`. No prompt content enters the root agent's context.

## Step 2 — retrieve, chunk, and rank

```bash
"$RUN" step2_retrieve --run "<RUN_DIR>"
```

No model session is involved. The script validates the generated Europe PMC
queries, searches Europe PMC, fetches available full text, chunks papers into
paragraphs, BM25-ranks each sub-query pool, and writes the merged selection to
`02_paragraphs.json`.

Read only stdout.

### If Europe PMC doesn't come through

Two outcomes count as "doesn't come through," and both are treated the same
way:

- A clean run that reports `paragraphs=0` (the queries ran fine, Europe PMC
  just had nothing to give back), or
- The command itself failing (a non-zero exit, a traceback, a Europe PMC
  network/HTTP/timeout error).

Either way, the fault is Europe PMC's, not this skill's:

- Retry exactly once: re-run the identical `run.sh step2_retrieve --run <RUN_DIR>`
  command, unmodified.
- If the retry also comes back empty or fails, **stop the pipeline entirely**. Do
  not run step 3 or step 4.
- Reply to the user with exactly this message:
  > I'm extremely sorry — this isn't a problem with the librarian pipeline, it's Europe PMC itself: literature retrieval failed twice. I won't guess at an answer without literature evidence. Please try again later.

## Step 3 — judge paragraph relevance

```bash
"$RUN" step3_judge_prompts --run "<RUN_DIR>" --provider "$PROVIDER"
```

The script renders one or more files under `03_judge/`, starts one fresh direct
CLI session per file, and runs those sessions in parallel. Each session receives
only its own rendered prompt and writes a validated `{"relevant_ids": [...]}`
output through the local launcher. The root agent sees none of the prompts.

## Step 4 — assemble final evidence

```bash
"$RUN" step4_finalize --run "<RUN_DIR>"
```

No model session is involved. The script maps cited sentence IDs back to the
paragraphs, groups contiguous citations into evidence spans, and prints the
final report. Each paper's block carries a `Cite as:` line — the author-year
markdown link Step 5 cites it with. Do not answer from this report directly —
proceed to Step 5, staying within its evidence.

## Step 5 — synthesize the final answer

After every successful Step 4, read `prompts/summarizer.md` in full, then use
its structure, source-fidelity rules, and citation discipline to synthesize
the Step-4 evidence report into the final answer presented to the user. Always
use this step, whether the user's request was a broad search/overview/summary
or a narrow, specific question — the summarizer prompt already directs the
answer at the original user query, so it fits both. Feed it the user's exact
question so it answers that question directly, not just a generic overview of
the retrieved papers.

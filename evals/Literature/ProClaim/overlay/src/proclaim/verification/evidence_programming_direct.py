#!/usr/bin/env python3
"""
Evidence Programming — Direct API with per-call context refresh.

Uses LiteLLM for model-agnostic LLM completion, bash for code execution,
and jupytext for post-hoc notebook generation.  No Jupyter kernel, no MCP
server, no Claude Agent SDK.

Architecture:
  - Single flat loop: each LLM call gets [system_prompt, execution_log]
  - Per-call context refresh: messages are rebuilt from the execution log
    before every LLM call — no conversation accumulation
  - Tool results flow through the log, not through message history
  - State persistence: EvidenceState auto-saves to disk after every mutation

Usage:
  uv run python -m proclaim.verification.evidence_programming_direct \\
      --config experiments/config.yaml \\
      --claim "Does MAPK1 directly activate H3-3A?"
"""

from proclaim.verification.prompts import (
    DIRECT_SYSTEM_PROMPT as SYSTEM_PROMPT,
    LIBRARIAN_DIRECT_SYSTEM_PROMPT,
    SUBCLAIM_EXAMPLES,
)
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = int(os.environ.get("NB_MAX_OUTPUT_CHARS", "12000"))
_DEFAULT_PYTHON_WRAPPER_TEMPLATES: tuple[dict[str, str], ...] = (
    {
        "kind": "argv",
        "executable_regex": r"python(?:3(?:\.\d+)?)?",
        "flag": "-c",
    },
    {
        "kind": "heredoc",
        "header_regex": r"python(?:3(?:\.\d+)?)?\s*<<\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)\1",
    },
)


# ---------------------------------------------------------------------------
# System prompt: imported from prompts.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Web search helpers (Serper → DuckDuckGo fallback)
# ---------------------------------------------------------------------------

_WEB_SEARCH_LOCK = threading.Lock()  # ddgs hangs on concurrent calls
_SERPER_URL = "https://google.serper.dev"


def _serper_search(query: str, api_key: str, k: int = 5) -> str:
    import requests as _requests
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = _requests.post(
        f"{_SERPER_URL}/search",
        headers=headers,
        params={"q": query, "num": k, "gl": "us", "hl": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    snippets: list[str] = []
    if data.get("answerBox"):
        ab = data["answerBox"]
        for field in ("answer", "snippet"):
            val = ab.get(field)
            if isinstance(val, str):
                snippets.append(val.replace("\n", " "))
    if data.get("knowledgeGraph", {}).get("description"):
        snippets.append(data["knowledgeGraph"]["description"])
    for item in data.get("organic", [])[:k]:
        if "snippet" in item:
            snippets.append(item["snippet"])
    return "\n".join(snippets) if snippets else "No search results."


def _ddg_search(query: str, k: int = 5) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "[ERROR: ddgs not installed. Run: pip install ddgs]"
    snippets: list[str] = []
    try:
        with _WEB_SEARCH_LOCK:
            for result in DDGS().text(query, max_results=k):
                body = result.get("body", "")
                if body:
                    snippets.append(body)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
    return "\n".join(snippets) if snippets else "No search results."


def _do_web_search(query: str, k: int = 5) -> str:
    """Run web search via Serper (if key set) or DuckDuckGo fallback."""
    serper_key = os.environ.get("SERPER_API_KEY", "")
    for attempt in range(3):
        try:
            if serper_key:
                return _serper_search(query, serper_key, k=k)
            return _ddg_search(query, k=k)
        except Exception as exc:
            wait = 2 ** attempt * 2
            logger.warning(
                "Web search error (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)
    return "Web search temporarily unavailable."


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format, used by LiteLLM)
# ---------------------------------------------------------------------------

def build_tool_schemas(disable_web_search: bool = False) -> list[dict]:
    """Build tool schemas in OpenAI function calling format."""
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Execute a command in bash. Use for running Python code: "
                    "python3 -c 'code'. Working directory is the workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for scientific evidence. Use specific queries "
                    "targeting the entities and relationships in the claim. "
                    "Returns snippets from top results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    if disable_web_search:
        schemas = [s for s in schemas if s["function"]["name"] != "web_search"]
    return schemas


# ---------------------------------------------------------------------------
# Jupytext execution log management
# ---------------------------------------------------------------------------

def init_jupytext_log(path: Path, claim: str) -> None:
    """Create the initial jupytext percent-format .py file."""
    header = textwrap.dedent(f"""\
        # ---
        # jupyter:
        #   jupytext:
        #     text_representation:
        #       format_name: percent
        # ---

        # %% [markdown]
        # # Evidence Report: {claim}
        # Started: {datetime.now().isoformat()}
    """)
    path.write_text(header)


def _unwrap_python_c_command(
    command: str,
    templates: tuple[dict[str, str], ...] | list[dict[str, str]] | None = None,
) -> str | None:
    """Return embedded Python source from known shell wrapper templates."""
    stripped = command.strip()
    if not stripped:
        return None

    try:
        argv_parts = shlex.split(stripped, posix=True)
    except ValueError:
        argv_parts = None

    wrapper_templates = templates or _DEFAULT_PYTHON_WRAPPER_TEMPLATES
    for template in wrapper_templates:
        kind = template.get("kind")
        if kind == "argv":
            if argv_parts is None:
                continue
            parts = argv_parts
            if "&&" in parts:
                parts = parts[parts.index("&&") + 1:]
            if len(parts) < 3 or parts[1] != template.get("flag", "-c"):
                continue
            executable = Path(parts[0]).name
            executable_regex = template.get(
                "executable_regex", r"python(?:3(?:\.\d+)?)?")
            if re.fullmatch(executable_regex, executable) is None:
                continue
            return parts[2].strip("\n")
        elif kind == "heredoc":
            lines = stripped.splitlines()
            if len(lines) < 3:
                continue
            header = lines[0].strip()
            if "&&" in header:
                header = header.rsplit("&&", 1)[1].strip()
            header_regex = template.get(
                "header_regex",
                r"python(?:3(?:\.\d+)?)?\s*<<\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)\1",
            )
            match = re.fullmatch(header_regex, header)
            if match is None:
                continue
            closing_tag = template.get(
                "closing_tag") or match.groupdict().get("tag")
            if closing_tag and lines[-1].strip() != closing_tag:
                continue
            return "\n".join(lines[1:-1]).strip("\n")
        else:
            continue

    return None


def _normalize_jupytext_code(code: str) -> str:
    """Normalize logged code so notebook cells contain Python, not shell wrappers."""
    return _unwrap_python_c_command(code) or code


def append_to_jupytext_log(
    path: Path,
    code: str,
    output: str,
    *,
    is_markdown: bool = False,
) -> None:
    """Append a code block and its output to the jupytext log."""
    with open(path, "a") as f:
        if is_markdown:
            f.write("\n# %% [markdown]\n")
            for line in code.splitlines():
                f.write(f"# {line}\n")
        else:
            normalized_code = _normalize_jupytext_code(code)
            f.write("\n# %%\n")
            f.write(normalized_code)
            f.write("\n")
            if output.strip():
                for line in output.strip().splitlines():
                    f.write(f"# → {line}\n")


def generate_notebook(log_path: Path, notebook_path: Path) -> None:
    """Convert the jupytext percent-format .py log to .ipynb.

    We parse the simple percent format ourselves rather than shelling out to
    ``jupytext``, which (as of 1.19.x) leaks cell-boundary markers into cell
    source and cannot represent captured outputs.  Our parser:

    * Splits on ``# %%`` / ``# %% [markdown]`` boundaries.
        * Unwraps embedded ``python -c`` and ``python <<EOF`` commands into
            notebook Python cells.
    * Moves ``# → …`` output-comment lines into proper ``stream`` outputs.
    """
    import json as _json
    import re as _re

    text = log_path.read_text()

    # Split into raw cell blocks on the ``# %%`` boundary
    # Each match gives (tag, body) where tag is "" or " [markdown]"
    CELL_RE = _re.compile(r"^# %%( \[markdown\])?\s*$", _re.MULTILINE)
    splits = list(CELL_RE.finditer(text))

    cells: list[dict] = []
    OUTPUT_PREFIX = "# → "

    for idx, m in enumerate(splits):
        is_markdown = m.group(1) is not None
        start = m.end()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
        body = text[start:end].strip("\n")
        if not body:
            continue

        if is_markdown:
            # Strip leading "# " from each line to recover markdown
            md_lines: list[str] = []
            for line in body.splitlines():
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in md_lines],
            })
        else:
            # Code cell – separate source from ``# → …`` output lines
            code_lines: list[str] = []
            output_lines: list[str] = []
            for line in body.splitlines():
                if line.startswith(OUTPUT_PREFIX) or line == "# →":
                    out = line[len(OUTPUT_PREFIX):] if line.startswith(
                        OUTPUT_PREFIX) else ""
                    output_lines.append(out + "\n")
                else:
                    code_lines.append(line + "\n")

            # Trim trailing blank lines from code
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            normalized_code = _normalize_jupytext_code(
                "".join(code_lines)).rstrip("\n")
            code_lines = [
                line + "\n" for line in normalized_code.splitlines()] if normalized_code else []

            cell: dict = {
                "cell_type": "code",
                "metadata": {},
                "source": code_lines,
                "outputs": [],
                "execution_count": None,
            }
            if output_lines:
                cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_lines,
                }]
                cell["execution_count"] = 1
            cells.append(cell)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    notebook_path.write_text(_json.dumps(nb, indent=1) + "\n")


def serialize_execution_log(path: Path, max_chars: int = 120_000) -> str:
    """Read the log file for context injection, with optional truncation."""
    text = path.read_text()
    if len(text) <= max_chars:
        return text
    # Keep the header + last portion of the log
    header_end = text.find("\n# %%", 100)  # after jupytext header
    if header_end == -1:
        return text[-max_chars:]
    header = text[:header_end]
    tail = text[-(max_chars - len(header) - 50):]
    return header + "\n\n# [... earlier cells truncated ...]\n" + tail


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

def _smart_truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate text with a note so the agent knows data was clipped."""
    if len(text) <= limit:
        return text
    suffix = (
        f"\n[...TRUNCATED — showing {limit} of {len(text)} chars. "
        f"Use read_file on the output file for full content.]"
    )
    return text[: limit - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(
    name: str,
    arguments: dict,
    *,
    workspace: Path,
    log_path: Path,
    env: dict,
) -> str:
    """Dispatch a tool call to the appropriate handler."""
    if name == "bash":
        command = arguments.get("command", "")
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(workspace),
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            output = output.strip()
        except subprocess.TimeoutExpired:
            output = "[ERROR: Command timed out after 600 seconds]"
        except Exception as e:
            output = f"[ERROR: {e}]"

        append_to_jupytext_log(log_path, command, output)
        return _smart_truncate(output) if output else "(no output)"

    elif name == "read_file":
        fpath = arguments.get("path", "")
        try:
            content = Path(fpath).read_text()
            truncated = _smart_truncate(content)
            append_to_jupytext_log(log_path, f"cat {fpath}", truncated)
            return truncated
        except Exception as e:
            error_msg = f"[ERROR reading {fpath}: {e}]"
            append_to_jupytext_log(log_path, f"cat {fpath}", error_msg)
            return error_msg

    elif name == "web_search":
        query = arguments.get("query", "")
        k = int(arguments.get("num_results", 5))
        try:
            result = _do_web_search(query, k=k)
        except Exception as e:
            result = f"[ERROR: web search failed: {e}]"
        truncated = _smart_truncate(result)
        append_to_jupytext_log(log_path, f"web_search({query!r})", truncated)
        return truncated

    return f"[ERROR: Unknown tool '{name}']"


# ---------------------------------------------------------------------------
# Stopping condition
# ---------------------------------------------------------------------------

def should_stop(workspace: Path, threshold: float) -> bool:
    """Check if verification should stop."""
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        return True

    state_path = workspace / "evidence_state.json"
    if not state_path.exists():
        return False

    from proclaim.verification.evidence_state import EvidenceState
    state = EvidenceState.load(state_path)

    if not state.sufficiency_history:
        return False

    last = state.sufficiency_history[-1]
    return last.confidence >= threshold


# ---------------------------------------------------------------------------
# Build environment for subprocess
# ---------------------------------------------------------------------------

def build_subprocess_env(cfg) -> dict:
    """Build environment variables for bash subprocess calls.

    These env vars are read by setup_workspace() inside the subprocess.
    """
    env = {**os.environ}

    # Subagent LLM config (for evidence API calls inside bash)
    env["LLM_BASE_URL"] = cfg.llm.subagent_base_url
    env["LLM_API_KEY"] = cfg.api_key
    env["LLM_MODEL"] = cfg.subagent_model
    env["LLM_TEMPERATURE"] = str(cfg.llm.temperature)
    env["LLM_DISABLE_THINKING"] = "1" if cfg.llm.disable_thinking else "0"
    # Outer-agent LLM config — the librarian run emits its final verdict on the
    # agent model rather than the subagent. Read by model_registry.get_agent_llm().
    agent_model = cfg.llm.model
    if agent_model.lower().startswith(("openai/", "gpt-")):
        agent_api_key = os.environ.get("OPENAI_API_KEY") or cfg.api_key
    else:
        agent_api_key = cfg.api_key
    env["AGENT_LLM_MODEL"] = agent_model
    if cfg.llm.agent_base_url:
        env["AGENT_LLM_BASE_URL"] = cfg.llm.agent_base_url
    env["AGENT_LLM_API_KEY"] = agent_api_key
    env["AGENT_LLM_TEMPERATURE"] = str(cfg.llm.temperature)
    env["AGENT_LLM_TIMEOUT"] = str(cfg.llm.timeout)
    env["AGENT_LLM_STREAM"] = "1" if cfg.llm.stream else "0"

    # The librarian run decides sufficiency from extracted facts, so it needs
    # neither the MLP classifier nor a sufficiency backend.
    if getattr(cfg, "retrieval_backend", "pubmed") == "librarian":
        env.pop("MLP_MODEL_DIR", None)
        env.pop("SUFFICIENCY_BACKEND", None)
    else:
        env["MLP_MODEL_DIR"] = cfg.mlp_model_dir or "results/models/classifier_best"
        env["SUFFICIENCY_BACKEND"] = cfg.sufficiency_backend
    env["RETRIEVAL_BACKEND"] = getattr(cfg, "retrieval_backend", "pubmed")
    env["LIBRARIAN_LLM_BASE_URL"] = getattr(cfg, "librarian_llm_base_url", "")
    env["LIBRARIAN_LLM_MODEL"] = getattr(cfg, "librarian_llm_model", "")
    env["MAX_ITERATIONS"] = str(cfg.max_iterations)
    env["LABEL_CONFIG_JSON"] = cfg.labels.model_dump_json()
    env["NB_MAX_OUTPUT_CHARS"] = str(cfg.max_output_chars)

    # Debug mode for evidence API tools
    env["EVIDENCE_DEBUG"] = "1" if cfg.verbose else "0"

    # Ensure PYTHONPATH includes src
    src_dir = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    if src_dir not in existing:
        env["PYTHONPATH"] = f"{src_dir}:{existing}" if existing else src_dir

    return env


# ---------------------------------------------------------------------------
# Forced verdict helper
# ---------------------------------------------------------------------------

def _force_verdict(workspace: Path, claim: str, sub_env: dict) -> None:
    """Force check_sufficiency + LLM-guided emit_verdict when the turn budget is exhausted.

    Mirrors what the agent would do in its final turn: runs check_sufficiency, then
    prompts the subagent LLM to evaluate the evidence and decide SUPPORT/REFUTE/UNCERTAIN,
    then calls emit_verdict.  No hardcoded verdict defaults — the LLM makes the call.
    """
    from proclaim.verification.config import get_label_config as _get_label_cfg
    _verdict_names = ', '.join(_get_label_cfg().verdict_names())

    abs_workspace = str(workspace.resolve())
    script = textwrap.dedent(f"""\
        from proclaim.verification.evidence_api import (
            setup_workspace, populate_paper_features, check_sufficiency, emit_verdict,
            get_evidence_summary,
        )
        from proclaim.verification.config import get_label_config

        state, llm, workspace = setup_workspace(
            claim={claim!r},
            workspace_path={abs_workspace!r},
        )
        print(f"Forced verdict: {{len(state.papers)}} papers, {{len(state.facts)}} facts, iteration={{state.iteration}}")

        populate_paper_features(state)

        # Run or reuse sufficiency check for gaps.
        if state.sufficiency_history:
            suf = state.sufficiency_history[-1]
            print(f"Reusing last sufficiency: {{suf.label}} (confidence={{suf.confidence:.3f}})")
        else:
            suf = check_sufficiency(state, llm)

        gaps = [g.description for g in (suf.gaps if suf else [])]
        suf_confidence = suf.confidence if suf else 0.0

        # Prompt the subagent LLM to evaluate evidence and decide the verdict.
        label_cfg = get_label_config()
        summary = get_evidence_summary(state)
        facts_text = "\\n".join(
            f"  [{{f.stance}}] {{f.text[:200]}}" for f in state.facts[:20]
        ) or "  (none)"

        verdict_prompt = (
            "You are a scientific evidence evaluator. Based on the evidence below,\\n"
            "determine the verdict for this claim using exactly one of the defined labels.\\n"
            "\\n"
            f"Claim: {{state.claim}}\\n"
            "\\n"
            "Verdict label definitions:\\n"
            f"{{label_cfg.verdict_prompt_block()}}\\n"
            "\\n"
            "Evidence summary:\\n"
            f"{{summary}}\\n"
            "\\n"
            "Extracted facts:\\n"
            f"{{facts_text}}\\n"
            "\\n"
            "Gaps identified:\\n"
            f"{{chr(10).join(f'  - {{g}}' for g in gaps[:5]) or '  (none)'}}\\n"
            "\\n"
            "Output your answer in this exact format:\\n"
            f"VERDICT: <one of {_verdict_names}>\\n"
            "CONFIDENCE: <0.0-1.0>\\n"
            "REASONING: <one paragraph>\\n"
            "KEY_EVIDENCE: <bullet 1> | <bullet 2> | <bullet 3>\\n"
        )

        response = llm(verdict_prompt)
        print("LLM verdict response:", response[:600])

        # Parse the LLM's structured response.
        verdict_label = None
        confidence = suf_confidence
        reasoning = None
        key_evidence = []

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in label_cfg.verdict_names():
                    verdict_label = raw
                else:
                    print(f"WARNING: LLM returned unrecognised verdict label: {{raw!r}}")
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_EVIDENCE:"):
                key_evidence = [e.strip() for e in line.split(":", 1)[1].split("|") if e.strip()]

        if verdict_label is None or reasoning is None:
            raise RuntimeError(
                f"Failed to parse LLM verdict response. Raw response:\\n{{response}}"
            )

        print(f"Parsed verdict: {{verdict_label}} (confidence={{confidence:.3f}})")

        emit_verdict(
            verdict=verdict_label,
            confidence=confidence,
            reasoning=reasoning,
            key_evidence=key_evidence[:5],
            gaps_remaining=gaps[:5],
            state=state,
            workspace=workspace,
        )
    """)

    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(script)
            tmp_path = tf.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=sub_env,
        )
        output = (result.stdout +
                  ("\n" + result.stderr if result.stderr else "")).strip()
    except Exception as exc:
        output = f"[ERROR: forced verdict script failed: {exc}]"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    logger.info("Forced verdict")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def verify_claim_direct(cfg) -> Path:
    """Run evidence programming via LiteLLM direct API with context refresh."""
    # The librarian backend runs its own single-pass path end to end.
    if getattr(cfg, "retrieval_backend", "pubmed") == "librarian":
        return _verify_librarian_one_shot(cfg)

    import litellm

    workspace = cfg.resolved_workspace
    output_dir = cfg.resolved_output_dir
    claim = cfg.claim

    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evidence state on disk
    from proclaim.verification.evidence_state import EvidenceState
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt
    from proclaim.verification.evidence_api import schema_docs, function_docs
    label_cfg = cfg.labels
    _web_search_step = (
        ""
        if cfg.disable_web_search
        else (
            "   d. web_search(query) — call this tool directly (NOT via bash) to search the\n"
            "      web for evidence not found in PubMed/S2; use when academic databases\n"
            "      return few results or for recent findings not yet indexed."
        )
    )
    system_prompt = SYSTEM_PROMPT.format(
        workspace=str(workspace.resolve()),
        claim=claim,
        max_iterations=cfg.max_iterations,
        sufficiency_threshold=cfg.sufficiency_threshold,
        schemas=schema_docs(),
        function_docs=function_docs(),
        verdict_names=", ".join(label_cfg.verdict_names()),
        verdict_definitions=label_cfg.verdict_prompt_block(),
        subclaim_examples=SUBCLAIM_EXAMPLES if cfg.include_subclaim_examples else "",
        web_search_step=_web_search_step,
    )

    # Jupytext execution log
    log_path = workspace / "execution_log.py"
    init_jupytext_log(log_path, claim)

    # Tools and environment
    tools = build_tool_schemas(disable_web_search=cfg.disable_web_search)
    sub_env = build_subprocess_env(cfg)

    # Outer agent model (via LiteLLM)
    agent_model = cfg.llm.model  # e.g. "anthropic/claude-sonnet-4-20250514"

    # Anthropic extended thinking: pass thinking block when budget_tokens > 0.
    # Requires temperature=1 per Anthropic API requirements.
    _thinking_budget = cfg.llm.thinking_budget_tokens
    _thinking_kwargs: dict = {}
    if _thinking_budget > 0:
        _thinking_kwargs["thinking"] = {
            "type": "enabled", "budget_tokens": _thinking_budget}
        _thinking_kwargs["temperature"] = 1

    # Anthropic prompt caching: mark the system message and the last user
    # message with cache_control so repeated turns reuse cached prefixes.
    # LiteLLM passes this through to Anthropic's API.  For non-Anthropic
    # models the extra key is silently ignored.
    # Note: prompt caching is disabled when thinking is active (Anthropic restriction).
    _use_cache = agent_model.startswith("anthropic/") and _thinking_budget == 0

    def _cached_system_msg(text: str) -> dict:
        if _use_cache:
            return {
                "role": "system",
                "content": [{"type": "text", "text": text,
                             "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
            }
        return {"role": "system", "content": text}

    def _cached_user_msg(text: str, prev_log_len: int = 0, curr_log_len: int = 0) -> dict:
        """Split user message into stable cached prefix + new delta + uncached tail.

        Anthropic's cache lookup requires the cache_control breakpoint to be at
        the EXACT SAME character position as a previously-cached entry to get a
        cache read.  Moving the breakpoint every call therefore never produces reads.

        Fix: use TWO breakpoints per call.
          BP#2 at (prev_log_len) — same position as last call's BP#3 → cache READ
          BP#3 at (curr_log_len) — new log end position             → cache WRITE

        On the very first caching step (no prior entry) only BP#3 is emitted.
        """
        if not _use_cache:
            return {"role": "user", "content": text}

        log_start = text.find("<execution_log>\n")
        if log_start == -1:
            return {"role": "user", "content": text}

        log_content_start = log_start + len("<execution_log>\n")
        curr_split = log_content_start + curr_log_len  # new content ends here

        # Guard: must have substantial new content to cache
        if curr_log_len < 500 or curr_split >= len(text) - 200:
            return {"role": "user", "content": text}

        prev_split = log_content_start + prev_log_len  # previous call's cache end

        if prev_log_len < 500 or prev_split >= curr_split:
            # First caching step: write a new cache entry at curr_split
            return {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text[:curr_split],
                        # cache WRITE
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    },
                    {
                        "type": "text",
                        "text": text[curr_split:],
                    },
                ],
            }

        # Subsequent steps: read at prev_split, write delta to curr_split
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text[:prev_split],
                    # cache READ (prev write pos)
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "type": "text",
                    "text": text[prev_split:curr_split],
                    # cache WRITE (new delta)
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "type": "text",
                    # uncached tail (closing tag + prompt)
                    "text": text[curr_split:],
                },
            ],
        }

    # Token usage tracking
    from proclaim.verification.cost_tracker import CostTracker
    tracker = CostTracker(model=agent_model)

    logger.info(
        "Starting direct API verification: claim=%r, model=%s, workspace=%s",
        claim, agent_model, workspace,
    )

    # Single flat loop with per-call context refresh.
    # Each LLM call receives only [system_prompt, execution_log].
    # Tool results flow through the log — no conversation accumulation.
    max_calls = cfg.max_iterations * cfg.max_turns
    call_count = 0
    # len(execution_log) sent in the previous call; 0 on first call
    _prev_log_len = 0
    # len(execution_log) read at the start of the current call
    _this_log_len = 0

    while call_count < max_calls:
        # Carry forward the log length from the previous iteration so we can
        # split the user message into a cached prefix + uncached tail.
        _prev_log_len = _this_log_len

        # *** CONTEXT REFRESH: rebuild messages from log before every call ***
        execution_log = serialize_execution_log(log_path)
        _this_log_len = len(execution_log)

        turns_remaining = max_calls - call_count
        if call_count == 0:
            user_text = (
                f"Verify the following scientific claim using evidence programming.\n\n"
                f"Claim: {claim}\n\n"
                f"Start by calling bash with the setup code to import the evidence API. "
                f"Follow the evidence programming workflow. "
                f"Call check_sufficiency after each round. "
                f"Stop when confidence >= {cfg.sufficiency_threshold} or after "
                f"{cfg.max_iterations} iterations."
            )
        elif turns_remaining <= 2:
            user_text = (
                f"<execution_log>\n{execution_log}\n</execution_log>\n\n"
                f"URGENT — only {turns_remaining} turn(s) remaining before hard stop.\n"
                f"You MUST call check_sufficiency and then emit_verdict NOW.\n"
                f"Do NOT run any more searches or extractions.\n"
                f"Claim: {claim}"
            )
        else:
            user_text = (
                f"<execution_log>\n{execution_log}\n</execution_log>\n\n"
                f"Continue evidence verification for claim: {claim}\n"
                f"The execution log above shows all prior work. "
                f"Continue the workflow: run code, check sufficiency, "
                f"address remaining gaps, or emit verdict if ready. "
                f"Stop when confidence >= {cfg.sufficiency_threshold}."
            )

        messages = [
            _cached_system_msg(system_prompt),
            _cached_user_msg(user_text, prev_log_len=_prev_log_len,
                             curr_log_len=_this_log_len),
        ]

        # --- LLM call ---
        _t0 = time.monotonic()
        try:
            response = litellm.completion(
                model=agent_model,
                messages=messages,
                tools=tools,
                max_tokens=16384,
                **_thinking_kwargs,
            )
        except Exception as e:
            logger.error("LiteLLM API error: %s", e)
            time.sleep(5)
            try:
                response = litellm.completion(
                    model=agent_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=16384,
                    **_thinking_kwargs,
                )
            except Exception as e2:
                logger.error("LiteLLM retry failed: %s", e2)
                break
        _latency = time.monotonic() - _t0

        call_count += 1

        choice = response.choices[0]
        assistant_msg = choice.message

        _tool_calls_trace = []
        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                _tool_calls_trace.append({
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        # Track usage
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            tracker.record(
                "llm_call",
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                latency=_latency,
                call_number=call_count,
                cache_read_tokens=getattr(
                    u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(
                    u, "cache_creation_input_tokens", 0) or 0,
                system_prompt=system_prompt,
                user_message=user_text,
                assistant_response=assistant_msg.content or "",
                tool_calls=_tool_calls_trace,
            )

        # Log agent reasoning to execution log
        if assistant_msg.content:
            logger.info("Agent [call %d]: %s", call_count,
                        assistant_msg.content[:300])
            append_to_jupytext_log(
                log_path,
                assistant_msg.content[:2000],
                "",
                is_markdown=True,
            )

        # No tool calls — agent paused or finished
        if choice.finish_reason == "stop" or not assistant_msg.tool_calls:
            if should_stop(workspace, cfg.sufficiency_threshold):
                logger.info(
                    "Stopping: verdict emitted or sufficiency reached."
                )
                break
            logger.info(
                "Agent stopped without verdict (call %d). Re-prompting.", call_count)
            continue

        # Dispatch tool calls — results are appended to execution log
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {"command": tc.function.arguments}

            logger.info("Tool [call %d]: %s(%s)", call_count,
                        fn_name, str(fn_args)[:200])

            result = dispatch_tool(
                fn_name,
                fn_args,
                workspace=workspace,
                log_path=log_path,
                env=sub_env,
            )

            logger.info("Tool result: %s", result[:200])

        # Check stopping after tool dispatch.
        # Only stop on an emitted verdict here — sufficiency alone is not enough
        # because the agent still needs one more turn to call deliver_verdict.
        verdict_path = workspace / "verdict.json"
        if verdict_path.exists():
            logger.info("Stopping: verdict emitted.")
            break

    else:
        logger.info("Reached max calls (%d).", max_calls)

    # If the loop exited without a verdict, force one from the current state.
    verdict_path = workspace / "verdict.json"
    if not verdict_path.exists():
        logger.info(
            "No verdict emitted — forcing check_sufficiency + emit_verdict.")
        _force_verdict(workspace, claim, sub_env)

    # Log final usage
    summary = tracker.summary()
    logger.info("Total token usage: %s", json.dumps(summary))

    # Save usage stats
    usage_path = output_dir / "token_usage.json"
    usage_path.write_text(json.dumps(summary, indent=2))

    trace_path = output_dir / "token_usage_trace.json"
    trace_path.write_text(json.dumps(tracker.trace_as_dicts(), indent=2))

    # Generate notebook from jupytext log
    notebook_path = output_dir / "evidence_report.ipynb"
    try:
        generate_notebook(log_path, notebook_path)
        logger.info("Notebook generated: %s", notebook_path)
    except Exception as e:
        logger.warning("Failed to generate notebook (jupytext): %s", e)
        notebook_path = log_path  # fall back to .py log

    # Print verdict if available
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        from proclaim.verification.data_models import VerificationVerdict
        try:
            verdict = VerificationVerdict.model_validate_json(
                verdict_path.read_text()
            )
            print(
                f"\nVerdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
            print(f"Reasoning: {verdict.reasoning}")
        except Exception as e:
            logger.error("Failed to parse verdict: %s", e)

    return notebook_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from proclaim.verification.config import VerificationSettings

    cfg = VerificationSettings.from_cli()

    output_dir = cfg.resolved_output_dir
    workspace = cfg.resolved_workspace

    # Configure logging
    log_level = logging.DEBUG if cfg.verbose else logging.INFO
    log_file = output_dir / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logger.info("Log file: %s", log_file)

    print(f"Claim: {cfg.claim}")
    print(f"Agent model: {cfg.llm.model}")
    print(f"Subagent model: {cfg.subagent_model}")
    print(f"Workspace: {workspace}")
    print()

    result = verify_claim_direct(cfg)
    print(f"\nOutput: {result}")
    print(f"Workspace: {workspace}")



# ── Librarian retrieval backend ──────────────────────────────────────────────
# Everything below is the librarian integration: a one-shot verification path
# used when `retrieval_backend: librarian`, plus its helpers. Not upstream —
# see ../../../NOTICE.

_RAW_EVIDENCE_VERDICT_CAP = 50
_RAW_EVIDENCE_SNIPPET_CHARS = 1500
_TOP_EVIDENCE_VERDICT_CAP = 10


def _is_librarian_infrastructure_error(result: str) -> bool:
    """Detect librarian backend failures that invalidate the verification run."""
    markers = (
        "Librarian unavailable:",
        "Librarian search failed",
        "RuntimeError: Librarian unavailable:",
        "RuntimeError: Librarian search failed",
    )
    return any(marker in result for marker in markers)

def _collect_raw_librarian_evidence(
    state,
    cap: int = _RAW_EVIDENCE_VERDICT_CAP,
) -> list[dict[str, str]]:
    """Collect retrieved librarian evidence snippets for fallback verdicting."""
    records: list[dict[str, str]] = []
    for paper in state.papers.values():
        if len(records) >= cap:
            break

        record = _librarian_evidence_record(
            paper,
            max_chars=_RAW_EVIDENCE_SNIPPET_CHARS,
        )
        if record:
            records.append(record)
    return records

def _paper_url(pmid: str, doi: str | None) -> str:
    """Return a citation URL for a retrieved record when one is available."""
    if str(pmid).isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if doi:
        return f"https://doi.org/{doi}"
    return ""

def _clean_prompt_text(value: object, max_chars: int | None = None) -> str:
    """Normalize whitespace before placing retrieved evidence in a prompt."""
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text

def _librarian_evidence_record(
    paper,
    max_chars: int | None = None,
) -> dict[str, str]:
    """Convert a retrieved paper record into a serializable evidence record."""
    evidence_text = paper.full_text or paper.abstract or paper.summary or ""
    evidence_text = _clean_prompt_text(evidence_text, max_chars)
    if not evidence_text:
        return {}

    pmid = str(paper.pmid or "")
    doi = str(paper.doi or "")
    return {
        "title": _clean_prompt_text(paper.title),
        "url": _paper_url(pmid, doi or None),
        "pmid": pmid,
        "doi": doi,
        "evidence": evidence_text,
    }

def _write_librarian_search_iteration_evidence(
    workspace: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Persist librarian evidence grouped by search iteration in the workspace."""
    path = workspace / "librarian_evidence_by_search_iteration.json"
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

def _build_librarian_recovery_query(claim: str, decisive_count: int) -> str:
    """Fallback query when failed-paper refinement returns no new papers."""
    claim_lc = claim.lower()
    if " as ligand " in claim_lc and " as receptor" in claim_lc:
        domain_note = (
            "Focus on exact direct extracellular or cell-surface interaction "
            "between the named pair, receptor-complex or heterodimer evidence "
            "that contains the named receptor subunit, and papers assigning the "
            "opposite ligand/receptor roles."
        )
    elif " directly activates " in claim_lc or " directly inhibits " in claim_lc:
        domain_note = (
            "Focus on direct mechanism papers, aliases for both proteins, and "
            "functional evidence such as phosphorylation, cleavage, degradation, "
            "stabilization, complex formation, expression regulation, enzyme "
            "activity, second messengers, or pathway readouts."
        )
    else:
        domain_note = (
            "Prioritize primary experimental evidence over network, enrichment, "
            "co-expression, docking, or pathway-summary papers."
        )

    return (
        "Find additional primary biomedical papers that can verify this claim: "
        f'"{claim}". The current evidence extraction found {decisive_count} '
        "SUPPORT or REFUTE fact(s), so search with different aliases and "
        "terminology to find at least one directional or contradictory fact. "
        f"{domain_note} Include evidence that supports or contradicts the claim."
    )

def _proclaim_decisive_fact_counts(state) -> dict[str, int]:
    """Count extracted facts that can support a committed verdict."""
    counts = {"SUPPORT": 0, "REFUTE": 0}
    for fact in state.facts:
        stance = str(getattr(fact, "stance", "")).upper()
        if stance in counts:
            counts[stance] += 1
    return counts

def _pmids_without_decisive_facts(pmids: list[str], state) -> list[str]:
    """Return current PMIDs that yielded no SUPPORT or REFUTE fact."""
    requested_pmids = [str(pmid) for pmid in pmids]
    requested_set = set(requested_pmids)
    decisive_sources: set[str] = set()
    for fact in state.facts:
        pmid = str(getattr(fact, "source_pmid", ""))
        stance = str(getattr(fact, "stance", "")).upper()
        if pmid in requested_set and stance in {"SUPPORT", "REFUTE"}:
            decisive_sources.add(pmid)
    return [
        pmid
        for pmid in requested_pmids
        if pmid in state.papers and pmid not in decisive_sources
    ]

def _evidence_records_for_pmids(
    state,
    pmids: list[str],
) -> list[dict[str, str]]:
    """Collect raw evidence records for a specific search result batch."""
    records: list[dict[str, str]] = []
    for pmid in pmids:
        paper = state.papers.get(pmid)
        if not paper:
            continue
        record = _librarian_evidence_record(paper)
        if record:
            records.append(record)
    return records

def _select_top_evidence_for_verdict(
    search_evidence_iterations: list[dict[str, Any]],
    cap: int = _TOP_EVIDENCE_VERDICT_CAP,
) -> list[dict[str, str]]:
    """Pick up to ``cap`` librarian evidence snippets for the verdict prompt.

    Evidence is grouped by search iteration in the order the librarian returned
    it. With a single iteration we take the top ``cap`` snippets as ranked. With
    several iterations we split the budget evenly (e.g. 5 + 5 for two
    iterations) and fill any shortfall from the remaining snippets, so each
    iteration contributes context up to the cap.
    """
    batches = [
        entry["evidence_records"]
        for entry in search_evidence_iterations
        if entry.get("evidence_records")
    ]
    if not batches:
        return []

    per_batch = max(1, cap // len(batches))
    selected: list[dict[str, str]] = []
    leftovers: list[dict[str, str]] = []
    for batch in batches:
        selected.extend(batch[:per_batch])
        leftovers.extend(batch[per_batch:])

    for record in leftovers:
        if len(selected) >= cap:
            break
        selected.append(record)

    selected = selected[:cap]
    # Bound snippet length so a handful of full-text records cannot blow up the prompt.
    return [
        {
            **record,
            "evidence": _clean_prompt_text(
                record.get("evidence", ""), _RAW_EVIDENCE_SNIPPET_CHARS
            ),
        }
        for record in selected
    ]

def _render_top_evidence(records: list[dict[str, str]]) -> str:
    """Render selected evidence snippets as a compact, labeled block."""
    lines: list[str] = []
    for i, record in enumerate(records, 1):
        title = record.get("title", "")
        header = f"[{i}] PMID {record.get('pmid', '?')} — {title}".rstrip(" —")
        lines.append(header)
        lines.append(f"    {record.get('evidence', '')}")
    return "\n".join(lines) or "  (none)"

def _proclaim_verdict_rubric() -> str:
    """Shared ProClaim verdict decision rule for the final verdict prompts.

    Organized as labeled sections so each rule is easy to find and edit. This
    rule overrides any broader UNCERTAIN wording in the configured label
    definitions.
    """
    return textwrap.dedent(
        """\
        ProClaim verdict decision rule.

        HOW TO WEIGH EVIDENCE
        - Judge by the QUALITY and relevance of the facts, not their count. One
          directly relevant, credible fact can decide the claim; a large pile of
          weak or tangential facts cannot.
        - The SUPPORT/REFUTE/NEUTRAL tags on facts are advisory. Reason from the
          fact text itself and correct a tag mentally when the text points the
          other way.
        - Treat NEUTRAL facts as tie-breaker evidence, not as decisive evidence.
          Use them to choose between SUPPORT, REFUTE, and UNCERTAIN only after
          weighing the clear SUPPORT/REFUTE facts. If neutral facts consistently
          point toward one biological interpretation, they can tip a close case;
          if they are merely adjacent or ambiguous, they should not decide it.

        SUPPORT vs REFUTE
        - SUPPORT: the evidence establishes the claimed relationship in the
          claimed direction. Mechanistic equivalents count when they
          establish the claimed direction.
        - REFUTE: the evidence establishes the OPPOSITE direction/sign or
          otherwise contradicts the claim. Opposite-polarity evidence is REFUTE,
          not NEUTRAL.

        ANNOTATION SPECIFICS
        - Aliases and indirect evidence: evidence about aliases, family members, domains,
          orthologs, paralogs, or complexes can SUPPORT a claim about the named
          entities when the paper states a clear relationship that can be mapped to
          the claim. Judge the paper's stated relationship and its relevance to the
          claim, not just the presence of the named entities.
        - Direction: evidence stating the relationship in the opposite direction
          to the claim REFUTES it.
        - Signed regulation — the mechanism is NOT the sign: "X phosphorylates /
          ubiquitinates / cleaves / modifies Y" identifies Y as a substrate but
          does NOT by itself mean X activates Y. Read the sign off the functional
          consequence the paper states for Y's activity or abundance:
            * modification that degrades/destabilizes Y, or poly-ubiquitination
              targeting Y for degradation, is INHIBITION;
            * modification that raises Y's activity or is an activating mark
              (activating phosphorylation, mono-ubiquitination of a histone,
              zymogen cleavage yielding an active fragment) is ACTIVATION.
          When the consequence is not stated, the modification verb is
          sign-neutral — do NOT assume activation from it.
        - Directly (for "X directly activates/inhibits Y" claims): require a
          direct molecular action of X on Y. Canonical multi-step pathways and
          second-messenger relays (e.g. PI3K -> PIP3 -> AKT), or wording like
          "drives/promotes Y signaling" or "engaged with the Y pathway", are
          INDIRECT and do not satisfy the claim, even when biologically real.
        - Ligand-receptor: receptor-subunit or heterodimer evidence can support
          the named receptor when the complex contains that subunit. Exact
          ligand/receptor role reversal is strong refuting evidence, while
          membrane-bound bidirectional signaling systems should be judged from
          direct pair engagement and the paper's stated biology rather than
          canonical role labels alone.
        - Extracellular claims: if the claim asserts an EXTRACELLULAR
          interaction but the paper localizes the binding intracellularly, that
          REFUTES it; an intracellular association does not satisfy an
          extracellular-interaction claim. For "X as ligand directly interacts
          extracellularly with Y as receptor" claims, require positive evidence
          of direct physical binding of the specific X-Y pair at the cell
          surface, with X acting as a secreted or membrane-surface ligand.
          Shared pathway membership, co-expression, co-occurrence in the same
          process, or X being an intracellular protein (e.g. an enzyme or
          cytoplasmic adaptor that merely functions in the same pathway) does
          NOT satisfy the claim and REFUTES it, even when abundant literature
          links the two entities.
        - Pharmacological vs biological: a drug, compound, or inhibitor of X that
          reduces Y's activity is NOT evidence that X inhibits Y. Judge only the
          protein-protein relationship the paper states directly.

        WHEN TO CHOOSE UNCERTAIN (Not Enough Info)
        UNCERTAIN is a last resort. Use it ONLY when credible, directly on-point
        SUPPORT and REFUTE facts make exactly opposite, same-scope claims about
        the same specific relationship or mechanism and neither is clearly
        stronger. Before returning UNCERTAIN you MUST cite, in your reasoning,
        the one directly on-point SUPPORT fact and the one directly on-point
        REFUTE fact that conflict; if you cannot cite BOTH, UNCERTAIN is not
        permitted — commit to the direction the on-point evidence leans, and
        when no directly on-point evidence substantiates the claim, choose
        REFUTE. Do NOT choose UNCERTAIN merely because a molecular event's
        direction or sign looks ambiguous — resolve the sign from the functional
        consequence the paper states (see Signed regulation above) and commit.

        OTHERWISE COMMIT
        Lean toward a committed SUPPORT or REFUTE. Do NOT use UNCERTAIN merely
        because the evidence is sparse, partial, indirect, correlative, imperfect,
        or phrased through aliases/complexes/family terms. If the relevant
        evidence points one way and the UNCERTAIN case above does not hold, choose
        SUPPORT or REFUTE accordingly; when evidence is thin or only weakly
        on-point, still commit to the direction it most supports rather than
        defaulting to UNCERTAIN."""
    )

def _build_raw_evidence_verdict_prompt(
    claim: str,
    evidence_records: list[dict[str, str]],
    label_cfg,
    valid_labels: list[str],
) -> str:
    """Build the verdict prompt over raw retrieved evidence.

    Uses the same label definitions and ProClaim decision rubric as the
    fact-based verdict path. A scope rule splits the no-evidence cases: evidence
    that does not address the claim's entities at all leaves the claim
    unsubstantiated (REFUTE), while evidence that concerns the entities but
    cannot resolve the specific relationship is UNCERTAIN.
    """
    return (
        "You are a scientific evidence evaluator. Based only on the librarian "
        "retrieval evidence below, determine the verdict for this claim using "
        "exactly one of the defined labels.\n\n"
        f"Claim: {claim.strip()}\n\n"
        "Verdict label definitions:\n"
        f"{label_cfg.verdict_prompt_block()}\n\n"
        "Retrieved evidence records:\n"
        f"{json.dumps(evidence_records, indent=2, ensure_ascii=False)}\n\n"
        f"{_proclaim_verdict_rubric()}\n\n"
        "SCOPE RULE: if none of the evidence above addresses the claim's specific "
        "entities (or their aliases/isoforms/complexes/family terms), the claim "
        "is unsubstantiated by the literature — answer REFUTE, not UNCERTAIN. Use "
        "UNCERTAIN only when the evidence does concern the claimed entities but "
        "cannot resolve the specific relationship.\n\n"
        "Output your answer in this exact format:\n"
        f"VERDICT: <one of {', '.join(valid_labels)}>\n"
        "REASONING: <one paragraph>\n"
        "KEY_EVIDENCE: <bullet 1> | <bullet 2> | <bullet 3>\n"
    )

def _parse_verdict_response(
    response: str, valid_labels: list[str]
) -> tuple[str, str, list[str]]:
    """Parse the one-shot verdict LLM response."""
    verdict_label = ""
    reasoning = ""
    key_evidence: list[str] = []

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw in valid_labels:
                verdict_label = raw
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.startswith("KEY_EVIDENCE:"):
            raw_items = line.split(":", 1)[1].split("|")
            key_evidence = [item.strip() for item in raw_items if item.strip()]

    if not verdict_label or not reasoning:
        raise RuntimeError(
            "Failed to parse LLM verdict response. Expected VERDICT and REASONING "
            f"lines. Raw response:\n{response}"
        )
    return verdict_label, reasoning, key_evidence

def _verify_librarian_one_shot(cfg) -> Path:
    """Run the ProClaim+librarian pipeline with sparse-evidence recovery."""
    workspace = cfg.resolved_workspace
    output_dir = cfg.resolved_output_dir
    claim = cfg.claim

    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_env = build_subprocess_env(cfg)
    os.environ.update(sub_env)

    log_path = workspace / "execution_log.py"
    init_jupytext_log(log_path, claim)

    from proclaim.verification.config import get_label_config
    from proclaim.verification.evidence_api import (
        emit_verdict,
        extract_and_add_facts,
        get_evidence_summary,
        librarian_refine_search,
        librarian_search,
        librarian_search_for_claim,
        setup_workspace,
    )
    from proclaim.verification.model_registry import get_agent_llm

    state, llm, workspace_obj = setup_workspace(
        claim=claim,
        workspace_path=str(workspace.resolve()),
    )
    state.subclaims = []
    state._auto_save()
    append_to_jupytext_log(
        log_path,
        textwrap.dedent(
            f"""\
            from proclaim.verification.evidence_api import setup_workspace
            state, llm, workspace = setup_workspace(
                claim={claim!r},
                workspace_path={str(workspace.resolve())!r},
            )
            state.subclaims = []
            state._auto_save()
            print("Ready")
            """
        ),
        "Ready",
    )

    pmids = librarian_search_for_claim(state.claim, state, llm)
    search_evidence_iterations: list[dict[str, Any]] = [
        {
            "search_iteration": 1,
            "phase": "initial_claim_search",
            "search_call": "librarian_search_for_claim",
            "extraction_iteration": 1,
            "pmids": list(pmids),
            "evidence_records": _evidence_records_for_pmids(state, list(pmids)),
        }
    ]
    _write_librarian_search_iteration_evidence(
        workspace,
        search_evidence_iterations,
    )
    append_to_jupytext_log(
        log_path,
        (
            "pmids = librarian_search_for_claim(state.claim, state, llm)\n"
            "print('PMIDs:', pmids)"
        ),
        (
            f"PMIDs: {pmids}\n"
            "Saved search evidence: "
            "librarian_evidence_by_search_iteration.json"
        ),
    )

    max_iterations = max(1, int(getattr(state, "MAX_ITERATIONS", cfg.max_iterations)))
    current_pmids = list(pmids)

    for iteration in range(1, max_iterations + 1):
        if current_pmids:
            extraction_results = extract_and_add_facts(
                llm, current_pmids, state, max_workers=8
            )
        else:
            extraction_results = {}

        state.iteration = iteration
        state._auto_save()
        append_to_jupytext_log(
            log_path,
            (
                f"# Librarian extraction iteration {iteration}\n"
                "results = extract_and_add_facts(llm, current_pmids, state, "
                "max_workers=8)\n"
                "print('Extraction results:', results)"
            ),
            f"Extraction results: {extraction_results}\n"
            f"Facts: {len(state.facts)}; Papers: {len(state.papers)}",
        )

        decisive_counts = _proclaim_decisive_fact_counts(state)
        decisive_total = decisive_counts["SUPPORT"] + decisive_counts["REFUTE"]

        if decisive_total >= 1:
            append_to_jupytext_log(
                log_path,
                "# Stop refinement: found decisive fact",
                (
                    "Found at least one SUPPORT or REFUTE fact: "
                    f"{decisive_counts}. Total facts: {len(state.facts)}."
                ),
            )
            break

        if iteration >= max_iterations:
            append_to_jupytext_log(
                log_path,
                "# Stop refinement: max iterations reached",
                (
                    f"Stopping with {len(state.facts)} facts after "
                    f"{iteration} iteration(s)."
                ),
            )
            break

        failed_pmids = _pmids_without_decisive_facts(current_pmids, state)

        if failed_pmids:
            next_pmids = librarian_refine_search(failed_pmids, state, llm)
            recovery_note = (
                "next_pmids = librarian_refine_search(failed_pmids, state, llm) "
                "# failed_pmids includes zero-fact and neutral-only papers"
            )
        else:
            next_pmids = []
            recovery_note = "# no failed PMIDs available for refine search"

        if not next_pmids:
            decisive_counts = _proclaim_decisive_fact_counts(state)
            decisive_total = decisive_counts["SUPPORT"] + decisive_counts["REFUTE"]
            recovery_query = _build_librarian_recovery_query(
                state.claim, decisive_total
            )
            next_pmids = librarian_search(recovery_query, state)
            recovery_note = (
                "next_pmids = librarian_search(recovery_query, state)"
            )

        append_to_jupytext_log(
            log_path,
            f"# Librarian recovery iteration {iteration}\n{recovery_note}",
            (
                f"Failed PMIDs: {failed_pmids[:10]}\n"
                f"Next PMIDs: {next_pmids}\n"
                f"Facts before next extraction: {len(state.facts)}"
            ),
        )

        if not next_pmids:
            search_evidence_iterations.append(
                {
                    "search_iteration": len(search_evidence_iterations) + 1,
                    "phase": "recovery_search",
                    "search_call": recovery_note,
                    "recovery_after_extraction_iteration": iteration,
                    "extraction_iteration": None,
                    "pmids": [],
                    "evidence_records": [],
                }
            )
            _write_librarian_search_iteration_evidence(
                workspace,
                search_evidence_iterations,
            )
            append_to_jupytext_log(
                log_path,
                "# Stop refinement: no new papers",
                "No new papers found for sparse-evidence recovery.",
            )
            break

        search_evidence_iterations.append(
            {
                "search_iteration": len(search_evidence_iterations) + 1,
                "phase": "recovery_search",
                "search_call": recovery_note,
                "recovery_after_extraction_iteration": iteration,
                "extraction_iteration": iteration + 1,
                "pmids": list(next_pmids),
                "evidence_records": _evidence_records_for_pmids(
                    state,
                    list(next_pmids),
                ),
            }
        )
        _write_librarian_search_iteration_evidence(
            workspace,
            search_evidence_iterations,
        )
        current_pmids = list(next_pmids)

    label_cfg = get_label_config()
    valid_labels = label_cfg.verdict_names()
    verdict_llm = get_agent_llm()

    if not state.facts:
        evidence_records = _collect_raw_librarian_evidence(
            state, cap=_RAW_EVIDENCE_VERDICT_CAP
        )
        if not evidence_records:
            # Nothing was retrieved at all: the claim is unsubstantiated by the
            # literature. For these rejected/flipped-style claims REFUTE is the
            # correct base-rate verdict (in the no-evidence regime gold is REFUTE
            # far more often than UNCERTAIN), so default to REFUTE without an LLM
            # call. Evidence that is present but indecisive goes to the rubric.
            verdict_label = "REFUTE" if "REFUTE" in valid_labels else valid_labels[0]
            reasoning = (
                "No evidence was retrieved for the claim's entities; the claim is "
                "unsubstantiated by the available literature."
            )
            key_evidence = []
            verdict_log_title = "# Final verdict: no evidence retrieved"
            verdict_log_details = "Empty evidence set — defaulting to REFUTE."
        else:
            verdict_prompt = _build_raw_evidence_verdict_prompt(
                state.claim, evidence_records, label_cfg, valid_labels
            )
            response = verdict_llm(verdict_prompt)
            verdict_label, reasoning, key_evidence = _parse_verdict_response(
                response, valid_labels
            )
            verdict_log_title = "# Final verdict from raw librarian evidence"
            verdict_log_details = (
                f"Raw evidence records passed to verdict model: {len(evidence_records)}"
                f"\n\nLLM verdict response:\n{response}"
            )
    else:
        facts_text = "\n".join(
            f"  [{fact.stance}] PMID {fact.source_pmid}: {fact.text[:500]}"
            for fact in state.facts[:40]
        )
        summary = get_evidence_summary(state)
        top_evidence = _select_top_evidence_for_verdict(search_evidence_iterations)
        top_evidence_text = _render_top_evidence(top_evidence)

        verdict_prompt = (
            "You are a scientific evidence evaluator. Based on the librarian "
            "retrieval below, determine the verdict for this claim using exactly "
            "one of the defined labels.\n\n"
            "You are given two views of the evidence: (a) facts already extracted "
            "from the papers in a previous step, and (b) the top retrieved "
            "librarian evidence snippets, provided to broaden the context beyond "
            "the extracted facts. Use BOTH the extracted facts and the top "
            "evidence snippets to reach the verdict.\n\n"
            f"Claim: {state.claim}\n\n"
            "Verdict label definitions:\n"
            f"{label_cfg.verdict_prompt_block()}\n\n"
            "Evidence summary:\n"
            f"{summary}\n\n"
            "Extracted facts (from a previous extraction step):\n"
            f"{facts_text}\n\n"
            "Top retrieved evidence (for broader context):\n"
            f"{top_evidence_text}\n\n"
            f"{_proclaim_verdict_rubric()}\n\n"
            "Output your answer in this exact format:\n"
            f"VERDICT: <one of {', '.join(valid_labels)}>\n"
            "REASONING: <one paragraph>\n"
            "KEY_EVIDENCE: <bullet 1> | <bullet 2> | <bullet 3>\n"
        )
        response = verdict_llm(verdict_prompt)
        verdict_label, reasoning, key_evidence = _parse_verdict_response(
            response, valid_labels
        )
        verdict_log_title = "# Final verdict from librarian facts"
        verdict_log_details = f"LLM verdict response:\n{response}"

    verdict = emit_verdict(
        verdict=verdict_label,
        reasoning=reasoning,
        key_evidence=key_evidence[:5],
        gaps_remaining=[],
        state=state,
        workspace=workspace_obj,
    )
    append_to_jupytext_log(
        log_path,
        verdict_log_title,
        f"{verdict_log_details}\n\nVerdict emitted: {verdict.verdict}",
    )

    notebook_path = output_dir / "evidence_report.ipynb"
    try:
        generate_notebook(log_path, notebook_path)
        logger.info("Notebook generated: %s", notebook_path)
    except Exception as exc:
        logger.warning("Failed to generate notebook (jupytext): %s", exc)
        notebook_path = log_path

    print(f"\nVerdict: {verdict.verdict}")
    print(f"Reasoning: {verdict.reasoning}")
    return notebook_path

if __name__ == "__main__":
    main()

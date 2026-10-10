"""Run a fresh Claude Code, Codex, or Antigravity session for one rendered prompt.

The child CLI receives the prompt through stdin and its raw response is returned
to ``CliClient`` (``search.py``), which hands it to ``LibrarianAgent`` as an LLM reply.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROVIDERS = ("claude", "codex", "antigravity")


class CliSessionError(RuntimeError):
    """A child CLI session failed.

    A ``RuntimeError``, not ``SystemExit``, so it fails one LLM call: the agent's
    judge logs it and splits the batch, and only a query-planning failure ends
    the run (``search.py`` turns that into the exit message).
    """


# Codex and AGY need a fixed reply schema, but the agent's prompts do not say
# which reply they want in a machine-readable way. One schema carries both keys
# the agent reads (query planning: ``queries``; judge: ``relevant_ids``); the
# prompt names the one to fill, and the other comes back empty and is ignored.
# Codex's strict schemas require every property, hence both in ``required``.
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        key: {"type": "array", "items": {"type": "string"}}
        for key in ("queries", "relevant_ids")
    },
    "required": ["queries", "relevant_ids"],
    "additionalProperties": False,
}


# Where installers and desktop apps put each CLI when it is not on the host's
# PATH: the official installers write to ~/.local/bin, and the ChatGPT/Codex
# desktop apps ship a working codex binary that shares ~/.codex authentication.
_FALLBACK_PATHS = {
    "claude": ["~/.local/bin/claude"],
    "codex": [
        "/Applications/ChatGPT.app/Contents/Resources/codex",
        "/Applications/Codex.app/Contents/Resources/codex",
    ],
    "agy": ["~/.local/bin/agy"],
}


def _find_cli(command: str) -> Optional[str]:
    executable = shutil.which(command)
    # Python 3.10/3.11 do not apply PATHEXT when given a full path.
    if executable is None and os.name == "nt" and os.path.dirname(command) and not os.path.splitext(command)[1]:
        for extension in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";"):
            executable = shutil.which(command + extension.lower())
            if executable:
                break
    return executable


def _require_cli(provider: str) -> str:
    """Return the installed CLI executable or stop with an actionable error."""
    cli = "agy" if provider == "antigravity" else provider
    override = os.environ.get("LIBRARIAN_CLI_PATH", "").strip()
    if override:
        executable = _find_cli(str(Path(override).expanduser()))
        if executable:
            return executable
        raise CliSessionError(f"LIBRARIAN_CLI_PATH does not name an executable: {override}")
    executable = _find_cli(cli)
    if executable is not None:
        return executable
    if provider == "claude":
        bundled = os.environ.get("CLAUDE_CODE_EXECPATH", "").strip()
        if bundled:
            executable = _find_cli(bundled)
            if executable:
                return executable
    for candidate in _FALLBACK_PATHS[cli]:
        # which also checks Windows PATHEXT (.exe/.cmd) for full paths.
        executable = _find_cli(str(Path(candidate).expanduser()))
        if executable:
            return executable
    raise CliSessionError(
        f"{provider} CLI ({cli}) is not installed or not on PATH. "
        "Calling agent: follow SKILL.md setup recovery. Locate or install this CLI "
        "through the host's normal permission flow, set LIBRARIAN_CLI_PATH if needed, "
        "and retry. If the user did not choose a provider, you may announce and try "
        "one installed alternative with --provider."
    )


def _claude_command(executable: str) -> List[str]:
    """Build a one-turn Claude Code command with no inherited project context.

    Thinking is the latency knob. Both prompts ask for a single JSON object from
    material already in front of the model — the planner picks query strings, the
    judge ranks paragraphs it can read — so extended thinking buys nothing. The
    child environment sets ``MAX_THINKING_TOKENS`` from
    ``LIBRARIAN_CLAUDE_THINKING_TOKENS`` (default ``0``): ``--effort low`` alone
    still spent 5-11k hidden tokens (75-105 s) on a ~100-token query reply, and
    ``0`` brings it to ~4 s. ``--effort`` is kept as a secondary cap, and keeps a
    large batch clear of the 180-second timeout in ``_run_command``. Override with
    ``LIBRARIAN_CLAUDE_EFFORT`` (low, medium, high, xhigh, max). An empty value
    for either variable restores the CLI default.
    """
    command = [
        executable,
        "-p",
        "--system-prompt",
        "Return only the JSON object requested by the user prompt. Do not use tools.",
        "--safe-mode",
        "--restricted",
        "--no-session-persistence",
        "--max-turns",
        "1",
        "--output-format",
        "text",
    ]
    effort = os.environ.get("LIBRARIAN_CLAUDE_EFFORT", "low").strip()
    if effort:
        command += ["--effort", effort]
    return command


def _antigravity_command(executable: str) -> List[str]:
    """Build a one-turn AGY command on a fast, explicitly chosen model.

    AGY bakes the reasoning tier into the model slug, so picking the model is
    picking the effort. On a 71-paragraph metformin judge batch, mean of two
    runs each, measured against Claude's 56 selected ids as the reference:

    ==========================  ========  ======
    model                       time      recall
    ==========================  ========  ======
    ``gemini-3.7-flash-low``    ~17 s     38%
    ``gemini-3.7-flash-medium`` ~35 s     45%
    ``gemini-3.7-flash-high``   ~35 s     58%
    ==========================  ========  ======

    ``medium`` is dominated — it costs what ``high`` costs and returns less —
    so the useful choice is the two ends. The default takes ``low`` for roughly
    twice the speed, accepting that the judge surfaces less evidence; this step
    is recall-oriented, so a miss drops a paragraph from the final answer for
    good. Set ``LIBRARIAN_ANTIGRAVITY_MODEL=gemini-3.7-flash-high`` when a
    question deserves the fuller sweep, or to an empty value to restore the
    account default. AGY latency also carries heavy provider variance (an
    identical batch ran 7.5 s once and 67.4 s on a rerun), so treat these times
    as ranks, not guarantees. ``LIBRARIAN_ANTIGRAVITY_EFFORT`` (low, medium,
    high) remains available for slugs that carry no tier suffix.
    """
    command = [
        executable,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--json-schema",
        json.dumps(_RESPONSE_SCHEMA),
    ]
    model = os.environ.get("LIBRARIAN_ANTIGRAVITY_MODEL", "gemini-3.7-flash-low").strip()
    if model:
        command += ["--model", model]
    effort = os.environ.get("LIBRARIAN_ANTIGRAVITY_EFFORT", "").strip()
    if effort:
        command += ["--effort", effort]
    return command


def _codex_command(
    executable: str,
    prompt_path: Path,
    schema_path: Path,
    response_path: Path,
) -> List[str]:
    """Build an isolated, low-latency Codex command.

    These sessions only return one small JSON object, so Luna at low effort is
    sufficient and materially faster. Empty environment values restore the
    Codex CLI defaults.
    """
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "-C",
        str(prompt_path.parent),
        "-s",
        "read-only",
    ]
    model = os.environ.get("LIBRARIAN_CODEX_MODEL", "gpt-5.6-luna").strip()
    effort = os.environ.get("LIBRARIAN_CODEX_EFFORT", "low").strip()
    if model:
        command += ["-m", model]
    if effort:
        command += ["-c", f'model_reasoning_effort="{effort}"']
    command += [
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
    ]
    command.append("-")
    return command


def _codex_environment(runtime_path: Path) -> Dict[str, str]:
    """Return a writable, isolated Codex home for one child CLI session.

    Codex initializes its state database even for an ephemeral, read-only run.
    A skill can be invoked from a sandbox where the user's normal ``~/.codex``
    directory is read-only, so the launcher must not rely on that directory.

    :param runtime_path: Temporary directory owned by the current child session.
    :type runtime_path: Path
    :return: Child-process environment with a writable ``CODEX_HOME``.
    :rtype: dict[str, str]
    """
    configured_home = os.environ.get("LIBRARIAN_CODEX_HOME", "").strip()
    codex_home = (
        Path(configured_home).expanduser()
        if configured_home
        else runtime_path / "codex-home"
    )
    codex_home.mkdir(parents=True, exist_ok=True)
    # Keep signed policy caches alongside auth so restricted children can start.
    source_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    for filename in (
        "auth.json",
        "cloud-config-bundle-cache.json",
        "cloud-requirements-cache.json",
    ):
        source = source_home / filename
        target = codex_home / filename
        try:
            if source.exists() and source.resolve() != target.resolve():
                shutil.copy2(source, target)
                target.chmod(0o600)
        except OSError as error:
            raise CliSessionError(
                f"Codex authentication/policy cache could not be read or copied: {error}. "
                "Calling agent: request approved file access through the host's normal "
                "permission flow, then retry the search."
            ) from None
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home)
    return environment


def _run_command(
    command: List[str],
    prompt_path: Path,
    environment: Optional[Dict[str, str]] = None,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    """Run one CLI session with the rendered prompt streamed from disk.

    :param command: Fully formed provider CLI command.
    :param prompt_path: Rendered brief provided as the CLI standard input.
    :param environment: Optional environment overrides for the child process.
    :param timeout: Seconds before the child is stopped.
    :return: Completed child process result.
    :rtype: subprocess.CompletedProcess[str]
    """
    print(f"[Librarian] Starting model session ({timeout:.0f}-second timeout).", file=sys.stderr)
    with prompt_path.open(encoding="utf-8") as prompt_file:
        try:
            return subprocess.run(
                command,
                stdin=prompt_file,
                capture_output=True,
                env=environment,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # Raw CLI diagnostics may echo the prompt; keep that context private.
            raise CliSessionError(
                f"Model session timed out after {timeout:.0f} seconds and was stopped. "
                "Check model-provider connectivity and retry with approved network "
                "access. Do not keep waiting or fall back to web search."
            ) from None
        except OSError as error:
            raise CliSessionError(f"Could not start provider CLI: {error}") from None


def _antigravity_response(raw_output: str) -> str:
    """Extract the single completed turn from AGY's documented NDJSON stream."""
    try:
        events = [json.loads(line) for line in raw_output.splitlines() if line.strip()]
        results = [event["result"] for event in events if event.get("event") == "result"]
        if len(results) != 1 or results[0].get("status") != "SUCCESS":
            raise ValueError
        structured = results[0].get("structured_output")
        if isinstance(structured, dict):
            return json.dumps(structured)
        response = results[0]["response"]
        if not isinstance(response, str):
            raise ValueError
        return response
    except (ValueError, KeyError, TypeError, AttributeError):
        raise CliSessionError("Antigravity did not return one successful model response.") from None


@functools.cache
def _check_codex_network() -> None:
    """Fail fast if chatgpt.com is unreachable, before Codex's long reconnect loop.

    Cached, so it probes once per process rather than once per judge batch; a
    failure raises and is not cached, so a later call probes again.
    """
    print("[Librarian] Checking Codex network access...", file=sys.stderr)
    try:
        with socket.create_connection(("chatgpt.com", 443), timeout=5):
            pass
    except OSError as error:
        raise CliSessionError(
            f"Codex cannot reach chatgpt.com:443: {error}. "
            "Retry this command with approved network access "
            "(exec_command sandbox_permissions='require_escalated'), "
            "or run it in your normal terminal. Do not fall back to web search."
        ) from None


def run_direct_session(provider: str, prompt_path: Path, timeout: float = 180) -> str:
    """Run a fresh CLI model session and return its raw response text.

    :param provider: CLI provider: ``claude``, ``codex``, or ``antigravity``.
    :param prompt_path: Rendered prompt file streamed directly to the child CLI.
    :param timeout: Seconds before the child session is stopped.
    :return: The model's reply, unparsed (``LibrarianAgent`` parses it).
    :rtype: str
    :raises CliSessionError: The CLI is missing, timed out, exited non-zero, or
        returned no usable reply.
    """
    if provider not in PROVIDERS:
        raise CliSessionError(f"provider must be one of {', '.join(PROVIDERS)}.")
    executable = _require_cli(provider)
    if provider == "codex":
        _check_codex_network()
    if provider == "antigravity":
        # AGY accepts stdin prompts as user events; EOF ends this fresh session.
        with tempfile.TemporaryDirectory(dir=prompt_path.parent) as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            input_path.write_text(json.dumps({
                "event": "user",
                "message": {"content": prompt_path.read_text(encoding="utf-8")},
            }) + "\n", encoding="utf-8")
            completed = _run_command(
                _antigravity_command(executable), input_path, timeout=timeout
            )
        raw_response = completed.stdout
    elif provider == "claude":
        thinking = os.environ.get("LIBRARIAN_CLAUDE_THINKING_TOKENS", "0").strip()
        environment = dict(os.environ)
        if thinking:
            environment["MAX_THINKING_TOKENS"] = thinking
        completed = _run_command(
            _claude_command(executable), prompt_path, environment, timeout
        )
        raw_response = completed.stdout
    elif provider == "codex":
        with tempfile.TemporaryDirectory(dir=prompt_path.parent) as temporary_directory:
            temporary_path = Path(temporary_directory)
            schema_path = temporary_path / "response_schema.json"
            response_path = temporary_path / "response.json"
            schema_path.write_text(json.dumps(_RESPONSE_SCHEMA), encoding="utf-8")
            command = _codex_command(
                executable,
                prompt_path,
                schema_path,
                response_path,
            )
            environment = _codex_environment(temporary_path)
            completed = _run_command(command, prompt_path, environment, timeout)
            raw_response = (
                response_path.read_text(encoding="utf-8")
                if response_path.exists()
                else ""
            )
    diagnostics = completed.stderr + "\n" + completed.stdout
    if provider == "claude" and any(
        marker in diagnostics.lower()
        for marker in ("not logged in", "please run /login", "authentication required")
    ):
        raise CliSessionError(
            "The Claude CLI used by Librarian is not signed in. "
            "Calling agent: apply SKILL.md authentication recovery. If sign-in is "
            f"required, launch \"{executable}\" in a host terminal when available, "
            "ask the user only to complete /login, then resume the search. "
            "The desktop app's login may not authenticate a separate child CLI."
        )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise CliSessionError(f"{provider} session failed: {message}")
    if provider == "antigravity":
        raw_response = _antigravity_response(raw_response)
    return raw_response

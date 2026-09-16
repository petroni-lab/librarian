"""Run an isolated Claude Code or Codex CLI session for one rendered prompt.

The root agent passes paths to this module, never the prompt contents. The child
CLI receives the prompt through stdin and its final JSON response is validated
and written locally, avoiding model Read and Write tool calls.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import _runs
from librarian.llm_client import parse_json_response


def _response_schema(key: str) -> Dict[str, Any]:
    """Build the Codex schema for the single list-valued response field."""
    return {
        "type": "object",
        "properties": {key: {"type": "array", "items": {"type": "string"}}},
        "required": [key],
        "additionalProperties": False,
    }


def _require_cli(provider: str) -> str:
    """Return the installed CLI executable or stop with an actionable error."""
    executable = shutil.which(provider)
    if executable is None:
        raise SystemExit(f"{provider} CLI is not installed or not on PATH.")
    return executable


def _validate_response(raw_response: str, key: str) -> Dict[str, List[str]]:
    """Parse and validate the one-field JSON object returned by the model."""
    parsed = parse_json_response(raw_response)
    values = parsed.get(key) if isinstance(parsed, dict) else None
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise SystemExit(
            f"Model response must be a JSON object with a string-array '{key}'."
        )
    return {key: values}


def _claude_command(executable: str) -> List[str]:
    """Build a one-turn Claude Code command with no inherited project context."""
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
    return command


def _codex_command(
    executable: str,
    prompt_path: Path,
    schema_path: Path,
    response_path: Path,
) -> List[str]:
    """Build an isolated Codex command whose final message goes to a file."""
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
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
            target.chmod(0o600)
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home)
    return environment


def _run_command(
    command: List[str], prompt_path: Path, environment: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess[str]:
    """Run one CLI session with the rendered prompt streamed from disk.

    :param command: Fully formed provider CLI command.
    :param prompt_path: Rendered brief provided as the CLI standard input.
    :param environment: Optional environment overrides for the child process.
    :return: Completed child process result.
    :rtype: subprocess.CompletedProcess[str]
    """
    print("[Librarian] Starting model session (180-second timeout).", flush=True)
    with prompt_path.open(encoding="utf-8") as prompt_file:
        try:
            return subprocess.run(
                command,
                stdin=prompt_file,
                capture_output=True,
                env=environment,
                text=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            # Raw CLI diagnostics may echo the prompt; keep that context private.
            raise SystemExit(
                "Model session timed out after 180 seconds and was stopped. "
                "Check model-provider connectivity and retry with approved network "
                "access. Do not keep waiting or fall back to web search."
            ) from None


def run_direct_session(
    provider: str,
    prompt_path: Path,
    output_path: Path,
    output_key: str,
) -> None:
    """Run a fresh CLI model session and save its validated JSON response.

    :param provider: CLI provider, either ``claude`` or ``codex``.
    :param prompt_path: Rendered prompt file streamed directly to the child CLI.
    :param output_path: JSON hand-off file consumed by the next pipeline step.
    :param output_key: Required list field in the model response.
    """
    if provider not in ("claude", "codex"):
        raise SystemExit("provider must be 'claude' or 'codex'.")
    executable = _require_cli(provider)
    if provider == "codex":
        # The isolated Codex CLI uses ChatGPT; fail before its long reconnect loop.
        print("[Librarian] Checking Codex network access...", flush=True)
        try:
            with socket.create_connection(("chatgpt.com", 443), timeout=5):
                pass
        except OSError as error:
            raise SystemExit(
                f"Codex cannot reach chatgpt.com:443: {error}. "
                "Retry this command with approved network access "
                "(exec_command sandbox_permissions='require_escalated'), "
                "or run it in your normal terminal. Do not fall back to web search."
            ) from None
    if provider == "claude":
        completed = _run_command(_claude_command(executable), prompt_path)
        raw_response = completed.stdout
    elif provider == "codex":
        with tempfile.TemporaryDirectory(dir=prompt_path.parent) as temporary_directory:
            temporary_path = Path(temporary_directory)
            schema_path = temporary_path / "response_schema.json"
            response_path = temporary_path / "response.json"
            _runs.write_json(schema_path, _response_schema(output_key))
            command = _codex_command(
                executable,
                prompt_path,
                schema_path,
                response_path,
            )
            environment = _codex_environment(temporary_path)
            completed = _run_command(command, prompt_path, environment)
            raw_response = (
                response_path.read_text(encoding="utf-8")
                if response_path.exists()
                else ""
            )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"{provider} session failed: {message}")
    _runs.write_json(output_path, _validate_response(raw_response, output_key))

"""macOS sandbox checks with installed CLIs, credentials, and network blocked.

Run with pytest and Librarian dependencies available. Stub CLIs test launcher
recovery only; they never authenticate or contact a model provider.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/librarian/scripts"
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS sandbox required"
)


@pytest.fixture
def sandbox(tmp_path):
    home = Path.home()
    blocked = [
        home / path
        for path in (".local/bin", ".cargo/bin", ".codex", ".claude", ".gemini")
    ]
    blocked += [
        tmp_path / "blocked-home",
        Path("/Applications/Codex.app/Contents/Resources/codex"),
        Path("/Applications/ChatGPT.app/Contents/Resources/codex"),
        Path("/opt/homebrew/bin/codex"),
    ]
    profile = "(version 1) (allow default) (deny network*)\n"
    for path in blocked:
        profile += f"(deny file-read* (subpath {json.dumps(str(path))}))\n"
        profile += f"(deny process-exec (subpath {json.dumps(str(path))}))\n"
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),  # Preserve HOME; access is restricted by the sandbox.
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(SCRIPTS),
    }

    def run(command, **overrides):
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, *command],
            env={**environment, **overrides},
            cwd=tmp_path,
            capture_output=True,
            encoding="utf-8",
            check=False,
            timeout=30,
        )

    return run


@pytest.mark.parametrize("provider", ("claude", "codex", "antigravity"))
def test_missing_uv_stops_at_python_setup(sandbox, provider):
    result = sandbox(
        [
            "/bin/bash",
            str(SCRIPTS / "run.sh"),
            "search",
            "--provider",
            provider,
            "β cells in mice",
        ]
    )
    assert result.returncode == 127, result.stderr
    assert "need uv" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("provider", ("claude", "codex", "antigravity"))
def test_missing_cli_stops_before_retrieval(sandbox, provider):
    result = sandbox(
        [
            "/bin/bash",
            str(SCRIPTS / "run.sh"),
            "search",
            "--provider",
            provider,
            "β cells in mice",
        ],
        LIBRARIAN_PYTHON=sys.executable,
    )
    assert result.returncode == 1, result.stderr
    assert f"Model/CLI setup: {provider} CLI" in result.stderr
    assert "--provider" in result.stderr
    assert "Starting model session" not in result.stderr
    assert not result.stdout


def make_stub(tmp_path, authenticated=True):
    executable = tmp_path / "stub-provider"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "prompt = sys.stdin.read()\n"
        "if '--input-format' in sys.argv: prompt = json.loads(prompt)['message']['content']\n"
        "assert 'β cells' in prompt\n"
        + (
            "print('Not logged in · Please run /login', file=sys.stderr)\nsys.exit(1)\n"
            if not authenticated
            else "payload = {'queries': ['β cells'], 'relevant_ids': []}\n"
            "if '--output-last-message' in sys.argv:\n"
            "    target = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
            "    pathlib.Path(target).write_text(json.dumps(payload), encoding='utf-8')\n"
            "elif '--input-format' in sys.argv:\n"
            "    print(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS', 'structured_output': payload}}))\n"
            "else:\n"
            "    print(json.dumps(payload))\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("provider", ("claude", "codex", "antigravity"))
def test_explicit_cli_path_recovers_with_stub(sandbox, tmp_path, provider):
    executable = make_stub(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("β cells", encoding="utf-8")
    probe = (
        "import os, sys; from pathlib import Path; import _direct_session as s; "
        # Stubs do not need connectivity or credentials. Those remain blocked.
        "s._check_codex_network = lambda: None; "
        "s._codex_environment = lambda p: dict(os.environ); "
        "print(s.run_direct_session(sys.argv[1], Path(sys.argv[2])))"
    )
    result = sandbox(
        [sys.executable, "-c", probe, provider, str(prompt)],
        LIBRARIAN_CLI_PATH=str(executable),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"queries": ["β cells"], "relevant_ids": []}




def test_codex_blocked_credentials_have_actionable_error(sandbox, tmp_path):
    fake_home = tmp_path / "blocked-home"
    credentials = fake_home / ".codex"
    credentials.mkdir(parents=True)
    (credentials / "auth.json").write_text("{}", encoding="utf-8")
    probe = (
        "import sys; from pathlib import Path; import _direct_session as s\n"
        "s.Path.home = lambda: Path(sys.argv[1])\n"
        "try: s._codex_environment(Path(sys.argv[2]))\n"
        "except s.CliSessionError as error: print(error)\n"
    )
    result = sandbox(
        [sys.executable, "-c", probe, str(fake_home), str(tmp_path / "runtime")]
    )
    assert result.returncode == 0, result.stderr
    assert (
        "Codex authentication/policy cache could not be read or copied" in result.stdout
    )
    assert "approved file access" in result.stdout

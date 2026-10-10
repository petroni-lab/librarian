"""Offline regressions for desktop CLI discovery and Windows skill setup."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/librarian/scripts"
sys.path.insert(0, str(SCRIPTS))
import _direct_session as sessions


class SessionTests(unittest.TestCase):
    def setUp(self):
        system_environment = {
            key: os.environ[key]
            for key in ("HOME", "USERPROFILE", "SYSTEMROOT", "PATHEXT")
            if key in os.environ
        }
        self.env = patch.dict(os.environ, system_environment, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_claude_desktop_binary(self):
        with (
            patch.dict(os.environ, {"CLAUDE_CODE_EXECPATH": "/desktop/claude.exe"}),
            patch.object(
                shutil,
                "which",
                side_effect=lambda p: p if p == "/desktop/claude.exe" else None,
            ),
        ):
            self.assertEqual(sessions._require_cli("claude"), "/desktop/claude.exe")

    def test_path_has_priority_over_desktop(self):
        with (
            patch.dict(os.environ, {"CLAUDE_CODE_EXECPATH": "/desktop/claude.exe"}),
            patch.object(shutil, "which", return_value="/signed-in/claude"),
        ):
            self.assertEqual(sessions._require_cli("claude"), "/signed-in/claude")

    def test_explicit_path_and_invalid_override(self):
        with patch.dict(os.environ, {"LIBRARIAN_CLI_PATH": "/custom/agy.exe"}):
            with patch.object(shutil, "which", return_value="/custom/agy.exe"):
                self.assertEqual(
                    sessions._require_cli("antigravity"), "/custom/agy.exe"
                )
            with (
                patch.object(shutil, "which", return_value=None),
                self.assertRaisesRegex(sessions.CliSessionError, "LIBRARIAN_CLI_PATH"),
            ):
                sessions._require_cli("antigravity")

    def test_antigravity_windows_installer_path(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": "/local app data"}):
            expected = str(Path("/local app data/agy/bin/agy"))
            with patch.object(
                shutil, "which", side_effect=lambda p: p if p == expected else None
            ):
                self.assertEqual(sessions._require_cli("antigravity"), expected)

    def test_missing_cli_explains_provider_selection(self):
        with (
            patch.object(shutil, "which", return_value=None),
            self.assertRaisesRegex(sessions.CliSessionError, "--provider"),
        ):
            sessions._require_cli("antigravity")

    def test_fallback_uses_platform_executable_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "agy"
            executable = base.with_suffix(".cmd") if os.name == "nt" else base
            executable.write_text(
                "@echo off\n" if os.name == "nt" else "#!/bin/sh\n", encoding="utf-8"
            )
            executable.chmod(0o755)
            # PATHEXT is normally supplied by Windows, but setUp clears the env.
            with (
                patch.dict(os.environ, {"PATHEXT": ".COM;.EXE;.BAT;.CMD"}),
                patch.dict(sessions._FALLBACK_PATHS, {"agy": [str(base)]}),
            ):
                self.assertEqual(sessions._require_cli("antigravity"), str(executable))

    def test_auth_failure_on_either_stream_and_zero_exit(self):
        for code, stdout, stderr in [
            (1, "Not logged in · Please run /login", ""),
            (1, "", "Not logged in"),
            (0, "Please run /login", ""),
        ]:
            with (
                self.subTest(code=code, stderr=stderr),
                patch.object(
                    sessions, "_require_cli", return_value="/desktop/claude.exe"
                ),
                patch.object(
                    sessions,
                    "_run_command",
                    return_value=subprocess.CompletedProcess([], code, stdout, stderr),
                ),
                self.assertRaisesRegex(sessions.CliSessionError, "desktop app.*login"),
            ):
                sessions.run_direct_session("claude", Path("unused"))

    def test_unicode_subprocess_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("β cells → mice 🐁", encoding="utf-8")
            result = sessions._run_command(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-c",
                    "import sys; print(sys.stdin.read())",
                ],
                prompt,
            )
            self.assertEqual(result.stdout.strip(), "β cells → mice 🐁")

    def test_explicit_utf8_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("β", encoding="utf-8")
            with patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "β", ""),
            ) as run:
                sessions._run_command(["fake"], prompt)
                self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")

    def test_startup_oserror_is_cli_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            with self.assertRaisesRegex(
                sessions.CliSessionError, "Could not start provider CLI"
            ):
                sessions._run_command([str(Path(tmp) / "missing")], prompt)

    def test_unreadable_codex_credentials_are_cli_failure(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(Path, "exists", side_effect=PermissionError("sandbox denied")),
            self.assertRaisesRegex(sessions.CliSessionError, "approved file access"),
        ):
            sessions._codex_environment(Path(tmp))


POWERSHELL = shutil.which(
    os.environ.get("LIBRARIAN_TEST_POWERSHELL", "pwsh")
) or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "PowerShell unavailable")
class PowerShellLauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="librarian launch ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.scripts = self.root / "skills/librarian/scripts"
        self.scripts.mkdir(parents=True)
        shutil.copy2(SCRIPTS / "run.ps1", self.scripts)
        (self.scripts / "search.py").write_text(
            'import json, os, sys\nprint(json.dumps([sys.argv[1:], os.environ["PYTHONUTF8"], os.environ["PYTHONIOENCODING"]]))\nprint("β → 🐁")\n',
            encoding="utf-8",
        )
        self.environment = dict(
            os.environ, LIBRARIAN_PYTHON="", UV_PYTHON="", MOCK_SETUP="link"
        )
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.environment["MOCK_LOG"] = str(self.log)
        if os.name == "nt":
            venv.EnvBuilder().create(self.root / ".venv")
            candidate = Path(sys.executable)
        else:
            python_dir = self.root / ".venv/Scripts"
            python_dir.mkdir(parents=True)
            (python_dir / "python.exe").symlink_to(sys.executable)
            candidate = python_dir / "python.exe"
        self.environment["MOCK_CANDIDATE"] = str(candidate.parent)
        fake = self.bin / "fake_uv.py"
        fake.write_text(
            """import json, os, sys
with open(os.environ['MOCK_LOG'], 'a', encoding='utf-8') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
mode = os.environ['MOCK_SETUP']
if mode == 'generic':
    print('dependency setup failed', file=sys.stderr)
    sys.exit(3)
if '--python' not in sys.argv:
    print('error: Missing expected target directory for Python minor version link at\\n"' + os.environ['MOCK_CANDIDATE'] + '"', file=sys.stderr)
    sys.exit(2)
sys.exit(4 if mode == 'retry-fail' else 0)
""",
            encoding="utf-8",
        )
        if os.name == "nt":
            (self.bin / "uv.cmd").write_text(
                f'@"{sys.executable}" "{fake}" %*\n', encoding="utf-8"
            )
        else:
            wrapper = self.bin / "uv"
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
            wrapper.chmod(0o755)
        self.environment["PATH"] = str(self.bin) + os.pathsep + os.environ["PATH"]

    def launch(self):
        return subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-File",
                str(self.scripts / "run.ps1"),
                "search",
                "--provider",
                "codex",
                "β cells in mice",
            ],
            env=self.environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=30,
        )

    def test_link_recovery_preserves_unicode_and_arguments(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("β → 🐁", result.stdout)
        self.assertIn("β cells in mice", json.loads(result.stdout.splitlines()[0])[0])
        self.assertEqual(json.loads(result.stdout.splitlines()[0])[1:], ["1", "utf-8"])
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1][-2:],
            ["--python", str(Path(self.environment["MOCK_CANDIDATE"]) / "python.exe")],
        )

    def test_unrelated_setup_failure_is_not_retried(self):
        self.environment["MOCK_SETUP"] = "generic"
        result = self.launch()
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("Python setup failed", result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 1)

    def test_missing_interpreter_is_not_retried(self):
        self.environment["MOCK_CANDIDATE"] = str(self.root / "missing")
        result = self.launch()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 1)

    def test_failed_recovery_stops_after_one_retry(self):
        self.environment["MOCK_SETUP"] = "retry-fail"
        result = self.launch()
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 2)

    def test_python_override_skips_uv(self):
        self.environment["LIBRARIAN_PYTHON"] = sys.executable
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())

    def test_global_skill_link_resolves_repository_root(self):
        global_dir = self.root / "global-skills"
        global_dir.mkdir()
        link = global_dir / "librarian"
        target = self.scripts.parent
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True,
                capture_output=True,
            )
        else:
            link.symlink_to(target, target_is_directory=True)
        self.scripts = link / "scripts"
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls[0], ["sync", "--project", str(self.root)])


if __name__ == "__main__":
    unittest.main()

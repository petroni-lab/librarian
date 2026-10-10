"""Search entrypoint checks: fail before retrieval and print Unicode reports."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/librarian/scripts")
)
import search
from _direct_session import CliSessionError


def test_missing_cli_stops_before_agent_setup(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search", "--provider", "claude", "mice tumors"])
    monkeypatch.setattr(
        search, "_require_cli", Mock(side_effect=CliSessionError("missing CLI"))
    )
    agent = Mock()
    monkeypatch.setattr(search, "LibrarianAgent", agent)
    with pytest.raises(SystemExit, match="Model/CLI setup: missing CLI"):
        search.main()
    agent.assert_not_called()


def test_report_output_overrides_cp1252(monkeypatch, tmp_path):
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "argv", ["search", "--provider", "claude", "β cells"])
    monkeypatch.setattr(search, "_require_cli", Mock(return_value="claude"))
    agent = SimpleNamespace(
        run=Mock(return_value=["paper"]),
        last_run_debug={"search_queries": ["β cells"], "paragraph_count": 1},
    )
    monkeypatch.setattr(search, "LibrarianAgent", Mock(return_value=agent))
    monkeypatch.setattr(search, "load_runtime_config", Mock(return_value=None))
    monkeypatch.setattr(search, "render_report", Mock(return_value="β cells → mice 🐁"))
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Unicode synthesis → β", encoding="utf-8")
    monkeypatch.setattr(search, "SUMMARIZER", instructions)
    assert search.main() == 0
    stream.flush()
    text = output.getvalue().decode("utf-8")
    assert "β cells → mice 🐁" in text
    assert "Unicode synthesis → β" in text

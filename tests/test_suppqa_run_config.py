"""The SuppQA eval's run_config.json never records the answering-model key (#97).

uv run --extra supplementary pytest tests
"""

import json
import sys

import pytest

from evals.Literature.SuppQA import run_suppqa_eval as ev

_KEY = "sk-dummy-key-0123456789"


@pytest.mark.parametrize(
    "key_args",
    [
        ["--api-key", _KEY],
        [f"--api-key={_KEY}"],
        ["--api", _KEY],
        [f"--api={_KEY}"],
    ],
    ids=["separate", "equals", "abbreviated", "abbreviated-equals"],
)
def test_run_config_masks_the_api_key(tmp_path, monkeypatch, key_args):
    data = tmp_path / "q.jsonl"
    data.write_text("")
    argv = [
        "--arm", "supp", "--out-dir", str(tmp_path / "out"), "--data-file", str(data),
        "--model", "m", *key_args,
    ]
    monkeypatch.setattr(sys, "argv", ["run_suppqa_eval.py", *argv])
    args = ev.parse_args(argv)
    assert args.api_key == _KEY
    ev.run(args)

    text = (tmp_path / "out" / "run_config.json").read_text()
    assert _KEY not in text
    assert "***" in json.loads(text)["command"][-1]

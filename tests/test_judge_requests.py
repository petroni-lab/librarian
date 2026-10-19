"""Stage-3 judge requests are logged on the run's state with an outcome.

uv run python -m unittest discover tests
"""

import unittest

from librarian import LibrarianAgent, load_runtime_config
from librarian.agent import _RunState


class ScriptedLLM:
    """Replies in order; an Exception instance is raised instead of returned."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat_completion(self, messages, temperature=0, max_tokens=0):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _items(n):
    return [
        {"id": f"paragraph_{i}", "sentences": [{"id": f"paragraph_{i}_A", "text": "x"}]}
        for i in range(n)
    ]


class JudgeRequestLogTest(unittest.TestCase):
    def _evaluate(self, replies, items):
        agent = LibrarianAgent(load_runtime_config(), llm_client=ScriptedLLM(replies))
        state = _RunState()
        ids = agent._evaluate_batch(
            "q", items, "{user_query} {paragraphs_batch}", state
        )
        return ids, state.judge_requests

    def test_unparseable_reply_is_split_and_every_outcome_recorded(self):
        ids, log = self._evaluate(
            [
                "not json",
                TimeoutError("read timed out"),
                '{"relevant_ids": ["paragraph_1_A"]}',
            ],
            _items(2),
        )
        self.assertEqual(ids, ["paragraph_1_A"])
        self.assertEqual(
            [(r["depth"], r["outcome"]) for r in log],
            [(0, "unparseable"), (1, "exception"), (1, "ok")],
        )
        self.assertIn("TimeoutError", log[1]["error"])
        self.assertEqual(log[0]["error"], "")

    def test_truncated_reply_is_salvaged(self):
        ids, log = self._evaluate(
            ['{"relevant_ids": ["paragraph_0_A", "parag'], _items(1)
        )
        self.assertEqual(ids, ["paragraph_0_A"])
        self.assertEqual([r["outcome"] for r in log], ["salvaged"])

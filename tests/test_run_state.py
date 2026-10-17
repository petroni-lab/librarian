"""Per-call results and progress stay isolated when agents are shared."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from librarian import (
    LibrarianAgent,
    RunResult,
    SynthesisAgent,
    SynthesisResult,
    load_runtime_config,
)
from librarian.literature_search import LiteratureSearchError
from librarian.scope_port import ScopeVerdict


def retrieval_agent(barrier=None):
    agent = LibrarianAgent(
        runtime_config=replace(load_runtime_config(), paragraphs_per_judge_batch=1),
        llm_client=object(),
        literature_source=object(),
    )

    def generate(query):
        if barrier:
            barrier.wait(timeout=10)
        return [query]

    def paragraphs(query):
        if query == "empty":
            return []
        return [
            {
                "paper": {"pmid": f"{query}-{i}", "title": query},
                "text": f"Evidence for {query}.",
                "source_paragraph_index": 0,
                "word_start": 0,
            }
            for i in range(1 if query == "alpha" else 2)
        ]

    def evaluate(query, batch, template):
        if barrier and batch[0]["id"] == "paragraph_0":
            barrier.wait(timeout=10)
        return [sentence["id"] for item in batch for sentence in item["sentences"]]

    agent._generate_queries = generate
    agent._validate_queries = lambda queries: queries
    agent._paragraphs_for_subquery = paragraphs
    agent._evaluate_batch = evaluate
    return agent


class RunStateTests(unittest.TestCase):
    def test_concurrent_retrieval_isolates_callbacks_counters_and_debug(self):
        agent = retrieval_agent(threading.Barrier(2))
        before = vars(agent).copy()
        messages = {query: [] for query in ("alpha", "beta")}
        steps = {query: [] for query in messages}

        def run(query):
            caller = threading.get_ident()
            return agent.run(
                query,
                on_progress=messages[query].append,
                progress_callback=lambda step: steps[query].append(
                    (step, threading.get_ident() == caller)
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(zip(messages, pool.map(run, messages)))
        for query, total in (("alpha", 1), ("beta", 2)):
            result = results[query]
            self.assertIsInstance(result, RunResult)
            self.assertEqual(result.debug["search_queries"], [query])
            self.assertEqual(result.debug["paragraph_count"], total)
            self.assertEqual(result.debug["relevant_count"], total)
            self.assertEqual(
                result.debug["final_pmids"], [f"{query}-{i}" for i in range(total)]
            )
            self.assertEqual(len(result.papers), total)
            self.assertTrue(all(paper["title"] == query for paper in result.papers))
            self.assertEqual(steps[query], [(0, True), (1, True), (2, True)])
            self.assertIn(f"Found {total} candidate paragraphs", messages[query])
            batch_messages = [m for m in messages[query] if m.endswith(" batches")]
            self.assertCountEqual(
                batch_messages,
                [
                    f"Judging paragraph relevance · {i}/{total} batches"
                    for i in range(1, total + 1)
                ],
            )
        self.assertEqual(vars(agent), before)
        self.assertIsNot(results["alpha"].debug, results["beta"].debug)

    def test_empty_and_failed_runs_do_not_change_previous_result(self):
        agent = retrieval_agent()
        first = agent.run("alpha")
        empty = agent.run("empty")
        self.assertEqual(empty, RunResult([], {"search_queries": ["empty"]}))

        def cancel(step):
            raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            agent.run("beta", progress_callback=cancel)
        self.assertEqual(first.debug["search_queries"], ["alpha"])
        self.assertEqual(agent.run("beta").debug["relevant_count"], 2)
        self.assertFalse(hasattr(agent, "last_run_debug"))

    def test_failed_search_preserves_partial_debug(self):
        agent = retrieval_agent()
        error = LiteratureSearchError("search unavailable")
        with patch.object(agent, "_paragraphs_for_subquery", side_effect=error):
            with self.assertRaises(LiteratureSearchError) as caught:
                agent.run("alpha")
        self.assertIs(caught.exception, error)
        self.assertEqual(error.run_debug, {"search_queries": ["alpha"]})
        agent.run("beta")
        self.assertEqual(error.run_debug, {"search_queries": ["alpha"]})

    def test_failed_synthesis_search_preserves_route_and_subqueries(self):
        librarian = retrieval_agent()
        agent = SynthesisAgent(llm_client=object(), librarian=librarian)
        error = LiteratureSearchError("search unavailable")
        decision = {"action": "search", "query": "rewritten", "language": "English"}
        with (
            patch.object(agent, "_route", return_value=decision),
            patch.object(librarian, "_paragraphs_for_subquery", side_effect=error),
        ):
            with self.assertRaises(LiteratureSearchError) as caught:
                agent.run("original")
        self.assertIs(caught.exception, error)
        self.assertEqual(
            error.run_debug,
            {
                "action": "search",
                "query": "rewritten",
                "language": "English",
                "passages": [],
                "search_queries": ["rewritten"],
            },
        )

    def test_concurrent_synthesis_with_shared_librarian(self):
        agent = SynthesisAgent(
            llm_client=object(), librarian=retrieval_agent(threading.Barrier(2))
        )
        barrier = threading.Barrier(2)
        agent._route = lambda query, history: {
            "action": "search",
            "query": query,
            "language": query,
        }

        def summarize(query, passages, **kwargs):
            barrier.wait(timeout=10)
            answer = f"answer {query}"
            kwargs["stream_callback"](answer)
            return answer

        agent._summarize = summarize
        before = vars(agent).copy()
        streams = {query: [] for query in ("alpha", "beta")}
        steps = {query: [] for query in streams}

        def run(query):
            return agent.run(
                query,
                stream_callback=streams[query].append,
                progress_callback=steps[query].append,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(zip(streams, pool.map(run, streams)))
        for query, result in results.items():
            self.assertIsInstance(result, SynthesisResult)
            self.assertEqual(result.answer, f"answer {query}")
            self.assertEqual(result.debug["action"], "search")
            self.assertEqual(result.debug["query"], query)
            self.assertEqual(result.debug["language"], query)
            self.assertEqual(result.debug["search_queries"], [query])
            self.assertTrue(all(p["title"] == query for p in result.debug["passages"]))
            self.assertEqual(streams[query], [result.answer])
            self.assertEqual(steps[query], [0, 1, 2, 3])
        self.assertEqual(vars(agent), before)

    def test_synthesis_early_paths_return_their_own_debug(self):
        agent = SynthesisAgent(llm_client=object())
        with patch.object(agent, "_summarize", return_value="summary"):
            summary = agent.run("supplied", passages=[])
        self.assertEqual(summary.answer, "summary")
        self.assertEqual(summary.debug["action"], "summarize")
        with patch.object(
            agent,
            "_route",
            return_value={"action": "reply", "query": "rewritten", "answer": "hello"},
        ):
            reply = agent.run("direct")
        self.assertEqual(reply.answer, "hello")
        self.assertEqual(reply.debug["action"], "reply")
        self.assertEqual(reply.debug["search_queries"], [])
        with patch.object(agent, "_screen_query", return_value="stop"):
            stopped = agent.run("screened")
        self.assertEqual(stopped.answer, "stop")
        self.assertEqual(stopped.debug["action"], "stopped")
        self.assertEqual(summary.debug["query"], "supplied")
        self.assertFalse(hasattr(agent, "last_run_debug"))

    def test_partial_scope_note_is_returned_and_streamed(self):
        class Guard:
            def check(self, query, conversation_history):
                return ScopeVerdict(in_scope=True, query="accepted", note="left out")

        agent = SynthesisAgent(llm_client=object(), scope_guard=Guard())
        stream = []
        with patch.object(
            agent, "_route", return_value={"action": "reply", "answer": "answer"}
        ):
            result = agent.run("mixed", stream_callback=stream.append)
        self.assertEqual(result.answer, "answer\n\nleft out")
        self.assertEqual(result.debug["query"], "accepted")
        self.assertEqual("".join(stream), result.answer)

    def test_orchestrator_preserves_response_shape(self):
        import orchestrator

        retrieval = RunResult([{"pmid": "1"}], {"search_queries": ["subquery"]})
        with (
            patch.object(orchestrator.LibrarianAgent, "run", return_value=retrieval),
            patch.object(
                orchestrator.SynthesisAgent,
                "run",
                return_value=SynthesisResult("summary", {}),
            ),
        ):
            result = orchestrator._run(
                orchestrator.RunRequest(query="question", agent="literature_synthesis")
            )
        self.assertEqual(
            result,
            {
                "query": "question",
                "search_queries": ["subquery"],
                "papers": [{"pmid": "1"}],
                "summary": "summary",
                "action": "search",
            },
        )


if __name__ == "__main__":
    unittest.main()

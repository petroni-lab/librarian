"""Record one live Librarian run as a replayable fixture.

Runs ``LibrarianAgent.run`` once against an OpenAI-compatible LLM and plain Europe
PMC (no disk cache), capturing what its external calls returned and what each
stage produced. The pipeline is never re-implemented here — the recording is a
side effect of a normal run — so a fixture cannot drift from what the agent
actually does.

Run from the repository root, with the LLM endpoint reachable:

    uv run python tests/record_fixture.py --case <name> --query "<question>"

The API key, if the endpoint needs one, comes from ``LLM_API_KEY`` as usual and is
never written to the fixture. A failed run leaves no partial case directory
behind; the tests report an incomplete one as such.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import shutil
import sys
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fixture_replay

from librarian.config import LibrarianRuntimeConfig
from librarian.literature_search import (
    EuropePmcSource,
    Fulltext,
    LiteratureSource,
)
from librarian.llm_client import create_llm_client

DEFAULT_CONFIG = Path(__file__).parent / "fixtures" / "recording.toml"

# Recorded runs use full text: it is the expensive path, and the one whose JATS
# extraction and chunking most needs pinning.
FULL_TEXT_ENRICHMENT = True
# "low" is GLM-5.3+'s minimum reasoning_effort. Only the live run sees it (replay
# never reaches an LLM), so it is written to 00_run.json as a record, not an input.
REASONING_EFFORT = "low"


@dataclass
class BoundaryRecorder:
    """What the outside world answered during one live ``run``.

    The recording half of ``fixture_replay``'s replay pair: ``RecordingLLM`` and
    ``RecordingSource`` fill it in, and ``ReplayLLM`` / ``ReplaySource`` (in
    ``test_recorded_run.py``) later answer from it, leaving the agent's own
    behaviour completely unchanged.
    """

    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    search_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    fulltext_xml: dict[str, str] = field(default_factory=dict)
    # Stage 2's search and full-text fetches, and Stage 3's judge calls, all run on
    # thread pools, so every collection here is written concurrently.
    lock: threading.Lock = field(default_factory=threading.Lock)


class RecordingLLM:
    """An ``llm_client`` that forwards to a real one and records every answer."""

    def __init__(self, inner: Any, recorder: BoundaryRecorder):
        self._inner = inner
        self._recorder = recorder

    def chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Record one LLM call under its replay key, with the parameters used."""
        response = self._inner.chat_completion(messages, **kwargs)
        with self._recorder.lock:
            self._recorder.llm_calls.append(
                {
                    "key": fixture_replay.llm_call_key(messages),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "response": response,
                }
            )
        return response


class RecordingSource:
    """A ``LiteratureSource`` that forwards to a real one and records every answer.

    Has no ``fetch_supplementary_many``: supplementary files are not recorded, and
    ``record_fixture`` refuses a config that would ask for them.
    """

    def __init__(self, inner: LiteratureSource, recorder: BoundaryRecorder):
        self._inner = inner
        self._recorder = recorder

    def search(self, query: str, page_size: int) -> list[dict[str, Any]]:
        """Record one Europe PMC search, keyed by the sub-query that ran it.

        Snapshotted as returned: Stage 3 later writes the evidence fields onto
        these same paper dicts, and a recording holding them would hand replay
        the very evidence it is meant to rebuild.
        """
        papers = self._inner.search(query, page_size=page_size)
        with self._recorder.lock:
            self._recorder.search_results[query] = copy.deepcopy(papers)
        return papers

    def fetch_fulltext_many(self, pmcids: Iterable[str]) -> dict[str, Fulltext]:
        """Record each paper's JATS XML in this batch, keyed by the id asked for.

        Two sub-queries can fetch the same paper. Replay answers both from one
        file, so a paper whose fetches came back differently (one succeeded, one
        failed) cannot be replayed, and the recording stops instead.

        :raises RuntimeError: if this paper was already fetched with another outcome.
        """
        results = self._inner.fetch_fulltext_many(pmcids)
        with self._recorder.lock:
            for pmcid, result in results.items():
                xml_text = result.xml if result.ok else ""
                recorded = self._recorder.fulltext_xml.setdefault(pmcid, xml_text)
                if recorded != xml_text:
                    raise RuntimeError(
                        f"{pmcid} was fetched twice with different outcomes, which "
                        f"a replay cannot reproduce. Re-run the recording."
                    )
        return results


def _case_directory(case_id: str) -> Path:
    """Resolve the case directory, keeping every write inside the fixture root."""
    if case_id in {"", ".", ".."} or Path(case_id).name != case_id:
        raise ValueError("case must be a single directory name")
    return fixture_replay.CASE_ROOT / case_id


def _check_case_directory(target: Path, overwrite: bool) -> None:
    """Fail before the run starts if this case cannot be written afterwards."""
    if target.is_symlink():
        raise ValueError(f"fixture path cannot be a symbolic link: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"fixture already exists: {target}")


def _replace_case_directory(target: Path) -> None:
    """Empty the case directory, called only once a run has produced everything."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def _write_case(
    target: Path,
    run_metadata: dict,
    stages: fixture_replay.StageCapture,
    boundaries: BoundaryRecorder,
    evidence: list[dict],
) -> None:
    """Write one completed recording: run metadata, stage outputs, and boundaries."""
    _replace_case_directory(target)
    write = fixture_replay.write_json
    write(target / fixture_replay.RUN_FILE, run_metadata)
    write(target / fixture_replay.GENERATED_QUERIES_FILE, stages.generated_queries)
    write(target / fixture_replay.VALIDATED_QUERIES_FILE, stages.validated_queries)
    write(
        target / fixture_replay.SUBQUERY_PARAGRAPHS_FILE,
        stages.subquery_paragraphs_in_order(stages.validated_queries),
    )
    write(target / fixture_replay.MERGED_PARAGRAPHS_FILE, stages.merged_paragraphs)
    write(target / fixture_replay.RELEVANT_PAPERS_FILE, stages.relevant_papers)
    write(target / fixture_replay.EVIDENCE_FILE, evidence)

    write(target / fixture_replay.LLM_CALLS_FILE, boundaries.llm_calls)
    write(target / fixture_replay.SEARCH_RESULTS_FILE, boundaries.search_results)
    # One file per paper rather than one JSON blob: JATS full text runs to ~100 KB
    # each, and keeping it out of the JSON leaves every other fixture readable.
    fulltext_root = target / fixture_replay.FULLTEXT_DIR
    fulltext_root.mkdir(parents=True, exist_ok=True)
    for pmcid, xml_text in boundaries.fulltext_xml.items():
        path = fulltext_root / fixture_replay.fulltext_filename(pmcid)
        path.write_text(xml_text, encoding="utf-8")


def record_fixture(
    *,
    case_id: str,
    query: str,
    config_path: Path,
    llm_base_url: str,
    llm_model_name: str,
    overwrite: bool,
    verbose: bool,
) -> Path:
    """Record one live Librarian run into a replayable case directory.

    :param case_id: Name of the fixture directory under ``integration/``.
    :param query: Public user question passed to the Librarian.
    :param config_path: TOML file defining the small recording configuration.
    :param llm_base_url: OpenAI-compatible LLM endpoint.
    :param llm_model_name: Model served by that endpoint.
    :param overwrite: Replace an existing case directory when ``True``.
    :param verbose: Enable the Librarian's normal progress logging.
    :returns: Directory containing the written case.
    :raises ValueError: if the config turns on supplementary enrichment, which
        replay cannot answer.
    :raises RuntimeError: if the run returned no evidence, which would record a
        fixture that pins nothing.
    """
    if not query.strip():
        raise ValueError("query cannot be empty")

    runtime_config = LibrarianRuntimeConfig.from_toml(
        config_path.read_text(encoding="utf-8")
    )
    if runtime_config.supplementary_enrichment:
        raise ValueError(
            "supplementary_enrichment must be false: supplementary files are not "
            "recorded, so a replay could not answer for them."
        )
    # Keep the recorded config honest about which model actually answered, even if
    # --llm-model-name overrode the TOML.
    recording_config = replace(runtime_config, default_model_name=llm_model_name)

    target = _case_directory(case_id)
    _check_case_directory(target, overwrite)

    llm = create_llm_client(base_url=llm_base_url, model_name=llm_model_name)
    llm.reasoning_effort = REASONING_EFFORT
    boundaries = BoundaryRecorder()
    case = fixture_replay.RecordedCase(
        directory=target,
        query=query,
        full_text_enrichment=FULL_TEXT_ENRICHMENT,
        bm25_stemming=fixture_replay.bm25_stemming_enabled(),
        runtime_config=recording_config,
        llm_calls=[],
        search_results={},
    )
    # Plain EuropePmcSource, never default_literature_source(): a recording must
    # not depend on whatever LIBRARIAN_CACHE_DIR holds on this machine.
    agent = fixture_replay.build_agent(
        case,
        llm_client=RecordingLLM(llm, boundaries),
        literature_source=RecordingSource(EuropePmcSource(), boundaries),
    )
    agent.verbose = verbose
    stages = fixture_replay.StageCapture()
    stages.install(agent)

    evidence = agent.run(query).papers
    if not evidence:
        raise RuntimeError(
            "the run returned no evidence; nothing worth recording. Check the LLM "
            "endpoint and Europe PMC connectivity, then retry."
        )

    run_metadata = {
        "query": query,
        "recorded_on": datetime.date.today().isoformat(),
        "model_name": llm_model_name,
        "reasoning_effort": REASONING_EFFORT,
        # Recorded so the tests rebuild the agent from the fixture instead of
        # repeating this constant and drifting from it.
        "full_text_enrichment": FULL_TEXT_ENRICHMENT,
        # Stage 2's BM25 ranking depends on it but it is not a config knob (see
        # agent._load_bm25_stemmer), so the tests skip a replay under the other one.
        "bm25_stemming": case.bm25_stemming,
        "runtime_config": asdict(recording_config),
    }
    _write_case(target, run_metadata, stages, boundaries, evidence)
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for one live fixture recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model-name", default="glm-5.3-flash")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    """Parse arguments, record one case, and print its directory."""
    arguments = build_parser().parse_args()
    target = record_fixture(
        case_id=arguments.case,
        query=arguments.query,
        config_path=arguments.config,
        llm_base_url=arguments.llm_base_url,
        llm_model_name=arguments.llm_model_name,
        overwrite=arguments.overwrite,
        verbose=arguments.verbose,
    )
    print(f"Recorded Librarian fixture: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

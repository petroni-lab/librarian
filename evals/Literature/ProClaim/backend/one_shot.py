"""The ProClaim + librarian verification path.

ProClaim retrieves with PubMed and Semantic Scholar. The paper's "ProClaim
0.80" row retrieves with the librarian instead, through a single-pass path that
exists nowhere upstream: one librarian search, fact extraction, a bounded
sparse-evidence recovery loop, and a verdict emitted on the outer-agent model.

This is GPL-3.0. It is a derived work of ProClaim (see ../NOTICE) even though it
lives outside the clone, because it is written against ProClaim's internals and
distributed with it. The repository's root LICENSE does not reach this
directory.

WHY IT IS HERE AND NOT IN THE CLONE

`../ProClaim_src/` is a pristine clone at a pinned commit, and
`../../setup.sh --check` asserts it stays that way. Nothing of ours is written
inside it. Upstream's own behaviour is therefore untouched: there is no
`retrieval_backend` switch to leave at "pubmed", because upstream never learns
about the librarian at all.

That is possible because this path is purely additive. It calls four helpers
out of upstream's `evidence_programming_direct` — the jupytext logging trio and
`generate_notebook` — plus `evidence_api`'s workspace, extraction and verdict
functions, and otherwise replaces `verify_claim_direct()` rather than modifying
it. ../direct_entry.py is what runs it in upstream's place.

`build_subprocess_env` is the one helper we could not use as-is: a librarian run
needs the AGENT_LLM_* variables and must not carry the MLP classifier. It is
wrapped here rather than patched there.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

# Upstream, out of the pristine clone. These are the only four things this path
# borrows from the module it stands in for.
from proclaim.verification.evidence_programming_direct import (
    append_to_jupytext_log,
    build_subprocess_env as _upstream_build_subprocess_env,
    generate_notebook,
    init_jupytext_log,
)

from evals.Literature.ProClaim.backend.librarian_search import (
    librarian_refine_search,
    librarian_search,
    librarian_search_for_claim,
)
from evals.Literature.ProClaim.backend.model_registry import get_agent_llm
from evals.Literature.ProClaim.backend.prompts import LIBRARIAN_DIRECT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_subprocess_env(cfg) -> dict[str, str]:
    """Upstream's subprocess environment, adjusted for a librarian run.

    Two differences from upstream's, and both matter:

    * The final verdict is emitted on the *outer agent* model rather than the
      Qwen subagent, so the AGENT_LLM_* variables have to be there for
      ``get_agent_llm()`` to read.
    * Sufficiency is decided from extracted facts, so the MLP classifier and
      the sufficiency backend are removed rather than left pointing at a model
      directory this run will never load.
    """
    env = dict(_upstream_build_subprocess_env(cfg))

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

    env.pop("MLP_MODEL_DIR", None)
    env.pop("SUFFICIENCY_BACKEND", None)

    env["RETRIEVAL_BACKEND"] = "librarian"
    env["LIBRARIAN_LLM_BASE_URL"] = getattr(cfg, "librarian_llm_base_url", "")
    env["LIBRARIAN_LLM_MODEL"] = getattr(cfg, "librarian_llm_model", "")
    return env


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

def verify_librarian_one_shot(cfg) -> Path:
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

    # Deferred: importing proclaim's evidence_api at module scope would pull in
    # its retrieval stack before build_subprocess_env has set the environment it
    # reads. The librarian_* names come from our backend, not from evidence_api
    # — upstream has no such functions, and this is why it needs no patching.
    from proclaim.verification.config import get_label_config
    from proclaim.verification.evidence_api import (
        emit_verdict,
        extract_and_add_facts,
        get_evidence_summary,
        setup_workspace,
    )

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

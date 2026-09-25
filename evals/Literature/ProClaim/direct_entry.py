"""Run ProClaim's direct verification with the librarian retrieval backend.

GPL-3.0: a derived work of ProClaim (see NOTICE), distributed with it.

    python -m evals.Literature.ProClaim.direct_entry --config <yaml> --claim "..."

This stands in for

    python -m proclaim.verification.evidence_programming_direct

and takes the same arguments, because it reuses ProClaim's own CLI parser. It
is what makes the librarian arm possible without touching the clone: upstream's
``verify_claim_direct()`` dispatches to PubMed retrieval and knows nothing about
the librarian, so rather than patch that dispatch we simply do not call it. The
one-shot path in backend/one_shot.py runs instead, borrowing four helpers out of
upstream's module and leaving everything else alone.

``../ProClaim_src/`` stays byte-for-byte its pinned commit, which
``../setup.sh --check`` asserts. Upstream's own behaviour is therefore not
merely unchanged but unreachable from here: there is no switch inside ProClaim
left at a safe default, because nothing inside ProClaim was altered.

PYTHONPATH must carry both this repository and the clone's ``src``;
proclaim_librarian.py sets that up for the subprocesses it spawns.
"""

from __future__ import annotations

import logging
import sys


def main(argv: list[str] | None = None) -> int:
    try:
        import proclaim  # noqa: F401
    except ImportError:
        print(
            "ERROR: cannot import `proclaim`.\n"
            "       ProClaim uses a src layout, so its source root has to be on\n"
            "       PYTHONPATH:\n\n"
            "         PYTHONPATH=evals/Literature/ProClaim/ProClaim_src/src:.\n\n"
            "       ./evals/Literature/setup.sh --bench proclaim clones it.",
            file=sys.stderr,
        )
        return 1

    from evals.Literature.ProClaim.backend.config import load_from_cli
    from evals.Literature.ProClaim.backend.one_shot import verify_librarian_one_shot

    cfg = load_from_cli(argv)

    output_dir = cfg.resolved_output_dir
    workspace = cfg.resolved_workspace
    log_file = output_dir / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
    )

    print(f"Claim: {cfg.claim}")
    print(f"Agent model: {cfg.llm.model}")
    print(f"Subagent model: {cfg.subagent_model}")
    print(f"Retrieval: librarian @ {cfg.librarian_llm_base_url} ({cfg.librarian_llm_model or 'agent default'})")
    print(f"Workspace: {workspace}")
    print()

    result = verify_librarian_one_shot(cfg)
    print(f"\nOutput: {result}")
    print(f"Workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

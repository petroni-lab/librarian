"""Assert that the pristine ProClaim clone still fits the librarian backend.

GPL-3.0; see NOTICE.

Because `backend/` stands beside ProClaim rather than inside it, the join is
made of imports and attribute names, not of a patch that would refuse to apply.
This checks that join.

It parses the clone's source rather than importing it, so it runs without
ProClaim's dependency stack -- on any machine, right after cloning, before the
pipeline environment exists.

    python evals/Literature/ProClaim/check_seam.py [--clone DIR]

It asserts that:

  * the four helpers one_shot.py borrows are still defined;
  * `VerificationSettings` still ignores unknown keys (which is why
    `LibrarianSettings` subclasses it) and still has no `retrieval_backend`
    field of its own (which would collide with ours);
  * `evidence_api` still exports the four functions the one-shot path calls.

That is not proof the run works, only that the parts we reach for are there.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Borrowed by backend/one_shot.py out of upstream's module.
EPD_HELPERS = (
    "append_to_jupytext_log",
    "build_subprocess_env",
    "generate_notebook",
    "init_jupytext_log",
)
# Called by verify_librarian_one_shot().
EVIDENCE_API_FUNCTIONS = (
    "emit_verdict",
    "extract_and_add_facts",
    "get_evidence_summary",
    "setup_workspace",
)


def _module(clone: Path, dotted: str) -> ast.Module:
    path = clone / "src" / Path(*dotted.split(".")).with_suffix(".py")
    if not path.is_file():
        raise SystemExit(f"FAIL: {dotted} not found at {path}")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_defs(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise SystemExit(f"FAIL: class {name} is gone from upstream")


def check(clone: Path) -> list[str]:
    """Return a list of problems; empty means the seam still fits."""
    problems: list[str] = []

    epd = _module(clone, "proclaim.verification.evidence_programming_direct")
    defined = _top_level_defs(epd)
    for name in EPD_HELPERS:
        if name not in defined:
            problems.append(
                f"evidence_programming_direct no longer defines {name}(), which "
                f"backend/one_shot.py imports"
            )

    api = _module(clone, "proclaim.verification.evidence_api")
    api_defined = _top_level_defs(api)
    for name in EVIDENCE_API_FUNCTIONS:
        if name not in api_defined:
            problems.append(
                f"evidence_api no longer defines {name}(), which the one-shot "
                f"path calls"
            )

    cfg = _module(clone, "proclaim.verification.config")
    settings = _class(cfg, "VerificationSettings")

    fields = {
        target.id
        for node in settings.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in [node.target]
    }
    if "retrieval_backend" in fields:
        problems.append(
            "VerificationSettings has grown its own retrieval_backend field; "
            "backend/config.py's subclass may now conflict with it"
        )

    # model_config = SettingsConfigDict(..., extra="ignore", ...)
    extra = None
    for node in settings.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "model_config" for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "extra" and isinstance(kw.value, ast.Constant):
                        extra = kw.value.value
    if extra != "ignore":
        problems.append(
            f"VerificationSettings.model_config extra={extra!r}, expected 'ignore'. "
            "backend/config.py subclasses precisely because unknown keys are "
            "dropped silently; revisit it if that has changed"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clone",
        type=Path,
        default=HERE / "ProClaim_src",
        help="the pristine ProClaim clone (default: ./ProClaim_src)",
    )
    args = parser.parse_args(argv)
    clone = args.clone.resolve()
    if not (clone / "src" / "proclaim").is_dir():
        print(
            f"ERROR: no ProClaim clone at {clone}.\n"
            "       Run ./evals/Literature/setup.sh --bench proclaim.",
            file=sys.stderr,
        )
        return 1

    problems = check(clone)
    if problems:
        print("The ProClaim clone no longer fits the librarian backend:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe clone is pinned, so this can only mean the pin moved. Either\n"
            "restore it or update backend/ to the new tree.",
            file=sys.stderr,
        )
        return 1
    print(f"ProClaim seam ok: {clone.name} still provides everything backend/ reaches for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

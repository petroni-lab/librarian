"""Reconstruct the LitQA2 EuropePMC-fulltext eval subset from a public id list.

Given a plain-text file of LitQA2 row ids (one per line, e.g.
data/litqa2_europepmc_fulltext/litqa2_public_subset_ids.txt), re-fetches the
full rows from the public `futurehouse/lab-bench` LitQA2 dataset on the Hugging
Face Hub and writes them out in the same shape as
litqa2_full_europepmc_fulltext.json, ready for check_litqa2_europepmc_fulltext.py
downstream tooling or evals.Literature.AstaBench.litqa2_open_judge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
except ImportError:
    pass

DEFAULT_IDS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "litqa2_europepmc_fulltext"
    / "litqa2_public_subset_ids.txt"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "litqa2_europepmc_fulltext"
    / "litqa2_full_europepmc_fulltext.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    ids = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}

    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dataset = load_dataset("futurehouse/lab-bench", "LitQA2", token=token)
    assert isinstance(dataset, Dataset | DatasetDict | IterableDatasetDict)
    rows = [dict(row) for row in dataset["train"] if str(row["id"]) in ids]

    missing = ids - {row["id"] for row in rows}
    if missing:
        print(f"warning: {len(missing)} id(s) not found in futurehouse/lab-bench: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

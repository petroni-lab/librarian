"""Reconstruct scholar_multi_biomed_eval.json from a public id list.

Filters the Scholar-Multi gold-reference file down to the ids in a plain-text
list, one per line, and writes them in the same shape. Defaults are the
committed id list and the gold file ``../setup.sh --bench sqa`` copies out of the
ScholarQABench clone; the id list alone carries no question or answer content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "scholarqa_multi"
DEFAULT_IDS_PATH = DATA_DIR / "scholar_multi_biomed_public_subset_ids.txt"
DEFAULT_SOURCE_PATH = DATA_DIR / "human_answers.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "scholar_multi_biomed_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    ids = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
    records = json.loads(args.source.read_text())
    matched = [record for record in records if record["id"] in ids]

    missing = ids - {record["id"] for record in matched}
    if missing:
        print(
            f"warning: {len(missing)} id(s) not found in {args.source}: {sorted(missing)}"
        )

    args.output.write_text(json.dumps(matched, ensure_ascii=True, indent=2) + "\n")
    print(f"wrote {len(matched)} records to {args.output}")


if __name__ == "__main__":
    main()

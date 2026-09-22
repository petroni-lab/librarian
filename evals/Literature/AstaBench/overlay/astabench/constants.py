import os

ASTA_BENCH_DATASET_REPO = "allenai/asta-bench"
ASTA_BENCH_DATASET_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"

# For large-ish data used by solvers (e.g. cached results, fewshot examples)
ASTA_SOLVER_DATA_REPO = "allenai/asta-bench-solver-data"
ASTA_SOLVER_DATA_REVISION = "112f457ad94b05fd377234feae363ae794550049"


def get_hf_token() -> str | None:
    """Return the first configured Hugging Face token env var."""

    return (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
    )

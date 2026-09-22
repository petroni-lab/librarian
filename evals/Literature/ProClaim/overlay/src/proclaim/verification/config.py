"""
Centralised configuration for the Evidence Verification system.

Uses ``pydantic-settings`` to unify:
  - Local YAML parameter files (for reproducible experiments)
  - Environment variables (from ``.env`` at project root)
  - CLI arguments (``--config``, ``--claim``, etc.)
  - Programmatic overrides (constructor kwargs)

Resolution order (highest priority first):
  1. Explicit constructor kwargs / CLI flags
  2. Environment variables (reads ``.env`` automatically)
  3. YAML config file (``--config path/to/config.yaml``)
  4. Field defaults

Usage — library code::

    from proclaim.verification.config import get_settings
    cfg = get_settings()
    print(cfg.api_key, cfg.model)

Usage — CLI script::

    from proclaim.verification.config import VerificationSettings
    cfg = VerificationSettings.from_cli()   # parses sys.argv

Usage — from YAML file::

    cfg = VerificationSettings.from_yaml("experiments/mapk1_h3.yaml")

Usage — save config for reproducibility::

    cfg.save_yaml("experiments/mapk1_h3.yaml")

Usage — programmatic override::

    cfg = VerificationSettings(model="gpt-4o", max_iterations=12)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from datetime import datetime
from uuid import uuid4

from pydantic import Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path constants (not configurable — derived from source tree)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent.parent  # repo root

# ---------------------------------------------------------------------------
# Settings models
# ---------------------------------------------------------------------------


class LabelConfig(BaseSettings):
    """User-configurable label definitions for stance and verdict categories.

    Allows users to define/redefine the label names and their detailed
    explanations used throughout the evidence programming system. All
    prompts (system prompt, subagent prompts) and parsers read from these
    definitions so the label taxonomy is consistent end-to-end.

    Override via YAML config::

        labels:
          stance_labels:
            SUPPORT: "Evidence directly corroborates the claim."
            REFUTE: "Evidence contradicts the claim."
            NEUTRAL: "Relevant but neither supports nor contradicts."
          verdict_labels:
            SUPPORT: "The retrieved evidence directly corroborates the claim."
            REFUTE: "The retrieved evidence contradicts the claim."
            UNCERTAIN: "The evidence is ambiguous or insufficient."
          default_stance: "NEUTRAL"
    """

    model_config = SettingsConfigDict(extra="ignore")

    stance_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "SUPPORT": "The fact directly supports or corroborates the claim.",
            "REFUTE": "The fact directly contradicts or refutes the claim.",
            "NEUTRAL": "The fact is relevant to the claim but neither clearly supports nor refutes it.",
        },
        description="Mapping from stance label name to its description. "
                    "Each key is used in LLM prompts for fact extraction; "
                    "each value is the explanation shown to the LLM.",
    )

    verdict_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "SUPPORT": (
                "The retrieved evidence contains statements that directly "
                "corroborate the claim. The evidence, taken at face value, is "
                "sufficient to conclude that the claim is true or highly likely true."
            ),
            "REFUTE": (
                "Either (a) the retrieved evidence contains statements that "
                "directly contradict the claim, or (b) given the scope of the "
                "retrieved corpus, a thorough search yields no evidence that "
                "substantiates the claim. In both cases, the evidence base does "
                "not support accepting the claim as true."
            ),
            "UNCERTAIN": (
                "The retrieved evidence is relevant to the claim but is ambiguous, "
                "incomplete, or internally conflicting such that neither a clear "
                "supportive nor a clear refutatory conclusion can be drawn. This "
                "includes cases where evidence partially supports the claim but "
                "with meaningful caveats, or where sources of comparable credibility "
                "disagree."
            ),
        },
        description="Mapping from verdict label name to its description. "
                    "These are used in the system prompt for the orchestrator agent.",
    )

    default_stance: str = Field(
        default="NEUTRAL",
        description="Fallback stance label when the LLM returns an unrecognised value.",
    )

    # --- Helpers for prompt construction ---

    def stance_names(self) -> list[str]:
        """Return the list of valid stance label names."""
        return list(self.stance_labels.keys())

    def stance_options_str(self) -> str:
        """Format stance labels as a pipe-separated options string, e.g. 'SUPPORT | REFUTE | NEUTRAL'."""
        return " | ".join(f'"{name}"' for name in self.stance_labels)

    def stance_prompt_block(self) -> str:
        """Build a multi-line label definition block for use in LLM prompts."""
        lines = []
        for name, desc in self.stance_labels.items():
            lines.append(f'- "{name}": {desc}')
        return "\n".join(lines)

    def verdict_names(self) -> list[str]:
        """Return the list of valid verdict label names."""
        return list(self.verdict_labels.keys())

    def verdict_prompt_block(self) -> str:
        """Build a multi-line verdict definition block for use in the system prompt."""
        lines = []
        for name, desc in self.verdict_labels.items():
            lines.append(f"{name} — {desc}")
        return "\n".join(lines)

    def validate_stance(self, value: str) -> str:
        """Validate a stance string against configured labels, returning the canonical name or default."""
        upper = value.upper().strip()
        for name in self.stance_labels:
            if upper == name.upper():
                return name
        return self.default_stance

    def validate_verdict(self, value: str) -> str:
        """Validate a verdict string against configured labels."""
        upper = value.upper().strip()
        for name in self.verdict_labels:
            if upper == name.upper():
                return name
        return upper  # pass through for free-form verdicts


class APISettings(BaseSettings):
    """LLM / external-service API keys and endpoints.

    Reads from env vars automatically (case-insensitive).
    The ``.env`` file at the project root is loaded first.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys -----------------------------------------------------------
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic/Claude API key.",
    )
    glm_api_key: Optional[str] = Field(
        default=None,
        description="Primary API key for Z.AI / GLM models.",
    )
    zai_api_key: Optional[str] = Field(
        default=None,
        description="Alias for glm_api_key (fallback).",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="Fallback OpenAI API key.",
    )

    # Full-text retrieval ------------------------------------------------
    unpaywall_email: str = Field(
        default="proclaim@example.com",
        description="Email address for Unpaywall API requests.",
    )
    elsevier_api_key: Optional[str] = Field(
        default=None,
        description="Elsevier API key for INDRA full-text via ScienceDirect.",
    )

    # Derived: resolved API key ------------------------------------------
    @property
    def api_key(self) -> str:
        """Resolve the best available API key (Anthropic > GLM > ZAI > OpenAI > EMPTY)."""
        return (
            self.anthropic_api_key
            or self.glm_api_key
            or self.zai_api_key
            or self.openai_api_key
            or "EMPTY"
        )


class LLMSettings(BaseSettings):
    """LLM endpoint and model configuration."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    # Agent endpoint (Anthropic-compatible, used by Claude Agent SDK)
    agent_base_url: str = Field(
        default="https://api.z.ai/api/anthropic",
        validation_alias="ANTHROPIC_BASE_URL",
        description="Anthropic-compatible base URL for the outer agent (Claude Agent SDK).",
    )

    # Subagent endpoint (OpenAI-compatible, used by REPL subagents / vLLM)
    subagent_base_url: str = Field(
        default="http://localhost:8000/v1/",
        validation_alias="LLM_BASE_URL",
        description="OpenAI-compatible base URL for subagent LLM calls (vLLM, etc.).",
    )

    # Model identifiers
    model: str = Field(
        default="glm-5",
        description="Model identifier for the outer/main agent.",
    )
    subagent_model: Optional[str] = Field(
        default=None,
        description="Model identifier for inner subagent calls. "
                    "Defaults to `model` when None.",
    )

    # Generation parameters
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    disable_thinking: bool = Field(
        default=True,
        description="Disable Qwen thinking mode to save tokens.",
    )
    thinking_budget_tokens: int = Field(
        default=0,
        ge=0,
        description="Enable Claude extended thinking when > 0. Sets budget_tokens for the thinking block. Requires temperature=1.",
    )
    timeout: int = Field(
        default=300,
        description="Connection and read timeout in seconds.",
    )
    stream: bool = Field(
        default=False,
        description="Use streaming API to prevent timeouts.",
    )

    @property
    def effective_subagent_model(self) -> str:
        """Return subagent_model if set, else fall back to model."""
        return self.subagent_model or self.model


class VerificationSettings(BaseSettings):
    """Top-level configuration merging API, LLM, and workflow settings.

    This is the single settings object that scripts and library code should use.
    """

    _auto_timestamp: str = PrivateAttr(
        default_factory=lambda: f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    )

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Nested settings (populated via env vars) ──────────────────────
    api: APISettings = Field(default_factory=APISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    labels: LabelConfig = Field(default_factory=LabelConfig)

    # ── Verification workflow ─────────────────────────────────────────
    claim: str = Field(
        default="",
        description="Scientific claim to verify. Always provided via CLI "
                    "(--claim).",
    )
    mode: Literal["sdk", "direct"] = Field(
        default="sdk",
        description=(
            "Orchestration mode. "
            "'sdk': Claude Agent SDK + Jupyter kernel (nb_execute). "
            "'direct': LiteLLM + bash + jupytext (no kernel, no MCP)."
        ),
    )

    max_iterations: int = Field(
        default=8,
        ge=1,
        description="Maximum sufficiency-check iterations.",
    )
    sufficiency_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for early stopping.",
    )
    mlp_model_dir: Optional[str] = Field(
        default=None,
        description="Path to the MLP model directory (relative to project root). "
                    "Defaults to 'results/models/classifier_best'.",
    )
    sufficiency_backend: Literal["mlp", "llm", "haiku"] = Field(
        default="mlp",
        description=(
            "Sufficiency classifier backend. "
            "'mlp' (default): trained MLP using aggregated numerical features. "
            "'llm': Qwen subagent via local vLLM endpoint. "
            "'haiku': Claude Haiku via native Anthropic API."
        ),
    )
    retrieval_backend: Literal["pubmed", "librarian"] = Field(
        default="pubmed",
        description=(
            "Retrieval layer backend. "
            "'pubmed' (default): the original PubMed + Semantic Scholar search. "
            "'librarian': an in-process LibrarianAgent provides papers and "
            "evidence snippets; PubMed/S2 search and full-text fetching are "
            "bypassed."
        ),
    )
    librarian_llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description=(
            "OpenAI-compatible base URL of the model service that backs the "
            "librarian agent's internal LLM (query planner / relevance filter). "
            "Only used when retrieval_backend is 'librarian'. Independent of the "
            "evidence-programmer model. Example: http://localhost:8000/v1"
        ),
    )
    librarian_llm_model: str = Field(
        default="",
        description=(
            "Model id served by `librarian_llm_base_url`, used by the librarian "
            "agent. Must be set when retrieval_backend is 'librarian'; if empty "
            "the agent falls back to its own default model resolution."
        ),
    )
    max_turns: int = Field(
        default=30,
        ge=1,
        description="Hard ceiling on LLM turns in REPL mode.",
    )
    max_output_chars: int = Field(
        default=12_000,
        description="Max chars returned to the agent from any notebook tool "
                    "(nb_execute, nb_render_*, nb_read_output).",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output workspace directory. "
                    "Defaults to results/verification/notebook_demo.",
    )
    notebook_path: Optional[Path] = Field(
        default=None,
        description="Path for the output notebook (Mode A only).",
    )

    # ── Prompt options ────────────────────────────────────────────────
    include_subclaim_examples: bool = Field(
        default=True,
        description="Inject few-shot subclaim decomposition examples into the system prompt.",
    )
    disable_web_search: bool = Field(
        default=False,
        description="Remove web_search from the agent's tool set (direct mode only).",
    )

    # ── Logging ───────────────────────────────────────────────────────
    verbose: bool = Field(
        default=False,
        description="Enable debug logging.",
    )

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def resolved_output_dir(self) -> Path:
        """Return output_dir or an auto-generated timestamped directory."""
        if self.output_dir is not None:
            return self.output_dir
        return PROJECT_ROOT / "results" / "verification" / self._auto_timestamp

    @property
    def resolved_workspace(self) -> Path:
        return self.resolved_output_dir / "workspace"

    @property
    def resolved_notebook_path(self) -> Path:
        if self.notebook_path is not None:
            return self.notebook_path
        return self.resolved_output_dir / "evidence_report.ipynb"

    # ── Shortcut properties (delegate to nested) ──────────────────────

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.llm.model

    @property
    def subagent_model(self) -> str:
        return self.llm.effective_subagent_model

    @property
    def api_key(self) -> str:
        return self.api.api_key

    @property
    def agent_base_url(self) -> str:
        return self.llm.agent_base_url

    @property
    def subagent_base_url(self) -> str:
        return self.llm.subagent_base_url

    @property
    def temperature(self) -> float:
        return self.llm.temperature

    @property
    def unpaywall_email(self) -> str:
        return self.api.unpaywall_email

    # ── LLM factory ───────────────────────────────────────────────────

    def make_subagent_llm(self):
        """Build the subagent ``llm(prompt) -> str`` callable from config.

        Uses ``subagent_base_url``, ``api_key``, and ``subagent_model``
        so callers don't need to pass any LLM parameters.

        If ``disable_thinking`` is True, passes extra_body to disable Qwen's
        built-in thinking mode (saves tokens).
        """
        from proclaim.verification.llm_factory import make_llm

        extra_body = None
        if self.llm.disable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        return make_llm(
            model=self.subagent_model,
            api_key=self.api_key,
            base_url=self.subagent_base_url,
            temperature=self.temperature,
            extra_body=extra_body,
            timeout=self.llm.timeout,
            stream=self.llm.stream,
        )

    # ── Builders ──────────────────────────────────────────────────────

    def build_sdk_env(self) -> dict[str, str]:
        """Build the environment dict for the Claude Agent SDK process.

        Includes ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``, and
        ``MLP_MODEL_DIR`` so that ``setup_kernel()`` in the MCP server's
        Jupyter kernel can resolve LLM config without agent involvement.
        """
        if self.api_key == "EMPTY":
            raise RuntimeError(
                "GLM_API_KEY not set. Add it to .env at project root.\n"
                "  echo 'GLM_API_KEY=<your-key>' >> .env"
            )
        env = {
            **os.environ,
            "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
            "ANTHROPIC_AUTH_TOKEN": self.api_key,
            "ANTHROPIC_BASE_URL": self.agent_base_url,
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "300000",
            # LLM config for setup_kernel() inside the Jupyter kernel
            "LLM_BASE_URL": self.subagent_base_url,
            "LLM_API_KEY": self.api_key,
            "LLM_MODEL": self.subagent_model,
            "LLM_TEMPERATURE": str(self.llm.temperature),
            "LLM_DISABLE_THINKING": "1" if self.llm.disable_thinking else "0",
            "LLM_TIMEOUT": str(self.llm.timeout),
            "LLM_STREAM": "1" if self.llm.stream else "0",
            "MLP_MODEL_DIR": self.mlp_model_dir or "results/models/classifier_best",
            "SUFFICIENCY_BACKEND": self.sufficiency_backend,
            "RETRIEVAL_BACKEND": self.retrieval_backend,
            "LIBRARIAN_LLM_BASE_URL": self.librarian_llm_base_url,
            "LIBRARIAN_LLM_MODEL": self.librarian_llm_model,
            "MAX_ITERATIONS": str(self.max_iterations),
            "SUFFICIENCY_BACKEND": self.sufficiency_backend,
            # Label config for setup_kernel() inside the Jupyter kernel
            "LABEL_CONFIG_JSON": self.labels.model_dump_json(),
            # Notebook MCP truncation limit
            "NB_MAX_OUTPUT_CHARS": str(self.max_output_chars),
        }
        # Keep ANTHROPIC_API_KEY in environment for Claude models
        return env

    # ── Factory: from YAML file ───────────────────────────────────────

    @classmethod
    def from_yaml(
        cls, path: str | Path, **overrides,
    ) -> "VerificationSettings":
        """Load settings from a YAML config file.

        Values in the YAML file are merged with env vars and defaults.
        Explicit ``overrides`` keyword arguments take highest priority.

        Note: ``claim`` is intentionally excluded from YAML — it varies
        per-run and should be passed as a keyword override or via CLI.

        Usage::

            cfg = VerificationSettings.from_yaml(
                "experiments/config.yaml",
                claim="Does MAPK1 directly activate H3-3A?",
            )
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config files.  "
                "Install it with: uv pip install pyyaml"
            )
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # claim is a per-run argument, not a config parameter — strip it
        # from YAML to avoid accidental reuse across different runs.
        if "claim" in raw:
            import warnings
            warnings.warn(
                "'claim' found in YAML config file and will be ignored. "
                "Pass --claim via CLI or as a keyword override instead.",
                UserWarning,
                stacklevel=2,
            )
            raw.pop("claim")

        # Resolve nested dicts into sub-model instances
        if "llm" in raw and isinstance(raw["llm"], dict):
            raw["llm"] = LLMSettings(**raw["llm"])
        if "api" in raw and isinstance(raw["api"], dict):
            raw["api"] = APISettings(**raw["api"])
        if "labels" in raw and isinstance(raw["labels"], dict):
            raw["labels"] = LabelConfig(**raw["labels"])

        # Convert path strings
        for key in ("output_dir", "notebook_path"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = Path(raw[key])

        # Overrides trump YAML
        raw.update(overrides)

        return cls(**raw)

    # ── Save config snapshot ──────────────────────────────────────────

    def save_yaml(self, path: str | Path) -> Path:
        """Dump current settings to a YAML file for reproducibility.

        API keys are redacted.  The file can be reloaded with ``from_yaml()``.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config files.  "
                "Install it with: uv pip install pyyaml"
            )
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(
            mode="json",
            exclude={
                "claim": True,  # per-run CLI argument, not a config param
                "api": {"anthropic_api_key", "glm_api_key", "zai_api_key",
                        "openai_api_key", "elsevier_api_key"},
            },
        )

        # Convert Path objects to strings for YAML
        for key in ("output_dir", "notebook_path"):
            if data.get(key) is not None:
                data[key] = str(data[key])

        with open(out_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

        return out_path

    # ── Factory: from CLI args ────────────────────────────────────────

    @classmethod
    def from_cli(cls, args: Optional[list[str]] = None) -> "VerificationSettings":
        """Parse CLI arguments and env vars into a VerificationSettings.

        This combines argparse (for familiarity / --help) with pydantic-settings
        (for env-var merging).  CLI flags take highest priority.
        """
        import argparse

        parser = argparse.ArgumentParser(
            description="Evidence Verification — RLM dual-mode orchestrator.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--config", "-c", default=None,
            help="Path to a YAML config file. Values are merged with env vars "
                 "and defaults. CLI flags override config file values.",
        )
        parser.add_argument(
            "--claim", required=True,
            help="Scientific claim to verify.",
        )
        parser.add_argument(
            "--mode", choices=["sdk", "direct"], default=None,
            help="Orchestration mode (default: sdk).",
        )
        parser.add_argument("--model", default=None, help="Model identifier for the outer agent.")
        parser.add_argument(
            "--subagent-model", default=None,
            help="Model for inner subagent calls (defaults to --model).",
        )
        parser.add_argument("--threshold", type=float, default=None, help="Sufficiency confidence threshold.")
        parser.add_argument("--max-iterations", type=int, default=None, help="Max sufficiency iterations.")
        parser.add_argument("--output-dir", default=None, help="Output workspace directory.")
        parser.add_argument("--notebook-path", default=None, help="Notebook path (Mode A only).")
        parser.add_argument("--subagent-base-url", default=None, help="OpenAI-compatible base URL for subagent LLM.")
        parser.add_argument("--agent-base-url", default=None, help="Anthropic-compatible base URL for outer agent.")
        parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Debug logging.")

        parsed = parser.parse_args(args)

        # ── Start from YAML base if provided ──────────────────────────
        yaml_base: dict = {}
        if parsed.config is not None:
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for --config.  "
                    "Install it with: uv pip install pyyaml"
                )
            config_path = Path(parsed.config)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}"
                )
            with open(config_path) as fh:
                yaml_base = yaml.safe_load(fh) or {}

            # claim is a per-run argument — strip from YAML
            yaml_base.pop("claim", None)

            # Resolve nested dicts into sub-model instances
            if "llm" in yaml_base and isinstance(yaml_base["llm"], dict):
                yaml_base["llm"] = LLMSettings(**yaml_base["llm"])
            if "api" in yaml_base and isinstance(yaml_base["api"], dict):
                yaml_base["api"] = APISettings(**yaml_base["api"])
            for key in ("output_dir", "notebook_path"):
                if key in yaml_base and isinstance(yaml_base[key], str):
                    yaml_base[key] = Path(yaml_base[key])

        # ── Build CLI overrides (only explicitly-provided values) ─────
        overrides: dict = {"claim": parsed.claim}
        if parsed.verbose:
            overrides["verbose"] = True
        if parsed.mode is not None:
            overrides["mode"] = parsed.mode
        if parsed.threshold is not None:
            overrides["sufficiency_threshold"] = parsed.threshold
        if parsed.max_iterations is not None:
            overrides["max_iterations"] = parsed.max_iterations
        if parsed.output_dir is not None:
            overrides["output_dir"] = Path(parsed.output_dir)
        if parsed.notebook_path is not None:
            overrides["notebook_path"] = Path(parsed.notebook_path)

        # LLM overrides
        llm_overrides: dict = {}
        if parsed.model is not None:
            llm_overrides["model"] = parsed.model
        if parsed.subagent_model is not None:
            llm_overrides["subagent_model"] = parsed.subagent_model
        if parsed.subagent_base_url is not None:
            llm_overrides["subagent_base_url"] = parsed.subagent_base_url
        if parsed.agent_base_url is not None:
            llm_overrides["agent_base_url"] = parsed.agent_base_url

        if llm_overrides:
            # Merge with YAML-level llm if present
            if "llm" in yaml_base and isinstance(yaml_base["llm"], LLMSettings):
                base_llm = yaml_base["llm"].model_dump()
                base_llm.update(llm_overrides)
                overrides["llm"] = LLMSettings(**base_llm)
            else:
                overrides["llm"] = LLMSettings(**llm_overrides)

        # Merge: yaml_base is the floor, overrides are the ceiling
        merged = {**yaml_base, **overrides}

        return cls(**merged)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> VerificationSettings:
    """Return the cached global settings instance.

    Reads from env vars / ``.env`` only.  For CLI usage, prefer
    ``VerificationSettings.from_cli()`` and pass the result explicitly.
    """
    return VerificationSettings()


# ---------------------------------------------------------------------------
# Module-level label config singleton
# ---------------------------------------------------------------------------

_label_config: LabelConfig | None = None


def set_label_config(config: LabelConfig) -> None:
    """Set the global label configuration (called during kernel setup).

    Also rebuilds the ``Stance`` enum in ``data_models`` so that
    Pydantic accepts the configured label names when constructing
    ``Fact`` objects.
    """
    global _label_config
    _label_config = config
    # Rebuild the Stance enum (and Fact model) to match new labels
    from proclaim.verification.data_models import rebuild_stance_enum
    rebuild_stance_enum(config.stance_labels)


def get_label_config() -> LabelConfig:
    """Return the current label configuration.

    Returns the config set by ``set_label_config()``, or the default
    ``LabelConfig()`` if none has been set.
    """
    if _label_config is not None:
        return _label_config
    return LabelConfig()

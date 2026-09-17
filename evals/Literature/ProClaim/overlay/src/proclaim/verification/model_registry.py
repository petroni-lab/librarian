"""
Global singleton registry for heavy ML models.

This module provides a centralized cache for all heavy models used in evidence
verification to avoid reloading them multiple times during verification loops.

Models managed:
- SemanticSimilarityComputer: SBERT model for semantic similarity (~100MB)
- NLIEntailmentComputer: Cross-encoder for NLI entailment (~400MB)
- MLP Classifier: Trained sufficiency classifier with FeatureAggregator
- PaperFeatureExtractor: Metadata extraction utilities

Usage:
    # Lazy loading (models loaded on first use)
    from proclaim.verification.model_registry import (
        get_semantic_similarity_computer,
        get_nli_entailment_computer,
        get_mlp_classifier,
        get_feature_aggregator,
        get_metadata_extractor,
    )

    sim_computer = get_semantic_similarity_computer()
    similarity = sim_computer.compute(claim, text)

    # Pre-warming (load all models upfront)
    from proclaim.verification.model_registry import prewarm_all_models
    load_times = prewarm_all_models()

Cache management:
    from proclaim.verification.model_registry import clear_model_cache, get_cache_status

    status = get_cache_status()  # Check which models are loaded
    clear_model_cache()          # Clear all cached models
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global cache for all models
_MODEL_CACHE: Dict[str, Any] = {}

# Cache key for the outer-agent LLM used by librarian-run verdicts.
_AGENT_LLM = "agent_llm"

# Model type identifiers
_SEMANTIC_SIM = "semantic_similarity_computer"
_NLI_COMPUTER = "nli_entailment_computer"
_MLP_CLASSIFIER = "mlp_classifier"
_FEATURE_AGGREGATOR = "feature_aggregator"
_METADATA_EXTRACTOR = "metadata_extractor"
_HAIKU_LLM = "haiku_sufficiency_llm"


# ---------------------------------------------------------------------------
# Lazy Getter Functions
# ---------------------------------------------------------------------------


def get_semantic_similarity_computer():
    """Get or load the SemanticSimilarityComputer (SBERT model).

    Returns:
        SemanticSimilarityComputer instance (cached after first call)
    """
    if _SEMANTIC_SIM in _MODEL_CACHE:
        logger.debug("Using cached SemanticSimilarityComputer")
        return _MODEL_CACHE[_SEMANTIC_SIM]

    logger.info("Loading SemanticSimilarityComputer (first use)...")
    start = time.time()

    from proclaim.verification.feature_tools import SemanticSimilarityComputer
    model = SemanticSimilarityComputer()

    _MODEL_CACHE[_SEMANTIC_SIM] = model
    elapsed = time.time() - start
    logger.info(f"SemanticSimilarityComputer loaded in {elapsed:.2f}s")

    return model


def get_nli_entailment_computer():
    """Get or load the NLIEntailmentComputer (cross-encoder model).

    Returns:
        NLIEntailmentComputer instance (cached after first call)
    """
    if _NLI_COMPUTER in _MODEL_CACHE:
        logger.debug("Using cached NLIEntailmentComputer")
        return _MODEL_CACHE[_NLI_COMPUTER]

    logger.info("Loading NLIEntailmentComputer (first use)...")
    start = time.time()

    from proclaim.verification.feature_tools import NLIEntailmentComputer
    model = NLIEntailmentComputer()

    _MODEL_CACHE[_NLI_COMPUTER] = model
    elapsed = time.time() - start
    logger.info(f"NLIEntailmentComputer loaded in {elapsed:.2f}s")

    return model


def get_mlp_classifier():
    """Get or load the MLP sufficiency classifier.

    Returns:
        Dictionary with keys:
        - model: SufficiencyMLP instance
        - config: Model configuration dict
        - expected_features: List of feature names
        - mean: NumPy array of feature means (for scaling)
        - scale: NumPy array of feature scales (for scaling)
        - aggregator: FeatureAggregator instance
    """
    if _MLP_CLASSIFIER in _MODEL_CACHE:
        logger.debug("Using cached MLP classifier")
        return _MODEL_CACHE[_MLP_CLASSIFIER]

    logger.info("Loading MLP sufficiency classifier (first use)...")
    start = time.time()

    import importlib
    import sys as _sys
    import json
    import numpy as np
    import torch
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    project_root = _THIS_DIR.parents[2]

    # Make scripts/sufficiency_classifier importable
    scripts_dir = str(project_root / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)

    inference_module = importlib.import_module("sufficiency_classifier.inference")
    feature_module = importlib.import_module("sufficiency_classifier.feature_aggregation")
    SufficiencyMLP = inference_module.SufficiencyMLP
    FeatureAggregator = feature_module.FeatureAggregator

    # Load config
    import os
    model_dir_rel = os.environ.get("MLP_MODEL_DIR", "results/models/classifier_best")
    model_dir = project_root / model_dir_rel
    with open(model_dir / "mlp_config.json") as f:
        config = json.load(f)

    # Load model
    model = SufficiencyMLP(
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 64),
    )
    weights_path = model_dir / "best_model.pth"
    if not weights_path.exists():
        weights_path = model_dir / "mlp_classifier_weights.pth"
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    mlp_state = {
        "model": model,
        "config": config,
        "expected_features": config["expected_features"],
        "mean": np.array(config["scaler_mean"]),
        "scale": np.array(config["scaler_scale"]),
        "aggregator": FeatureAggregator(),
    }

    _MODEL_CACHE[_MLP_CLASSIFIER] = mlp_state
    elapsed = time.time() - start
    logger.info(f"MLP sufficiency classifier loaded from {model_dir} in {elapsed:.2f}s")

    return mlp_state


def get_feature_aggregator():
    """Get the FeatureAggregator from the cached MLP classifier.

    Note: This loads the MLP classifier if not already loaded, since the
    aggregator is part of the MLP state.

    Returns:
        FeatureAggregator instance
    """
    mlp_state = get_mlp_classifier()
    return mlp_state["aggregator"]


def get_metadata_extractor():
    """Get or load the PaperFeatureExtractor for metadata extraction.

    Returns:
        PaperFeatureExtractor instance (cached after first call)
    """
    if _METADATA_EXTRACTOR in _MODEL_CACHE:
        logger.debug("Using cached PaperFeatureExtractor")
        return _MODEL_CACHE[_METADATA_EXTRACTOR]

    logger.info("Loading PaperFeatureExtractor (first use)...")
    start = time.time()

    from proclaim.verification.feature_tools import PaperFeatureExtractor
    extractor = PaperFeatureExtractor()

    _MODEL_CACHE[_METADATA_EXTRACTOR] = extractor
    elapsed = time.time() - start
    logger.info(f"PaperFeatureExtractor loaded in {elapsed:.2f}s")

    return extractor


# ---------------------------------------------------------------------------
# Haiku LLM Singleton
# ---------------------------------------------------------------------------


def get_haiku_llm():
    """Get or create the Claude Haiku LLM callable for sufficiency checks.

    Reads the Anthropic API key from environment variables:
        ANTHROPIC_API_KEY, CLAUDE_API_KEY, or LLM_API_KEY (in that order).

    The callable is cached as a singleton — the Anthropic client is constructed
    only once per kernel session to avoid repeated connection overhead.

    Returns:
        LLMCallable wrapping claude-haiku-4-5-20251001 via the native Anthropic API.
    """
    if _HAIKU_LLM in _MODEL_CACHE:
        logger.debug("Using cached Haiku LLM callable")
        return _MODEL_CACHE[_HAIKU_LLM]

    import os
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "EMPTY"
    )

    from proclaim.verification.llm_factory import make_anthropic_llm

    logger.info("Creating Claude Haiku sufficiency LLM callable (model=claude-haiku-4-5-20251001)...")
    haiku_llm = make_anthropic_llm(api_key=api_key)
    _MODEL_CACHE[_HAIKU_LLM] = haiku_llm
    return haiku_llm


# ---------------------------------------------------------------------------
# Pre-warming and Cache Management
# ---------------------------------------------------------------------------


def prewarm_all_models() -> Dict[str, float]:
    """Pre-load all ML models to avoid first-call latency.

    This function loads all heavy models upfront. It's mandatory to call this
    during kernel setup to ensure consistent performance across all verification
    sessions.

    Returns:
        Dictionary mapping model names to their load times in seconds.
        Example: {"semantic_similarity": 2.3, "nli_entailment": 8.5, ...}
    """
    logger.info("Pre-warming all models...")
    load_times = {}

    # Load each model and record time
    start = time.time()
    get_semantic_similarity_computer()
    load_times["semantic_similarity"] = time.time() - start

    start = time.time()
    get_nli_entailment_computer()
    load_times["nli_entailment"] = time.time() - start

    import os
    backend = os.environ.get("SUFFICIENCY_BACKEND", "mlp").lower()
    if backend not in ("llm", "haiku"):
        start = time.time()
        get_mlp_classifier()
        load_times["mlp_classifier"] = time.time() - start
    else:
        logger.info("Skipping MLP classifier preload (SUFFICIENCY_BACKEND=%s)", backend)

    start = time.time()
    get_metadata_extractor()
    load_times["metadata_extractor"] = time.time() - start

    total_time = sum(load_times.values())
    logger.info(f"All models pre-warmed in {total_time:.2f}s total")

    return load_times


def clear_model_cache():
    """Clear all cached models from memory.

    Useful for testing, debugging, or when you need to free up memory.
    Next call to any getter will reload the model.
    """
    global _MODEL_CACHE
    count = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    logger.info(f"Cleared {count} models from cache")


def get_cache_status() -> Dict[str, bool]:
    """Get the current cache status.

    Returns:
        Dictionary mapping model names to whether they are currently loaded.
        Example: {"semantic_similarity": True, "nli_entailment": False, ...}
    """
    return {
        "semantic_similarity": _SEMANTIC_SIM in _MODEL_CACHE,
        "nli_entailment": _NLI_COMPUTER in _MODEL_CACHE,
        "mlp_classifier": _MLP_CLASSIFIER in _MODEL_CACHE,
        "metadata_extractor": _METADATA_EXTRACTOR in _MODEL_CACHE,
    }


def get_agent_llm():
    """Get or create the outer-agent LLM callable for librarian-run verdicts.

    The ProClaim+librarian pipeline emits the final verdict on the *outer agent*
    model (e.g. ``openai/gpt-5.4``) rather than the Qwen subagent. This builds a
    LiteLLM callable from the ``AGENT_LLM_*`` env vars exported by
    ``build_subprocess_env`` (model, base URL, API key, etc.).

    The callable is cached as a singleton per kernel session.

    Returns:
        LLMCallable wrapping the outer-agent model via LiteLLM.
    """
    if _AGENT_LLM in _MODEL_CACHE:
        logger.debug("Using cached agent LLM callable")
        return _MODEL_CACHE[_AGENT_LLM]

    import os

    model = os.environ.get("AGENT_LLM_MODEL") or os.environ.get("LLM_MODEL", "glm-5")
    base_url = os.environ.get("AGENT_LLM_BASE_URL") or None
    api_key = (
        os.environ.get("AGENT_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "EMPTY"
    )
    temperature = float(os.environ.get("AGENT_LLM_TEMPERATURE", "0.0"))
    timeout = int(os.environ.get("AGENT_LLM_TIMEOUT", "300"))
    stream = os.environ.get("AGENT_LLM_STREAM", "1") == "1"

    from proclaim.verification.llm_factory import make_llm

    logger.info("Creating agent-model LLM callable (model=%s)...", model)
    agent_llm = make_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        stream=stream,
    )
    _MODEL_CACHE[_AGENT_LLM] = agent_llm
    return agent_llm

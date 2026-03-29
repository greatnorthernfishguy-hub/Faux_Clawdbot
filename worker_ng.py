# ---- Changelog ----
# [2026-03-29] Forge (TQB) — Worker NeuroGraph configuration and lifecycle
# What: Dedicated NG instance for the Faux_Clawdbot worker with code-judgment-optimized SNN params
# Why: Worker needs its own isolated substrate tuned for code pattern learning, not conversation
# How: Wraps NeuroGraphMemory with worker-specific config, separate workspace, three-factor learning
# -------------------

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openclaw_hook import NeuroGraphMemory

logger = logging.getLogger("worker_ng")

# Worker-specific SNN config — tuned for code-judgment learning
WORKER_SNN_CONFIG = {
    # Higher learning rate — code patterns are more structured than conversation
    "learning_rate": 0.03,
    # Shorter causal windows — code dependencies are tighter
    "tau_plus": 10.0,
    "tau_minus": 10.0,
    # Stronger LTP/LTD — code patterns should be learned faster
    "A_plus": 1.2,
    "A_minus": 1.4,
    # Standard decay
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 100,
    "weight_threshold": 0.01,
    "grace_period": 500,
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    # Predictive coding
    "prediction_threshold": 3.0,
    "prediction_pre_charge_factor": 0.3,
    "prediction_window": 10,
    "prediction_chain_decay": 0.7,
    "prediction_max_chain_depth": 3,
    "prediction_confirm_bonus": 0.01,
    "prediction_error_penalty": 0.02,
    "prediction_max_active": 1000,
    "surprise_sprouting_weight": 0.1,
    # THREE-FACTOR LEARNING ENABLED — worker should learn from delayed feedback
    "three_factor_enabled": True,
    # Hypergraph
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_experience_threshold": 100,
}

WORKER_NG_WORKSPACE = os.getenv("NEUROGRAPH_WORKSPACE_DIR", "/data/neurograph_worker")


def get_worker_ng() -> NeuroGraphMemory:
    """Get or create the worker's dedicated NeuroGraph instance.

    Uses a separate workspace dir from any ecosystem NG.
    Persists to /data/ on HF Spaces (survives container restarts).
    """
    # Reset singleton to ensure worker gets its own config
    # (The singleton pattern in NeuroGraphMemory is designed for single-instance use;
    #  we override it here because the worker IS the only instance in this container)
    instance = NeuroGraphMemory.get_instance(
        workspace_dir=WORKER_NG_WORKSPACE,
        config=WORKER_SNN_CONFIG,
    )
    # Override auto-save interval — more frequent for build sessions
    instance.auto_save_interval = 5  # every 5 tool calls, not every 10
    return instance


def ingest_tool_result(ng: NeuroGraphMemory, tool_name: str, args: dict, result: str):
    """Ingest a tool execution as raw experience into the worker's substrate.

    Raw experience in. No classification. Law 7.
    """
    # Truncate large results to avoid flooding the substrate
    result_preview = result[:2000] if len(result) > 2000 else result
    raw_experience = f"Tool: {tool_name}\nInput: {args}\nResult: {result_preview}"
    ng.on_message(raw_experience)


def recall_context(ng: NeuroGraphMemory, tool_name: str, context: str, k: int = 3) -> list:
    """Recall relevant past experience before a tool call.

    Drains the bucket. Returns what the substrate has learned.
    """
    query = f"{tool_name} {context}"
    return ng.recall(query, k=k, threshold=0.4)

# ---- Changelog ----
# [2026-03-29] Forge (TQB) — Worker NeuroGraph configuration and lifecycle
# What: Dedicated NG instance for the Faux_Clawdbot worker with code-judgment-optimized SNN params
# Why: Worker needs its own isolated substrate tuned for code pattern learning, not conversation
# How: Wraps NeuroGraphMemory with worker-specific config, separate workspace, three-factor learning
# [2026-04-12] Codemine (BLK-CM-DUALPASS-001) — Dual-pass outcome recording
# What: WorkerEcosystem proxy + record_tool_outcome wired into spec_executor
# Why: Success/failure outcome signals never reached substrate — three_factor_enabled was wired but never fired
# How: WorkerEcosystem.record_outcome calls on_message (STDP traces) then inject_reward (factor 3)
# [2026-03-30] Josh + Claude — Dual-pass ingestion via ng_embed
# What: Tool results ingested with forest (gestalt) + trees (concepts) dual-pass methodology
# Why: Multi-resolution semantic search. Single-pass was operating with one eye closed.
# How: ng_embed.NGEmbed for embedding, OpenRouter for concept extraction, both passes through on_message
# -------------------

from __future__ import annotations

import json
import logging
import os
import requests
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

# Default to local data/ dir. HF Spaces sets NEUROGRAPH_WORKSPACE_DIR=/data/neurograph_worker
_DEFAULT_WORKSPACE = str(Path(__file__).resolve().parent / "data" / "neurograph_worker")
WORKER_NG_WORKSPACE = os.getenv("NEUROGRAPH_WORKSPACE_DIR", _DEFAULT_WORKSPACE)


def get_worker_ng() -> NeuroGraphMemory:
    """Get or create the worker's dedicated NeuroGraph instance.

    Uses a separate workspace dir from any ecosystem NG.
    Persists to /data/ on HF Spaces (survives container restarts).
    """
    instance = NeuroGraphMemory.get_instance(
        workspace_dir=WORKER_NG_WORKSPACE,
        config=WORKER_SNN_CONFIG,
    )
    # Save after every 100 messages — dual-pass sends dozens of on_message calls
    # per tool result (1 forest + N concept trees), so 5 was causing constant disk I/O
    instance.auto_save_interval = 100
    return instance


# ---------------------------------------------------------------------------
# Concept extraction config — OpenRouter for dual-pass Pass 2
# ---------------------------------------------------------------------------

def _get_extraction_config():
    """Lazy config — reads env vars at call time, after dotenv has loaded."""
    return {
        "endpoint": os.getenv("EXTRACTION_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"),
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": os.getenv("EXTRACTION_MODEL", "google/gemini-2.0-flash-001"),
        "max_content": 8000,
        "max_concepts": 100,
        "timeout": 30,
        "temperature": 0.2,
        "max_tokens": 2000,
    }

_EXTRACTION_PROMPT = """Extract the key concepts, terms, and specific references from this text. Return them as a JSON array of short strings, each one a distinct concept or term mentioned in the text.

Focus on:
- Specific technical terms
- Named entities (people, tools, systems)
- Domain-specific concepts
- Action descriptions
- Relationships between things

Return ONLY a JSON array of strings. No explanation. Example: ["concept one", "concept two", "specific term"]

Text:
{content}"""


def _extract_concepts(text: str) -> List[str]:
    """Extract concepts via OpenRouter LLM call (Pass 2 of dual-pass).

    Returns list of concept strings, or empty list on failure (non-fatal).
    """
    cfg = _get_extraction_config()
    api_key = cfg["api_key"]
    if not api_key:
        logger.debug("No OPENROUTER_API_KEY — skipping concept extraction (single-pass only)")
        return []

    content = text[:cfg["max_content"]]
    prompt = _EXTRACTION_PROMPT.format(content=content)

    try:
        resp = requests.post(
            cfg["endpoint"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": "You extract concepts from text. Return only a JSON array of strings."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
            },
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_concepts(response_text)[:cfg["max_concepts"]]
    except Exception as exc:
        logger.debug("Concept extraction failed (non-fatal): %s", exc)
        return []


def _parse_concepts(text: str) -> List[str]:
    """Parse a JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(c).strip() for c in result if str(c).strip()]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
                if isinstance(result, list):
                    return [str(c).strip() for c in result if str(c).strip()]
            except json.JSONDecodeError:
                pass
    return []


# ---------------------------------------------------------------------------
# Dual-pass ingestion
# ---------------------------------------------------------------------------

def ingest_tool_result(ng: NeuroGraphMemory, tool_name: str, args: dict, result: str):
    """Dual-pass ingest a tool execution into the worker's substrate.

    Pass 1 (Forest): Gestalt — the full tool experience as one ingestion.
    Pass 2 (Trees): Concepts extracted via OpenRouter — each concept ingested
    separately, creating concept-level semantic nodes in the substrate.

    Raw experience in for both passes. No classification. Law 7.
    Falls back to single-pass (forest only) if OpenRouter unavailable.
    """
    # Truncate large results to avoid flooding the substrate
    result_preview = result[:2000] if len(result) > 2000 else result
    raw_experience = f"Tool: {tool_name}\nInput: {args}\nResult: {result_preview}"

    # Pass 1: Forest — the gestalt experience
    ng.on_message(raw_experience)

    # Pass 2: Trees — concept extraction via OpenRouter
    concepts = _extract_concepts(raw_experience)
    if concepts:
        for concept in concepts:
            # Each concept enters the substrate as its own raw experience
            # Linked to the forest naturally through temporal co-activation
            # (they fire close together in time → Hebbian → synapses form)
            ng.on_message(f"concept: {concept}")
        logger.info("  Dual-pass: forest + %d trees for %s", len(concepts), tool_name)
    else:
        logger.info("  Single-pass (no OPENROUTER_API_KEY) for %s", tool_name)


def recall_context(ng: NeuroGraphMemory, tool_name: str, context: str, k: int = 3) -> list:
    """Recall relevant past experience before a tool call.

    Drains the bucket. Returns what the substrate has learned.
    Uses query prefix for better retrieval (ng_embed convention).
    """
    from ng_embed import embed
    query = f"{tool_name} {context}"
    return ng.recall(query, k=k, threshold=0.4)


# ---------------------------------------------------------------------------
# Dual-pass outcome recording
# ---------------------------------------------------------------------------

class WorkerEcosystem:
    """Minimal NGEcosystem proxy for dual_record_outcome compatibility.

    NeuroGraphMemory (vendored) wraps neuro_foundation.py's Graph, which
    does not have record_outcome(). This proxy implements the interface
    that ng_embed.dual_record_outcome() expects using the full SNN
    three-factor learning loop that the worker was built for.

    Law 7: raw experience in — no classification before substrate ingestion.
    The outcome-labeled message is semantic content. The substrate learns
    'tool:X passed' / 'tool:X failed' as raw experience, which is correct.
    """

    def __init__(self, ng: NeuroGraphMemory):
        self._ng = ng

    def record_outcome(
        self,
        embedding,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Record outcome via full SNN three-factor learning loop.

        Factor 1+2 (STDP): ng.on_message() ingests the outcome experience,
        fires nodes, and builds eligibility traces via STDP.

        Factor 3 (Reward): ng.graph.inject_reward() broadcasts the reward
        signal — positive for success, negative (half-strength) for failure.
        Final weight change: Δw = eligibility_trace × reward × learning_rate.

        This activates three_factor_enabled=True in WORKER_SNN_CONFIG, which
        has been wired but never triggered until now.
        """
        outcome_label = "success" if success else "failure"
        experience = f"outcome: {target_id} {outcome_label}"
        # Tree-level calls include the concept for richer semantic signal
        if metadata and metadata.get("_concept"):
            concept = metadata["_concept"]
            experience = f"outcome: {target_id} concept:{concept} {outcome_label}"
        # Factor 1+2: ingest experience, STDP builds eligibility traces
        self._ng.on_message(experience)
        # Factor 3: inject reward — confirms or rejects the eligibility traces
        # Half-strength penalty on failure avoids catastrophic forgetting
        reward = strength if success else -strength * 0.5
        self._ng.graph.inject_reward(reward)
        return {"target_id": target_id, "success": success, "reward": reward, "ingested": True}


def record_tool_outcome(
    ng: NeuroGraphMemory,
    tool_name: str,
    target_id: str,
    success: bool,
    strength: float = 1.0,
    context: str = "",
) -> None:
    """Record tool execution outcome via dual-pass to the worker substrate.

    Called after step validation — success=True if validation passed,
    success=False if failed. Uses dual_record_outcome() from ng_embed.py
    for forest (gestalt) + tree (concept) outcome recording.

    Silent on failure — a dead embedding service must never interrupt
    the spec executor's flow.
    """
    try:
        from ng_embed import NGEmbed, embed
        content = f"tool:{tool_name} step:{target_id} {'PASS' if success else 'FAIL'}"
        if context:
            content += f" context:{context[:200]}"
        embedding = embed(content)
        eco = WorkerEcosystem(ng)
        NGEmbed.get_instance().dual_record_outcome(
            ecosystem=eco,
            content=content,
            embedding=embedding,
            target_id=target_id,
            success=success,
            strength=strength,
        )
        logger.info(
            "  Outcome recorded: %s %s", target_id, "PASS" if success else "FAIL"
        )
    except Exception as exc:
        logger.warning("  Outcome recording failed (non-fatal): %s", exc)

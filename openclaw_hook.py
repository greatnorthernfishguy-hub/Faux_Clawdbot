# ---- Changelog ----
# [2026-04-26] Codemine (BLK-FC-215) — Three targeted bug fixes
# What: (A) on_message() wraps ingestor.ingest() in try/except; result=None on failure; return dict gated.
#        (B) graph.config.update(snn_config) re-applied after restore() so checkpoint cannot overwrite code defaults.
#        (C) Unexpected TonicEngine init Exception escalated to logger.warning; _tonic_thread cleared to None.
# Why:  (A) Unbound NameError on result.nodes_created if ingest throws inside the concurrent lock.
#        (B) Pre-tuning config values bleed in from checkpoint on every container restart.
#        (C) Unexpected engine failures were silently swallowed at info level, leaving half-configured tonic state.
# How:  Targeted edits only. Intentional scope differences (no CES, no River, no BrainSwitcher) preserved.
# [2026-04-20] Codemine (BLK-NG-193) — Wire SimpleVectorDB persistence into save/load
#   What: _vector_db_path added; __init__ loads sidecar if exists; save() writes it.
#   Why:  vector_db was recreated empty on every restart — recall() returned
#         nothing after cold start. Fix: vector_db.npz sidecar alongside main.msgpack.
#   How:  SimpleVectorDB.save/load added in universal_ingestor.py (same spec).
# [2026-04-16] Claude (Sonnet 4.6) — Tonic wiring (heuristic mode)
# What: TonicThread + TonicEngine wired into NeuroGraphMemory. _concurrent_lock added
#       to graph. ouroboros_cycle() called on every on_message(). Tonic status in stats().
# Why: Worker NG was dormant between spec executions — nodes never warmed up, nothing
#      fired, zeros everywhere. TonicEngine background thread (2s idle / 0.5s active)
#      keeps substrate alive via heuristic_inference(): thread continuity + attractor
#      pull + prediction tension + exploration. No transformer weights on HF Spaces →
#      _use_heuristic=True automatically. First-class inference path, not a stub.
# How: tonic_thread.py + tonic_engine.py vendored from NeuroGraph canonical.
#      Simplified wiring vs canonical: no BTF River deposit (Codemine has no River),
#      no BrainSwitcher body sharing (no ProtoUniBrain on HF).
# [2026-04-15] Claude (Sonnet 4.6) — v0.4.1 homeostasis audit + three-factor enable
# What: OPENCLAW_SNN_CONFIG: scaling_interval 100→25, threshold_ceiling 5.0 added,
#       three_factor_enabled False→True, tonic disabled, stats() version bumped to 0.4.2
# Why: scaling_interval=100 means homeostatic scaling never fires in ephemeral workers.
#      three_factor_enabled was False — reward learning never fired even with inject_reward.
#      Tonic requires a persistent process; workers are ephemeral.
# -------------------

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuro_foundation import Graph, CheckpointMode
from universal_ingestor import (
    UniversalIngestor,
    SimpleVectorDB,
    SourceType,
    get_ingestor_config,
)

logger = logging.getLogger("neurograph")


# OpenClaw-tuned SNN config: fast learning, tight causal windows
OPENCLAW_SNN_CONFIG = {
    "learning_rate": 0.02,
    "tau_plus": 15.0,
    "tau_minus": 15.0,
    "A_plus": 1.0,
    "A_minus": 1.2,
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 25,        # v0.4.1: lowered from 100 — homeostatic scaling fires more often
    "threshold_ceiling": 5.0,      # v0.4.1: prevents runaway threshold growth
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
    "three_factor_enabled": True,   # reward learning enabled — inject_reward wired in worker_ng
    # Tonic enabled — Gradio process is long-lived; TonicEngine runs in heuristic mode
    # (no transformer weights on HF Spaces → _use_heuristic=True automatically).
    # Background thread fires every 2s (idle) / 0.5s (active spec execution).
    "tonic": {"enabled": True},
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


class NeuroGraphMemory:
    """Singleton cognitive memory layer for OpenClaw integration.

    Wraps NeuroGraph's Graph + UniversalIngestor + SimpleVectorDB into a
    single interface for message-level ingestion, learning, and recall.

    Auto-saves every ``auto_save_interval`` messages (default 10).
    Loads from the latest checkpoint on initialization if one exists.
    """

    _instance: Optional[NeuroGraphMemory] = None

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._workspace_dir = Path(
            workspace_dir
            or os.environ.get("NEUROGRAPH_WORKSPACE_DIR", "~/.openclaw/neurograph")
        ).expanduser()

        self._checkpoint_dir = self._workspace_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoint_path = self._checkpoint_dir / "main.msgpack"
        self._vector_db_path = self._checkpoint_dir / "vector_db.npz"

        # Merge user config over OpenClaw defaults
        snn_config = {**OPENCLAW_SNN_CONFIG, **(config or {})}
        self.graph = Graph(config=snn_config)

        # Concurrent lock — Tonic engine acquires non-blocking; hook ops
        # acquire blocking (waiting for Tonic to finish before mutating graph).
        if not hasattr(self.graph, '_concurrent_lock'):
            self.graph._concurrent_lock = threading.RLock()

        # Restore from checkpoint if one exists
        if self._checkpoint_path.exists():
            try:
                self.graph.restore(str(self._checkpoint_path))
                logger.info(
                    "Restored graph from %s (%d nodes, %d synapses)",
                    self._checkpoint_path,
                    len(self.graph.nodes),
                    len(self.graph.synapses),
                )
                self.graph.config.update(snn_config)
            except Exception as exc:
                logger.warning("Failed to restore checkpoint: %s", exc)

        # Vector DB for semantic search
        self.vector_db = SimpleVectorDB()
        if self._vector_db_path.exists():
            try:
                self.vector_db.load(str(self._vector_db_path))
                logger.info(
                    "Restored vector DB from %s (%d entries)",
                    self._vector_db_path, self.vector_db.count(),
                )
            except Exception as exc:
                logger.warning("Failed to restore vector DB: %s", exc)

        # Ingestor with OpenClaw project config
        ingestor_config = get_ingestor_config("openclaw")
        self.ingestor = UniversalIngestor(
            self.graph, self.vector_db, config=ingestor_config
        )

        self._message_count = 0
        self.auto_save_interval = 10

        # --- The Tonic: Latent Thread + Engine ---
        # Keeps the substrate alive between spec executions via continuous
        # heuristic inference (thread continuity + attractor pull + prediction
        # tension + exploration). No transformer weights needed — heuristic
        # mode is the designed path for Codemine.
        self._tonic_thread = None
        tonic_conf = snn_config.get("tonic", {})
        if tonic_conf.get("enabled", True):
            try:
                from tonic_thread import TonicThread, TonicConfig
                tonic_config = TonicConfig()
                for k, v in tonic_conf.items():
                    if k != "enabled" and hasattr(tonic_config, k):
                        setattr(tonic_config, k, v)
                self._tonic_thread = TonicThread(
                    self.graph, self.vector_db, tonic_config
                )
                logger.info("The Tonic initialized — latent thread live")
                # Latent engine — heuristic inference loop between spec executions.
                # No BrainSwitcher body sharing on Codemine (no ProtoUniBrain).
                try:
                    from tonic_engine import TonicEngine
                    engine = TonicEngine(
                        self.graph, self.vector_db, self._tonic_thread,
                    )
                    self._tonic_thread.set_latent_engine(engine)
                    engine.start()
                    logger.info("Tonic engine running — heuristic mode active")
                except ImportError:
                    logger.info("Tonic engine not available — ouroboros-only mode")
                except Exception as exc:
                    logger.warning("Tonic engine init error: %s — ouroboros-only mode", exc)
                    self._tonic_thread = None
            except Exception as exc:
                logger.info("The Tonic not available: %s", exc)

    @classmethod
    def get_instance(
        cls,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NeuroGraphMemory:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(workspace_dir=workspace_dir, config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def on_message(self, text: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a message, run one STDP learning step, and auto-save.

        Args:
            text: Raw message content to ingest.
            source_type: Override auto-detection (TEXT, MARKDOWN, CODE, etc.).

        Returns:
            Dict with ingestion stats and learning results.
        """
        if not text or not text.strip():
            return {"status": "skipped", "reason": "empty_input"}

        # Stage 1-5: Extract → Chunk → Embed → Register → Associate
        # Acquire graph lock — waits for Tonic engine to finish its current
        # token before mutating graph state (RLock so re-entrant calls are safe).
        with self.graph._concurrent_lock:
            try:
                result = self.ingestor.ingest(text, source_type=source_type)
            except Exception as exc:
                logger.warning("Ingest error: %s", exc)
                result = None

            # Run SNN learning step
            step_result = self.graph.step()

            # Update novelty probation for ingested nodes
            graduated = self.ingestor.update_probation()

            # The Tonic: signal message arrival + ouroboros cycle
            if self._tonic_thread is not None:
                try:
                    self._tonic_thread.message_received()
                    self._tonic_thread.ouroboros_cycle()
                except Exception as exc:
                    logger.debug("Tonic cycle error: %s", exc)

        self._message_count += 1

        # Auto-save
        if self._message_count % self.auto_save_interval == 0:
            self.save()

        if result is None:
            return {"status": "error", "reason": "ingest_failed", "message_count": self._message_count}

        return {
            "status": "ingested",
            "nodes_created": len(result.nodes_created),
            "synapses_created": len(result.synapses_created),
            "hyperedges_created": len(result.hyperedges_created),
            "chunks": result.chunks_created,
            "fired": len(step_result.fired_node_ids),
            "graduated": len(graduated),
            "message_count": self._message_count,
        }

    def recall(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic similarity search over ingested knowledge.

        Args:
            query: Text to search for.
            k: Maximum results to return.
            threshold: Minimum similarity score (0-1).

        Returns:
            List of dicts with 'content', 'similarity', 'node_id', 'metadata'.
        """
        return self.ingestor.query_similar(query, k=k, threshold=threshold)

    def step(self, n: int = 1) -> List[Any]:
        """Run N SNN learning steps without ingestion."""
        results = []
        for _ in range(n):
            results.append(self.graph.step())
        return results

    def save(self) -> str:
        """Save graph state to checkpoint. Returns the checkpoint path."""
        self.graph.checkpoint(str(self._checkpoint_path), mode=CheckpointMode.FULL)
        try:
            self.vector_db.save(str(self._vector_db_path))
        except Exception as exc:
            logger.warning("Vector DB save failed (non-fatal): %s", exc)
        logger.info("Checkpoint saved to %s", self._checkpoint_path)
        return str(self._checkpoint_path)

    def stats(self) -> Dict[str, Any]:
        """Return current graph statistics and telemetry."""
        tel = self.graph.get_telemetry()
        result = {
            "version": "0.4.2",
            "timestep": tel.timestep,
            "nodes": tel.total_nodes,
            "synapses": tel.total_synapses,
            "hyperedges": tel.total_hyperedges,
            "firing_rate": round(tel.global_firing_rate, 4),
            "mean_weight": round(tel.mean_weight, 4),
            "predictions_made": tel.total_predictions_made,
            "predictions_confirmed": tel.total_predictions_confirmed,
            "prediction_accuracy": round(tel.prediction_accuracy, 4),
            "novel_sequences": tel.total_novel_sequences,
            "pruned": tel.total_pruned,
            "sprouted": tel.total_sprouted,
            "vector_db_count": self.vector_db.count(),
            "checkpoint": str(self._checkpoint_path),
            "message_count": self._message_count,
        }
        if self._tonic_thread is not None:
            result["tonic"] = self._tonic_thread.status
        return result

    def ingest_file(self, path: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a file from disk."""
        p = Path(path).expanduser()
        if not p.exists():
            return {"status": "error", "reason": f"File not found: {path}"}

        content = p.read_text(errors="replace")

        # Auto-detect source type from extension
        if source_type is None:
            ext = p.suffix.lower()
            type_map = {
                ".py": SourceType.CODE,
                ".js": SourceType.CODE,
                ".ts": SourceType.CODE,
                ".md": SourceType.MARKDOWN,
                ".html": SourceType.URL,
                ".htm": SourceType.URL,
                ".pdf": SourceType.PDF,
            }
            source_type = type_map.get(ext, SourceType.TEXT)

        return self.on_message(content, source_type=source_type)

    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ingest all matching files from a directory.

        Args:
            directory: Path to directory.
            extensions: File extensions to include (e.g. ['.py', '.md']).
                       Default: ['.py', '.js', '.ts', '.md', '.txt']
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of ingestion results per file.
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".md", ".txt"]

        d = Path(directory).expanduser()
        if not d.is_dir():
            return [{"status": "error", "reason": f"Not a directory: {directory}"}]

        results = []
        pattern = "**/*" if recursive else "*"
        for fp in sorted(d.glob(pattern)):
            if fp.is_file() and fp.suffix.lower() in extensions:
                res = self.ingest_file(str(fp))
                res["file"] = str(fp)
                results.append(res)

        # Save after batch ingestion
        self.save()
        return results

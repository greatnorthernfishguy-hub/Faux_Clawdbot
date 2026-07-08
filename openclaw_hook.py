# ---- Changelog ----
# [2026-07-08] Meridian (TQB/QB build) — #256 anticipatory pre-activation + GSG surfacing re-score
# What: (1) _anticipate(): after each on_message() SNN step, walk outgoing
#       synapses from the just-fired set, score neighbors by accumulated edge
#       weight, keep top-K with a TTL (instance state: self._primed_nodes).
#       (2) _harvest_associations(): surfaced items still primed (unexpired)
#       gain _ANTICIPATE_BONUS on strength; then _gsg_rescore_surfaced() adds
#       _GSG_SCORE_BONUS/(1+geodesic_dist) for stamped nodes (Poincaré geodesic
#       for hyperbolic, great-circle for spherical); single re-sort by strength
#       desc ONLY if a bonus was applied, before [:max_surfaced] truncation —
#       otherwise the existing (latency, -strength) ordering stands untouched.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.4/§7.4 (#256 was never
#      built in the fork) and §2.5/§7.5 (GSG re-score absent). Retrieval
#      boundary only — prime_and_propagate, _score_node, STDP and all
#      learning math untouched.
# How: _anticipate() is canonical neurograph_rpc.py:2716-2742 verbatim, fork-
#      adapted per PRD §7.4: primed state on the NeuroGraphMemory singleton
#      (self._primed_nodes) instead of a module global. _poincare_distance()
#      and the re-score block are canonical rpc.py:2991-3038 (CC-port shape,
#      §7.5). Constants copied VERBATIM, source-annotated, test-pinned
#      (tests/test_anticipate_and_gsg_rescore.py): top-K 15, TTL 120.0s,
#      bonus 0.25, layer norms (0.70, 0.50, 0.30), GSG bonus 0.30. Every
#      block independently fail-soft: anticipate, bonus application, and
#      re-score each degrade to current behavior on error — none can empty
#      the harvest or break a worker turn.
# [2026-07-08] Meridian (TQB/QB build) — GSG poincare_dir ingest stamping + backfill
# What: (1) Module-level _embed_to_poincare_dir() (verbatim shape from canonical
#       neurograph_rpc.py). (2) on_message() stamps node.metadata["poincare_dir"]
#       on each id in result.nodes_created after a successful ingest, from the
#       node's stored vdb embedding (SimpleVectorDB.insert() L2-normalizes on
#       storage, so stored embeddings ARE unit directions — zero model calls).
#       (3) _gsg_backfill_poincare_dirs(): one-time idempotent backfill at the
#       end of __init__ (after graph restore + vdb load) for pre-existing nodes.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.3/§7.3 — the fork's
#      node-creation path never wrote poincare_dir, so shipped GSG Phase 3
#      (non-Euclidean message passing) and Phase 4 (spherical manifolds) were
#      permanent no-ops: the #358 audit found zero write sites. poincare_dir is
#      a geometric projection of the raw embedding, not a label — no LAW 7
#      classification-before-deposit.
# How: Both write sites are hook-side (universal_ingestor.py is canonical-synced
#      and untouched). Each is independently fail-soft: a stamping failure
#      degrades to current behavior and cannot affect the harvest, the step, or
#      the return dict. PERSISTENCE CHOICE (PRD §2.3 requires deciding +
#      documenting): backfill is STAMP-ONLY — NO force-save. Canonical's
#      backfill force-saves; Codemine deliberately does not, because
#      on_message() already autosaves every auto_save_interval messages, which
#      persists stamps on the normal cadence without adding a boot-time full
#      checkpoint write on a live Space. Tests: tests/test_gsg_stamping.py.
# [2026-07-08] Meridian (TQB/QB build) — Orphan-seed harvest guard
# What: _harvest_associations() seed loop now skips vdb hits whose graph node no
#       longer exists (orphaned by pruning): `entry_id in self.graph.nodes`.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.1/§7.1 — a single dead seed ID
#      entered prime_and_propagate's working set, raised KeyError in its decay
#      loop, and the harvest's own fail-soft swallowed it at debug level: the
#      ENTIRE surfacing result for that call silently became []. This is
#      Codemine's production recall path. Same one-line membership guard
#      canonical took (openclaw_hook.py, 2026-07-07, commit b2213df).
# How: Membership guard at the seed boundary — restores graph primacy over raw
#      vdb hits. Regression test: tests/test_harvest_orphan_seeds.py.
# [2026-07-05] Claude Code (Sonnet 5) — Torn-read fix + grace_period fix + CES wiring
# What: (1) _wait_for_stable_checkpoint() ported from canonical, wired into both restore
#       call sites (main.msgpack, vector_db.npz) — closes the same torn-read-during-
#       autosave bug that caused real checkpoint collapses on the VPS and laptop CC-NG
#       instances. (2) grace_period 500->5000, porting Syl's 2026-06-25 fix (never
#       applied here). (3) CES (StreamParser, ActivationPersistence, SurfacingMonitor,
#       CESMonitor) added as a new, additive block — was entirely absent (not staleness;
#       never wired into the original build, same "Syl-specific, not needed" mistake
#       named in docs/concepts/NeuroGraph Is a Mind, Not a Database.md). Feed/after_step
#       wired into on_message(); save/restore wired into save()/__init__(). Dashboard
#       stays opt-in (NEUROGRAPH_CES_DASHBOARD unset here) so it never binds a port on
#       the HF Space.
# Why:  All three are core substrate-safety/completeness fixes, not Codemine-specific
#       feature work — same class of gap this file's own changelog history already
#       treats seriously (see 2026-05-30 entry: "omission, not design choice").
# How:  Surgical, additive only. Every existing Codemine-specific adaptation preserved
#       verbatim: .npz vector format, ~/.openclaw/neurograph internal default (dead code
#       in the real path — verified via app.py -> worker_ng.py -> NEUROGRAPH_WORKSPACE_DIR
#       env var, real value is /data/neurograph_worker on the HF Space), heuristic-only
#       Tonic (no River, no BrainSwitcher), all three existing BLK-FC-215 bug fixes.
#       Nothing removed, nothing replaced.
# [2026-05-30] Claude Code (Sonnet 4.6) — Transplant _harvest_associations() + surprise-weighted surfacing
#   What: Added _harvest_associations() — full SNN spreading activation harvest with GSG-aware geodesic
#         routing (spherical arccos for attractor nodes, Poincaré for hyperbolic), surprise-weighted
#         parameter scaling via _substrate_novelty_ema (EMA α=0.1 of predictions_surprised/total).
#         recall() now delegates to _harvest_associations() instead of plain ingestor.query_similar().
#         on_message() harvests auto-knowledge associations after ingest + updates novelty EMA after step.
#   Why:  FC was discarding the most valuable part of the live SNN: topology-traversal recall.
#         _harvest_associations() didn't exist when FC was built — omission, not design choice.
#         With neuro_foundation.py now carrying full GSG (manifold_type, prime_and_propagate), FC
#         has everything needed to support it.
#   How:  _harvest_associations() verbatim from NeuroGraph canonical openclaw_hook.py (c32cc9e).
#         Novelty EMA update lives in on_message() (FC has no neurograph_rpc.py to own it).
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
from typing import Any, Dict, List, Optional, Tuple

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
    "grace_period": 5000,  # [2026-07-05] 500->5000, porting Syl's 2026-06-25 fix (openclaw_hook.py
                            # OPENCLAW_SNN_CONFIG canonical) — age-cull was reaping connections in
                            # ~17min of graph-time before they could double their weight, starving
                            # the graph. Same bug, never ported to Codemine's config until now.
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


# #256 anticipatory pre-activation + GSG surfacing re-score constants — copied
# VERBATIM from canonical neurograph_rpc.py (_anticipate, rpc.py:2716-2742;
# GSG assemble block, rpc.py:2991-3038) per PRD
# 2026-07-08-codemine-surfacing-parity §7.4/§7.5. Do NOT tune — exact values
# pinned by tests/test_anticipate_and_gsg_rescore.py.
_ANTICIPATE_TOP_K = 15
_ANTICIPATE_TTL_S = 120.0
_ANTICIPATE_BONUS = 0.25
_GSG_LAYER_NORMS = (0.70, 0.50, 0.30)   # diffpc_layer 0/1/2 -> Poincaré norm
_GSG_SCORE_BONUS = 0.30


def _wait_for_stable_checkpoint(path: str, max_wait: float = 10.0, check_interval: float = 0.5) -> bool:
    """Poll a checkpoint file's size until it stops changing.

    Returns True once two consecutive size reads agree (write complete, or
    file untouched during the poll window). Returns False if the file never
    stabilizes within max_wait — caller must NOT read it in that case; a
    file that's still growing/shrinking is mid-write, and reading it now
    risks a torn deserialization that then gets silently treated as an
    empty checkpoint. Ported from NeuroGraph canonical (2026-07-03) after
    the same class of bug caused real checkpoint collapses on the VPS and
    laptop CC-NG instances. Missing file is not instability — returns True
    immediately (existing exists() checks at call sites handle that case).
    """
    if not os.path.exists(path):
        return True
    deadline = time.time() + max_wait
    last_size = -1
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(check_interval)
            continue
        if size == last_size:
            return True
        last_size = size
        time.sleep(check_interval)
    logger.warning('%s did not stabilize within %.1fs — likely mid-write, skipping restore', path, max_wait)
    return False


def _embed_to_poincare_dir(embedding):
    """Project a raw embedding to a unit direction for GSG Poincaré placement.

    Verbatim shape from canonical neurograph_rpc.py _embed_to_poincare_dir
    (CC's identical port is _cc_embed_to_poincare_dir) — PRD
    2026-07-08-codemine-surfacing-parity §7.3. A geometric projection of the
    raw embedding, not a label (LAW 7 clean). Returns None for zero-norm
    input so callers skip stamping rather than storing a degenerate direction.
    """
    import numpy as _np
    v = _np.asarray(embedding, dtype=_np.float32)
    n = float(_np.linalg.norm(v))
    if n <= 0:
        return None
    return v / n


def _poincare_distance(x, y):
    """Poincaré-ball geodesic distance between two points.

    Verbatim from canonical neurograph_rpc.py's GSG assemble block
    (rpc.py:2991-3038) — PRD 2026-07-08-codemine-surfacing-parity §7.5.
    """
    import numpy as _np
    import math
    nx2 = min(float(_np.dot(x, x)), 0.9999)
    ny2 = min(float(_np.dot(y, y)), 0.9999)
    diff = x - y
    num = 2.0 * float(_np.dot(diff, diff))
    denom = (1.0 - nx2) * (1.0 - ny2)
    arg = 1.0 + num / max(denom, 1e-9)
    return math.acosh(max(1.0, arg))


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
        if self._checkpoint_path.exists() and not _wait_for_stable_checkpoint(str(self._checkpoint_path)):
            logger.warning(
                "Checkpoint %s mid-write — skipping restore this init (graph starts empty)",
                self._checkpoint_path,
            )
        elif self._checkpoint_path.exists():
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
        if self._vector_db_path.exists() and not _wait_for_stable_checkpoint(str(self._vector_db_path)):
            logger.warning(
                "Vector DB %s mid-write — skipping restore this init (vdb starts empty)",
                self._vector_db_path,
            )
        elif self._vector_db_path.exists():
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
        self._substrate_novelty_ema: float = 0.5  # MMN EMA for surprise-weighted surfacing (#255)
        # #256 anticipatory pre-activation (PRD §2.4/§7.4): node_id ->
        # (score, expiry). Instance state — the fork owns its singleton, so
        # canonical's module-global _primed_nodes becomes an instance attr.
        self._primed_nodes: Dict[str, Tuple[float, float]] = {}

        # --- CES: Cognitive Enhancement Suite ---
        # Ported from NeuroGraph canonical (2026-07-05) — was entirely absent from this
        # file (not staleness; the original build never wired it in). StreamParser
        # (real-time attention pre-activation), ActivationPersistence (cross-session
        # voltage continuity — directly relevant here since Codemine already backs up
        # NG state every N saves specifically to persist across separate spec runs;
        # without this the substrate still starts voltage-cold each run despite that),
        # SurfacingMonitor (associative recall without explicit search), and CESMonitor
        # (health context string; HTTP dashboard stays opt-in via NEUROGRAPH_CES_DASHBOARD,
        # unset here so it never binds a port on the HF Space). None of this is
        # OpenClaw/Syl-specific — constructors take explicit graph/vector_db/config,
        # no global state, same shape as the Tonic wiring already in this file.
        self._ces_config = None
        self._stream_parser = None
        self._activation_persistence = None
        self._surfacing_monitor = None
        self._ces_monitor = None

        ces_conf = (config or {}).get("ces", {})
        if ces_conf.get("enabled", True):
            try:
                from ces_config import load_ces_config
                from stream_parser import StreamParser
                from activation_persistence import ActivationPersistence
                from surfacing import SurfacingMonitor
                from ces_monitoring import CESMonitor

                self._ces_config = load_ces_config(ces_conf)
                self._stream_parser = StreamParser(
                    self.graph,
                    self.vector_db,
                    self._ces_config,
                    fallback_embedder=self.ingestor.embedder.embed_text,
                )
                self._activation_persistence = ActivationPersistence(self._ces_config)
                self._surfacing_monitor = SurfacingMonitor(
                    self.graph, self.vector_db, self._ces_config
                )
                self._ces_monitor = CESMonitor(self, self._ces_config)
                self._ces_monitor._surfacing_monitor = self._surfacing_monitor

                if self._checkpoint_path.exists():
                    self._activation_persistence.restore(
                        self.graph, str(self._checkpoint_path)
                    )

                if os.environ.get("NEUROGRAPH_CES_DASHBOARD", "0") == "1":
                    self._ces_monitor.start()
                logger.info("CES modules initialized")
            except Exception as exc:
                logger.info("CES not available: %s", exc)

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

        # --- GSG poincare_dir backfill (PRD §2.3/§7.3) ---
        # One-time, idempotent, STAMP-ONLY (no force-save — deliberate
        # persistence choice, see changelog: on_message()'s autosave cadence
        # persists stamps). Runs after graph restore + vdb load so pre-existing
        # nodes finally feed GSG Phases 3/4, which shipped inert (zero
        # poincare_dir write sites — #358 audit). Fail-soft: a backfill
        # failure degrades to current (unstamped) behavior.
        try:
            _stamped = self._gsg_backfill_poincare_dirs()
            if _stamped:
                logger.info(
                    "GSG backfill: stamped poincare_dir on %d existing nodes", _stamped
                )
        except Exception as exc:
            logger.warning("GSG poincare_dir backfill failed (non-fatal): %s", exc)

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

            # GSG poincare_dir ingest stamping (PRD §2.3/§7.3, success path
            # only): stamp each node created this message from its stored vdb
            # embedding — SimpleVectorDB.insert() L2-normalized it on storage,
            # so it IS the unit direction (zero model calls). Independently
            # fail-soft: a stamping failure cannot affect the harvest, the
            # step, or the return dict.
            if result is not None:
                try:
                    for _nid in result.nodes_created:
                        _node = self.graph.nodes.get(_nid)
                        if _node is None or (_node.metadata or {}).get("poincare_dir"):
                            continue
                        _emb = self.vector_db.embeddings.get(_nid)
                        if _emb is None:
                            continue
                        _dir = _embed_to_poincare_dir(_emb)
                        if _dir is None:
                            continue
                        if _node.metadata is None:
                            _node.metadata = {}
                        _node.metadata["poincare_dir"] = _dir.tolist()
                except Exception as exc:
                    logger.debug("GSG ingest stamping failed (non-fatal): %s", exc)

            # Auto-knowledge: spreading activation harvest before step (success path only)
            surfaced: List[Dict[str, Any]] = []
            if result is not None and self.graph.config.get("auto_knowledge_enabled", True) and self.vector_db.count() > 0:
                surfaced = self._harvest_associations(text, set(result.nodes_created))

            # Run SNN learning step
            step_result = self.graph.step()

            # #256 anticipatory pre-activation (PRD §2.4/§7.4): walk outgoing
            # synapses from the just-fired set and prime likely-next nodes
            # (top-K by accumulated edge weight, TTL'd). Independently
            # fail-soft: on error, degrade to no anticipation — never break
            # the turn.
            try:
                self._anticipate(step_result.fired_node_ids)
            except Exception as exc:
                logger.debug("Anticipate failed (non-fatal): %s", exc)
                self._primed_nodes = {}

            # Update MMN novelty EMA for surprise-weighted surfacing (#255)
            _pc_total = getattr(step_result, "predictions_confirmed", 0) + getattr(step_result, "predictions_surprised", 0)
            if _pc_total > 0:
                _raw_novelty = getattr(step_result, "predictions_surprised", 0) / _pc_total
                self._substrate_novelty_ema = 0.9 * self._substrate_novelty_ema + 0.1 * _raw_novelty

            # Update novelty probation for ingested nodes
            graduated = self.ingestor.update_probation()

            # The Tonic: signal message arrival + ouroboros cycle
            if self._tonic_thread is not None:
                try:
                    self._tonic_thread.message_received()
                    self._tonic_thread.ouroboros_cycle()
                except Exception as exc:
                    logger.debug("Tonic cycle error: %s", exc)

            # CES: feed stream parser (success path only)
            if self._stream_parser is not None and result is not None:
                self._stream_parser.feed(text)

            # CES: surfacing monitor — scan fired nodes for relevant concepts
            ces_surfaced: List[Dict[str, Any]] = []
            if self._surfacing_monitor is not None:
                self._surfacing_monitor.after_step(step_result)
                ces_surfaced = self._surfacing_monitor.get_surfaced()

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
            "surfaced": len(surfaced),
            "ces_surfaced": ces_surfaced,
            "graduated": len(graduated),
            "message_count": self._message_count,
        }

    def recall(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic + SNN spreading activation recall.

        Args:
            query: Text to search for.
            k: Maximum results to return.
            threshold: Minimum similarity gate for seed node selection.

        Returns:
            List of dicts with 'content', 'metadata', 'node_id', 'latency', 'strength'.
        """
        old_max = self.graph.config.get("max_surfaced", 10)
        old_thresh = self.graph.config.get("prime_threshold", 0.4)
        self.graph.config["max_surfaced"] = k
        self.graph.config["prime_threshold"] = threshold
        try:
            return self._harvest_associations(query, novelty=getattr(self, "_substrate_novelty_ema", 0.5))
        finally:
            self.graph.config["max_surfaced"] = old_max
            self.graph.config["prime_threshold"] = old_thresh

    def _harvest_associations(
        self,
        text: str,
        exclude_node_ids: Optional[set] = None,
        novelty: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Semantic priming + spreading activation harvest with GSG-aware geodesic routing.

        Embeds the input text, finds similar existing nodes via the vector DB,
        injects current into those nodes, runs N SNN steps, and harvests
        everything that fires. The result is knowledge the network
        *associatively connects* with the input — no explicit search needed.

        Returns:
            List of surfaced knowledge dicts sorted by association strength.
        """
        if exclude_node_ids is None:
            exclude_node_ids = set()

        snn_config = self.graph.config
        prime_k = snn_config.get("prime_k", 10)
        prime_threshold = snn_config.get("prime_threshold", 0.4)
        prime_strength = snn_config.get("prime_strength", 1.0)
        propagation_steps = snn_config.get("propagation_steps", 3)
        max_surfaced = snn_config.get("max_surfaced", 10)

        # Surprise-weighted surfacing (#255): scale retrieval aggressiveness by MMN novelty.
        # novelty ∈ [0,1]; novelty_scale ∈ [-1,+1]. High novelty → cast wider/deeper.
        ns = (novelty - 0.5) * 2.0  # novelty_scale
        prime_k = max(5, round(prime_k * (1.0 + ns * 0.5)))
        prime_threshold = max(0.15, prime_threshold * (1.0 - ns * 0.3))
        propagation_steps = max(1, round(propagation_steps * (1.0 + ns * 0.4)))
        max_surfaced = max(5, round(max_surfaced * (1.0 + ns * 0.3)))

        try:
            query_vec = self.ingestor.embedder.embed_text(text)
            similar = self.vector_db.search(
                query_vec, k=prime_k, threshold=prime_threshold
            )

            # Filter out newly created nodes (they ARE the input) and vdb
            # entries whose graph node no longer exists (orphaned by
            # pruning) — a single dead seed ID entered prime_and_propagate's
            # working set, raised KeyError in its decay loop, and silently
            # killed the ENTIRE harvest (caught below, returned [], logged
            # at debug only).
            prime_ids = []
            prime_currents = []
            for entry_id, sim_score in similar:
                if entry_id not in exclude_node_ids and entry_id in self.graph.nodes:
                    prime_ids.append(entry_id)
                    prime_currents.append(sim_score * prime_strength)

            if not prime_ids:
                return []

            propagation = self.graph.prime_and_propagate(
                node_ids=prime_ids,
                currents=prime_currents,
                steps=propagation_steps,
            )

            surfaced = []
            seen = set()
            for entry in propagation.fired_entries:
                if entry.node_id in exclude_node_ids:
                    continue
                if entry.node_id in seen:
                    continue
                seen.add(entry.node_id)

                db_entry = self.vector_db.get(entry.node_id)
                if db_entry is not None:
                    surfaced.append({
                        "node_id": entry.node_id,
                        "content": db_entry.get("content", ""),
                        "metadata": db_entry.get("metadata", {}),
                        "latency": entry.firing_step,
                        "strength": entry.voltage_at_fire,
                        "was_predicted": entry.was_predicted,
                    })

            surfaced.sort(key=lambda x: (x["latency"], -x["strength"]))

            # --- #256 anticipate bonus (PRD §2.4/§7.4) ---
            # Surfaced nodes still primed (unexpired) from the last step's
            # fired-set walk gain _ANTICIPATE_BONUS on strength.
            # Independently fail-soft: an error here degrades to the
            # un-bonused list — it must never reach the outer except (which
            # would empty the whole harvest).
            _bonus_applied = False
            try:
                _primed = getattr(self, "_primed_nodes", None)
                if _primed:
                    _now = time.time()
                    for item in surfaced:
                        _pentry = _primed.get(item["node_id"])
                        if _pentry is not None and _now < _pentry[1]:
                            item["strength"] += _ANTICIPATE_BONUS
                            _bonus_applied = True
            except Exception as exc:
                logger.debug("Anticipate bonus failed (non-fatal): %s", exc)

            # --- GSG surfacing re-score (PRD §2.5/§7.5) ---
            # Independently fail-soft: any error returns the list un-rescored.
            try:
                if self._gsg_rescore_surfaced(surfaced, query_vec):
                    _bonus_applied = True
            except Exception as exc:
                logger.debug("GSG re-score failed (non-fatal): %s", exc)

            # Canonical's shape: single re-sort by strength desc — ONLY if a
            # bonus actually changed strengths. Otherwise the existing
            # (latency, -strength) ordering stands untouched.
            if _bonus_applied:
                try:
                    surfaced.sort(key=lambda x: x["strength"], reverse=True)
                except Exception as exc:
                    logger.debug("Post-bonus re-sort failed (non-fatal): %s", exc)

            return surfaced[:max_surfaced]

        except Exception as exc:
            logger.debug("Auto-knowledge harvest failed: %s", exc)
            return []

    def _anticipate(self, fired_node_ids: List[str]) -> None:
        """#256 anticipatory pre-activation: prime likely-next nodes.

        Walk outgoing synapses from the just-fired set, score non-fired
        neighbors by accumulated edge weight, keep the top
        _ANTICIPATE_TOP_K with a _ANTICIPATE_TTL_S expiry. Port of canonical
        _anticipate() (neurograph_rpc.py:2716-2742) — PRD
        2026-07-08-codemine-surfacing-parity §2.4/§7.4. Fork adaptation per
        the PRD: primed state lives on this instance (self._primed_nodes)
        instead of canonical's module global, so canonical's `_memory is
        None` guard is unnecessary here; the walk logic and constants are
        verbatim. Read-only over the graph — no learning math touched.
        """
        if not fired_node_ids:
            self._primed_nodes = {}
            return
        fired_set = set(fired_node_ids)
        candidates: Dict[str, float] = {}
        for nid in fired_node_ids:
            for sid in self.graph._outgoing.get(nid, ()):
                syn = self.graph.synapses.get(sid)
                if syn is None:
                    continue
                target = syn.post_node_id
                if target not in fired_set and target in self.graph.nodes:
                    candidates[target] = candidates.get(target, 0.0) + syn.weight
        top_k = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:_ANTICIPATE_TOP_K]
        expiry = time.time() + _ANTICIPATE_TTL_S
        self._primed_nodes = {nid: (score, expiry) for nid, score in top_k}

    def _gsg_rescore_surfaced(
        self,
        surfaced: List[Dict[str, Any]],
        query_vec: Any,
    ) -> bool:
        """Re-score surfaced items by geodesic proximity to the query.

        Canonical's assemble-block shape (neurograph_rpc.py:2991-3038,
        CC-port) — PRD 2026-07-08-codemine-surfacing-parity §2.5/§7.5.
        Stamped nodes gain _GSG_SCORE_BONUS/(1+dist): Poincaré geodesic for
        hyperbolic nodes (layer-normed positions), great-circle for
        spherical. Unstamped nodes are untouched. Bonuses are computed for
        ALL items first, then committed — an exception mid-compute leaves
        every strength untouched (canonical: any exception returns the list
        un-rescored). Sorting stays with the caller.

        Returns:
            True if at least one bonus was applied.
        """
        import math
        import numpy as _np

        query_dir = _embed_to_poincare_dir(query_vec)
        if query_dir is None:
            return False

        bonuses = []
        for item in surfaced:
            node = self.graph.nodes.get(item["node_id"])
            if node is None:
                continue
            pd = (node.metadata or {}).get("poincare_dir")
            if not pd:
                continue
            node_dir = _np.asarray(pd, dtype=_np.float32)
            layer = max(0, min(2, getattr(node, "diffpc_layer", 0)))
            mtype = getattr(node, "manifold_type", "hyperbolic")
            if mtype == "spherical":
                cos = min(1.0 - 1e-7, max(-1.0 + 1e-7,
                          float(_np.dot(query_dir, node_dir))))
                bonus = _GSG_SCORE_BONUS / (1.0 + math.acos(cos))
            else:
                node_pt = node_dir * _GSG_LAYER_NORMS[layer]
                query_pt = query_dir * _GSG_LAYER_NORMS[0]
                bonus = _GSG_SCORE_BONUS / (1.0 + _poincare_distance(query_pt, node_pt))
            bonuses.append((item, bonus))

        for item, bonus in bonuses:
            item["strength"] += bonus
        return bool(bonuses)

    def _gsg_backfill_poincare_dirs(self) -> int:
        """Stamp poincare_dir on existing nodes from their stored vdb embeddings.

        Verbatim shape from canonical _gsg_backfill_existing_nodes
        (neurograph_rpc.py:2683-2713) MINUS its trailing force-save — PRD
        2026-07-08-codemine-surfacing-parity §2.3/§7.3. Stored embeddings are
        already L2-normalized (SimpleVectorDB.insert()), so they ARE unit
        directions — zero model calls. Idempotent: already-stamped nodes are
        skipped, so a second run stamps 0. STAMP-ONLY: persistence rides
        on_message()'s normal autosave cadence (documented persistence choice,
        see changelog).

        Returns:
            Number of nodes stamped this run.
        """
        stamped = 0
        for node_id, node in list(self.graph.nodes.items()):
            if (node.metadata or {}).get("poincare_dir"):
                continue
            emb = self.vector_db.embeddings.get(node_id)
            if emb is None:
                continue
            if node.metadata is None:
                node.metadata = {}
            node.metadata["poincare_dir"] = emb.tolist()
            stamped += 1
        return stamped

    def step(self, n: int = 1) -> List[Any]:
        """Run N SNN learning steps without ingestion."""
        results = []
        for _ in range(n):
            results.append(self.graph.step())
        return results

    def save(self) -> str:
        """Save graph state to checkpoint. Returns the checkpoint path."""
        self.graph.checkpoint(str(self._checkpoint_path), mode=CheckpointMode.FULL)
        # CES: save activation sidecar alongside checkpoint (restore side is in __init__)
        if self._activation_persistence is not None:
            try:
                self._activation_persistence.save(self.graph, str(self._checkpoint_path))
            except Exception as exc:
                logger.warning("Activation sidecar save failed (non-fatal): %s", exc)
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

# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: WorkspaceTool
# What: map_repository_structure, get_stats extracted from RecursiveContextManager
# Why: PRD Block C — single-responsibility tool classes
# How: AST-based repo mapping and NG stats; ng instance passed for stats retrieval
# [2026-03-30] QB — Block D: Error handling hardening
# What: Specific exception types, logger replaces print()
# Why: PRD Block D — no broad except Exception, structured logging
# How: Catch OSError/SyntaxError specifically; use logger.warning for NG stats
# -------------------

import ast
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("tools.workspace")


class WorkspaceTool:
    """Repository mapping and statistics."""

    def __init__(self, repo_path: Path, ng, policy_engine=None):
        self.repo_path = repo_path
        self.ng = ng
        self.policy_engine = policy_engine

    def map_repository_structure(self) -> str:
        graph = {"nodes": [], "edges": []}
        try:
            file_count = 0
            for file_path in self.repo_path.rglob('*.py'):
                if 'venv' in str(file_path):
                    continue
                rel_path = str(file_path.relative_to(self.repo_path))
                content = file_path.read_text(errors='ignore')
                file_count += 1
                graph["nodes"].append({"id": rel_path, "type": "file"})
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            node_id = f"{rel_path}::{node.name}"
                            graph["nodes"].append({"id": node_id, "type": "function"})
                except SyntaxError:
                    continue
            return f"Map Generated: {file_count} files, {len(graph['nodes'])} nodes."
        except (OSError, PermissionError) as e:
            logger.error("[workspace] map_repository_structure failed: %s: %s", type(e).__name__, e, exc_info=True)
            return {"status": "error", "tool": "workspace", "error": str(e), "type": type(e).__name__}

    def get_stats(self) -> Dict:
        ng_stats = {}
        try:
            ng_stats = self.ng.stats()
        except (OSError, ValueError, AttributeError) as e:
            logger.warning("NG stats retrieval failed: %s: %s", type(e).__name__, e)
        return {
            "total_files": len(list(self.repo_path.rglob("*"))),
            "conversations": ng_stats.get("message_count", 0),
            "ng_nodes": ng_stats.get("nodes", 0),
            "ng_synapses": ng_stats.get("synapses", 0),
            "ng_firing_rate": ng_stats.get("firing_rate", 0.0),
            "ng_prediction_accuracy": ng_stats.get("prediction_accuracy", 0.0),
        }

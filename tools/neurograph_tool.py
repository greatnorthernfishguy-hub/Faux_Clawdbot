# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: NeurographTool
# What: search_conversations, search_code, search_testament, ingest_workspace extracted
# Why: PRD Block C — DRY the three identical search methods into one _search(), single responsibility
# How: _search(query, n, domain) is the single recall path; public methods format differently
# [2026-03-30] QB — Block D: Error handling hardening
# What: Specific exception types, logger instead of broad catches, fixed can_access_path call
# Why: PRD Block D — no broad except Exception, structured logging
# How: Catch (OSError, ValueError, KeyError) for NG ops; fix policy_engine.can_access_path signature
# -------------------

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("tools.neurograph")


class NeurographTool:
    """NeuroGraph-backed search and workspace ingestion."""

    def __init__(self, repo_path: Path, ng, policy_engine=None):
        self.repo_path = repo_path
        self.ng = ng
        self.policy_engine = policy_engine

    def _search(self, query: str, n: int, domain: str) -> List[Dict]:
        """Single recall path — all search methods delegate here."""
        try:
            results = self.ng.recall(query, k=n, threshold=0.3)
            if domain == "conversations":
                return [{
                    "content": r.get("content", ""),
                    "similarity": r.get("similarity", 0),
                    "id": r.get("node_id", "")
                } for r in results]
            else:
                # code and testament share the same format
                return [{
                    "file": r.get("metadata", {}).get("source", "memory"),
                    "snippet": r.get("content", "")[:500]
                } for r in results]
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.error("[neurograph] _search(%s) failed: %s: %s", domain, type(e).__name__, e, exc_info=True)
            return [{"status": "error", "tool": "neurograph", "error": str(e), "type": type(e).__name__}]

    def search_conversations(self, query: str, n: int = 5) -> List[Dict]:
        """Semantic recall from NeuroGraph memory."""
        return self._search(query, n, "conversations")

    def search_code(self, query: str, n: int = 5) -> List[Dict]:
        """Semantic code search via NeuroGraph recall."""
        return self._search(query, n, "code")

    def search_testament(self, query: str, n: int = 5) -> List[Dict]:
        """Search docs/markdown via NeuroGraph recall."""
        return self._search(query, n, "testament")

    def ingest_workspace(self) -> str:
        try:
            if self.policy_engine:
                from policy_engine import check_tool_call
                allowed, reason = check_tool_call("ingest_workspace", {}, self.repo_path)
                if not allowed:
                    return {"status": "error", "tool": "neurograph", "error": reason, "type": "PermissionError"}
            results = self.ng.ingest_directory(str(self.repo_path), extensions=[".py", ".md", ".txt"])
            return f"Indexed {len(results)} files into NeuroGraph."
        except (OSError, ValueError, TypeError) as e:
            logger.error("[neurograph] ingest_workspace failed: %s: %s", type(e).__name__, e, exc_info=True)
            return {"status": "error", "tool": "neurograph", "error": str(e), "type": type(e).__name__}

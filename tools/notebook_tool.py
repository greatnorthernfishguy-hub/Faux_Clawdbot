# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: NotebookTool
# What: notebook_read, notebook_add, notebook_delete + _load_notebook, _save_local extracted
# Why: PRD Block C — single-responsibility tool classes
# How: Notebook I/O isolated here; HF sync callback injected from facade for _save_notebook
# [2026-03-30] QB — Block D: Error handling hardening
# What: Specific exception types, logger replaces print()
# Why: PRD Block D — no broad except Exception, structured logging
# How: Catch json.JSONDecodeError/OSError/IndexError specifically; use logger.warning
# -------------------

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Callable

logger = logging.getLogger("tools.notebook")


class NotebookTool:
    """Notebook read/add/delete operations with local persistence."""

    def __init__(self, repo_path: Path, policy_engine=None,
                 notebook_file: Path = None, save_callback: Callable = None):
        self.repo_path = repo_path
        self.policy_engine = policy_engine
        self.memory_path = repo_path / "memory"
        self.notebook_file = notebook_file or (self.memory_path / "notebook.json")
        # save_callback is called with (notes) after writes — facade uses it for HF sync
        self._save_callback = save_callback

    def _load_notebook(self) -> List[Dict]:
        if not self.notebook_file.exists():
            return []
        try:
            return json.loads(self.notebook_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Notebook load failed: %s", e)
            return []

    def _save_local(self, notes: List[Dict]):
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.notebook_file.write_text(json.dumps(notes, indent=2), encoding='utf-8')

    def notebook_read(self) -> str:
        try:
            notes = self._load_notebook()
            if not notes:
                return "Notebook is empty."
            return "\n".join(
                [f"[{i}] {n.get('timestamp', '')}: {n.get('content', '')}" for i, n in enumerate(notes)]
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("[notebook] read failed: %s: %s", type(e).__name__, e, exc_info=True)
            return {"status": "error", "tool": "notebook", "error": str(e), "type": type(e).__name__}

    def notebook_add(self, content: str) -> str:
        try:
            notes = self._load_notebook()
            notes.append({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "content": content})
            if len(notes) > 50:
                notes = notes[-50:]
            self._save_local(notes)
            if self._save_callback:
                self._save_callback(notes)
            return f"Note added & synced. ({len(notes)} items)"
        except (json.JSONDecodeError, OSError) as e:
            logger.error("[notebook] add failed: %s: %s", type(e).__name__, e, exc_info=True)
            return {"status": "error", "tool": "notebook", "error": str(e), "type": type(e).__name__}

    def notebook_delete(self, index: int) -> str:
        try:
            notes = self._load_notebook()
            removed = notes.pop(int(index))
            self._save_local(notes)
            if self._save_callback:
                self._save_callback(notes)
            return f"Deleted note: '{removed.get('content', '')[:20]}...'"
        except IndexError:
            return "Invalid index."
        except (json.JSONDecodeError, OSError) as e:
            logger.error("[notebook] delete failed: %s: %s", type(e).__name__, e, exc_info=True)
            return {"status": "error", "tool": "notebook", "error": str(e), "type": type(e).__name__}

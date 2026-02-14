import os
import json
import subprocess
import time
import shutil
import ast
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, hf_hub_download

from openclaw_hook import NeuroGraphMemory

class RecursiveContextManager:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.memory_path = self.repo_path / "memory"
        self.notebook_file = self.memory_path / "notebook.json"

        # --- AUTHENTICATION ---
        self.token = os.getenv("HF_TOKEN")
        self.dataset_id = os.getenv("DATASET_ID", "Executor-Tyrant-Framework/clawdbot-memory")

        # --- NEUROGRAPH MEMORY ---
        neurograph_workspace = os.getenv(
            "NEUROGRAPH_WORKSPACE_DIR",
            str(self.repo_path / ".neurograph")
        )
        self.ng = NeuroGraphMemory.get_instance(workspace_dir=neurograph_workspace)
        print("✅ NeuroGraph Memory Loaded.")

        # Debounce for HF checkpoint uploads
        self._saves_since_ng_backup = 0
        self.NG_BACKUP_EVERY_N = 10

        # --- RESTORE NOTEBOOK ---
        self._init_memory()

    # =========================================================================
    # 🧠 SYNC LOGIC (Notebook + NeuroGraph Checkpoint Upload)
    # =========================================================================
    def _init_memory(self):
        """STARTUP: Download Notebook from HF Dataset."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        if self.token:
            try:
                hf_hub_download(
                    repo_id=self.dataset_id, filename="notebook.json", repo_type="dataset",
                    token=self.token, local_dir=self.memory_path, local_dir_use_symlinks=False
                )
            except Exception:
                self._save_local([])

    def _backup_ng_checkpoint_to_dataset(self):
        """Upload NeuroGraph checkpoint to HF Dataset for persistence."""
        if not self.token:
            return
        checkpoint_path = Path(self.ng.save())
        if not checkpoint_path.exists():
            return
        try:
            api = HfApi(token=self.token)
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo="neurograph/main.msgpack",
                repo_id=self.dataset_id,
                repo_type="dataset",
                commit_message=f"🧠 NeuroGraph checkpoint ({self.ng.stats()['nodes']} nodes)"
            )
            print(f"☁️ NeuroGraph checkpoint uploaded.")
        except Exception as e:
            print(f"⚠️ NeuroGraph checkpoint upload failed: {e}")

    # =========================================================================
    # 📓 NOTEBOOK
    # =========================================================================
    def _save_local(self, notes: List[Dict]):
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.notebook_file.write_text(json.dumps(notes, indent=2), encoding='utf-8')

    def _save_notebook(self, notes: List[Dict]):
        self._save_local(notes)
        if self.token and self.dataset_id:
            try:
                api = HfApi(token=self.token)
                api.upload_file(
                    path_or_fileobj=self.notebook_file, path_in_repo="notebook.json",
                    repo_id=self.dataset_id, repo_type="dataset",
                    commit_message=f"Notebook Update: {len(notes)}"
                )
            except Exception: pass

    def _load_notebook(self) -> List[Dict]:
        if not self.notebook_file.exists(): return []
        try: return json.loads(self.notebook_file.read_text(encoding='utf-8'))
        except: return []

    def notebook_read(self) -> str:
        notes = self._load_notebook()
        if not notes: return "Notebook is empty."
        return "\n".join([f"[{i}] {n.get('timestamp','')}: {n.get('content','')}" for i, n in enumerate(notes)])

    def notebook_add(self, content: str) -> str:
        notes = self._load_notebook()
        notes.append({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "content": content})
        if len(notes) > 50: notes = notes[-50:]
        self._save_notebook(notes)
        return f"✅ Note added & synced. ({len(notes)} items)"

    def notebook_delete(self, index: int) -> str:
        notes = self._load_notebook()
        try:
            removed = notes.pop(int(index))
            self._save_notebook(notes)
            return f"🗑️ Deleted note: '{removed.get('content', '')[:20]}...'"
        except IndexError: return "❌ Invalid index."

    # =========================================================================
    # 🔍 SEARCH & MEMORY (NeuroGraph-backed)
    # =========================================================================
    def save_conversation_turn(self, user_msg: str, assist_msg: str, turn_id: int):
        """Ingest a conversation turn into NeuroGraph and debounce checkpoint upload."""
        from openclaw_hook import NeuroGraphMemory
        from universal_ingestor import SourceType
        combined = f"USER: {user_msg}\n\nASSISTANT: {assist_msg}"
        self.ng.on_message(combined, source_type=SourceType.TEXT)
        # Also run a few extra STDP learning steps after each turn
        self.ng.step(5)

        self._saves_since_ng_backup += 1
        if self._saves_since_ng_backup >= self.NG_BACKUP_EVERY_N:
            self._backup_ng_checkpoint_to_dataset()
            self._saves_since_ng_backup = 0

    def search_conversations(self, query: str, n: int = 5) -> List[Dict]:
        """Semantic recall from NeuroGraph memory."""
        results = self.ng.recall(query, k=n, threshold=0.3)
        return [{
            "content": r.get("content", ""),
            "similarity": r.get("similarity", 0),
            "id": r.get("node_id", "")
        } for r in results]

    def search_code(self, query: str, n: int = 5) -> List[Dict]:
        """Semantic code search via NeuroGraph recall."""
        results = self.ng.recall(query, k=n, threshold=0.3)
        return [{
            "file": r.get("metadata", {}).get("source", "memory"),
            "snippet": r.get("content", "")[:500]
        } for r in results]

    def search_testament(self, query: str, n: int = 5) -> List[Dict]:
        """Search docs/markdown via NeuroGraph recall."""
        results = self.ng.recall(query, k=n, threshold=0.3)
        return [{
            "file": r.get("metadata", {}).get("source", "memory"),
            "snippet": r.get("content", "")[:500]
        } for r in results]

    def ingest_workspace(self) -> str:
        """Index the workspace codebase into NeuroGraph on demand."""
        results = self.ng.ingest_directory(str(self.repo_path), extensions=[".py", ".md", ".txt"])
        return f"✅ Indexed {len(results)} files into NeuroGraph."

    # =========================================================================
    # 🛠️ STANDARD TOOLS
    # =========================================================================
    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        try:
            target = self.repo_path / path
            content = target.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            if start_line is not None and end_line is not None:
                lines = lines[start_line:end_line]
            return "\n".join(lines)
        except Exception as e: return str(e)

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        try:
            target = self.repo_path / path
            if not target.exists(): return "Path not found."
            files = []
            for p in target.rglob("*"):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    files.append(str(p.relative_to(self.repo_path)))
            return "\n".join(files[:50])
        except Exception as e: return str(e)

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self.repo_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return f"✅ Written to {path}"
        except Exception as e: return str(e)

    def shell_execute(self, command: str) -> str:
        try:
            if any(x in command for x in ["rm -rf /", ":(){ :|:& };:"]): return "❌ Blocked."
            result = subprocess.run(command, shell=True, cwd=str(self.repo_path), capture_output=True, text=True, timeout=10)
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e: return f"Error: {e}"

    def map_repository_structure(self) -> str:
        graph = {"nodes": [], "edges": []}
        try:
            file_count = 0
            for file_path in self.repo_path.rglob('*.py'):
                if 'venv' in str(file_path): continue
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
                except SyntaxError: continue
            return f"✅ Map Generated: {file_count} files, {len(graph['nodes'])} nodes."
        except Exception as e: return f"❌ Mapping failed: {e}"

    def push_to_github(self, message: str) -> str:
        """Push current state to the connected HF Space (Git)."""
        try:
            subprocess.run(["git", "config", "user.email", "clawdbot@system.local"], check=False)
            subprocess.run(["git", "config", "user.name", "Clawdbot"], check=False)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            # Note: 'git push' requires the token to be in the remote URL or credential helper
            return "✅ Changes committed (Push requires configured remote with token)."
        except Exception as e: 
            return f"Git Error: {e}"

    def pull_from_github(self, branch: str) -> str:
        """Pull latest from remote."""
        try:
            subprocess.run(["git", "pull", "origin", branch], check=True)
            return f"✅ Pulled {branch}"
        except Exception as e: 
            return f"Git Pull Error: {e}"

    def create_shadow_branch(self) -> str:
        """Create timestamped backup branch."""
        ts = int(time.time())
        try:
            subprocess.run(["git", "checkout", "-b", f"shadow_{ts}"], check=True)
            return f"✅ Created branch shadow_{ts}"
        except Exception as e: 
            return f"Error: {e}"
        
    def get_stats(self) -> Dict:
        ng_stats = {}
        try:
            ng_stats = self.ng.stats()
        except Exception:
            pass
        return {
            "total_files": len(list(self.repo_path.rglob("*"))),
            "conversations": ng_stats.get("message_count", 0),
            "ng_nodes": ng_stats.get("nodes", 0),
            "ng_synapses": ng_stats.get("synapses", 0),
            "ng_firing_rate": ng_stats.get("firing_rate", 0.0),
            "ng_prediction_accuracy": ng_stats.get("prediction_accuracy", 0.0),
        }

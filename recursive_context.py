import os
import json
import subprocess
import time
import shutil
import ast
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, hf_hub_download, InferenceClient

class RecursiveContextManager:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.memory_path = self.repo_path / "memory"
        self.notebook_file = self.memory_path / "notebook.json"
        
        # --- AUTHENTICATION ---
        self.token = os.getenv("HF_TOKEN")
        self.dataset_id = os.getenv("DATASET_ID", "Executor-Tyrant-Framework/clawdbot-memory")
        
        # --- XET / DATABASE INIT ---
        self.xet_store = None
        try:
            # Try to load the Xet store if the file exists
            if (self.repo_path / "xet_storage.py").exists():
                import sys
                sys.path.append(str(self.repo_path))
                from xet_storage import XetVectorStore
                # Assuming Xet Repo URL is in env or default
                xet_url = os.getenv("XET_REPO_URL", "local/xet-repo")
                self.xet_store = XetVectorStore(xet_url)
                print("✅ Xet Storage Driver Loaded.")
        except Exception as e:
            print(f"⚠️ Xet Driver not loaded: {e}")

        # --- MEMORY INIT ---
        self._init_memory()

    def _init_memory(self):
        """STARTUP: Download the brain from the Dataset."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        if self.token:
            try:
                hf_hub_download(
                    repo_id=self.dataset_id,
                    filename="notebook.json",
                    repo_type="dataset",
                    token=self.token,
                    local_dir=self.memory_path,
                    local_dir_use_symlinks=False
                )
                print(f"🧠 Brain restored from {self.dataset_id}")
            except Exception as e:
                print(f"⚠️ Memory download failed (Creating Fresh): {e}")
                self._save_local([])

    def _save_local(self, notes: List[Dict]):
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.notebook_file.write_text(json.dumps(notes, indent=2), encoding='utf-8')

    def _save_notebook(self, notes: List[Dict]):
        """SAVE: Disk + Cloud Sync."""
        self._save_local(notes)
        if self.token and self.dataset_id:
            try:
                api = HfApi(token=self.token)
                api.upload_file(
                    path_or_fileobj=self.notebook_file,
                    path_in_repo="notebook.json",
                    repo_id=self.dataset_id,
                    repo_type="dataset",
                    commit_message=f"🧠 Notebook Update: {len(notes)} items"
                )
            except Exception as e:
                print(f"⚠️ Dataset sync failed: {e}")

    def _load_notebook(self) -> List[Dict]:
        if not self.notebook_file.exists(): return []
        try: return json.loads(self.notebook_file.read_text(encoding='utf-8'))
        except: return []

    # =========================================================================
    # 🧠 NOTEBOOK TOOLS
    # =========================================================================
    def notebook_read(self) -> str:
        notes = self._load_notebook()
        if not notes: return "Notebook is empty."
        return "\n".join([f"[{i}] {n.get('timestamp','')}: {n.get('content','')}" for i, n in enumerate(notes)])

    def notebook_add(self, content: str) -> str:
        notes = self._load_notebook()
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        notes.append({"timestamp": timestamp, "content": content})
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
    # 🗺️ CARTOGRAPHER (The Graph Tool)
    # =========================================================================
    def map_repository_structure(self) -> str:
        """Scans codebase to build a structural graph (AST-based)."""
        graph = {"nodes": [], "edges": []}
        try:
            # Initialize Client for semantic tagging (if available)
            client = InferenceClient(token=self.token) if self.token else None
            
            file_count = 0
            for file_path in self.repo_path.rglob('*.py'):
                if 'venv' in str(file_path) or 'site-packages' in str(file_path): continue
                rel_path = str(file_path.relative_to(self.repo_path))
                content = file_path.read_text(errors='ignore')
                file_count += 1
                
                graph["nodes"].append({"id": rel_path, "type": "file"})
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            node_id = f"{rel_path}::{node.name}"
                            graph["nodes"].append({
                                "id": node_id, 
                                "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                                "lineno": node.lineno
                            })
                            graph["edges"].append({"source": rel_path, "target": node_id, "relation": "defines"})
                            
                            for child in ast.walk(node):
                                if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                                    graph["edges"].append({
                                        "source": node_id,
                                        "target": child.func.id, 
                                        "relation": "calls"
                                    })
                except SyntaxError: continue
            
            # Save the Map locally (and ideally push to dataset later)
            map_path = self.memory_path / "repository_map.json"
            map_path.write_text(json.dumps(graph, indent=2))
            return f"✅ Map Generated: {file_count} files, {len(graph['nodes'])} nodes. Saved to memory/repository_map.json"

        except Exception as e:
            return f"❌ Mapping failed: {e}"

    # =========================================================================
    # 🛠️ STANDARD TOOLS
    # =========================================================================
    def search_code(self, query: str, n: int=5) -> List[Dict]:
        results = []
        try:
            # 1. Try Xet Semantic Search first
            if self.xet_store:
                # Mock embedding for now, real one would go here
                vector = [0.1] * 128 
                return self.xet_store.similarity_search(vector, n)
            
            # 2. Fallback to Text Search
            for f in self.repo_path.rglob("*.py"):
                txt = f.read_text(errors='ignore')
                if query in txt:
                    results.append({"file": f.name, "snippet": txt[:300]})
        except: pass
        return results[:n]

    def search_conversations(self, query: str, n: int=5) -> List[Dict]:
        # Connect to Xet or memory store here
        # For now, return recent history from log if Xet fails
        return []

    def search_testament(self, query: str, n: int=5) -> List[Dict]:
        return []

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

    def read_file(self, path: str, start: int = None, end: int = None) -> str:
        try:
            target = self.repo_path / path
            content = target.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            if start is not None and end is not None: lines = lines[start:end]
            return "\n".join(lines)
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

    def push_to_github(self, message: str) -> str:
        """Push current state to the connected HF Space (Git)."""
        try:
            subprocess.run(["git", "config", "user.email", "clawdbot@system.local"], check=False)
            subprocess.run(["git", "config", "user.name", "Clawdbot"], check=False)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            # Note: 'git push' requires the token to be in the remote URL or credential helper
            return "✅ Changes committed (Push requires configured remote with token)."
        except Exception as e: return f"Git Error: {e}"

    def pull_from_github(self, branch: str) -> str:
        try:
            subprocess.run(["git", "pull", "origin", branch], check=True)
            return f"✅ Pulled {branch}"
        except Exception as e: return f"Git Pull Error: {e}"

    def create_shadow_branch(self) -> str:
        ts = int(time.time())
        try:
            subprocess.run(["git", "checkout", "-b", f"shadow_{ts}"], check=True)
            return f"✅ Created branch shadow_{ts}"
        except Exception as e: return f"Error: {e}"
        
    def get_stats(self) -> Dict:
        return {"total_files": len(list(self.repo_path.rglob("*"))), "conversations": 0}

    def save_conversation_turn(self, user_msg, assist_msg, turn_id):
        # Optional: Log turn to a file for ingestion
        pass

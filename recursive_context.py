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
        self.client = InferenceClient(token=self.token) if self.token else None
        
        # --- XET / DATABASE INIT ---
        self.xet_root = self.repo_path / "xet_data"
        self.xet_dataset_file = "xet_vectors.json"
        self.xet_store = None
        
        # DEBOUNCE CONTROLS (NEW)
        self._saves_since_xet_backup = 0
        self.XET_BACKUP_EVERY_N = 5  # Backup every 5 conversations
        
        try:
            if (self.repo_path / "xet_storage.py").exists():
                import sys
                sys.path.append(str(self.repo_path))
                from xet_storage import XetVectorStore
                self.xet_store = XetVectorStore(repo_path=str(self.xet_root))
                print("✅ Xet Storage Driver Loaded.")
        except Exception as e:
            print(f"⚠️ Xet Driver not loaded: {e}")

        # --- RESTORE MEMORY ---
        self._init_memory()
        self._init_xet_memory()

    # =========================================================================
    # 🧠 SYNC LOGIC (Notebook + Xet JSON)
    # =========================================================================
    def _init_memory(self):
        """STARTUP: Download Notebook."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        if self.token:
            try:
                hf_hub_download(
                    repo_id=self.dataset_id, filename="notebook.json", repo_type="dataset",
                    token=self.token, local_dir=self.memory_path, local_dir_use_symlinks=False
                )
            except Exception: self._save_local([])

    def _init_xet_memory(self):
        """STARTUP: Download Xet Vectors (JSON)."""
        if not self.token or not self.xet_store: return
        try:
            local_path = hf_hub_download(
                repo_id=self.dataset_id, filename=self.xet_dataset_file, repo_type="dataset",
                token=self.token, local_dir=self.memory_path, local_dir_use_symlinks=False
            )
            # Restore to Xet Store
            vectors = json.loads(Path(local_path).read_text())
            for v in vectors:
                self.xet_store.store_vector(v["id"], v["vector"], v["metadata"])
            print(f"🧠 Restored {len(vectors)} vectors from Dataset")
        except Exception as e:
            print(f"⚠️ Xet restore failed (New dataset?): {e}")

    def _backup_xet_to_dataset(self):
        """Sync only NEW vectors since last backup (incremental)."""
        if not self.token or not self.xet_store: 
            return
        
        # Track what we've already backed up via manifest
        manifest_path = self.memory_path / "xet_manifest.json"
        try:
            known_hashes = set(json.loads(manifest_path.read_text()))
        except:
            known_hashes = set()
        
        # Find new vectors (filename IS the content hash in Xet storage)
        new_vectors = []
        current_hashes = set()
        
        for f in self.xet_store.vectors_path.glob("*/*/*"):
            if not f.is_file(): 
                continue
            file_hash = f.name
            current_hashes.add(file_hash)
            
            if file_hash not in known_hashes:
                try:
                    new_vectors.append(json.loads(f.read_text()))
                except:
                    pass
        
        if not new_vectors:
            # Nothing new to sync
            return
        
        try:
            # Download existing vectors from dataset to merge
            existing = []
            try:
                local_path = hf_hub_download(
                    repo_id=self.dataset_id,
                    filename=self.xet_dataset_file,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=self.memory_path,
                    local_dir_use_symlinks=False
                )
                existing = json.loads(Path(local_path).read_text())
            except:
                pass  # File doesn't exist yet
            
            # Merge with deduplication by id
            existing_ids = {v["id"] for v in existing}
            for v in new_vectors:
                if v["id"] not in existing_ids:
                    existing.append(v)
            
            # Upload merged file
            backup_path = self.memory_path / self.xet_dataset_file
            backup_path.write_text(json.dumps(existing, indent=2))
            
            api = HfApi(token=self.token)
            api.upload_file(
                path_or_fileobj=backup_path,
                path_in_repo=self.xet_dataset_file,
                repo_id=self.dataset_id,
                repo_type="dataset",
                commit_message=f"🧠 Xet: +{len(new_vectors)} vectors (total: {len(existing)})"
            )
            
            # Update manifest so we don't re-upload these
            manifest_path.write_text(json.dumps(list(current_hashes)))
            print(f"☁️ Backed up {len(new_vectors)} new vectors")
            
        except Exception as e:
            print(f"⚠️ Xet backup failed: {e}")

    # =========================================================================
    # 🧬 EMBEDDINGS
    # =========================================================================
    def _get_embedding(self, text: str) -> List[float]:
        if not self.client: return [0.0] * 384
        try:
            # feature-extraction returns list of floats
            response = self.client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
            return response[0] if isinstance(response[0], list) else response
        except Exception: return [0.0] * 384

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
    # 🔍 SEARCH & MEMORY
    # =========================================================================
    def save_conversation_turn(self, user_msg, assist_msg, turn_id):
        if not self.xet_store: return
        combined = f"USER: {user_msg}\n\nASSISTANT: {assist_msg}"
        vector = self._get_embedding(combined)
        
        self.xet_store.store_vector(
            id=f"conv_{turn_id}_{int(time.time())}",
            vector=vector,
            metadata={
                "type": "conversation",
                "user": user_msg[:500],
                "assistant": assist_msg[:500],
                "content": combined,
                "timestamp": time.time()
            }
        )
        
        # Debounced backup - not every turn
        self._saves_since_xet_backup += 1
        if self._saves_since_xet_backup >= self.XET_BACKUP_EVERY_N:
            self._backup_xet_to_dataset()
            self._saves_since_xet_backup = 0

    def search_conversations(self, query: str, n: int=5) -> List[Dict]:
        if not self.xet_store: return []
        query_vector = self._get_embedding(query)
        results = self.xet_store.similarity_search(query_vector, n)
        
        # Format strictly for app.py
        return [{
            "content": r.get("metadata", {}).get("content", ""),
            "similarity": r.get("similarity", 0),
            "id": r.get("id", "")
        } for r in results]

    def search_code(self, query: str, n: int=5) -> List[Dict]:
        results = []
        try:
            for f in self.repo_path.rglob("*.py"):
                if "venv" in str(f): continue
                txt = f.read_text(errors='ignore')
                if query in txt:
                    results.append({"file": f.name, "snippet": txt[:300]})
        except: pass
        return results[:n]
        
    def search_testament(self, query: str, n: int=5) -> List[Dict]:
        results = []
        try:
            for f in self.repo_path.rglob("*.md"):
                txt = f.read_text(errors='ignore')
                if query.lower() in txt.lower():
                    results.append({"file": f.name, "snippet": txt[:300]})
        except: pass
        return results[:n]

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
        conv_count = 0
        if self.xet_store:
            try:
                # Count files in the vectors/shard/hash structure
                conv_count = len(list(self.xet_store.vectors_path.glob("*/*/*")))
            except: pass
        return {"total_files": len(list(self.repo_path.rglob("*"))), "conversations": conv_count}

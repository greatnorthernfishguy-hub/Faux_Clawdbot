"""
Recursive Context Manager for Clawdbot

CHANGELOG [2026-01-31 - Gemini]
ADDED: Phase 1 Orchestrator tools: create_shadow_branch, write_file, shell_execute.
ADDED: Documentation Scanner to mandate Living Changelog headers.
FIXED: PermissionError on /.cache by forcing ONNXMiniLM_L6_V2.DOWNLOAD_PATH.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
import hashlib
import json
import os
import time
import threading
import subprocess
import re

def _select_chroma_path():
    """HF Spaces Docker containers wipe everything EXCEPT /data on restart."""
    data_path = Path("/data/chroma_db")
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        test_file = data_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return str(data_path)
    except (OSError, PermissionError):
        workspace_path = Path("/workspace/chroma_db")
        workspace_path.mkdir(parents=True, exist_ok=True)
        return str(workspace_path)

CHROMA_DB_PATH = _select_chroma_path()

class HFDatasetPersistence:
    """Handles durable cloud storage via your 1TB PRO Dataset repository."""
    def __init__(self, repo_id: str = None):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id = repo_id or os.getenv("MEMORY_REPO")
        self.token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self._repo_ready = False
        
        if self.repo_id and self.token:
            self._ensure_repo_exists()

    def _ensure_repo_exists(self):
        if self._repo_ready: return
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="dataset", token=self.token)
            self._repo_ready = True
        except Exception:
            try:
                self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=True, token=self.token)
                self._repo_ready = True
            except Exception: pass

    @property
    def is_configured(self):
        return bool(self.repo_id and self.token)

    def save_conversations(self, data: List[Dict]):
        if not self.is_configured: return
        temp = Path("/tmp/conv_backup.json")
        temp.write_text(json.dumps(data, indent=2))
        try:
            self.api.upload_file(
                path_or_fileobj=str(temp), 
                path_in_repo="conversations.json", 
                repo_id=self.repo_id, 
                repo_type="dataset", 
                token=self.token
            )
        except Exception: pass

    def load_conversations(self) -> List[Dict]:
        if not self.is_configured: return []
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(repo_id=self.repo_id, filename="conversations.json", repo_type="dataset", token=self.token)
            with open(local_path, 'r') as f: return json.load(f)
        except Exception: return []

class RecursiveContextManager:
    """Manages unlimited context and vibe-coding tools for E-T Systems."""
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.persistence = HFDatasetPersistence()
        
        # Embedding Config (Fixes /.cache PermissionError)
        self.embedding_function = ONNXMiniLM_L6_V2()
        cache_dir = os.getenv("CHROMA_CACHE_DIR", "/tmp/.cache/chroma")
        self.embedding_function.DOWNLOAD_PATH = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        c_name = self._get_collection_name()
        self.collection = self.chroma_client.get_or_create_collection(
            name=c_name, 
            embedding_function=self.embedding_function
        )
        self.conversations = self.chroma_client.get_or_create_collection(
            name=f"conv_{c_name.split('_')[1]}", 
            embedding_function=self.embedding_function
        )

        if self.conversations.count() == 0:
            self._restore_from_cloud()

    def _restore_from_cloud(self):
        data = self.persistence.load_conversations()
        for conv in data:
            try:
                self.conversations.add(documents=[conv["document"]], metadatas=[conv["metadata"]], ids=[conv["id"]])
            except Exception: pass

    def _get_collection_name(self) -> str:
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"

    # --- PHASE 1 ORCHESTRATOR TOOLS ---

    def create_shadow_branch(self):
        """Creates a timestamped backup branch of the E-T Systems Space."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        branch_name = f"vibe-backup-{timestamp}"
        try:
            repo_id = os.getenv("ET_SYSTEMS_SPACE", "Executor-Tyrant-Framework/Executor-Framworks_Full_VDB")
            self.persistence.api.create_branch(
                repo_id=repo_id, 
                branch=branch_name, 
                repo_type="space", 
                token=self.persistence.token
            )
            return f"🛡️ Shadow branch created: {branch_name}"
        except Exception as e:
            return f"⚠️ Shadow branch failed: {e}"

    def write_file(self, path: str, content: str):
        """Writes file strictly if valid CHANGELOG is present."""
        if not re.search(r"CHANGELOG \[\d{4}-\d{2}-\d{2} - \w+\]", content):
            return "REJECTED: Missing mandatory CHANGELOG [YYYY-MM-DD - AgentName] header."
        
        try:
            full_path = self.repo_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"✅ Successfully wrote {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def shell_execute(self, command: str):
        """Runs shell commands in the /workspace directory."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.repo_path, timeout=30)
            return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        except Exception as e:
            return f"Execution Error: {e}"

    # --- RECURSIVE SEARCH TOOLS ---
    def search_code(self, query: str, n: int = 5):
        if self.collection.count() == 0: return []
        res = self.collection.query(query_texts=[query], n_results=min(n, self.collection.count()))
        return [{"file": m['path'], "snippet": d[:500]} for d, m in zip(res['documents'][0], res['metadatas'][0])]

    def read_file(self, path: str):
        p = self.repo_path / path
        return p.read_text() if p.exists() else "File not found."

    def save_conversation_turn(self, u, a, t_id):
        combined = f"USER: {u}\n\nASSISTANT: {a}"
        u_id = f"turn_{int(time.time())}"
        self.conversations.add(documents=[combined], metadatas=[{"turn": t_id}], ids=[u_id])
        self.persistence.save_conversations([{"document": combined, "metadata": {"turn": t_id}, "id": u_id}])

"""
Recursive Context Manager for Clawdbot
[Corrected version to fix SyntaxError and /.cache PermissionError]
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

def _select_chroma_path():
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
    def __init__(self, repo_id: str = None):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id = repo_id or os.getenv("MEMORY_REPO")
        self.token = (
            os.getenv("HF_TOKEN") or
            os.getenv("HUGGING_FACE_HUB_TOKEN") or
            os.getenv("HUGGINGFACE_TOKEN")
        )
        self._repo_ready = False
        self._save_lock = threading.Lock()
        self._pending_save = False
        self._last_save_time = 0
        self.SAVE_DEBOUNCE_SECONDS = 10 

        if self.repo_id and self.token:
            self._ensure_repo_exists()
            self._verify_write_permissions()

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

    def _verify_write_permissions(self):
        try:
            self.api.whoami(token=self.token)
        except Exception: pass

    def save_conversations(self, conversations_data: List[Dict], force: bool = False):
        if not self.is_configured or not self._repo_ready: return False
        current_time = time.time()
        if not force and (current_time - self._last_save_time < self.SAVE_DEBOUNCE_SECONDS):
            self._pending_save = True
            return False
        with self._save_lock:
            try:
                temp_path = Path("/tmp/conversations_backup.json")
                temp_path.write_text(json.dumps(conversations_data, indent=2))
                self.api.upload_file(
                    path_or_fileobj=str(temp_path),
                    path_in_repo="conversations.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                    commit_message=f"Backup {len(conversations_data)} conversations"
                )
                self._last_save_time = current_time
                self._pending_save = False
                return True
            except Exception: return False

    def load_conversations(self) -> List[Dict]:
        if not self.is_configured: return []
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(repo_id=self.repo_id, filename="conversations.json", repo_type="dataset", token=self.token)
            with open(local_path, 'r') as f: return json.load(f)
        except Exception: return []

class RecursiveContextManager:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.persistence = HFDatasetPersistence()
        
        # FIX: Explicitly configure embedding model path to prevent PermissionError
        self.embedding_function = ONNXMiniLM_L6_V2()
        cache_dir = os.getenv("CHROMA_CACHE_DIR", "/tmp/.cache/chroma")
        self.embedding_function.DOWNLOAD_PATH = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        collection_name = self._get_collection_name()
        # Ensure the collection uses the custom embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "E-T Systems codebase"}
        )

        conversations_name = f"conversations_{collection_name.split('_')[1]}"
        self.conversations = self.chroma_client.get_or_create_collection(
            name=conversations_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Clawdbot conversation history"}
        )

        if self.conversations.count() == 0:
            self._restore_from_cloud()

        self._saves_since_backup = 0
        self.BACKUP_EVERY_N_SAVES = 1 # Sync frequently for reliability
        self._is_first_save = True

    def _restore_from_cloud(self):
        cloud_data = self.persistence.load_conversations()
        if not cloud_data: return
        for conv in cloud_data:
            try:
                self.conversations.add(documents=[conv["document"]], metadatas=[conv["metadata"]], ids=[conv["id"]])
            except Exception: pass

    def _backup_to_cloud(self, force: bool = False):
        if self.conversations.count() == 0: return
        all_convs = self.conversations.get(include=["documents", "metadatas"])
        backup_data = [{"id": id_, "document": doc, "metadata": meta} 
                       for doc, meta, id_ in zip(all_convs["documents"], all_convs["metadatas"], all_convs["ids"])]
        self.persistence.save_conversations(backup_data, force=force)

    def _get_collection_name(self) -> str:
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"

    def _index_codebase(self):
        code_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', '.json', '.yaml', '.yml'}
        skip_dirs = {'node_modules', '.git', '__pycache__', 'venv', 'env', '.venv', 'dist', 'build'}
        documents, metadatas, ids = [], [], []
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_dir() or any(skip in file_path.parts for skip in skip_dirs) or file_path.suffix not in code_extensions:
                continue
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if not content.strip() or len(content) > 100000: continue
                rel = str(file_path.relative_to(self.repo_path))
                documents.append(content); ids.append(rel)
                metadatas.append({"path": rel, "type": file_path.suffix[1:], "size": len(content)})
            except Exception: continue
        if documents:
            for i in range(0, len(documents), 100):
                self.collection.add(documents=documents[i:i+100], metadatas=metadatas[i:i+100], ids=ids[i:i+100])

    def search_code(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.collection.count() == 0: return []
        results = self.collection.query(query_texts=[query], n_results=min(n_results, self.collection.count()))
        return [{"file": m['path'], "snippet": d[:500], "relevance": round(1-dist, 3)} 
                for d, m, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]

    def read_file(self, path: str, lines: Optional[Tuple[int, int]] = None) -> str:
        full_path = self.repo_path / path
        if not full_path.exists(): return "Error: File not found"
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            if lines:
                l_list = content.split('\n')
                return '\n'.join(l_list[lines[0]-1:lines[1]])
            return content
        except Exception as e: return str(e)

    def search_testament(self, query: str) -> str:
        t_path = self.repo_path / "TESTAMENT.md"
        if not t_path.exists(): return "Testament not found"
        try:
            sections = t_path.read_text(encoding='utf-8').split('\n## ')
            relevant = [('## ' + s if not s.startswith('#') else s) for s in sections if query.lower() in s.lower()]
            return '\n\n'.join(relevant) if relevant else "No matches"
        except Exception as e: return str(e)

    def list_files(self, directory: str = ".") -> List[str]:
        d_path = self.repo_path / directory
        if not d_path.exists(): return ["Error: Not found"]
        try:
            return [(f.name + '/' if f.is_dir() else f.name) for f in sorted(d_path.iterdir()) if not f.name.startswith('.')]
        except Exception as e: return [str(e)]

    def save_conversation_turn(self, user_message: str, assistant_message: str, turn_id: int):
        # FIX: Ensure all brackets and quotes are closed correctly
        combined = f"USER: {user_message}\n\nASSISTANT: {assistant_message}"
        u_id = f"turn_{int(time.time())}_{turn_id}"
        self.conversations.add(
            documents=[combined], 
            metadatas=[{"user": user_message[:500], "assistant": assistant_message[:500], "turn": turn_id}], 
            ids=[u_id]
        )
        if self._is_first_save:
            self._backup_to_cloud(force=True)
            self._is_first_save = False
        else:
            self._saves_since_backup += 1
            if self._saves_since_backup >= self.BACKUP_EVERY_N_SAVES:
                self._backup_to_cloud()
                self._saves_since_backup = 0

    def search_conversations(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.conversations.count() == 0: return []
        res = self.conversations.query(query_texts=[query], n_results=min(n_results, self.conversations.count()))
        return [{"turn": m.get("turn"), "full_text": d} for d, m in zip(res['documents'][0], res['metadatas'][0])]

    def get_conversation_count(self) -> int:
        return self.conversations.count()

    def get_stats(self) -> Dict:
        return {"total_files": self.collection.count(), "conversations": self.conversations.count(), "storage_path": CHROMA_DB_PATH, "cloud_backup_configured": self.persistence.is_configured, "cloud_backup_repo": self.persistence.repo_id}

    def force_backup(self):
        self._backup_to_cloud(force=True)

    def shutdown(self):
        self.force_backup()

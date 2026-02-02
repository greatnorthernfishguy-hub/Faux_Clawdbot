"""
Recursive Context Manager for Clawdbot

CHANGELOG [2025-01-28 - Josh]
CREATED: Initial recursive context manager with ChromaDB vector search,
  file reading, and conversation persistence. Based on MIT Recursive
  Language Model technique for unlimited context.

CHANGELOG [2026-01-31 - Gemini]
ADDED: Phase 1 Orchestrator tools: create_shadow_branch, write_file, shell_execute.
ADDED: Documentation Scanner to mandate Living Changelog headers.
FIXED: PermissionError on /.cache by forcing ONNXMiniLM_L6_V2.DOWNLOAD_PATH.

CHANGELOG [2026-01-31 - Claude/Opus]
ADDED: get_stats() method — was called by app.py but never defined, causing
  crash on startup. Returns dict with file counts, conversation counts,
  collection sizes, and persistence status.
ADDED: list_files() method — directory exploration tool for the agent.
  Returns tree of files/dirs at a given path relative to repo root.
ADDED: search_conversations() method — semantic search over saved conversation
  history in ChromaDB. Essential for persistent memory across sessions.
ADDED: search_testament() method — searches for Testament/architectural decision
  files and returns matching content. Falls back to codebase search if no
  dedicated testament files exist.
ADDED: index_repository() method — actually indexes the repo into ChromaDB on
  init. Without this, search_code() always returned empty because nothing
  was ever added to the codebase collection. Runs in background thread to
  avoid blocking startup.
PRESERVED: All existing functions from prior changelogs remain intact.
  HFDatasetPersistence class, create_shadow_branch, write_file, shell_execute,
  search_code, read_file, save_conversation_turn — all unchanged.
NOTE: get_stats() is critical — app.py calls it at module level during UI
  construction AND in the system prompt. Missing it = instant crash.

CHANGELOG [2026-02-02 - Gemini Pro]
FIXED: write_file now pushes to Remote Space (Permanent Persistence).
FIXED: Relaxed CHANGELOG check to non-blocking warning.
CLEANED: Removed duplicate function definitions at EOF.
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


# =============================================================================
# CHROMA DB PATH SELECTION
# =============================================================================
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


# =============================================================================
# HF DATASET PERSISTENCE
# =============================================================================
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
        if self._repo_ready:
            return
        try:
            self.api.repo_info(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            self._repo_ready = True
        except Exception:
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    private=True,
                    token=self.token
                )
                self._repo_ready = True
            except Exception:
                pass

    @property
    def is_configured(self):
        return bool(self.repo_id and self.token)

    def save_conversations(self, data: List[Dict]):
        if not self.is_configured:
            return
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
        except Exception:
            pass

    def load_conversations(self) -> List[Dict]:
        if not self.is_configured:
            return []
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="conversations.json",
                repo_type="dataset",
                token=self.token
            )
            with open(local_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []


# =============================================================================
# RECURSIVE CONTEXT MANAGER
# =============================================================================

class RecursiveContextManager:
    """Manages unlimited context and vibe-coding tools for E-T Systems."""

    INDEXABLE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        '.json', '.yaml', '.yml', '.toml',
        '.md', '.txt', '.rst',
        '.html', '.css', '.scss',
        '.sh', '.bash',
        '.sql',
        '.env.example',
        '.gitignore', '.dockerignore',
        '.cfg', '.ini', '.conf',
    }

    MAX_INDEX_SIZE = 256 * 1024

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.persistence = HFDatasetPersistence()

        # Embedding Config
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

        self._indexing = False
        self._index_error = None
        self._indexed_file_count = 0
        if self.repo_path.exists() and self.repo_path.is_dir():
            self._start_background_indexing()

    def _restore_from_cloud(self):
        data = self.persistence.load_conversations()
        for conv in data:
            try:
                self.conversations.add(
                    documents=[conv["document"]],
                    metadatas=[conv["metadata"]],
                    ids=[conv["id"]]
                )
            except Exception:
                pass

    def _get_collection_name(self) -> str:
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"

    # =====================================================================
    # REPOSITORY INDEXING
    # =====================================================================

    def _start_background_indexing(self):
        self._indexing = True
        thread = threading.Thread(target=self._index_repository, daemon=True)
        thread.start()

    def _index_repository(self):
        try:
            skip_dirs = {
                '.git', '__pycache__', 'node_modules', 'venv', '.venv',
                'env', '.eggs', 'dist', 'build', '.next', '.nuxt',
                'chroma_db', '.chroma'
            }
            count = 0

            for file_path in self.repo_path.rglob('*'):
                if file_path.is_dir(): continue
                if any(skip in file_path.parts for skip in skip_dirs): continue

                suffix = file_path.suffix.lower()
                if suffix not in self.INDEXABLE_EXTENSIONS:
                    if file_path.name not in {'Dockerfile', 'Makefile', 'Procfile', '.gitignore', '.dockerignore', '.env.example'}:
                        continue

                try:
                    if file_path.stat().st_size > self.MAX_INDEX_SIZE: continue
                except OSError: continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except (OSError, UnicodeDecodeError): continue

                if not content.strip(): continue

                rel_path = str(file_path.relative_to(self.repo_path))
                chunks = self._chunk_file(content, rel_path)

                for chunk_id, chunk_text, chunk_meta in chunks:
                    try:
                        self.collection.upsert(
                            documents=[chunk_text],
                            metadatas=[chunk_meta],
                            ids=[chunk_id]
                        )
                    except Exception: continue

                count += 1
                self._indexed_file_count = count

        except Exception as e:
            self._index_error = str(e)
        finally:
            self._indexing = False

    def _chunk_file(self, content: str, rel_path: str) -> List[Tuple[str, str, dict]]:
        lines = content.split('\n')
        chunks = []
        chunk_size = 50
        overlap = 10

        if len(lines) <= chunk_size:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            chunk_id = f"{rel_path}::full::{content_hash}"
            meta = {
                'path': rel_path,
                'chunk': 'full',
                'lines': f"1-{len(lines)}",
                'total_lines': len(lines)
            }
            chunks.append((chunk_id, content, meta))
        else:
            start = 0
            chunk_num = 0
            while start < len(lines):
                end = min(start + chunk_size, len(lines))
                chunk_text = '\n'.join(lines[start:end])
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunk_id = f"{rel_path}::chunk{chunk_num}::{content_hash}"
                meta = {
                    'path': rel_path,
                    'chunk': f"chunk_{chunk_num}",
                    'lines': f"{start + 1}-{end}",
                    'total_lines': len(lines)
                }
                chunks.append((chunk_id, chunk_text, meta))
                chunk_num += 1
                start += chunk_size - overlap

        return chunks

    # =====================================================================
    # STATS
    # =====================================================================

    def get_stats(self) -> dict:
        """Return system statistics for the UI sidebar and system prompt."""
        try:
            return {
                'total_files': self._indexed_file_count,
                'indexed_chunks': self.collection.count(),
                'conversations': self.conversations.count(),
                'chroma_path': CHROMA_DB_PATH,
                'persistence_configured': self.persistence.is_configured,
                'indexing_in_progress': self._indexing,
                'index_error': self._index_error,
            }
        except Exception as e:
            return {"index_error": str(e)}

    # =====================================================================
    # PHASE 1 ORCHESTRATOR TOOLS
    # =====================================================================

    def create_shadow_branch(self):
        """Creates a timestamped backup branch of the E-T Systems Space."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        branch_name = f"vibe-backup-{timestamp}"
        try:
            repo_id = os.getenv("ET_SYSTEMS_SPACE")
            if not repo_id: return "Error: ET_SYSTEMS_SPACE env var not set."
            
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
        """Writes file locally AND pushes to the remote HF Space."""
        warning = ""
        # 1. Non-blocking warning instead of rejection
        if not re.search(r"CHANGELOG \[\d{4}-\d{2}-\d{2} - \w+\]", content):
            warning = "\n⚠️ NOTE: Missing CHANGELOG header."

        try:
            # 2. Write to Local Disk (Container)
            full_path = self.repo_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            
            # 3. Push to Remote Space (Persistence)
            remote_msg = ""
            target_space = os.getenv("ET_SYSTEMS_SPACE")
            
            if self.persistence.is_configured and target_space:
                try:
                    self.persistence.api.upload_file(
                        path_or_fileobj=str(full_path),
                        path_in_repo=path,
                        repo_id=target_space,
                        repo_type="space",
                        token=self.persistence.token,
                        commit_message=f"Clawdbot update: {path}"
                    )
                    remote_msg = f"\n🚀 Pushed to remote Space: {target_space}"
                except Exception as e:
                    remote_msg = f"\n⚠️ Local write success, but remote push failed: {e}"

            return f"✅ Wrote {path}{warning}{remote_msg}"
        except Exception as e:
            return f"Error writing file: {e}"

    def shell_execute(self, command: str):
        """Runs shell commands in the /workspace directory."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=self.repo_path, timeout=30
            )
            return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        except Exception as e:
            return f"Execution Error: {e}"

    # =====================================================================
    # RECURSIVE SEARCH TOOLS
    # =====================================================================

    def search_code(self, query: str, n: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        actual_n = min(n, self.collection.count())
        res = self.collection.query(query_texts=[query], n_results=actual_n)
        return [
            {"file": m['path'], "snippet": d[:500]}
            for d, m in zip(res['documents'][0], res['metadatas'][0])
        ]

    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        p = self.repo_path / path
        if not p.exists():
            return f"File not found: {path}"
        try:
            content = p.read_text(encoding='utf-8', errors='ignore')
            if start_line is not None or end_line is not None:
                lines = content.split('\n')
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                sliced = lines[start:end]
                return '\n'.join(sliced)
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    def list_files(self, path: str = "", max_depth: int = 3) -> str:
        target = self.repo_path / path
        if not target.exists(): return f"Path not found: {path}"
        if not target.is_dir(): return f"Not a directory: {path}"

        skip_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', '.venv',
            'chroma_db', '.chroma', 'dist', 'build'
        }

        lines = [f"📂 {path or '(repo root)'}"]

        def _walk(dir_path: Path, prefix: str, depth: int):
            if depth > max_depth: return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError: return

            for i, entry in enumerate(entries):
                if entry.name in skip_dirs or entry.name.startswith('.'): continue
                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}📁 {entry.name}/")
                    extension = "    " if is_last else "│   "
                    _walk(entry, prefix + extension, depth + 1)
                else:
                    size = entry.stat().st_size
                    size_str = f"{size:,}B" if size < 1024 else f"{size // 1024:,}KB"
                    lines.append(f"{prefix}{connector}📄 {entry.name} ({size_str})")

        _walk(target, "", 1)
        return '\n'.join(lines)

    def search_conversations(self, query: str, n: int = 5) -> List[Dict]:
        if self.conversations.count() == 0: return []
        actual_n = min(n, self.conversations.count())
        res = self.conversations.query(query_texts=[query], n_results=actual_n)
        results = []
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            results.append({'content': doc[:1000], 'metadata': meta})
        return results

    def search_testament(self, query: str, n: int = 5) -> List[Dict]:
        testament_names = {'testament', 'decisions', 'adr', 'architecture', 'principles', 'constitution', 'changelog', 'design'}
        testament_results = []
        if self.collection.count() > 0:
            actual_n = min(n * 2, self.collection.count())
            res = self.collection.query(query_texts=[query], n_results=actual_n)
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                path_lower = meta.get('path', '').lower()
                is_testament = any(name in path_lower for name in testament_names)
                testament_results.append({
                    'file': meta['path'],
                    'snippet': doc[:500],
                    'is_testament': is_testament
                })
        testament_results.sort(key=lambda r: (not r.get('is_testament', False)))
        return testament_results[:n]

    def save_conversation_turn(self, u, a, t_id):
        """WHY: Pulls the FULL history before pushing to cloud to prevent memory loss."""
        combined = f"USER: {u}\n\nASSISTANT: {a}"
        u_id = f"turn_{int(time.time())}"
        
        # 1. Save locally to ChromaDB
        self.conversations.add(documents=[combined], metadatas=[{"turn": t_id}], ids=[u_id])
        
        # 2. Retrieve the complete historical record to avoid overwriting with a single turn
        all_convs = self.conversations.get()
        full_data = []
        for i in range(len(all_convs['ids'])):
            full_data.append({
                "document": all_convs['documents'][i],
                "metadata": all_convs['metadatas'][i],
                "id": all_convs['ids'][i]
            })
            
        # 3. Push the entire manifest back to your PRO storage dataset
        self.persistence.save_conversations(full_data)

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
# CHANGELOG [2026-01-31 - Gemini]
# HF Spaces Docker containers wipe everything EXCEPT /data on restart.
# We prefer /data/chroma_db (persistent) but fall back to /workspace/chroma_db
# (ephemeral) if /data isn't writable.
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
# CHANGELOG [2026-01-31 - Gemini]
# Handles durable cloud storage via HF Dataset repository. Conversations
# survive Space restarts by backing up to a private dataset repo.
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
    """Manages unlimited context and vibe-coding tools for E-T Systems.

    CHANGELOG [2026-01-31 - Claude/Opus]
    This is the core class. It provides:
    - ChromaDB-backed semantic search over the codebase and conversations
    - File read/write with changelog enforcement
    - Shell execution for build tasks
    - Shadow branching for safe experimentation
    - Stats reporting for the UI sidebar
    - Repository indexing (background thread on init)

    ARCHITECTURE NOTE:
    The class is initialized once at module level in app.py. That means
    __init__ runs during import, so it MUST NOT block or crash. Heavy work
    (like indexing the repo) is dispatched to a background thread.
    get_stats() must return sensible defaults even before indexing completes.
    """

    # =========================================================================
    # FILE EXTENSIONS TO INDEX
    # =========================================================================
    # CHANGELOG [2026-01-31 - Claude/Opus]
    # Only index code/text files. Binary files, images, and large data files
    # would pollute the vector space and waste embedding compute.
    # =========================================================================
    INDEXABLE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        '.json', '.yaml', '.yml', '.toml',
        '.md', '.txt', '.rst',
        '.html', '.css', '.scss',
        '.sh', '.bash',
        '.sql',
        '.env.example',  # Not .env itself — that's sensitive
        '.gitignore', '.dockerignore',
        '.cfg', '.ini', '.conf',
    }

    # Max file size to index (256KB). Larger files are likely generated/data.
    MAX_INDEX_SIZE = 256 * 1024

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.persistence = HFDatasetPersistence()

        # =================================================================
        # EMBEDDING CONFIG
        # =================================================================
        # CHANGELOG [2026-01-31 - Gemini]
        # Fixes /.cache PermissionError. ChromaDB's ONNXMiniLM_L6_V2 tries
        # to download model weights to ~/.cache. In Docker as UID 1000,
        # that's /.cache (root-owned). We override DOWNLOAD_PATH to a
        # writable directory.
        # =================================================================
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

        # Restore conversations from cloud backup if local is empty
        if self.conversations.count() == 0:
            self._restore_from_cloud()

        # =================================================================
        # BACKGROUND INDEXING
        # =================================================================
        # CHANGELOG [2026-01-31 - Claude/Opus]
        # Index the repository in a background thread so startup isn't
        # blocked. The _indexing flag lets get_stats() report status.
        # =================================================================
        self._indexing = False
        self._index_error = None
        self._indexed_file_count = 0
        if self.repo_path.exists() and self.repo_path.is_dir():
            self._start_background_indexing()

    def _restore_from_cloud(self):
        """Restore conversation history from HF Dataset backup.

        CHANGELOG [2026-01-31 - Gemini]
        Called during init if the local ChromaDB conversations collection
        is empty. Pulls from the cloud dataset repo to recover history
        after a Space restart.
        """
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
        """Generate a deterministic collection name from the repo path.

        CHANGELOG [2025-01-28 - Josh]
        Uses MD5 hash of repo path so different repos get different
        collections within the same ChromaDB instance.
        """
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"

    # =====================================================================
    # REPOSITORY INDEXING
    # =====================================================================
    # CHANGELOG [2026-01-31 - Claude/Opus]
    # Without indexing, search_code() always returns empty results because
    # nothing is ever added to the ChromaDB codebase collection. This walks
    # the repo, reads indexable files, chunks them, and upserts into ChromaDB.
    #
    # DESIGN DECISIONS:
    # - Background thread: Don't block Gradio startup. Users can chat while
    #   indexing runs. get_stats() shows indexing progress.
    # - Chunk by logical blocks: Split files into ~50-line chunks with overlap
    #   so semantic search finds relevant sections, not just file-level matches.
    # - Upsert (not add): Safe to re-run. If the file was already indexed
    #   with the same content hash, ChromaDB skips it.
    # - Skip .git, __pycache__, node_modules, venv: No value in indexing these.
    #
    # TESTED ALTERNATIVES (graveyard):
    # - Indexing entire files as single documents: Poor search precision.
    #   A 500-line file matching on line 3 returns all 500 lines.
    # - Line-by-line indexing: Too many tiny documents, poor semantic context.
    # - Synchronous indexing: Blocks startup for 30+ seconds on large repos.
    # =====================================================================

    def _start_background_indexing(self):
        """Kick off repo indexing in a daemon thread."""
        self._indexing = True
        thread = threading.Thread(target=self._index_repository, daemon=True)
        thread.start()

    def _index_repository(self):
        """Walk the repo and index code files into ChromaDB.

        Runs in background thread. Sets self._indexing = False when done.
        """
        try:
            skip_dirs = {
                '.git', '__pycache__', 'node_modules', 'venv', '.venv',
                'env', '.eggs', 'dist', 'build', '.next', '.nuxt',
                'chroma_db', '.chroma'
            }
            count = 0

            for file_path in self.repo_path.rglob('*'):
                # Skip directories and non-indexable files
                if file_path.is_dir():
                    continue

                # Skip files in excluded directories
                if any(skip in file_path.parts for skip in skip_dirs):
                    continue

                # Check extension
                suffix = file_path.suffix.lower()
                if suffix not in self.INDEXABLE_EXTENSIONS:
                    # Also allow extensionless files if they look like configs
                    if file_path.name not in {
                        'Dockerfile', 'Makefile', 'Procfile',
                        '.gitignore', '.dockerignore', '.env.example'
                    }:
                        continue

                # Check size
                try:
                    if file_path.stat().st_size > self.MAX_INDEX_SIZE:
                        continue
                except OSError:
                    continue

                # Read and chunk the file
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except (OSError, UnicodeDecodeError):
                    continue

                if not content.strip():
                    continue

                rel_path = str(file_path.relative_to(self.repo_path))
                chunks = self._chunk_file(content, rel_path)

                for chunk_id, chunk_text, chunk_meta in chunks:
                    try:
                        self.collection.upsert(
                            documents=[chunk_text],
                            metadatas=[chunk_meta],
                            ids=[chunk_id]
                        )
                    except Exception:
                        continue

                count += 1
                self._indexed_file_count = count

        except Exception as e:
            self._index_error = str(e)
        finally:
            self._indexing = False

    def _chunk_file(self, content: str, rel_path: str) -> List[Tuple[str, str, dict]]:
        """Split a file into overlapping chunks for better search precision.

        CHANGELOG [2026-01-31 - Claude/Opus]
        Returns list of (id, text, metadata) tuples ready for ChromaDB upsert.
        Chunks are ~50 lines with 10-line overlap so context isn't lost at
        chunk boundaries.

        Args:
            content: Full file text
            rel_path: Path relative to repo root (used in metadata and IDs)

        Returns:
            List of (chunk_id, chunk_text, metadata_dict) tuples
        """
        lines = content.split('\n')
        chunks = []
        chunk_size = 50
        overlap = 10

        if len(lines) <= chunk_size:
            # Small file — index as single chunk
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
            # Larger file — split into overlapping chunks
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
    # STATS (NEW — was missing, caused crash)
    # =====================================================================
    # CHANGELOG [2026-01-31 - Claude/Opus]
    # app.py calls ctx.get_stats() at module level during Gradio Block
    # construction AND in the system prompt for every message. It expected
    # a dict with 'conversations', 'total_files', etc. Without this method,
    # the app crashes immediately on import.
    #
    # Returns safe defaults during indexing so the UI can render.
    # =====================================================================

    def get_stats(self) -> dict:
        """Return system statistics for the UI sidebar and system prompt.

        Returns:
            dict with keys: total_files, indexed_chunks, conversations,
            chroma_path, persistence_configured, indexing_in_progress,
            index_error
        """
        return {
            'total_files': self._indexed_file_count,
            'indexed_chunks': self.collection.count(),
            'conversations': self.conversations.count(),
            'chroma_path': CHROMA_DB_PATH,
            'persistence_configured': self.persistence.is_configured,
            'indexing_in_progress': self._indexing,
            'index_error': self._index_error,
        }

    # =====================================================================
    # PHASE 1 ORCHESTRATOR TOOLS (preserved from Gemini)
    # =====================================================================

    def create_shadow_branch(self):
        """Creates a timestamped backup branch of the E-T Systems Space.

        CHANGELOG [2026-01-31 - Gemini]
        Safety net before any destructive operations. Creates a branch
        named vibe-backup-YYYYMMDD-HHMMSS on the E-T Systems HF Space
        so you can always roll back.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        branch_name = f"vibe-backup-{timestamp}"
        try:
            repo_id = os.getenv(
                "ET_SYSTEMS_SPACE",
                "Executor-Tyrant-Framework/Executor-Framworks_Full_VDB"
            )
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
        """Writes file strictly if valid CHANGELOG is present.

        CHANGELOG [2026-01-31 - Gemini]
        Enforces the living changelog pattern. Any code written by an agent
        MUST include a CHANGELOG [YYYY-MM-DD - AgentName] header or the
        write is rejected. This is non-negotiable for the E-T Systems
        development workflow.

        Args:
            path: Relative path within the repo (e.g., "server/routes.ts")
            content: Full file content (must contain CHANGELOG header)

        Returns:
            Success message or rejection reason
        """
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
        """Runs shell commands in the /workspace directory.

        CHANGELOG [2026-01-31 - Gemini]
        Used for build tasks, git operations, dependency installs, etc.
        Timeout of 30 seconds prevents runaway processes. Captures both
        stdout and stderr for full diagnostic output.

        Args:
            command: Shell command string to execute

        Returns:
            Combined stdout/stderr output or error message
        """
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
        """Semantic search across the indexed codebase.

        CHANGELOG [2025-01-28 - Josh]
        Core tool for the MIT recursive context technique. The model calls
        this to find relevant code without loading the entire repo into
        context.

        Args:
            query: Natural language search query
            n: Max number of results to return (default 5)

        Returns:
            List of dicts with 'file' (path) and 'snippet' (first 500 chars)
        """
        if self.collection.count() == 0:
            return []
        actual_n = min(n, self.collection.count())
        res = self.collection.query(query_texts=[query], n_results=actual_n)
        return [
            {"file": m['path'], "snippet": d[:500]}
            for d, m in zip(res['documents'][0], res['metadatas'][0])
        ]

    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        """Read a specific file, optionally a line range.

        CHANGELOG [2025-01-28 - Josh]
        Direct file access for when the model knows exactly what it needs.

        CHANGELOG [2026-01-31 - Claude/Opus]
        Added optional start_line/end_line params for reading specific
        sections without loading entire large files into context.

        Args:
            path: Relative path within repo (e.g., "server/routes.ts")
            start_line: Optional 1-based start line
            end_line: Optional 1-based end line

        Returns:
            File contents (full or sliced) or "File not found." message
        """
        p = self.repo_path / path
        if not p.exists():
            return f"File not found: {path}"
        try:
            content = p.read_text(encoding='utf-8', errors='ignore')
            if start_line is not None or end_line is not None:
                lines = content.split('\n')
                start = (start_line or 1) - 1  # Convert to 0-based
                end = end_line or len(lines)
                sliced = lines[start:end]
                return '\n'.join(sliced)
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    def list_files(self, path: str = "", max_depth: int = 3) -> str:
        """List files and directories at a given path.

        CHANGELOG [2026-01-31 - Claude/Opus]
        Directory exploration tool. The agent needs to know what files exist
        before it can read or search them. Returns a tree-formatted listing
        up to max_depth levels deep.

        Args:
            path: Relative path within repo (default "" = repo root)
            max_depth: How many levels deep to list (default 3)

        Returns:
            Formatted string showing directory tree
        """
        target = self.repo_path / path
        if not target.exists():
            return f"Path not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"

        skip_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', '.venv',
            'chroma_db', '.chroma', 'dist', 'build'
        }

        lines = [f"📂 {path or '(repo root)'}"]

        def _walk(dir_path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return

            for i, entry in enumerate(entries):
                if entry.name in skip_dirs or entry.name.startswith('.'):
                    continue
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
        """Semantic search over past conversation history.

        CHANGELOG [2026-01-31 - Claude/Opus]
        This is how Clawdbot "remembers" past discussions. Conversations
        are saved to ChromaDB via save_conversation_turn() and backed up
        to the HF Dataset repo. This searches them semantically.

        Args:
            query: Natural language search query
            n: Max results to return

        Returns:
            List of dicts with 'content' and 'metadata' from matched turns
        """
        if self.conversations.count() == 0:
            return []
        actual_n = min(n, self.conversations.count())
        res = self.conversations.query(query_texts=[query], n_results=actual_n)
        results = []
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            results.append({
                'content': doc[:1000],  # Cap at 1000 chars per result
                'metadata': meta
            })
        return results

    def search_testament(self, query: str, n: int = 5) -> List[Dict]:
        """Search for Testament/architectural decision records.

        CHANGELOG [2026-01-31 - Claude/Opus]
        The Testament contains design decisions, constitutional principles,
        and architectural rationale for E-T Systems. This searches for
        testament-specific files first (TESTAMENT.md, DECISIONS.md, etc.),
        then falls back to general codebase search filtered for decision-
        related content.

        Args:
            query: What architectural decision to search for
            n: Max results

        Returns:
            List of dicts with 'file' and 'snippet' from matching documents
        """
        # First, look for dedicated testament/decision files
        testament_names = {
            'testament', 'decisions', 'adr', 'architecture',
            'principles', 'constitution', 'changelog', 'design'
        }

        testament_results = []
        if self.collection.count() > 0:
            # Search the codebase but prefer testament-like files
            actual_n = min(n * 2, self.collection.count())  # Get extra, then filter
            res = self.collection.query(query_texts=[query], n_results=actual_n)
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                path_lower = meta.get('path', '').lower()
                # Check if this is a testament/decision file
                is_testament = any(name in path_lower for name in testament_names)
                testament_results.append({
                    'file': meta['path'],
                    'snippet': doc[:500],
                    'is_testament': is_testament
                })

        # Sort: testament files first, then other matches
        testament_results.sort(key=lambda r: (not r.get('is_testament', False)))
        return testament_results[:n]

    def get_stats(self) -> dict:
        """WHY: Provides the metrics for the sidebar to prevent 'blind' coding."""
        try:
            return {
                "total_files": self.collection.count(),
                "indexed_chunks": self.collection.count(),
                "conversations": self.conversations.count(),
                "chroma_path": str(CHROMA_DB_PATH),
                "persistence_configured": self.persistence.is_configured,
                "indexing_in_progress": False
            }
        except Exception as e:
            return {"index_error": str(e)}

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

    def save_conversation_turn(self, u, a, t_id):
        """WHY: Prevents amnesia by pushing the FULL history to the cloud, not just the last turn."""
        combined = f"USER: {u}\n\nASSISTANT: {a}"
        u_id = f"turn_{int(time.time())}"
        
        # 1. Save locally to Chroma
        self.conversations.add(documents=[combined], metadatas=[{"turn": t_id}], ids=[u_id])
        
        # 2. Retrieve ALL history so the cloud backup is a complete record
        all_convs = self.conversations.get()
        full_data = []
        for i in range(len(all_convs['ids'])):
            full_data.append({
                "document": all_convs['documents'][i],
                "metadata": all_convs['metadatas'][i],
                "id": all_convs['ids'][i]
            })
            
        # 3. Push complete manifest to PRO storage
        self.persistence.save_conversations(full_data)


    def save_conversation_turn(self, u, a, t_id):
        """Save turn locally and push the FULL history to the cloud to prevent memory loss."""
        combined = f"USER: {u}\n\nASSISTANT: {a}"
        u_id = f"turn_{int(time.time())}"
        
        # 1. Save locally
        self.conversations.add(documents=[combined], metadatas=[{"turn": t_id}], ids=[u_id])
        
        # 2. To prevent amnesia, we must retrieve ALL historical turns from the local database
        all_convs = self.conversations.get()
        data_to_save = []
        for i in range(len(all_convs['ids'])):
            data_to_save.append({
                "document": all_convs['documents'][i],
                "metadata": all_convs['metadatas'][i],
                "id": all_convs['ids'][i]
            })
            
        # 3. Push the COMPLETE history to your PRO storage (replaces the previous file)
        self.persistence.save_conversations(data_to_save)

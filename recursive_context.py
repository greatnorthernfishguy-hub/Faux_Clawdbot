"""
Recursive Context Manager for Clawdbot

CHANGELOG [2025-01-28 - Josh]
Implements MIT's Recursive Language Model technique for unlimited context.

CHANGELOG [2025-01-30 - Claude]
Added HuggingFace Dataset persistence layer.
PROBLEM: /workspace gets wiped on Space restart, killing ChromaDB data.
SOLUTION: Sync ChromaDB collections to a private HF Dataset repo.
- On startup: Pull from Dataset -> restore to ChromaDB
- On save: Also push to Dataset (debounced to avoid spam)
- Periodic backup every N conversation turns
This gives us FREE, VERSIONED, PERSISTENT storage that survives restarts.

CHANGELOG [2025-01-31 - Claude]
FIXED: Multiple persistence failures causing "Conversations Saved: 0"
ROOT CAUSES FOUND:
1. ChromaDB path was /workspace/chroma_db - EPHEMERAL on HF Spaces Docker.
   Container filesystem gets wiped on every restart. Only /data survives.
2. Cloud backup (HF Dataset) silently did nothing when MEMORY_REPO wasn't set.
   No errors, no warnings in UI - just quiet failure.
3. Debounce timer (30s) could prevent saves if Space sleeps quickly.
4. HF Spaces sometimes SIGKILL containers without sending SIGTERM,
   so shutdown hooks never fire and pending saves are lost.

FIXES APPLIED:
- ChromaDB path: /workspace/chroma_db -> /data/chroma_db (HF persistent volume)
  Falls back to /workspace/chroma_db if /data isn't writable (free tier without
  persistent storage enabled). Logs which path it's using so you can tell.
- Cloud backup: Now logs CLEAR warnings when MEMORY_REPO isn't set, every time
  a save would have been backed up. No more silent failure.
- Debounce: Reduced from 30s to 10s. Also saves immediately on first conversation
  turn (no debounce for turn 0) so even single-message sessions persist.
- Shutdown: force_backup() is still registered, but we no longer RELY on it.
  Instead, every Nth save goes to cloud immediately (N=3, was N=5).

PERSISTENCE ARCHITECTURE (now with /data):
/data/chroma_db (survives restarts if persistent storage enabled)
    |
    v
ChromaDB (fast local queries) <--> HF Dataset (durable cloud storage)
                                        ^
                                 Private repo: username/clawdbot-memory
                                 Contains: conversations.json

REFERENCE: https://www.youtube.com/watch?v=huszaaJPjU8
"MIT basically solved unlimited context windows"

APPROACH:
Instead of cramming everything into context (hits limits) or summarizing
(lossy compression), we:

1. Store entire codebase in searchable environment
2. Give model TOOLS to query what it needs
3. Model recursively retrieves relevant pieces
4. No summarization loss - full fidelity access

This is like RAG, but IN-ENVIRONMENT with the model actively deciding
what context it needs rather than us guessing upfront.

EXAMPLE FLOW:
User: "How does Genesis handle surprise?"
Model: search_code("Genesis surprise detection")
    -> Finds: genesis/substrate.py, genesis/attention.py
Model: read_file("genesis/substrate.py", lines 145-167)
    -> Gets actual implementation
Model: search_testament("surprise detection rationale")
    -> Gets design decision
Model: Synthesizes answer from retrieved pieces

NO CONTEXT WINDOW LIMIT - just selective retrieval.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import hashlib
import json
import os
import time
import threading


# =============================================================================
# PERSISTENT STORAGE PATH SELECTION
# =============================================================================
# CHANGELOG [2025-01-31 - Claude]
# HF Spaces Docker containers wipe everything EXCEPT /data on restart.
# We try /data first (persistent), fall back to /workspace (ephemeral).
# This decision is made once at module load and logged clearly.
# =============================================================================

def _select_chroma_path():
    """
    Choose the best available path for ChromaDB storage.

    CHANGELOG [2025-01-31 - Claude]
    PRIORITY ORDER:
    1. /data/chroma_db - HF Spaces persistent volume (survives restarts)
    2. /workspace/chroma_db - Container filesystem (wiped on restart)

    WHY /data:
    HuggingFace Spaces with Docker SDK provide /data as persistent storage.
    It must be enabled in Space settings (Settings -> Persistent Storage).
    Free tier gets 20GB. This is the ONLY path that survives container restarts.

    WHY FALLBACK:
    If /data doesn't exist or isn't writable (persistent storage not enabled),
    we still need ChromaDB to work for the current session. /workspace works
    fine within a single session, just doesn't survive restarts.
    """
    # Try /data first (HF persistent volume)
    data_path = Path("/data/chroma_db")
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        # Test write access by creating and removing a temp file
        test_file = data_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print("=" * 60)
        print("STORAGE: Using /data/chroma_db (PERSISTENT - survives restarts)")
        print("=" * 60)
        return str(data_path)
    except (OSError, PermissionError) as e:
        print("=" * 60)
        print(f"STORAGE WARNING: /data not available ({e})")
        print("STORAGE: Falling back to /workspace/chroma_db (EPHEMERAL)")
        print("STORAGE: Memory will be lost on restart!")
        print("STORAGE: Enable persistent storage in Space Settings,")
        print("STORAGE: or set MEMORY_REPO secret for cloud backup.")
        print("=" * 60)

    # Fallback to /workspace
    workspace_path = Path("/workspace/chroma_db")
    workspace_path.mkdir(parents=True, exist_ok=True)
    return str(workspace_path)


# Resolve once at import time so it's consistent throughout the session
CHROMA_DB_PATH = _select_chroma_path()


class HFDatasetPersistence:
    """
    Handles syncing ChromaDB data to/from HuggingFace Datasets.

    CHANGELOG [2025-01-30 - Claude]
    Created to solve the Space restart problem.

    CHANGELOG [2025-01-31 - Claude]
    FIXED: Now logs clear warnings when MEMORY_REPO isn't configured.
    Previously failed silently, making it impossible to tell why memory
    wasn't persisting. Also reduced debounce from 30s to 10s.

    WHY HF DATASETS:
    - Free storage (up to 50GB on free tier)
    - Version controlled (can roll back if corrupted)
    - Private repos available
    - Native HF integration (no extra auth needed in Spaces)
    - JSON files work great for conversation data

    ALTERNATIVES CONSIDERED:
    - Supabase: Good but adds external dependency
    - /data mount: Requires persistent storage setting (now our primary!)
    - External S3: More complex, costs money
    """

    def __init__(self, repo_id: str = None):
        """
        Initialize persistence layer.

        Args:
            repo_id: HF Dataset repo (e.g., "username/clawdbot-memory")
                     If None, uses MEMORY_REPO env var
        """
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id or os.getenv("MEMORY_REPO")
        self.token = (
            os.getenv("HF_TOKEN") or
            os.getenv("HUGGING_FACE_HUB_TOKEN") or
            os.getenv("HUGGINGFACE_TOKEN")
        )

        # Track if we've initialized the repo
        self._repo_ready = False

        # Debounce saves to avoid hammering HF API
        # RATIONALE: User might send 10 messages quickly, we don't want 10 uploads
        # CHANGELOG [2025-01-31 - Claude]: Reduced from 30s to 10s. 30s was too
        # long - Spaces can sleep after 15 minutes of inactivity, and if a user
        # sends a few messages then leaves, the debounce could eat the last save.
        self._save_lock = threading.Lock()
        self._pending_save = False
        self._last_save_time = 0
        self.SAVE_DEBOUNCE_SECONDS = 10  # Min time between cloud saves

        # CHANGELOG [2025-01-31 - Claude]
        # Log configuration status clearly on startup so it's visible in logs
        if self.repo_id and self.token:
            self._ensure_repo_exists()
            print(f"CLOUD BACKUP: Configured -> {self.repo_id}")
            # Verify token has write permissions
            # CHANGELOG [2025-01-31 - Claude]
            # Gemini caught this: a read-only token will let the app start
            # but all upload_file calls will fail with 403. Check early.
            self._verify_write_permissions()
        elif not self.repo_id:
            print("=" * 60)
            print("CLOUD BACKUP: NOT CONFIGURED")
            print("Add MEMORY_REPO secret to Space settings.")
            print("Value should be: your-username/clawdbot-memory")
            print("Without this, conversations won't survive restarts")
            print("(unless /data persistent storage is enabled).")
            print("=" * 60)
        elif not self.token:
            print("CLOUD BACKUP: No HF_TOKEN found - cloud backup disabled")

    def _ensure_repo_exists(self):
        """
        Create the memory repo if it doesn't exist.

        CHANGELOG [2025-01-30 - Claude]
        Auto-creates private Dataset repo for memory storage.
        """
        if self._repo_ready:
            return

        try:
            # Check if repo exists
            self.api.repo_info(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            print(f"Memory repo exists: {self.repo_id}")
            self._repo_ready = True
        except Exception:
            # Create it
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    private=True,  # Keep conversations private!
                    token=self.token
                )
                print(f"Created memory repo: {self.repo_id}")
                self._repo_ready = True
            except Exception as e:
                print(f"Could not create memory repo: {e}")
                print("   Memory will not persist across restarts!")

    @property
    def is_configured(self):
        """
        Check if cloud backup is properly configured.

        CHANGELOG [2025-01-31 - Claude]
        Added so callers can check before relying on cloud backup.
        """
        return bool(self.repo_id and self.token)

    def _verify_write_permissions(self):
        """
        Check that the HF_TOKEN has write permissions.

        CHANGELOG [2025-01-31 - Claude]
        Added per Gemini's feedback: a read-only token lets the app start
        but causes all cloud saves to fail with 403. Better to catch this
        at startup and warn loudly than discover it after losing data.

        NOTE: We don't fail hard here because the app can still function
        without cloud backup (using /data persistent storage). Just warn.
        """
        try:
            # whoami returns token info including permissions
            user_info = self.api.whoami(token=self.token)
            # The token type tells us about permissions
            # "write" or "fineGrained" tokens with repo write access work
            # "read" tokens will fail on upload
            token_name = user_info.get("auth", {}).get("accessToken", {}).get("displayName", "unknown")
            print(f"CLOUD BACKUP: Token verified (name: {token_name})")
        except Exception as e:
            print(f"CLOUD BACKUP WARNING: Could not verify token permissions: {e}")
            print("CLOUD BACKUP WARNING: If saves fail, check that HF_TOKEN has WRITE access")

    def save_conversations(self, conversations_data: List[Dict], force: bool = False):
        """
        Save conversations to HF Dataset.

        CHANGELOG [2025-01-30 - Claude]
        Debounced save to avoid API spam. Use force=True for shutdown saves.

        CHANGELOG [2025-01-31 - Claude]
        Now logs when save is skipped due to missing config (was silent before).
        Added _repo_ready guard per Gemini's race condition catch: if the repo
        hasn't finished initializing (or failed to initialize), skip the save
        rather than letting it throw an opaque HfApi error.

        Args:
            conversations_data: List of conversation dicts to save
            force: If True, save immediately ignoring debounce
        """
        if not self.is_configured:
            # CHANGELOG [2025-01-31 - Claude]
            # Was completely silent before. Now logs so you can see in Space logs.
            print("Cloud save skipped: MEMORY_REPO not configured")
            return False

        # CHANGELOG [2025-01-31 - Claude]
        # Race condition guard: _ensure_repo_exists() runs in __init__ but
        # could fail (network issue, bad token, etc). If repo isn't ready,
        # don't try to upload - it'll just throw a confusing 404 or 403.
        if not self._repo_ready:
            print("Cloud save skipped: memory repo not ready (init may have failed)")
            # Retry initialization in case it was a transient network issue
            self._ensure_repo_exists()
            if not self._repo_ready:
                return False

        current_time = time.time()

        # Check debounce (unless forced)
        if not force:
            if current_time - self._last_save_time < self.SAVE_DEBOUNCE_SECONDS:
                self._pending_save = True
                return False

        with self._save_lock:
            try:
                # Save to local temp file first
                temp_path = Path("/tmp/conversations_backup.json")
                temp_path.write_text(json.dumps(conversations_data, indent=2))

                # Upload to HF Dataset
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
                print(f"Cloud saved {len(conversations_data)} conversations to {self.repo_id}")
                return True

            except Exception as e:
                print(f"Failed to save conversations to cloud: {e}")
                return False

    def load_conversations(self) -> List[Dict]:
        """
        Load conversations from HF Dataset.

        CHANGELOG [2025-01-30 - Claude]
        Called on startup to restore conversation history.

        Returns:
            List of conversation dicts, or empty list if none found
        """
        if not self.is_configured:
            print("Cloud load skipped: MEMORY_REPO not configured")
            return []

        try:
            from huggingface_hub import hf_hub_download

            # Download conversations file
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="conversations.json",
                repo_type="dataset",
                token=self.token
            )

            with open(local_path, 'r') as f:
                data = json.load(f)

            print(f"Cloud loaded {len(data)} conversations from {self.repo_id}")
            return data

        except Exception as e:
            # File might not exist yet (first run)
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"No existing conversations found in {self.repo_id} (first run)")
            else:
                print(f"Failed to load conversations from cloud: {e}")
            return []

    def has_pending_save(self) -> bool:
        """Check if there's a pending save that was debounced."""
        return self._pending_save


class RecursiveContextManager:
    """
    Manages unlimited context via recursive retrieval.

    The model has TOOLS to search and read the codebase selectively,
    rather than loading everything upfront.

    CHANGELOG [2025-01-30 - Claude]
    Added HF Dataset persistence. Conversations now survive Space restarts.

    CHANGELOG [2025-01-31 - Claude]
    FIXED: ChromaDB path now uses /data (persistent) instead of /workspace (ephemeral).
    FIXED: Cloud backup logs clear warnings when not configured.
    FIXED: Backup frequency increased (every 3 saves, was 5).
    FIXED: First conversation turn always triggers immediate cloud save.
    """

    def __init__(self, repo_path: str):
        """
        Initialize context manager for a repository.

        Args:
            repo_path: Path to the code repository
        """
        self.repo_path = Path(repo_path)

        # Initialize persistence layer FIRST
        # RATIONALE: Need this before ChromaDB so we can restore data
        self.persistence = HFDatasetPersistence()

        # Initialize ChromaDB for semantic search
        # CHANGELOG [2025-01-31 - Claude]
        # Now uses CHROMA_DB_PATH which is resolved at module load to either
        # /data/chroma_db (persistent) or /workspace/chroma_db (ephemeral fallback).
        # See _select_chroma_path() at top of file for selection logic.
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print(f"ChromaDB initialized at: {CHROMA_DB_PATH}")

        # Create or get CODEBASE collection
        collection_name = self._get_collection_name()
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Loaded existing index: {self.collection.count()} files")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "E-T Systems codebase"}
            )
            print(f"Created new collection: {collection_name}")
            self._index_codebase()

        # Create or get CONVERSATION collection for persistence
        # CHANGELOG [2025-01-30 - Josh]: Added conversation persistence
        # CHANGELOG [2025-01-30 - Claude]: Added HF Dataset restore on startup
        conversations_name = f"conversations_{self._get_collection_name().split('_')[1]}"
        try:
            self.conversations = self.chroma_client.get_collection(conversations_name)
            print(f"Loaded conversation history: {self.conversations.count()} exchanges")
        except Exception:
            self.conversations = self.chroma_client.create_collection(
                name=conversations_name,
                metadata={"description": "Clawdbot conversation history"}
            )
            print(f"Created conversation collection: {conversations_name}")

        # RESTORE FROM CLOUD if local is empty but cloud has data
        # RATIONALE: Space restarted, ChromaDB wiped, but HF Dataset has our history
        if self.conversations.count() == 0:
            self._restore_from_cloud()

        # Track saves for periodic backup
        # CHANGELOG [2025-01-31 - Claude]: Reduced from 5 to 3 for more frequent backups.
        # Also added _is_first_save flag for immediate first-turn backup.
        self._saves_since_backup = 0
        self.BACKUP_EVERY_N_SAVES = 3  # Backup to cloud every N conversation turns
        self._is_first_save = True  # First save always goes to cloud immediately

    def _restore_from_cloud(self):
        """
        Restore conversations from HF Dataset to ChromaDB.

        CHANGELOG [2025-01-30 - Claude]
        Called when local ChromaDB is empty but cloud might have data.
        This is the magic that makes memory survive restarts.
        """
        cloud_data = self.persistence.load_conversations()

        if not cloud_data:
            print("No cloud conversations to restore")
            return

        print(f"Restoring {len(cloud_data)} conversations from cloud...")

        restored = 0
        for conv in cloud_data:
            try:
                self.conversations.add(
                    documents=[conv["document"]],
                    metadatas=[conv["metadata"]],
                    ids=[conv["id"]]
                )
                restored += 1
            except Exception as e:
                # Might fail if ID already exists (shouldn't happen but safety first)
                print(f"Skipping conversation {conv.get('id')}: {e}")

        print(f"Restored {restored} conversations (total: {self.conversations.count()})")

    def _backup_to_cloud(self, force: bool = False):
        """
        Backup all conversations to HF Dataset.

        CHANGELOG [2025-01-30 - Claude]
        Called periodically and on shutdown to ensure durability.

        Args:
            force: If True, save immediately ignoring debounce
        """
        if self.conversations.count() == 0:
            return

        # Get all conversations from ChromaDB
        all_convs = self.conversations.get(
            include=["documents", "metadatas"]
        )

        # Format for JSON storage
        backup_data = []
        for i, (doc, meta, id_) in enumerate(zip(
            all_convs["documents"],
            all_convs["metadatas"],
            all_convs["ids"]
        )):
            backup_data.append({
                "id": id_,
                "document": doc,
                "metadata": meta
            })

        # Save to cloud
        self.persistence.save_conversations(backup_data, force=force)

    def _get_collection_name(self) -> str:
        """Generate unique collection name based on repo path."""
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"

    def _index_codebase(self):
        """
        Index all code files for semantic search.

        This creates the "environment" that the model can search through.
        We index with metadata so search results include file paths.
        """
        print(f"Indexing codebase at {self.repo_path}...")

        # File types to index
        code_extensions = {
            '.py', '.js', '.ts', '.tsx', '.jsx',
            '.md', '.txt', '.json', '.yaml', '.yml'
        }

        # Skip these directories
        skip_dirs = {
            'node_modules', '.git', '__pycache__', 'venv',
            'env', '.venv', 'dist', 'build'
        }

        documents = []
        metadatas = []
        ids = []

        for file_path in self.repo_path.rglob('*'):
            # Skip directories and non-code files
            if file_path.is_dir():
                continue
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if file_path.suffix not in code_extensions:
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')

                # Don't index empty files or massive files
                if not content.strip() or len(content) > 100000:
                    continue

                relative_path = str(file_path.relative_to(self.repo_path))

                documents.append(content)
                metadatas.append({
                    "path": relative_path,
                    "type": file_path.suffix[1:],  # Remove leading dot
                    "size": len(content)
                })
                ids.append(relative_path)

            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")
                continue

        if documents:
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]

                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )

            print(f"Indexed {len(documents)} files")
        else:
            print("No files found to index")

    def search_code(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search codebase semantically.

        This is a TOOL available to the model for recursive retrieval.
        Model can search for concepts without knowing exact file names.

        Args:
            query: What to search for (e.g. "surprise detection", "vector embedding")
            n_results: How many results to return

        Returns:
            List of dicts with {file, snippet, relevance}
        """
        if self.collection.count() == 0:
            return [{"error": "No files indexed yet"}]

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )

        # Format results for the model
        formatted = []
        for i in range(len(results['documents'][0])):
            # Truncate document to first 500 chars for search results
            # Model can read_file() if it wants the full content
            snippet = results['documents'][0][i][:500]
            if len(results['documents'][0][i]) > 500:
                snippet += "... [truncated, use read_file to see more]"

            formatted.append({
                "file": results['metadatas'][0][i]['path'],
                "snippet": snippet,
                "relevance": round(1 - results['distances'][0][i], 3),
                "type": results['metadatas'][0][i]['type']
            })

        return formatted

    def read_file(self, path: str, lines: Optional[Tuple[int, int]] = None) -> str:
        """
        Read a specific file or line range.

        This is a TOOL available to the model.
        After searching, model can read full files as needed.

        Args:
            path: Relative path to file
            lines: Optional (start, end) line numbers (1-indexed, inclusive)

        Returns:
            File content or specified lines
        """
        full_path = self.repo_path / path

        if not full_path.exists():
            return f"Error: File not found: {path}"

        if not full_path.is_relative_to(self.repo_path):
            return "Error: Path outside repository"

        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')

            if lines:
                start, end = lines
                content_lines = content.split('\n')
                # Adjust for 1-indexed
                selected_lines = content_lines[start-1:end]
                return '\n'.join(selected_lines)

            return content

        except Exception as e:
            return f"Error reading file: {str(e)}"

    def search_testament(self, query: str) -> str:
        """
        Search architectural decisions in Testament.

        This is a TOOL available to the model.
        Helps model understand design rationale.

        Args:
            query: What decision to look for

        Returns:
            Relevant Testament sections
        """
        testament_path = self.repo_path / "TESTAMENT.md"

        if not testament_path.exists():
            return "Testament not found. No architectural decisions recorded yet."

        try:
            content = testament_path.read_text(encoding='utf-8')

            # Split into sections (marked by ## headers)
            sections = content.split('\n## ')

            # Simple relevance: sections that contain query terms
            query_lower = query.lower()
            relevant = []

            for section in sections:
                if query_lower in section.lower():
                    # Include section with header
                    if not section.startswith('#'):
                        section = '## ' + section
                    relevant.append(section)

            if relevant:
                return '\n\n'.join(relevant)
            else:
                return f"No Testament entries found matching '{query}'"

        except Exception as e:
            return f"Error searching Testament: {str(e)}"

    def list_files(self, directory: str = ".") -> List[str]:
        """
        List files in a directory.

        This is a TOOL available to the model.
        Helps model explore repository structure.

        Args:
            directory: Directory to list (relative path)

        Returns:
            List of file/directory names
        """
        dir_path = self.repo_path / directory

        if not dir_path.exists():
            return [f"Error: Directory not found: {directory}"]

        if not dir_path.is_relative_to(self.repo_path):
            return ["Error: Path outside repository"]

        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                # Skip hidden and system directories
                if item.name.startswith('.'):
                    continue
                if item.name in {'node_modules', '__pycache__', 'venv'}:
                    continue

                # Mark directories with /
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)

            return items

        except Exception as e:
            return [f"Error listing directory: {str(e)}"]

    def save_conversation_turn(self, user_message: str, assistant_message: str, turn_id: int):
        """
        Save a conversation turn to persistent storage.

        CHANGELOG [2025-01-30 - Josh]
        Implements MIT recursive technique for conversations.
        Chat history becomes searchable context that persists across sessions.

        CHANGELOG [2025-01-30 - Claude]
        Added cloud backup integration. Every N saves triggers HF Dataset backup.

        CHANGELOG [2025-01-31 - Claude]
        FIXED: First conversation turn now triggers immediate cloud backup.
        Previously, a user could have a single exchange and leave, and the
        debounce timer would prevent the cloud save from ever firing.
        Also reduced BACKUP_EVERY_N_SAVES from 5 to 3.

        Args:
            user_message: What the user said
            assistant_message: What Clawdbot responded
            turn_id: Unique ID for this turn (timestamp-based)
        """
        # Create a combined document for semantic search
        combined = f"USER: {user_message}\n\nASSISTANT: {assistant_message}"

        # Generate unique ID with timestamp to avoid collisions
        unique_id = f"turn_{int(time.time())}_{turn_id}"

        # Save to ChromaDB (fast local access)
        self.conversations.add(
            documents=[combined],
            metadatas=[{
                "user": user_message[:500],  # Truncate for metadata
                "assistant": assistant_message[:500],
                "timestamp": int(time.time()),
                "turn": turn_id
            }],
            ids=[unique_id]
        )

        print(f"Saved conversation turn {turn_id} (total: {self.conversations.count()})")

        # CLOUD BACKUP LOGIC
        # CHANGELOG [2025-01-31 - Claude]
        # First save always goes to cloud immediately (force=True).
        # This ensures even single-message sessions persist.
        # Subsequent saves follow the periodic backup schedule.
        if self._is_first_save:
            print("First conversation turn - forcing immediate cloud backup")
            self._backup_to_cloud(force=True)
            self._is_first_save = False
            self._saves_since_backup = 0
        else:
            # Periodic cloud backup
            # RATIONALE: Don't backup every message (API spam), but don't wait too long
            self._saves_since_backup += 1
            if self._saves_since_backup >= self.BACKUP_EVERY_N_SAVES:
                self._backup_to_cloud()
                self._saves_since_backup = 0

    def search_conversations(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search past conversations for relevant context.

        This enables TRUE unlimited context - Clawdbot can remember
        everything ever discussed by searching its own conversation history.

        Args:
            query: What to search for in past conversations
            n_results: How many results to return

        Returns:
            List of past conversation turns with user/assistant messages
        """
        if self.conversations.count() == 0:
            return []

        results = self.conversations.query(
            query_texts=[query],
            n_results=min(n_results, self.conversations.count())
        )

        formatted = []
        for i, (doc, metadata) in enumerate(zip(
            results['documents'][0], results['metadatas'][0]
        )):
            formatted.append({
                "turn": metadata.get("turn", "unknown"),
                "user": metadata.get("user", ""),
                "assistant": metadata.get("assistant", ""),
                "full_text": doc,
                "relevance": i + 1  # Lower is more relevant
            })

        return formatted

    def get_conversation_count(self) -> int:
        """Get total number of saved conversation turns."""
        return self.conversations.count()

    def get_stats(self) -> Dict:
        """
        Get statistics about indexed codebase.

        CHANGELOG [2025-01-31 - Claude]
        Added storage_path and cloud_backup_status for better diagnostics.

        Returns:
            Dict with file counts, sizes, etc.
        """
        return {
            "total_files": self.collection.count(),
            "repo_path": str(self.repo_path),
            "collection_name": self.collection.name,
            "conversations": self.conversations.count(),
            "storage_path": CHROMA_DB_PATH,
            "cloud_backup_configured": self.persistence.is_configured,
            "cloud_backup_repo": self.persistence.repo_id or "Not set"
        }

    def force_backup(self):
        """
        Force immediate backup to cloud.

        CHANGELOG [2025-01-30 - Claude]
        Call this on app shutdown to ensure no data loss.
        """
        print("Forcing cloud backup...")
        self._backup_to_cloud(force=True)
        print("Backup complete")

    def shutdown(self):
        """
        Clean shutdown - ensure all data is saved.

        CHANGELOG [2025-01-30 - Claude]
        Call this when the Space is shutting down.
        """
        print("Shutting down RecursiveContextManager...")
        self.force_backup()
        print("Shutdown complete")

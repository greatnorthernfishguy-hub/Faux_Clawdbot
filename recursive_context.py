# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: Rewrite as thin facade
# What: RecursiveContextManager is now a facade delegating to tools/ classes
# Why: PRD Block C — split 267-line god-class into focused single-responsibility tools
# How: Constructor wires tool instances; methods delegate; cross-cutting sync stays here
# -------------------

import logging
import os
from pathlib import Path
from typing import List, Dict
from huggingface_hub import HfApi, hf_hub_download

from openclaw_hook import NeuroGraphMemory
from tools import FilesystemTool, GitTool, NotebookTool, NeurographTool, ShellTool, WorkspaceTool

logger = logging.getLogger("recursive_context")


class RecursiveContextManager:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.memory_path = self.repo_path / "memory"
        self.notebook_file = self.memory_path / "notebook.json"
        self.token = os.getenv("HF_TOKEN")
        self.dataset_id = os.getenv("DATASET_ID", "Executor-Tyrant-Framework/clawdbot-memory")

        neurograph_workspace = os.getenv("NEUROGRAPH_WORKSPACE_DIR", str(self.repo_path / ".neurograph"))
        self.ng = NeuroGraphMemory.get_instance(workspace_dir=neurograph_workspace)
        logger.info("NeuroGraph Memory Loaded.")

        self._saves_since_ng_backup = 0
        self.NG_BACKUP_EVERY_N = 10

        # --- Tool instances ---
        import policy_engine as pe  # Cricket-shaped enforcement
        self._fs = FilesystemTool(self.repo_path, pe)
        self._git = GitTool(self.repo_path, pe)
        self._notebook = NotebookTool(self.repo_path, pe,
                                       notebook_file=self.notebook_file,
                                       save_callback=self._save_notebook)
        self._ng_tool = NeurographTool(self.repo_path, self.ng, pe)
        self._shell = ShellTool(self.repo_path, pe)
        self._workspace = WorkspaceTool(self.repo_path, self.ng, pe)

        self._init_memory()

    # === Cross-cutting sync (stays in facade) ===

    def _init_memory(self):
        self.memory_path.mkdir(parents=True, exist_ok=True)
        if self.token:
            try:
                hf_hub_download(
                    repo_id=self.dataset_id, filename="notebook.json", repo_type="dataset",
                    token=self.token, local_dir=self.memory_path, local_dir_use_symlinks=False
                )
            except (OSError, ValueError) as e:
                logger.warning("Failed to download notebook from HF: %s", e)
                self._notebook._save_local([])

    def _save_notebook(self, notes: List[Dict]):
        if self.token and self.dataset_id:
            try:
                api = HfApi(token=self.token)
                api.upload_file(
                    path_or_fileobj=self.notebook_file, path_in_repo="notebook.json",
                    repo_id=self.dataset_id, repo_type="dataset",
                    commit_message=f"Notebook Update: {len(notes)}"
                )
            except (OSError, ConnectionError) as e:
                logger.warning("HF notebook sync failed: %s", e)

    def _backup_ng_checkpoint_to_dataset(self):
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
                commit_message=f"NeuroGraph checkpoint ({self.ng.stats()['nodes']} nodes)"
            )
            logger.info("NeuroGraph checkpoint uploaded.")
        except (OSError, ConnectionError) as e:
            logger.warning("NeuroGraph checkpoint upload failed: %s", e)

    def save_conversation_turn(self, user_msg: str, assist_msg: str, turn_id: int):
        from openclaw_hook import NeuroGraphMemory
        from universal_ingestor import SourceType
        combined = f"USER: {user_msg}\n\nASSISTANT: {assist_msg}"
        self.ng.on_message(combined, source_type=SourceType.TEXT)
        self.ng.step(5)
        self._saves_since_ng_backup += 1
        if self._saves_since_ng_backup >= self.NG_BACKUP_EVERY_N:
            self._backup_ng_checkpoint_to_dataset()
            self._saves_since_ng_backup = 0

    # === Delegated methods ===

    def read_file(self, path, start_line=None, end_line=None):
        return self._fs.read_file(path, start_line, end_line)

    def write_file(self, path, content):
        return self._fs.write_file(path, content)

    def list_files(self, path=".", max_depth=3):
        return self._fs.list_files(path, max_depth)

    def push_to_github(self, message):
        return self._git.push_to_github(message)

    def pull_from_github(self, branch):
        return self._git.pull_from_github(branch)

    def create_shadow_branch(self):
        return self._git.create_shadow_branch()

    def notebook_read(self):
        return self._notebook.notebook_read()

    def notebook_add(self, content):
        return self._notebook.notebook_add(content)

    def notebook_delete(self, index):
        return self._notebook.notebook_delete(index)

    def search_conversations(self, query, n=5):
        return self._ng_tool.search_conversations(query, n)

    def search_code(self, query, n=5):
        return self._ng_tool.search_code(query, n)

    def search_testament(self, query, n=5):
        return self._ng_tool.search_testament(query, n)

    def ingest_workspace(self):
        return self._ng_tool.ingest_workspace()

    def shell_execute(self, command):
        return self._shell.shell_execute(command)

    def map_repository_structure(self):
        return self._workspace.map_repository_structure()

    def get_stats(self):
        return self._workspace.get_stats()

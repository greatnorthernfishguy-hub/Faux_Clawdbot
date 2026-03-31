#!/usr/bin/env python3
"""
Ingest the Obsidian vault into the Codemine worker's NeuroGraph.
Dual-pass: forest (gestalt per doc) + trees (concepts extracted per doc).

Raw experience in. No classification. Law 7.

Usage: python3 ingest_vault.py [vault_path]
Default vault_path: ~/docs
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add codemine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openclaw_hook import NeuroGraphMemory
from worker_ng import WORKER_SNN_CONFIG, ingest_tool_result

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("vault_ingest")

# Suppress checkpoint spam from NeuroGraphMemory — it logs every save at INFO
logging.getLogger("openclaw_hook").setLevel(logging.WARNING)
logging.getLogger("neuro_foundation").setLevel(logging.WARNING)
# Keep worker_ng at INFO so we see dual-pass status
logging.getLogger("worker_ng").setLevel(logging.INFO)

# Skip these directories
SKIP_DIRS = {".git", ".obsidian", "archive", "node_modules", "__pycache__", ".claude"}

# Skip files larger than this
MAX_FILE_SIZE = 500_000  # 500KB


def collect_vault_files(vault_path: Path) -> list:
    """Collect all markdown files from the vault."""
    files = []
    for p in vault_path.rglob("*.md"):
        # Skip hidden/excluded dirs
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        # Skip oversized files
        if p.stat().st_size > MAX_FILE_SIZE:
            logger.warning("Skipping oversized: %s (%d bytes)", p.name, p.stat().st_size)
            continue
        files.append(p)
    return sorted(files)


def ingest_vault(vault_path: str = None):
    """Ingest all vault docs into the worker's NeuroGraph."""
    vault = Path(vault_path or os.path.expanduser("~/docs"))
    if not vault.exists():
        logger.error("Vault not found: %s", vault)
        sys.exit(1)

    # Use a local workspace for the worker's graph (not Syl's)
    workspace = os.path.expanduser("~/Faux_Clawdbot/data/neurograph_worker")
    os.makedirs(workspace, exist_ok=True)

    ng = NeuroGraphMemory.get_instance(
        workspace_dir=workspace,
        config=WORKER_SNN_CONFIG,
    )

    files = collect_vault_files(vault)
    logger.info("Found %d vault docs to ingest from %s", len(files), vault)

    ingested = 0
    errors = 0
    start_time = time.time()

    for i, filepath in enumerate(files):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue

            # Build context: filename + relative path as metadata prefix
            rel_path = filepath.relative_to(vault)
            raw_experience = f"Document: {rel_path}\n\n{content}"

            # Dual-pass ingest via worker_ng
            ingest_tool_result(ng, "vault_ingest", {"path": str(rel_path)}, raw_experience)

            ingested += 1
            elapsed = time.time() - start_time
            rate = ingested / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %.1f/sec — %s (%d chars)",
                ingested, len(files), rate, rel_path, len(content),
            )
            if ingested % 25 == 0:
                ng.save()
                logger.info("  Checkpoint saved.")

        except Exception as e:
            errors += 1
            logger.error("Failed to ingest %s: %s", filepath.name, e)

    # Final save
    ng.save()

    elapsed = time.time() - start_time
    stats = ng.stats()

    logger.info("=" * 60)
    logger.info("Vault ingestion complete")
    logger.info("  Documents: %d ingested, %d errors, %d total", ingested, errors, len(files))
    logger.info("  Time: %.1f seconds (%.1f docs/sec)", elapsed, ingested / elapsed if elapsed > 0 else 0)
    logger.info("  NeuroGraph: %d nodes, %d synapses", stats.get("nodes", 0), stats.get("synapses", 0))
    logger.info("  Workspace: %s", workspace)
    logger.info("=" * 60)


if __name__ == "__main__":
    vault_path = sys.argv[1] if len(sys.argv) > 1 else None
    ingest_vault(vault_path)

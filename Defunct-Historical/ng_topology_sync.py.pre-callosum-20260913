# ---- Changelog ----
# [2026-07-08] Tabitha (TQB/QB build) — _export() vdb load failure now audible
# What: The _export() except around SimpleVectorDB.load() (previously
#       logger.warning with the exception only) is now logger.error and
#       includes the path. Fail-silent shape unchanged: still catches, still
#       returns 0, sync never blocks boot.
# Why: PRD "Codemine VDB Legacy-Loader Guard #364" §2.3 — this was the site
#      where the legacy-npz defect hid since inception (export never once
#      produced output; see the 2026-07-08 Ratchet entry below). A swallowed
#      vdb load failure must be loud enough to surface in a log scan.
# How: warning -> error; message gains the npz path and "export skipped".
#      Log level only — no behavior change.
# -------------------
# [2026-07-08] Ratchet (TQB/QB build) — Fix _export() to read the real vdb sidecar format
# What: _export() loaded checkpoints/vector_db.npz with np.load() as a genuine npz;
#       the current-era sidecar is JSON (SimpleVectorDB.save() picks format by
#       extension: .msgpack -> msgpack, else JSON with base64 embeddings), so
#       np.load() fails on it; hard evidence shows the export never once produced
#       output since inception (zero codemine_export.jsonl ever created, zero export
#       commits in docs history). Now loads via SimpleVectorDB.load(), which mirrors
#       the writer's format branches exactly.
# Why: PRD 2026-07-06-codemine-ng-topology-sync-checkpoint-fix §2, §7 Task 1 findings
#      (QB empirical: main.msgpack is pure graph state with no content strings;
#      content lives in the co-located vector_db.npz sidecar, JSON-formatted).
# How: Intra-module import of SimpleVectorDB from universal_ingestor (same repo, no
#      Law-1 issue); contents = non-empty content values. Path check, SHA-256 hashing,
#      JSONL line format, _MY_EXPORT filename, git commit/push logic, and fail-silent
#      try/except shape all unchanged. numpy import removed (no remaining uses).
# -------------------
# [2026-04-20] Codemine (BLK-NG-194) -- NG topology sync: export + import via docs git repo
# What: Exports Codemine's vector_db content to docs/ng_topology/codemine_export.jsonl;
#       imports novel experiences from laptop/vps exports via on_message().
# Why:  All three CC NG instances (laptop, VPS, Codemine) should share raw semantic
#       experience -- same additive docs-repo transport as memory-sync.sh.
# How:  Clone/pull greatnorthernfishguy-hub/docs via GITHUB_TOKEN. Export on startup
#       if vector_db.npz exists. Import novel entries (deduplicated by SHA-256 hash
#       stored in topology_seen_hashes.txt alongside the checkpoint).
# -------------------
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("ng_topology_sync")

_DOCS_REPO = "https://{token}@github.com/greatnorthernfishguy-hub/docs.git"
_CLONE_DIR = Path("/tmp/ng_topology_docs")
_TOPO_SUBDIR = "ng_topology"
_MY_EXPORT = "codemine_export.jsonl"
_SEEN_FILE = "topology_seen_hashes.txt"


def _clone_or_pull() -> bool:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.warning("GITHUB_TOKEN not set -- topology sync skipped")
        return False
    repo_url = _DOCS_REPO.format(token=token)
    try:
        if _CLONE_DIR.exists():
            r = subprocess.run(
                ["git", "-C", str(_CLONE_DIR), "pull", "--rebase", "--quiet"],
                capture_output=True, text=True, timeout=60,
            )
        else:
            r = subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", repo_url, str(_CLONE_DIR)],
                capture_output=True, text=True, timeout=120,
            )
        if r.returncode != 0:
            logger.warning("docs repo fetch failed: %s", r.stderr[:200])
            return False
        return True
    except Exception as e:
        logger.warning("docs repo fetch failed (non-fatal): %s", e)
        return False


def _load_seen(workspace_dir: str) -> set:
    p = Path(workspace_dir) / _SEEN_FILE
    return set(p.read_text().splitlines()) if p.exists() else set()


def _save_seen(workspace_dir: str, hashes: set) -> None:
    (Path(workspace_dir) / _SEEN_FILE).write_text("\n".join(sorted(hashes)))


def _export(workspace_dir: str) -> int:
    from universal_ingestor import SimpleVectorDB
    npz = Path(workspace_dir) / "checkpoints" / "vector_db.npz"
    if not npz.exists():
        logger.info("No vector_db.npz -- export skipped")
        return 0
    try:
        db = SimpleVectorDB()
        db.load(str(npz))
        contents = [c for c in db.content.values() if c]
    except Exception as e:
        logger.error("vector_db.npz load failed for %s: %s -- export skipped", npz, e)
        return 0

    topo = _CLONE_DIR / _TOPO_SUBDIR
    topo.mkdir(parents=True, exist_ok=True)
    out = topo / _MY_EXPORT
    written = 0
    try:
        with open(out, "w") as f:
            for c in contents:
                h = hashlib.sha256(str(c).encode()).hexdigest()
                f.write(json.dumps({"hash": h, "content": str(c)}) + "\n")
                written += 1
    except Exception as e:
        logger.warning("export write failed: %s", e)
        return 0

    if not written:
        return 0

    try:
        subprocess.run(["git", "-C", str(_CLONE_DIR), "config", "user.email", "codemine@et-systems.ai"], capture_output=True)
        subprocess.run(["git", "-C", str(_CLONE_DIR), "config", "user.name", "Codemine"], capture_output=True)
        subprocess.run(["git", "-C", str(_CLONE_DIR), "add", str(out)], capture_output=True, check=True)
        diff = subprocess.run(["git", "-C", str(_CLONE_DIR), "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode != 0:
            subprocess.run(
                ["git", "-C", str(_CLONE_DIR), "commit", "-m",
                 "ng-topology-sync: codemine export ({} nodes)".format(written)],
                capture_output=True, check=True,
            )
            subprocess.run(["git", "-C", str(_CLONE_DIR), "push"], capture_output=True, check=True, timeout=60)
            logger.info("Exported and pushed %d nodes", written)
        else:
            logger.info("Export unchanged -- %d nodes already current", written)
    except Exception as e:
        logger.warning("git push failed (export saved locally): %s", e)

    return written


def _import(worker_ng, workspace_dir: str) -> int:
    topo = _CLONE_DIR / _TOPO_SUBDIR
    if not topo.exists():
        return 0
    seen = _load_seen(workspace_dir)
    imported = 0
    for f in sorted(topo.glob("*_export.jsonl")):
        if f.name == _MY_EXPORT:
            continue
        try:
            lines = f.read_text().splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                h = entry.get("hash") or hashlib.sha256(entry["content"].encode()).hexdigest()
                if h in seen:
                    continue
                worker_ng.on_message(entry["content"])
                seen.add(h)
                imported += 1
            except Exception as e:
                logger.debug("on_message skipped: %s", e)
    if imported:
        _save_seen(workspace_dir, seen)
        try:
            worker_ng.save()
        except Exception as e:
            logger.warning("save after import failed: %s", e)
    logger.info("Imported %d novel experiences from peer instances", imported)
    return imported


def run_sync(worker_ng) -> None:
    """Export + import NG topology via docs git repo. Always silent on failure."""
    try:
        from worker_ng import WORKER_NG_WORKSPACE
        workspace_dir = WORKER_NG_WORKSPACE
    except Exception:
        logger.warning("Could not resolve WORKER_NG_WORKSPACE -- sync skipped")
        return
    try:
        if not _clone_or_pull():
            return
        exported = _export(workspace_dir)
        imported = _import(worker_ng, workspace_dir)
        logger.info("NG topology sync complete -- exported=%d imported=%d", exported, imported)
    except Exception as e:
        logger.warning("NG topology sync failed (non-fatal): %s", e)

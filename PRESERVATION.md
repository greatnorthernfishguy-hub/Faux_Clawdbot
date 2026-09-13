# Codemine NeuroGraph Preservation Gate

<!--
Changelog
2026-09-12 — Codex
What: Added the controlled preservation and independent-copy verification runbook.
Why: Codemine's HF-hosted third CC hemisphere must survive modernization intact.
How: A preservation-only boot archives `/data/neurograph_worker`; Dev Mode SSH
     retrieves that archive; the same tool verifies every file locally.
-->

Run this gate before allowing the modernized Codemine application to start on
its Hugging Face Space. Preservation does not authorize restoration, migration,
or a normal application boot.

## What preservation mode guarantees

- It runs before the entrypoint's first write probe, cache creation, import,
  repository update, Gradio startup, NeuroGraph construction, sync, or Tonic.
- It refuses to create a missing graph.
- It requires `/data` to appear as a distinct mount in
  `/proc/self/mountinfo`; a writable directory is insufficient.
- It requires the existing `main.msgpack` topology and `vector_db.npz` vector
  sidecar to be present and nonempty.
- It copies the complete graph workspace, including activation state and any
  tracts, to a temporary directory; hashes every regular file before and after;
  and atomically publishes the archive only when both inventories match.
- It records the rectangular file inventory in `MANIFEST.csv` and the archive
  metadata in `MANIFEST.json`.

The archive under `/data/codemine-preservation` is the first safety copy. It is
on the same attached bucket as the live graph, so it is not independent
redundancy. A normal boot remains blocked until that archive has also been
copied off the Space and verified.

## Runtime configuration

The Docker image owns the path configuration:

```text
NEUROGRAPH_WORKSPACE_DIR=/data/neurograph_worker
CODEMINE_PRESERVATION_DIR=/data/codemine-preservation
CODEMINE_PERSISTENT_ROOT=/data
CODEMINE_BOOT_MODE=normal
```

Set `CODEMINE_BOOT_MODE=preserve` as a Space variable before the preservation
build is delivered. Set `CODEMINE_PRESERVATION_DRY_RUN=1` for the first boot.
The script refuses to run when any required path variable is missing.

## Controlled sequence

1. Keep the Space sleeping. Set `CODEMINE_BOOT_MODE=preserve` and
   `CODEMINE_PRESERVATION_DRY_RUN=1` in its settings.
2. Deliver the reviewed preservation commit. Its GitHub `main` workflow pushes
   to the Space, whose first new boot can only run the dry-run path.
3. Check the runtime log for `DRY RUN: validated ...`. Any refusal blocks the
   procedure; do not substitute a path or create a checkpoint.
4. Set `CODEMINE_PRESERVATION_DRY_RUN=0` and restart once. Check the log for
   `PRESERVED: ...` and record the full archive path.
5. Enable Hugging Face Dev Mode. From this laptop, connect without placing a
   token in a command or transcript:

   ```bash
   hf spaces ssh Executor-Tyrant-Framework/Working-in-a-Codemine
   ```

6. In the Space shell, confirm the named archive and create a transport copy in
   ephemeral storage:

   ```bash
   ls -la /data/codemine-preservation
   tar -C /data/codemine-preservation -czf /tmp/codemine-ng-backup.tar.gz ARCHIVE_NAME
   ```

7. On the laptop, ask the HF CLI for the exact SSH endpoint, copy the tarball to
   `/home/josh/backups/codemine-neurograph/`, and extract it there. The CLI may
   include identity options that must also be used by `scp`:

   ```bash
   hf spaces ssh Executor-Tyrant-Framework/Working-in-a-Codemine --dry-run
   ```

8. Verify the extracted archive with the exact preservation script from the
   delivered commit:

   ```bash
   python3 preserve_codemine.py --verify /home/josh/backups/codemine-neurograph/ARCHIVE_NAME
   ```

   `VERIFIED: ...` is the required result. Retain the Space log, archive name,
   local path, file count, byte count, and deployed code revision as the receipt.
9. Keep `CODEMINE_BOOT_MODE=preserve` until the graph has also passed offline
   compatibility inspection and the normal-start modernization has its own
   review gate.

Hugging Face documents that attached buckets are mounted into the container at
their configured paths and that Dev Mode supplies SSH access for Pro accounts.
The Space image includes the Docker Dev Mode prerequisites for that retrieval
path.

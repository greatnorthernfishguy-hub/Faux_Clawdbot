# Codemine deployment gates

<!--
Changelog
2026-09-12 — Codex
What: Replaced the obsolete generic Space quick-start with the actual gated
      delivery sequence for the existing Codemine instance.
Why: Pushing GitHub main can update and wake the live HF Space, whose existing
     third-hemisphere NeuroGraph has not yet been independently preserved.
How: Preservation, offline verification, compatibility, normal-start, and
     mission smoke-test gates are separate and produce reviewable evidence.
-->

The target is the existing private Hugging Face Space
`Executor-Tyrant-Framework/Working-in-a-Codemine`. This is an upgrade of a
stateful system, not a new stateless Space.

## Current stop condition

Do not push the modernization branch to the GitHub branch that mirrors the
Space. That push can rebuild or wake the Space. Normal startup remains blocked
until the existing graph has two verified copies and the runtime compatibility
review is complete.

No deployment step may create a replacement graph merely because a checkpoint
is absent or unreadable. Preserve and diagnose the existing state.

## Gate 1: code review while the Space sleeps

Before any delivery:

1. Review the complete modernization branch against its merge base.
2. Run the full repository test suite and Python compilation checks.
3. Run the NeuroGraph Law Enforcer against substrate boundaries, raw experience
   flow, shared-Graph assumptions, vendored files, and autonomic behavior.
4. Confirm the diff contains no secrets and no accidental edits to checkpoint,
   activation, vector, or vendored files.
5. Record the reviewed commit hash.

## Gate 2: preservation-only delivery

Follow [PRESERVATION.md](PRESERVATION.md). The only permitted first delivery
mode is:

```text
CODEMINE_BOOT_MODE=preserve
CODEMINE_PRESERVATION_DRY_RUN=1
```

The entrypoint checks this mode before cache probes, repository pulls, Python
imports, NeuroGraph construction, topology sync, Tonic, or Gradio. A successful
dry run establishes paths and mount identity but does not create an archive.

After reviewing that receipt, run preservation mode once with the dry-run flag
disabled. Keep the normal application stopped.

## Gate 3: independent backup

The archive under `/data/codemine-preservation` shares the attached storage
volume with the live graph. Copy that archive to the laptop and verify every
file against `MANIFEST.csv` and `MANIFEST.json` with `preserve_codemine.py
--verify`.

The gate receipt must include:

- deployed code revision;
- source graph path;
- Space archive path;
- laptop archive path;
- file and byte counts;
- verification result.

## Gate 4: offline compatibility review

Inspect a copy, never the live graph. Establish:

- checkpoint and vector sidecar formats;
- topology/node/edge counts and metadata expectations;
- activation and tract artifacts;
- the exact Codemine Tonic and embedder path;
- compatibility with the current shared CC/Codemine graph standard;
- whether a migration is required and its reversible procedure.

Heuristic execution is not an embedding-compatible fallback. Do not silently
switch the restored Space to a heuristic path.

## Gate 5: normal-start readiness

Before setting `CODEMINE_BOOT_MODE=normal`, verify the image owns these runtime
paths and provider routes:

```text
NEUROGRAPH_WORKSPACE_DIR=/data/neurograph_worker
CODEMINE_PRESERVATION_DIR=/data/codemine-preservation
CODEMINE_PERSISTENT_ROOT=/data
CODEMINE_MISSION_ROOT=/data/codemine-missions
CODEMINE_WORKTREE_ROOT=/tmp/codemine-worktrees
CODEMINE_DEFAULT_WORKSPACE=/workspace/e-t-systems
CODEMINE_WORKSPACE_ROOTS=/workspace
CODEMINE_VALIDATION_CATALOG_PATH=/app/validation_catalog.json
```

Also verify `/data` is the attached persistent mount, the target Git repository
exists, configured provider credentials are present as Space secrets, and the
selected model IDs are available.

## Gate 6: bounded smoke test

After normal-start approval:

1. Open Chat and perform one read-only repository inspection.
2. Use Mission Control to create a read-only MissionPlan.
3. Confirm the proposal appears under
   `/data/codemine-missions/proposals` and includes route receipts.
4. Execute a disposable or documentation-only standard-risk plan.
5. Confirm each block has a separate worktree, selected validations ran, Reviewer
   inspected the actual diff, and the integration worktree reached
   `ready_for_external_review`.
6. Confirm the source checkout status before and after is identical.
7. Do not merge or push the smoke-test integration branch from Codemine.

## Delivery authority

A `ready_for_external_review` result means Codemine completed its bounded build.
It is not permission to merge, push, deploy, restart a service, modify a
vendored canonical file, or propagate that file to modules. Those actions stay
with the external development manager and the applicable human/Law gates.

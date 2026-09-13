"""Static guards for Codemine's fail-closed container startup contract."""

# ---- Changelog ----
# [2026-09-13] Codex — Pin preservation dry-run as the image and shell default.
# What: Verify both Docker configuration and the entrypoint fail closed.
# Why: A missing Space variable must never wake an unpreserved legacy Graph.
# How: Inspect the shipped deployment files without launching the application.
# -------------------

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_docker_image_defaults_to_preservation_dry_run():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV CODEMINE_BOOT_MODE=preserve" in dockerfile
    assert "ENV CODEMINE_PRESERVATION_DRY_RUN=1" in dockerfile
    assert "ENV CODEMINE_NORMAL_BOOT_ACK" not in dockerfile
    assert "COPY boot_mode.py ." in dockerfile
    assert "COPY archive_safety.py ." in dockerfile
    assert "COPY spec_executor.py ." not in dockerfile


def test_entrypoint_defaults_to_preservation_when_variable_is_absent():
    entrypoint = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")

    assert '${CODEMINE_BOOT_MODE:-preserve}' in entrypoint
    assert '${CODEMINE_BOOT_MODE:-normal}' not in entrypoint
    assert "python3 /app/boot_mode.py --check-normal" in entrypoint
    assert "Invalid CODEMINE_BOOT_MODE" in entrypoint


def test_worker_wrapper_is_the_only_operational_graph_constructor():
    guarded_owner = (ROOT / "worker_ng.py").read_text(encoding="utf-8")
    assert "NeuroGraphMemory.get_instance(" in guarded_owner

    for relative in ("app.py", "recursive_context.py", "ingest_vault.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "NeuroGraphMemory.get_instance(" not in source

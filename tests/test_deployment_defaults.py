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


def test_entrypoint_defaults_to_preservation_when_variable_is_absent():
    entrypoint = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")

    assert '${CODEMINE_BOOT_MODE:-preserve}' in entrypoint
    assert '${CODEMINE_BOOT_MODE:-normal}' not in entrypoint

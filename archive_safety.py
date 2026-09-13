"""Pure validation helpers for untrusted ZIP uploads."""

# ---- Changelog ----
# [2026-09-13] Codex — Refuse ZIP traversal and symlink members before extraction.
# What: Validate every member against the resolved extraction root.
# Why: Size checks alone do not prevent an uploaded archive from writing elsewhere.
# How: Reject absolute, parent, backslash, NUL, symlink, and resolved escape paths.
# -------------------

from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo


def validated_zip_members(archive: ZipFile, extract_root: Path) -> list[ZipInfo]:
    """Return safe members or raise ``ValueError`` before any filesystem write."""
    root = Path(extract_root).resolve()
    members = archive.infolist()
    for info in members:
        name = info.filename
        member_path = Path(name)
        file_type = (info.external_attr >> 16) & 0o170000
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or member_path.is_absolute()
            or ".." in member_path.parts
            or stat.S_ISLNK(file_type)
        ):
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        destination = (root / member_path).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"ZIP member escapes extraction root: {name!r}")
    return members

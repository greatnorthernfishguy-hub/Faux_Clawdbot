"""Hostile archive members must be refused before extraction starts."""

# ---- Changelog ----
# [2026-09-13] Codex — Cover archive destination-root confinement.
# What: Test unsafe basenames, repository-local roots, and safe scratch roots.
# Why: Member validation alone did not constrain the parent extraction path.
# How: Exercise the pure upload-root resolver without filesystem mutation.
# -------------------

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from archive_safety import safe_upload_extract_root, validated_zip_members


def make_zip(tmp_path, members):
    archive_path = tmp_path / "upload.zip"
    with ZipFile(archive_path, "w") as archive:
        for member, content in members:
            if isinstance(member, ZipInfo):
                archive.writestr(member, content)
            else:
                archive.writestr(member, content)
    return archive_path


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "dir\\escape.py"])
def test_traversal_and_nonportable_paths_are_rejected(tmp_path, name):
    archive_path = make_zip(tmp_path, [(name, "bad")])
    with ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe ZIP"):
            validated_zip_members(archive, tmp_path / "extract")


def test_symlink_member_is_rejected(tmp_path):
    member = ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive_path = make_zip(tmp_path, [(member, "../outside")])
    with ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe ZIP"):
            validated_zip_members(archive, tmp_path / "extract")


def test_safe_members_are_returned_without_writing(tmp_path):
    archive_path = make_zip(tmp_path, [("src/module.py", "VALUE = 1\n")])
    extract_root = tmp_path / "extract"
    with ZipFile(archive_path) as archive:
        members = validated_zip_members(archive, extract_root)
    assert [member.filename for member in members] == ["src/module.py"]
    assert not extract_root.exists()


@pytest.mark.parametrize("name", ["..zip", ".zip", "../.zip"])
def test_unsafe_archive_basename_cannot_widen_destination(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe ZIP archive name"):
        safe_upload_extract_root(
            tmp_path / "uploads", name, repository_root=tmp_path / "repo",
        )


def test_upload_root_must_be_outside_repository(tmp_path):
    repository = tmp_path / "repo"
    with pytest.raises(ValueError, match="outside the source repository"):
        safe_upload_extract_root(
            repository / "uploads", "safe.zip", repository_root=repository,
        )


def test_safe_archive_uses_environment_owned_upload_root(tmp_path):
    repository = tmp_path / "repo"
    uploads = tmp_path / "uploads"
    destination = safe_upload_extract_root(
        uploads, "bundle.zip", repository_root=repository,
    )
    assert destination == uploads / "bundle"

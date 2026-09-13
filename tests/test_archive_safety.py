"""Hostile archive members must be refused before extraction starts."""

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from archive_safety import validated_zip_members


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

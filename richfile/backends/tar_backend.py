from __future__ import annotations

from typing import List, Union

from pathlib import Path
import io
import tarfile
import time

from .archive_common import ArchiveBackendBase
from . import helpers


class TarBackend(ArchiveBackendBase):
    """
    Plain TAR archive backend (no compression in v1).
    """

    def _backend_name(self) -> str:
        """Return the backend identifier string."""
        return "tar"

    def _open_reader(self, path_archive: Union[str, Path]):
        """Open a TAR archive for reading (uncompressed stream mode)."""
        return tarfile.open(name=str(path_archive), mode="r:")

    def _open_writer(self, path_archive: Union[str, Path]):
        """Open a new TAR archive for writing."""
        return tarfile.open(name=str(path_archive), mode="w")

    def _iter_raw_members(self, reader: tarfile.TarFile) -> List[str]:
        """Return all member names and cache TarInfo objects on the reader."""
        members_by_name = {member.name: member for member in reader.getmembers()}
        setattr(reader, "_richfile_members_by_name", members_by_name)
        return list(members_by_name.keys())

    def _is_raw_member_dir(self, reader: tarfile.TarFile, raw_name: str) -> bool:
        """Check whether a raw member name represents a directory entry."""
        members_by_name = getattr(reader, "_richfile_members_by_name", None)
        if members_by_name is None:
            member = reader.getmember(raw_name)
        else:
            member = members_by_name[raw_name]
        return member.isdir()

    def _read_raw_member_bytes(self, reader: tarfile.TarFile, raw_name: str) -> bytes:
        """Read the raw bytes of a TAR member. Raises if the member is not a regular file."""
        members_by_name = getattr(reader, "_richfile_members_by_name", None)
        if members_by_name is None:
            member = reader.getmember(raw_name)
        else:
            member = members_by_name[raw_name]
        fileobj = reader.extractfile(member)
        if fileobj is None:
            raise FileNotFoundError(f"TAR member '{raw_name}' is not a regular file.")
        return fileobj.read()

    def _write_file_member(self, writer: tarfile.TarFile, member_name: str, data: bytes) -> None:
        """Write a file member to the TAR archive with current timestamp."""
        member_name = helpers.normalize_archive_path(path_in_archive=member_name)
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = 0o644
        writer.addfile(tarinfo=info, fileobj=io.BytesIO(data))

    def _write_dir_member(self, writer: tarfile.TarFile, member_name: str) -> None:
        """Write an empty directory entry to the TAR archive."""
        path_dir = helpers.normalize_archive_path(path_in_archive=member_name)
        if path_dir == "":
            return
        info = tarfile.TarInfo(name=path_dir)
        info.type = tarfile.DIRTYPE
        info.mtime = int(time.time())
        info.mode = 0o755
        writer.addfile(tarinfo=info)

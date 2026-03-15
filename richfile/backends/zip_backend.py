from __future__ import annotations

from typing import List, Union

from pathlib import Path
import zipfile

from .archive_common import ArchiveBackendBase
from . import helpers


class ZipBackend(ArchiveBackendBase):
    """
    ZIP archive backend using uncompressed members (ZIP_STORED).
    """

    def _backend_name(self) -> str:
        return "zip"

    def _open_reader(self, path_archive: Union[str, Path]):
        return zipfile.ZipFile(file=str(path_archive), mode="r")

    def _open_writer(self, path_archive: Union[str, Path]):
        return zipfile.ZipFile(
            file=str(path_archive),
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )

    def _iter_raw_members(self, reader: zipfile.ZipFile) -> List[str]:
        infos_by_name = {info.filename: info for info in reader.infolist()}
        setattr(reader, "_richfile_infos_by_name", infos_by_name)
        return list(infos_by_name.keys())

    def _is_raw_member_dir(self, reader: zipfile.ZipFile, raw_name: str) -> bool:
        infos_by_name = getattr(reader, "_richfile_infos_by_name")
        info = infos_by_name.get(raw_name)
        if info is None:
            return raw_name.endswith("/")
        return info.is_dir()

    def _read_raw_member_bytes(self, reader: zipfile.ZipFile, raw_name: str) -> bytes:
        return reader.read(raw_name)

    def _write_file_member(self, writer: zipfile.ZipFile, member_name: str, data: bytes) -> None:
        member_name = helpers.normalize_archive_path(path_in_archive=member_name)
        info = zipfile.ZipInfo(filename=member_name)
        info.compress_type = zipfile.ZIP_STORED
        writer.writestr(zinfo_or_arcname=info, data=data)

    def _write_dir_member(self, writer: zipfile.ZipFile, member_name: str) -> None:
        path_dir = helpers.normalize_archive_path(path_in_archive=member_name)
        if path_dir == "":
            return
        info = zipfile.ZipInfo(filename=f"{path_dir}/")
        info.compress_type = zipfile.ZIP_STORED
        writer.writestr(zinfo_or_arcname=info, data=b"")

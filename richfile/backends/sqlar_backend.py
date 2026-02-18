from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple, Union

from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import time

from . import helpers
from .archive_common import ArchiveBackendBase

_SQLAR_MODE_FILE = 0o100644
_SQLAR_MODE_DIR = 0o040755


class SQLARBackend(ArchiveBackendBase):
    """
    SQLite SQLAR backend.

    This backend intentionally reuses shared archive logic for save/load/lazy
    traversal. The SQLAR-specific code here only defines database primitives.
    """

    def _backend_name(self) -> str:
        return "sqlar"

    def _open_reader(self, path_archive: Union[str, Path]):
        return closing(self._connect(path_archive=path_archive))

    @contextmanager
    def _open_writer(self, path_archive: Union[str, Path]):
        conn = self._connect(path_archive=path_archive)
        try:
            self._initialize_schema(conn=conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _iter_raw_members(self, reader: sqlite3.Connection):
        rows = reader.execute("SELECT name FROM sqlar").fetchall()
        return [row[0] for row in rows]

    def _is_raw_member_dir(self, reader: sqlite3.Connection, raw_name: str) -> bool:
        row = self._read_row(conn=reader, row_name=raw_name)
        if row is None:
            return False
        return (row[4] is None) or (row[1] == _SQLAR_MODE_DIR)

    def _read_raw_member_bytes(self, reader: sqlite3.Connection, raw_name: str) -> bytes:
        row = self._read_row(conn=reader, row_name=raw_name)
        if row is None:
            raise FileNotFoundError(f"SQLAR row not found: {raw_name}")
        if row[4] is None:
            raise ValueError(f"SQLAR row has no data: {raw_name}")
        return row[4]

    def _write_file_member(self, writer: sqlite3.Connection, member_name: str, data: bytes) -> None:
        self._write_row(
            conn=writer,
            row_name=member_name,
            data=data,
            mode=_SQLAR_MODE_FILE,
        )

    def _write_dir_member(self, writer: sqlite3.Connection, member_name: str) -> None:
        path_dir = helpers.validate_archive_path(
            path_in_archive=member_name,
            allow_empty=True,
        )
        if path_dir == "":
            return
        self._write_row(
            conn=writer,
            row_name=path_dir,
            data=None,
            mode=_SQLAR_MODE_DIR,
        )

    def _build_index(self, reader: sqlite3.Connection) -> Dict[str, Any]:
        rows_raw = reader.execute("SELECT name, mode, data IS NULL FROM sqlar").fetchall()

        files: Dict[str, str] = {}
        dirs: Set[str] = set()

        for row_name_raw, mode, is_data_null in rows_raw:
            path_name = helpers.validate_archive_path(
                path_in_archive=row_name_raw,
                allow_empty=True,
            )
            if path_name == "":
                continue
            if bool(is_data_null) or (int(mode) == _SQLAR_MODE_DIR):
                dirs.add(path_name)
            else:
                files[path_name] = path_name

        for path_file in list(files.keys()):
            path_parent, _ = helpers.split_archive_path(path_in_archive=path_file)
            while path_parent != "":
                dirs.add(path_parent)
                path_parent, _ = helpers.split_archive_path(path_in_archive=path_parent)

        children = self._build_children_index(
            names_all=set(files.keys()) | set(dirs),
        )
        return {
            "files": files,
            "dirs": dirs,
            "children": children,
        }

    def _connect(self, path_archive: Union[str, Path]) -> sqlite3.Connection:
        return sqlite3.connect(str(path_archive))

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sqlar(
                name TEXT PRIMARY KEY,
                mode INT,
                mtime INT,
                sz INT,
                data BLOB
            )
            """
        )
        conn.execute("DELETE FROM sqlar")

    def _write_row(
        self,
        conn: sqlite3.Connection,
        row_name: str,
        data: Optional[bytes],
        mode: int,
    ) -> None:
        row_name = helpers.validate_archive_path(
            path_in_archive=row_name,
            allow_empty=False,
        )
        size_bytes = 0 if data is None else len(data)
        conn.execute(
            """
            INSERT OR REPLACE INTO sqlar(name, mode, mtime, sz, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row_name,
                int(mode),
                int(time.time()),
                int(size_bytes),
                data,
            ),
        )

    def _read_row(
        self,
        conn: sqlite3.Connection,
        row_name: str,
    ) -> Optional[Tuple[str, int, int, int, Optional[bytes]]]:
        row_name = helpers.validate_archive_path(
            path_in_archive=row_name,
            allow_empty=False,
        )
        row = conn.execute(
            "SELECT name, mode, mtime, sz, data FROM sqlar WHERE name = ? LIMIT 1",
            (row_name,),
        ).fetchone()
        return row

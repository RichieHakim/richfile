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
        """Return the backend identifier string."""
        return "sqlar"

    def _open_reader(self, path_archive: Union[str, Path]):
        """
        Open a read-only SQLite connection to the SQLAR archive.
        Raises ``FileNotFoundError`` if the archive does not exist (prevents
        ``sqlite3.connect`` from silently creating an empty file).
        """
        path_archive = Path(path_archive)
        if not path_archive.exists():
            raise FileNotFoundError(f"SQLAR archive not found: {path_archive}")
        conn = sqlite3.connect(f"file:{path_archive}?mode=ro", uri=True)
        return closing(conn)

    @contextmanager
    def _open_writer(self, path_archive: Union[str, Path]):
        """Open a read-write SQLite connection, initialize schema, and commit on success."""
        conn = self._connect(path_archive=path_archive)
        try:
            self._initialize_schema(conn=conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _iter_raw_members(self, reader: sqlite3.Connection):
        """Return all row names from the sqlar table."""
        rows = reader.execute("SELECT name FROM sqlar").fetchall()
        return [row[0] for row in rows]

    def _is_raw_member_dir(self, reader: sqlite3.Connection, raw_name: str) -> bool:
        """Check whether a row represents a directory (null data or dir mode)."""
        row = self._read_row(conn=reader, row_name=raw_name)
        if row is None:
            return False
        return (row[4] is None) or (row[1] == _SQLAR_MODE_DIR)

    def _read_raw_member_bytes(self, reader: sqlite3.Connection, raw_name: str) -> bytes:
        """Read the data blob for a file row. Raises if the row is missing or has no data."""
        row = self._read_row(conn=reader, row_name=raw_name)
        if row is None:
            raise FileNotFoundError(f"SQLAR row not found: {raw_name}")
        if row[4] is None:
            raise ValueError(f"SQLAR row has no data: {raw_name}")
        return row[4]

    def _write_file_member(self, writer: sqlite3.Connection, member_name: str, data: bytes) -> None:
        """Write a file row to the sqlar table."""
        self._write_row(
            conn=writer,
            row_name=member_name,
            data=data,
            mode=_SQLAR_MODE_FILE,
        )

    def _write_dir_member(self, writer: sqlite3.Connection, member_name: str) -> None:
        """Write a directory row (null data) to the sqlar table. Skips root."""
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
        """
        Build a files/dirs/children index from all sqlar rows. Overrides the
        base class to use a single SQL query instead of per-member calls.
        """
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
        """Open a read-write SQLite connection (used by the writer)."""
        return sqlite3.connect(str(path_archive))

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        """Create the sqlar table if needed and clear all existing rows."""
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
        """
        Insert or replace a single row in the sqlar table.

        Args:
            conn (sqlite3.Connection):
                Active database connection.
            row_name (str):
                Archive member name (validated before insertion).
            data (Optional[bytes]):
                File contents, or ``None`` for directory entries.
            mode (int):
                Unix-style mode bits (e.g. ``0o100644`` for files).
        """
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
        """
        Read a single row from the sqlar table by name.

        Args:
            conn (sqlite3.Connection):
                Active database connection.
            row_name (str):
                Archive member name to look up.

        Returns:
            (Optional[Tuple[str, int, int, int, Optional[bytes]]]):
                row (Optional[Tuple]):
                    ``(name, mode, mtime, sz, data)`` or ``None`` if not found.
        """
        row_name = helpers.validate_archive_path(
            path_in_archive=row_name,
            allow_empty=False,
        )
        row = conn.execute(
            "SELECT name, mode, mtime, sz, data FROM sqlar WHERE name = ? LIMIT 1",
            (row_name,),
        ).fetchone()
        return row

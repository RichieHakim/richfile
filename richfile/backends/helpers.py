from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import copy
import inspect
import re
from pathlib import Path

from .. import FILENAME_METADATA

_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def normalize_archive_path(path_in_archive: Optional[str]) -> str:
    """
    Normalize an internal archive path to a forward-slash relative form.
    Strips leading/trailing slashes, converts backslashes, and collapses
    empty/dot-only paths to ``""``.

    Args:
        path_in_archive (Optional[str]):
            Raw archive member path. ``None`` is treated as empty.

    Returns:
        (str):
            path_normalized (str):
                Cleaned forward-slash relative path, or ``""`` for root.
    """
    if path_in_archive is None:
        return ""
    path_clean = str(path_in_archive).replace("\\", "/").strip("/")
    return "" if path_clean in {"", ".", "/"} else path_clean


def validate_archive_path(path_in_archive: Optional[str], allow_empty: bool = True) -> str:
    """
    Validate and normalize an archive member path, rejecting absolute paths,
    drive prefixes, and parent-traversal segments (``..``).

    Args:
        path_in_archive (Optional[str]):
            Raw archive member path.
        allow_empty (bool):
            If ``True``, ``None`` and ``""`` are accepted and return ``""``.

    Returns:
        (str):
            path_validated (str):
                Normalized, validated path safe for archive operations.

    Raises:
        ValueError:
            If the path is absolute, contains ``..``, or is empty when
            ``allow_empty=False``.
    """
    if path_in_archive is None:
        if allow_empty:
            return ""
        raise ValueError("Archive path cannot be empty.")

    path_raw = str(path_in_archive)
    if path_raw == "":
        if allow_empty:
            return ""
        raise ValueError("Archive path cannot be empty.")

    path_posix = path_raw.replace("\\", "/")
    if path_posix.startswith("/"):
        raise ValueError(f"Archive path must be relative, found absolute path: {path_raw}")
    if _WINDOWS_DRIVE_PREFIX.match(path_posix):
        raise ValueError(f"Archive path must not include drive prefixes: {path_raw}")

    path_norm = normalize_archive_path(path_in_archive=path_posix)
    if path_norm == "":
        if allow_empty:
            return ""
        raise ValueError("Archive path cannot be empty.")

    for part in path_norm.split("/"):
        if part in {"", ".", ".."}:
            raise ValueError(f"Invalid archive path segment '{part}' in path: {path_raw}")
    return path_norm


def safe_resolve_materialized_path(path_root: Path, path_relative: str) -> Path:
    """
    Resolve a materialized output path and enforce that it stays inside
    ``path_root``. Prevents archive extraction from writing outside the
    intended directory via path traversal.

    Args:
        path_root (Path):
            Root directory that all materialized paths must stay within.
        path_relative (str):
            Relative archive path to resolve.

    Returns:
        (Path):
            path_resolved (Path):
                Fully resolved path guaranteed to be under ``path_root``.

    Raises:
        ValueError:
            If the resolved path escapes ``path_root``.
    """
    path_relative = validate_archive_path(path_in_archive=path_relative, allow_empty=False)
    path_root_resolved = path_root.resolve()
    path_target_resolved = (path_root_resolved / path_relative).resolve()

    try:
        path_target_resolved.relative_to(path_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Materialized path escapes root directory. root={path_root_resolved}, "
            f"relative={path_relative}, resolved={path_target_resolved}"
        ) from exc
    return path_target_resolved


def join_archive_path(path_parent: str, path_child: str) -> str:
    """
    Join two archive-internal paths with forward-slash separator.

    Args:
        path_parent (str):
            Parent path segment.
        path_child (str):
            Child path segment.

    Returns:
        (str):
            path_joined (str):
                Combined path. If either is empty, returns the other.
    """
    path_parent = normalize_archive_path(path_in_archive=path_parent)
    path_child = normalize_archive_path(path_in_archive=path_child)
    if path_parent == "":
        return path_child
    if path_child == "":
        return path_parent
    return f"{path_parent}/{path_child}"


def split_archive_path(path_in_archive: str) -> Tuple[str, str]:
    """
    Split an archive-internal path into ``(parent, name)``.

    Args:
        path_in_archive (str):
            Archive-internal path to split.

    Returns:
        (Tuple[str, str]):
            path_parent (str):
                Parent directory path, or ``""`` if at root.
            name (str):
                Final segment name, or ``""`` if input is empty.
    """
    path_in_archive = normalize_archive_path(path_in_archive=path_in_archive)
    if path_in_archive == "":
        return "", ""
    path_parts = path_in_archive.split("/")
    if len(path_parts) == 1:
        return "", path_parts[0]
    return "/".join(path_parts[:-1]), path_parts[-1]


def metadata_row_name(path_in_archive: str) -> str:
    """
    Return the archive member name for the metadata file at a given internal
    path. For root, returns ``FILENAME_METADATA``; for nested paths, returns
    ``"path/to/.metadata.richfile"``.

    Args:
        path_in_archive (str):
            Archive-internal path (``""`` for root).

    Returns:
        (str):
            name_metadata (str):
                Full archive member name for the metadata file.
    """
    path_in_archive = normalize_archive_path(path_in_archive=path_in_archive)
    return FILENAME_METADATA if path_in_archive == "" else f"{path_in_archive}/{FILENAME_METADATA}"


def get_direct_child_name(path_parent: str, row_name: str) -> Optional[str]:
    """
    Return the direct child name under ``path_parent`` represented by
    ``row_name``, or ``None`` if ``row_name`` is not a descendant.

    Args:
        path_parent (str):
            Parent archive path (``""`` for root).
        row_name (str):
            Full archive member name to inspect.

    Returns:
        (Optional[str]):
            name_child (Optional[str]):
                First path segment under ``path_parent``, or ``None`` if
                ``row_name`` is not a descendant.
    """
    path_parent = normalize_archive_path(path_in_archive=path_parent)
    row_name = normalize_archive_path(path_in_archive=row_name)

    if path_parent == "":
        rel = row_name
    else:
        prefix = f"{path_parent}/"
        if not row_name.startswith(prefix):
            return None
        rel = row_name[len(prefix) :]

    if rel == "":
        return None
    return rel.split("/", maxsplit=1)[0]


def serialize_type_lookup(type_lookup_properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert callable/class values in a type lookup table to JSON-serializable
    strings. Functions are converted via ``inspect.getsource``; classes via
    ``str()``. The original list is not modified.

    Args:
        type_lookup_properties (List[Dict[str, Any]]):
            List of type property dicts from ``TypeLookup.properties``.

    Returns:
        (List[Dict[str, Any]]):
            properties_serialized (List[Dict[str, Any]]):
                Deep copy with ``function_load``, ``function_save``, and
                ``object_class`` replaced by string representations.
    """
    out = copy.deepcopy(type_lookup_properties)
    for prop in out:
        try:
            prop["function_load"] = inspect.getsource(prop["function_load"])
        except (OSError, TypeError):
            prop["function_load"] = str(prop["function_load"])

        try:
            prop["function_save"] = inspect.getsource(prop["function_save"])
        except (OSError, TypeError):
            prop["function_save"] = str(prop["function_save"])

        prop["object_class"] = str(prop["object_class"])
    return out

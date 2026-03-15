from __future__ import annotations

from typing import Any, Optional, Union

import copy
import shutil
import tempfile
from pathlib import Path

from . import FILENAME_METADATA
from . import util
from .backends.archive_common import ArchiveBackendBase

_BACKENDS_SUPPORTED = {"directory", "sqlar", "zip", "tar"}


def _validate_backend_name(backend: str) -> None:
    """
    Validate that the backend name is supported.
    """
    if backend not in _BACKENDS_SUPPORTED:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Supported backends: {sorted(_BACKENDS_SUPPORTED)}"
        )


def _prepare_target_path(path_target: Path, overwrite: bool) -> None:
    """
    Prepare a target path according to overwrite policy.
    """
    if path_target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target path already exists and overwrite=False: {path_target}"
            )
        util.delete_file_or_folder(path=path_target)


def _ensure_directory_style_root(path_directory: Path) -> None:
    """
    Ensure a directory path appears to be a directory-style richfile root.
    """
    if not path_directory.exists():
        raise FileNotFoundError(f"Directory-style richfile path not found: {path_directory}")
    if not path_directory.is_dir():
        raise ValueError(f"Expected directory-style richfile directory. Found file: {path_directory}")
    path_metadata = path_directory / FILENAME_METADATA
    if not path_metadata.exists():
        raise FileNotFoundError(
            f"Missing metadata file in directory-style richfile: {path_metadata}"
        )


def _copy_directory_tree(path_source: Path, path_target: Path) -> None:
    """
    Copy one directory tree to another path.
    """
    path_target.mkdir(parents=True, exist_ok=True)
    for path_item in sorted(path_source.rglob("*")):
        path_relative = path_item.relative_to(path_source)
        path_out = path_target / path_relative
        if path_item.is_dir():
            path_out.mkdir(parents=True, exist_ok=True)
        else:
            path_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=path_item, dst=path_out)


def extract_backend_to_directory(
    path_source: Union[str, Path],
    backend_source: str,
    path_directory_out: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Extract a backend payload into directory-style richfile files.

    Args:
        path_source (Union[str, Path]):
            Source backend root path.
        backend_source (str):
            Source backend name.
        path_directory_out (Union[str, Path]):
            Output directory path.
        overwrite (bool):
            Whether to overwrite the output path if it already exists.

    Returns:
        (Path):
            path_directory_out (Path):
                Directory containing directory-style richfile files.
    """
    _validate_backend_name(backend=backend_source)
    path_source = Path(path_source).resolve()
    path_directory_out = Path(path_directory_out).resolve()

    if path_source == path_directory_out:
        raise ValueError(
            f"Source and target paths must differ. Both resolve to: {path_source}"
        )

    _prepare_target_path(path_target=path_directory_out, overwrite=overwrite)
    path_directory_out.parent.mkdir(parents=True, exist_ok=True)

    if backend_source == "directory":
        _ensure_directory_style_root(path_directory=path_source)
        _copy_directory_tree(
            path_source=path_source,
            path_target=path_directory_out,
        )
    else:
        if not path_source.exists():
            raise FileNotFoundError(f"Source path not found: {path_source}")
        if not path_source.is_file():
            raise ValueError(f"Archive backend source must be a file. Found: {path_source}")

        richfile_source = util.RichFile(
            path=path_source,
            backend=backend_source,
            check=True,
        )
        backend_impl = richfile_source._get_backend_impl()
        if not isinstance(backend_impl, ArchiveBackendBase):
            raise TypeError(
                "Expected archive backend implementation for non-directory source. "
                f"Found: {type(backend_impl)}"
            )

        path_directory_out.mkdir(parents=True, exist_ok=True)
        with backend_impl._open_reader(path_archive=path_source) as reader:
            index = backend_impl._build_index(reader=reader)
            backend_impl._materialize_subtree(
                reader=reader,
                index=index,
                path_in_archive="",
                path_out=path_directory_out,
            )

    _ensure_directory_style_root(path_directory=path_directory_out)
    return path_directory_out


def pack_directory_to_backend(
    path_directory_in: Union[str, Path],
    backend_target: str,
    path_target: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Pack a directory-style richfile tree into a target backend layout.

    Args:
        path_directory_in (Union[str, Path]):
            Input directory-style richfile root directory.
        backend_target (str):
            Target backend name.
        path_target (Union[str, Path]):
            Target backend root path.
        overwrite (bool):
            Whether to overwrite target path if it already exists.

    Returns:
        (Path):
            path_target (Path):
                Created backend root path.
    """
    _validate_backend_name(backend=backend_target)
    path_directory_in = Path(path_directory_in).resolve()
    path_target = Path(path_target).resolve()
    _ensure_directory_style_root(path_directory=path_directory_in)

    if path_directory_in == path_target:
        raise ValueError(
            f"Source and target paths must differ. Both resolve to: {path_directory_in}"
        )

    _prepare_target_path(path_target=path_target, overwrite=overwrite)
    path_target.parent.mkdir(parents=True, exist_ok=True)

    if backend_target == "directory":
        _copy_directory_tree(
            path_source=path_directory_in,
            path_target=path_target,
        )
        return path_target

    richfile_target = util.RichFile(
        path=path_target,
        backend=backend_target,
        check=True,
    )
    backend_impl = richfile_target._get_backend_impl()
    if not isinstance(backend_impl, ArchiveBackendBase):
        raise TypeError(
            "Expected archive backend implementation for non-directory target. "
            f"Found: {type(backend_impl)}"
        )

    with backend_impl._open_writer(path_archive=path_target) as writer:
        for path_item in sorted(path_directory_in.rglob("*")):
            path_relative = path_item.relative_to(path_directory_in).as_posix()
            if path_item.is_dir():
                backend_impl._write_dir_member(
                    writer=writer,
                    member_name=path_relative,
                )
            else:
                backend_impl._write_file_member(
                    writer=writer,
                    member_name=path_relative,
                    data=path_item.read_bytes(),
                )

    return path_target


def convert_backend(
    path_source: Union[str, Path],
    backend_source: str,
    path_target: Union[str, Path],
    backend_target: str,
    overwrite: bool = False,
    mode: str = "raw",
    check: bool = True,
    type_lookup: Optional[Any] = None,
) -> Path:
    """
    Convert a richfile payload between backend layouts.

    Conversion modes:
        * ``raw``:
            Byte-preserving conversion by materializing directory-style files and
            re-packing them. This does not deserialize custom Python objects.
        * ``semantic``:
            Object-level conversion via ``load()`` followed by ``save()``.
            This requires all relevant custom type registrations.

    Args:
        path_source (Union[str, Path]):
            Source backend root path.
        backend_source (str):
            Source backend name.
        path_target (Union[str, Path]):
            Target backend root path.
        backend_target (str):
            Target backend name.
        overwrite (bool):
            Whether to overwrite target path if it exists.
        mode (str):
            Conversion mode. Supported values: ``"raw"``, ``"semantic"``.
        check (bool):
            Validation flag used in semantic mode.
        type_lookup (Optional[Any]):
            Optional type lookup object used in semantic mode for custom types.

    Returns:
        (Path):
            path_target (Path):
                Created target backend path.
    """
    _validate_backend_name(backend=backend_source)
    _validate_backend_name(backend=backend_target)
    path_source = Path(path_source)
    path_target = Path(path_target)

    if mode == "raw":
        if backend_source == "directory":
            return pack_directory_to_backend(
                path_directory_in=path_source,
                backend_target=backend_target,
                path_target=path_target,
                overwrite=overwrite,
            )
        if backend_target == "directory":
            return extract_backend_to_directory(
                path_source=path_source,
                backend_source=backend_source,
                path_directory_out=path_target,
                overwrite=overwrite,
            )
        with tempfile.TemporaryDirectory() as dir_tmp:
            path_tmp_directory = Path(dir_tmp) / "richfile_directory"
            extract_backend_to_directory(
                path_source=path_source,
                backend_source=backend_source,
                path_directory_out=path_tmp_directory,
                overwrite=False,
            )
            return pack_directory_to_backend(
                path_directory_in=path_tmp_directory,
                backend_target=backend_target,
                path_target=path_target,
                overwrite=overwrite,
            )

    if mode == "semantic":
        richfile_source = util.RichFile(
            path=path_source,
            backend=backend_source,
            check=check,
        )
        richfile_target = util.RichFile(
            path=path_target,
            backend=backend_target,
            check=check,
            overwrite=overwrite,
        )
        if type_lookup is not None:
            richfile_source.type_lookup = copy.deepcopy(type_lookup)
            richfile_target.type_lookup = copy.deepcopy(type_lookup)

        object_loaded = richfile_source.load()
        richfile_target.save(obj=object_loaded)
        return path_target

    raise ValueError("Unsupported conversion mode. Expected one of: ['raw', 'semantic'].")

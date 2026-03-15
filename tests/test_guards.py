"""
Tests for safety guards added during multi-agent code review:
- Same-path guards in conversion functions
- Descendant-path guard in _copy_directory_tree
- SQLAR read-only reader (no file creation on missing path)
- versions_supported warning (previously dead code)
"""
import sqlite3
import tempfile
import warnings
from pathlib import Path

import pytest

import richfile as rf
from richfile import conversion


DATA_BASIC = {
    "name": "John Doe",
    "age": 25,
    "address": {
        "street": "1234 Elm St",
        "zip": None,
    },
}


# ---------------------------------------------------------------------------
# Same-path guards in extract / pack
# ---------------------------------------------------------------------------


def test_extract_same_path_raises():
    """extract_backend_to_directory must reject source == target."""
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "data.zip"
        rf.RichFile(archive, backend="zip").save(DATA_BASIC)

        with pytest.raises(ValueError, match="Source and target paths must differ"):
            rf.extract_backend_to_directory(
                path_source=archive,
                backend_source="zip",
                path_directory_out=archive,
                overwrite=True,
            )


def test_pack_same_path_raises():
    """pack_directory_to_backend must reject source == target."""
    with tempfile.TemporaryDirectory() as d:
        directory = Path(d) / "data.richfile"
        rf.RichFile(directory, backend="directory").save(DATA_BASIC)

        with pytest.raises(ValueError, match="Source and target paths must differ"):
            rf.pack_directory_to_backend(
                path_directory_in=directory,
                backend_target="zip",
                path_target=directory,
                overwrite=True,
            )


def test_extract_same_path_via_symlink_raises():
    """Same-path guard should catch symlink aliases via resolve()."""
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "data.zip"
        rf.RichFile(archive, backend="zip").save(DATA_BASIC)

        link = Path(d) / "link_to_data.zip"
        link.symlink_to(archive)

        with pytest.raises(ValueError, match="Source and target paths must differ"):
            rf.extract_backend_to_directory(
                path_source=archive,
                backend_source="zip",
                path_directory_out=link,
                overwrite=True,
            )


# ---------------------------------------------------------------------------
# Descendant-path guard in _copy_directory_tree
# ---------------------------------------------------------------------------


def test_copy_directory_tree_rejects_target_inside_source():
    """_copy_directory_tree must reject target inside source."""
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "source"
        source.mkdir()
        (source / "a.txt").write_text("hello")

        target_inside = source / "output"

        with pytest.raises(ValueError, match="Target path must not be inside source"):
            conversion._copy_directory_tree(source, target_inside)


def test_extract_directory_to_subdirectory_raises():
    """Extracting a directory backend into its own subtree must fail."""
    with tempfile.TemporaryDirectory() as d:
        source_dir = Path(d) / "source.richfile"
        rf.RichFile(source_dir, backend="directory").save(DATA_BASIC)

        target_inside = source_dir / "nested_output"

        with pytest.raises(ValueError, match="Target path must not be inside source"):
            rf.extract_backend_to_directory(
                path_source=source_dir,
                backend_source="directory",
                path_directory_out=target_inside,
            )


# ---------------------------------------------------------------------------
# SQLAR read-only reader: no file creation on missing path
# ---------------------------------------------------------------------------


def test_sqlar_reader_does_not_create_file_on_missing_path():
    """Opening a non-existent SQLAR path for reading must raise, not create a file."""
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "does_not_exist.sqlar"
        assert not missing.exists()

        with pytest.raises(FileNotFoundError):
            rf.RichFile(missing, backend="sqlar").load()

        assert not missing.exists(), "Reader must not create a file as a side effect"


def test_sqlar_reader_opens_existing_file_readonly():
    """An existing SQLAR file should load fine via the read-only reader."""
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "data.sqlar"
        rf.RichFile(archive, backend="sqlar").save(DATA_BASIC)

        loaded = rf.RichFile(archive, backend="sqlar").load()
        assert loaded["name"] == "John Doe"
        assert loaded["age"] == 25


# ---------------------------------------------------------------------------
# versions_supported warning (previously dead code at util.py:294)
# ---------------------------------------------------------------------------


def test_empty_versions_supported_emits_warning():
    """Types with empty versions_supported should trigger a warning on load."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.richfile"

        r = rf.RichFile(path, backend="directory")
        r.type_lookup.add_property({
            "type_name": "custom_test_type",
            "function_load": lambda path, **kwargs: Path(path).read_bytes(),
            "function_save": lambda obj, path, **kwargs: Path(path).write_bytes(obj),
            "object_class": bytes,
            "suffix": "bin",
            "library": "python",
            "versions_supported": [],
        })
        r.save({"payload": b"test data"})

        r2 = rf.RichFile(path, backend="directory")
        r2.type_lookup.add_property({
            "type_name": "custom_test_type",
            "function_load": lambda path, **kwargs: Path(path).read_bytes(),
            "function_save": lambda obj, path, **kwargs: Path(path).write_bytes(obj),
            "object_class": bytes,
            "suffix": "bin",
            "library": "python",
            "versions_supported": [],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r2.load()
            version_warnings = [
                x for x in w
                if "versions_supported" in str(x.message)
                and "custom_test_type" in str(x.message)
            ]
            assert len(version_warnings) >= 1, (
                "Expected warning about empty versions_supported for custom_test_type"
            )

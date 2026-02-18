import sqlite3
import tempfile
import json
from pathlib import Path

import pytest

import richfile as rf


DATA_BASIC = {
    "name": "John Doe",
    "age": 25,
    "address": {
        "street": "1234 Elm St",
        "zip": None,
    },
    "siblings": [
        "Jane",
        "Jim",
    ],
    "data": [1, 2, 3],
    (1, 2, 3): "complex key",
}


def assert_equivalence(original, loaded):
    def compare(a, b):
        if isinstance(a, dict):
            assert isinstance(b, dict)
            assert set(a.keys()) == set(b.keys())
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for ai, bi in zip(a, b):
                compare(ai, bi)
        elif isinstance(a, (set, frozenset)):
            assert a == b
        else:
            assert a == b

    compare(original, loaded)


def test_sqlar_basic_roundtrip_and_single_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_data.sqlar"
        rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        assert path_archive.exists()
        assert path_archive.is_file()
        assert not path_archive.is_dir()

        loaded_data = rf.RichFile(path_archive, backend="sqlar").load()
        assert_equivalence(DATA_BASIC, loaded_data)


def test_sqlar_lazy_loading():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_data_lazy.sqlar"
        rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        r = rf.RichFile(path_archive, backend="sqlar")
        assert r["siblings"][0].load() == "Jane"
        assert r["address"]["street"].load() == "1234 Elm St"


def test_sqlar_metadata_and_typelookup_rows():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_data_metadata.sqlar"
        r = rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        metadata_root = r.get_metadata(path_dir=path_archive)
        assert metadata_root["type"] == "dict"
        assert "name.dict_item" in metadata_root["elements"]

        metadata_tree = r.get_metadata_tree()
        assert metadata_tree["metadata"]["type"] == "dict"

        with sqlite3.connect(path_archive) as conn:
            has_typelookup = conn.execute(
                "SELECT 1 FROM sqlar WHERE name = ? LIMIT 1",
                (rf.FILENAME_TYPELOOKUP,),
            ).fetchone()
            assert has_typelookup is not None

            n_compressed_rows = conn.execute(
                "SELECT COUNT(*) FROM sqlar WHERE data IS NOT NULL AND sz != length(data)"
            ).fetchone()[0]
            assert n_compressed_rows == 0


def test_sqlar_overwrite_behavior():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_data_overwrite.sqlar"

        rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        with pytest.raises(FileExistsError):
            rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        rf.RichFile(path_archive, backend="sqlar", overwrite=True).save({"value": 42})
        assert rf.RichFile(path_archive, backend="sqlar").load() == {"value": 42}


def test_sqlar_custom_type_roundtrip():
    class CustomPayload:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, CustomPayload) and self.value == other.value

    type_name = "custom_payload_sqlar"
    rf.functions.register_type(
        type_name=type_name,
        function_load=lambda path: CustomPayload(value=int(Path(path).read_text())),
        function_save=lambda path, obj: Path(path).write_text(str(obj.value)),
        object_class=CustomPayload,
        library="python",
        suffix="txt",
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            path_archive = Path(temp_dir) / "test_custom.sqlar"
            data = {"payload": CustomPayload(7)}
            rf.RichFile(path_archive, backend="sqlar").save(data)
            loaded = rf.RichFile(path_archive, backend="sqlar").load()
            assert loaded["payload"] == data["payload"]
    finally:
        rf.functions.remove_type(type_name)


def test_sqlar_explicit_backend_on_non_sqlar_suffix():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_non_sqlar_suffix = Path(temp_dir) / "test_data.richfile"
        rf.RichFile(path_non_sqlar_suffix, backend="sqlar").save(DATA_BASIC)
        assert path_non_sqlar_suffix.is_file()
        assert_equivalence(
            DATA_BASIC,
            rf.RichFile(path_non_sqlar_suffix, backend="sqlar").load(),
        )

        path_sqlar_suffix = Path(temp_dir) / "directory_mode.sqlar"
        rf.RichFile(path_sqlar_suffix, backend="directory").save(DATA_BASIC)
        assert path_sqlar_suffix.is_dir()
        assert_equivalence(
            DATA_BASIC,
            rf.RichFile(path_sqlar_suffix, backend="directory").load(),
        )


def test_sqlar_nested_save_not_supported_yet():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_nested_save.sqlar"
        r = rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)
        with pytest.raises(NotImplementedError):
            r["address"].save({"street": "new"}, overwrite=True)


def test_sqlar_keys_raises_on_corrupted_metadata_when_check_true():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "test_keys_corrupted.sqlar"
        rf.RichFile(path_archive, backend="sqlar").save(DATA_BASIC)

        payload_bad = b"not-json"
        with sqlite3.connect(path_archive) as conn:
            conn.execute(
                "UPDATE sqlar SET data = ?, sz = ? WHERE name = ?",
                (payload_bad, len(payload_bad), rf.FILENAME_METADATA),
            )
            conn.commit()

        with pytest.raises(json.JSONDecodeError):
            rf.RichFile(path_archive, backend="sqlar", check=True).keys()

        assert rf.RichFile(path_archive, backend="sqlar", check=False).keys() == []


def test_sqlar_rejects_parent_traversal_member_on_custom_materialization():
    class CustomRootPayload:
        def __init__(self, value):
            self.value = value

    type_name = "custom_root_payload_sqlar_security"
    rf.functions.register_type(
        type_name=type_name,
        function_load=lambda path: CustomRootPayload(value=Path(path).read_text()),
        function_save=lambda path, obj: Path(path).write_text(obj.value),
        object_class=CustomRootPayload,
        library="python",
        suffix="txt",
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            path_archive = Path(temp_dir) / "malicious.sqlar"
            metadata_root = {
                "elements": {},
                "type": type_name,
                "library": "python",
                "version": rf.util._get_python_version(),
                "version_richfile": rf.__version__,
            }
            with sqlite3.connect(path_archive) as conn:
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
                payload_metadata = json.dumps(metadata_root).encode("utf-8")
                conn.execute(
                    "INSERT INTO sqlar(name, mode, mtime, sz, data) VALUES (?, ?, 0, ?, ?)",
                    (rf.FILENAME_METADATA, 0o100644, len(payload_metadata), payload_metadata),
                )
                payload_evil = b"payload"
                conn.execute(
                    "INSERT INTO sqlar(name, mode, mtime, sz, data) VALUES (?, ?, 0, ?, ?)",
                    ("../evil.txt", 0o100644, len(payload_evil), payload_evil),
                )
                conn.commit()

            with pytest.raises(ValueError, match="Invalid archive path segment"):
                rf.RichFile(path_archive, backend="sqlar").load()
    finally:
        rf.functions.remove_type(type_name)

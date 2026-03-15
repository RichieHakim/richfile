import io
import json
import tarfile
import tempfile
import zipfile
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


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_basic_roundtrip_and_single_file(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"test_data{suffix}"
        rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        assert path_archive.exists()
        assert path_archive.is_file()
        assert not path_archive.is_dir()

        loaded_data = rf.RichFile(path_archive, backend=backend).load()
        assert_equivalence(DATA_BASIC, loaded_data)


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_lazy_loading(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"test_data_lazy{suffix}"
        rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        r = rf.RichFile(path_archive, backend=backend)
        assert r["siblings"][0].load() == "Jane"
        assert r["address"]["street"].load() == "1234 Elm St"


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_metadata_and_typelookup_members(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"test_data_metadata{suffix}"
        r = rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        metadata_root = r.get_metadata(path_dir=path_archive)
        assert metadata_root["type"] == "dict"
        assert "name.dict_item" in metadata_root["elements"]

        metadata_tree = r.get_metadata_tree()
        assert metadata_tree["metadata"]["type"] == "dict"

        listed_elements = r.list_elements(path_dir=path_archive)
        assert "name.dict_item" in listed_elements

        keys = r.keys()
        assert "name" in keys
        assert "address" in keys

        if backend == "zip":
            with zipfile.ZipFile(path_archive, mode="r") as zip_reader:
                names_members = zip_reader.namelist()
                assert rf.FILENAME_TYPELOOKUP in names_members
                for info in zip_reader.infolist():
                    assert info.compress_type == zipfile.ZIP_STORED
        else:
            with tarfile.open(path_archive, mode="r:") as tar_reader:
                names_members = tar_reader.getnames()
                assert rf.FILENAME_TYPELOOKUP in names_members

            bytes_prefix = path_archive.read_bytes()[:6]
            assert not bytes_prefix.startswith(b"\x1f\x8b")
            assert not bytes_prefix.startswith(b"BZh")
            assert not bytes_prefix.startswith(b"\xfd7zXZ")


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_overwrite_behavior(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"test_data_overwrite{suffix}"

        rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        with pytest.raises(FileExistsError):
            rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        rf.RichFile(path_archive, backend=backend, overwrite=True).save({"value": 42})
        assert rf.RichFile(path_archive, backend=backend).load() == {"value": 42}


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_custom_type_roundtrip(backend, suffix):
    class CustomPayload:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, CustomPayload) and self.value == other.value

    type_name = f"custom_payload_{backend}"
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
            path_archive = Path(temp_dir) / f"test_custom{suffix}"
            data = {"payload": CustomPayload(7)}
            rf.RichFile(path_archive, backend=backend).save(data)
            loaded = rf.RichFile(path_archive, backend=backend).load()
            assert loaded["payload"] == data["payload"]
    finally:
        rf.functions.remove_type(type_name)


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_explicit_backend_on_non_matching_suffix(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_non_matching_suffix = Path(temp_dir) / "test_data.richfile"
        rf.RichFile(path_non_matching_suffix, backend=backend).save(DATA_BASIC)
        assert path_non_matching_suffix.is_file()
        assert_equivalence(
            DATA_BASIC,
            rf.RichFile(path_non_matching_suffix, backend=backend).load(),
        )

        path_archive_suffix = Path(temp_dir) / f"directory_mode{suffix}"
        rf.RichFile(path_archive_suffix, backend="directory").save(DATA_BASIC)
        assert path_archive_suffix.is_dir()
        assert_equivalence(
            DATA_BASIC,
            rf.RichFile(path_archive_suffix, backend="directory").load(),
        )


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_nested_save_not_supported_yet(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"test_nested_save{suffix}"
        r = rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)
        with pytest.raises(NotImplementedError):
            r["address"].save({"street": "new"}, overwrite=True)


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_keys_raises_on_corrupted_metadata_when_check_true(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"corrupted_meta{suffix}"
        payload_bad = b"not-json"

        if backend == "zip":
            with zipfile.ZipFile(path_archive, mode="w", compression=zipfile.ZIP_STORED) as writer:
                writer.writestr(rf.FILENAME_METADATA, payload_bad)
        else:
            with tarfile.open(path_archive, mode="w") as writer:
                info = tarfile.TarInfo(name=rf.FILENAME_METADATA)
                info.size = len(payload_bad)
                writer.addfile(tarinfo=info, fileobj=io.BytesIO(payload_bad))

        with pytest.raises(json.JSONDecodeError):
            rf.RichFile(path_archive, backend=backend, check=True).keys()

        assert rf.RichFile(path_archive, backend=backend, check=False).keys() == []


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_rejects_parent_traversal_member_paths(backend, suffix):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"malicious{suffix}"
        payload_metadata = json.dumps(
            {
                "elements": {},
                "type": "list",
                "library": "python",
                "version": rf.util._get_python_version(),
                "version_richfile": rf.__version__,
            }
        ).encode("utf-8")
        payload_evil = b"payload"

        if backend == "zip":
            with zipfile.ZipFile(path_archive, mode="w", compression=zipfile.ZIP_STORED) as writer:
                writer.writestr(rf.FILENAME_METADATA, payload_metadata)
                writer.writestr("../evil.txt", payload_evil)
        else:
            with tarfile.open(path_archive, mode="w") as writer:
                info_meta = tarfile.TarInfo(name=rf.FILENAME_METADATA)
                info_meta.size = len(payload_metadata)
                writer.addfile(tarinfo=info_meta, fileobj=io.BytesIO(payload_metadata))

                info_evil = tarfile.TarInfo(name="../evil.txt")
                info_evil.size = len(payload_evil)
                writer.addfile(tarinfo=info_evil, fileobj=io.BytesIO(payload_evil))

        with pytest.raises(ValueError, match="Invalid archive member path"):
            rf.RichFile(path_archive, backend=backend).load()


@pytest.mark.parametrize("backend,suffix", [("zip", ".zip"), ("tar", ".tar")])
def test_archive_index_cache_reused_across_lazy_chain(backend, suffix, monkeypatch):
    if backend == "zip":
        from richfile.backends.zip_backend import ZipBackend as BackendClass
    else:
        from richfile.backends.tar_backend import TarBackend as BackendClass

    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"cache_lazy{suffix}"
        rf.RichFile(path_archive, backend=backend).save(DATA_BASIC)

        counter_calls = {"n": 0}
        fn_build_index_orig = BackendClass._build_index

        def _build_index_counting(self, reader):
            counter_calls["n"] += 1
            return fn_build_index_orig(self=self, reader=reader)

        monkeypatch.setattr(BackendClass, "_build_index", _build_index_counting)

        r = rf.RichFile(path_archive, backend=backend)
        assert r["address"]["street"].load() == "1234 Elm St"
        assert counter_calls["n"] == 1

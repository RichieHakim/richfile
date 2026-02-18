import tempfile
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


def backend_path(path_root: Path, backend: str) -> Path:
    suffix_map = {
        "directory": ".richfile",
        "sqlar": ".sqlar",
        "zip": ".zip",
        "tar": ".tar",
    }
    return path_root / f"payload{suffix_map[backend]}"


@pytest.mark.parametrize("backend", ["directory", "sqlar", "zip", "tar"])
def test_auto_backend_detects_load_and_lazy(backend):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_target = backend_path(path_root=Path(temp_dir), backend=backend)
        rf.RichFile(path_target, backend=backend).save(DATA_BASIC)

        loaded = rf.RichFile(path_target).load()
        assert_equivalence(DATA_BASIC, loaded)

        lazy_value = rf.RichFile(path_target)["address"]["street"].load()
        assert lazy_value == "1234 Elm St"


def test_auto_backend_detects_directory_leaf_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_directory = Path(temp_dir) / "payload.richfile"
        rf.RichFile(path_directory, backend="directory").save(DATA_BASIC)

        path_leaf = path_directory / "name.dict_item" / "key.json"
        loaded_leaf = rf.RichFile(path_leaf).load()
        assert loaded_leaf == "name"


def test_auto_backend_save_defaults_to_directory_for_new_paths():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_target = Path(temp_dir) / "new_payload.sqlar"
        rf.RichFile(path_target).save(DATA_BASIC)

        assert path_target.is_dir()
        assert_equivalence(DATA_BASIC, rf.RichFile(path_target).load())


def test_auto_backend_raises_on_unknown_file_type():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_bad = Path(temp_dir) / "unknown.bin"
        path_bad.write_bytes(b"not-a-richfile")

        with pytest.raises(ValueError, match="Could not detect backend"):
            rf.RichFile(path_bad).load()

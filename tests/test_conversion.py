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


def backend_suffix(backend: str) -> str:
    suffix_map = {
        "directory": ".richfile",
        "sqlar": ".sqlar",
        "zip": ".zip",
        "tar": ".tar",
    }
    return suffix_map[backend]


@pytest.mark.parametrize("backend_source", ["sqlar", "zip", "tar"])
def test_extract_archive_to_directory_is_readable(backend_source):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / f"source{backend_suffix(backend_source)}"
        rf.RichFile(path_archive, backend=backend_source).save(DATA_BASIC)

        path_out = Path(temp_dir) / "extracted.richfile"
        rf.extract_backend_to_directory(
            path_source=path_archive,
            backend_source=backend_source,
            path_directory_out=path_out,
        )

        assert path_out.exists()
        assert path_out.is_dir()
        assert (path_out / rf.FILENAME_METADATA).exists()
        loaded = rf.RichFile(path_out, backend="directory").load()
        assert_equivalence(DATA_BASIC, loaded)


@pytest.mark.parametrize("backend_target", ["sqlar", "zip", "tar"])
def test_pack_directory_to_archive_is_readable(backend_target):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_directory_source = Path(temp_dir) / "source.richfile"
        rf.RichFile(path_directory_source, backend="directory").save(DATA_BASIC)

        path_archive = Path(temp_dir) / f"packed{backend_suffix(backend_target)}"
        rf.pack_directory_to_backend(
            path_directory_in=path_directory_source,
            backend_target=backend_target,
            path_target=path_archive,
        )

        assert path_archive.exists()
        assert path_archive.is_file()
        loaded = rf.RichFile(path_archive, backend=backend_target).load()
        assert_equivalence(DATA_BASIC, loaded)


@pytest.mark.parametrize(
    "backend_source,backend_target",
    [
        ("directory", "zip"),
        ("directory", "sqlar"),
        ("zip", "directory"),
        ("tar", "directory"),
        ("sqlar", "directory"),
        ("zip", "tar"),
        ("tar", "sqlar"),
    ],
)
def test_convert_backend_raw_matrix_roundtrip(backend_source, backend_target):
    with tempfile.TemporaryDirectory() as temp_dir:
        if backend_source == "directory":
            path_source = Path(temp_dir) / "source.richfile"
        else:
            path_source = Path(temp_dir) / f"source{backend_suffix(backend_source)}"
        rf.RichFile(path_source, backend=backend_source).save(DATA_BASIC)

        if backend_target == "directory":
            path_target = Path(temp_dir) / "target.richfile"
        else:
            path_target = Path(temp_dir) / f"target{backend_suffix(backend_target)}"

        rf.convert_backend(
            path_source=path_source,
            backend_source=backend_source,
            path_target=path_target,
            backend_target=backend_target,
            mode="raw",
        )

        loaded = rf.RichFile(path_target, backend=backend_target).load()
        assert_equivalence(DATA_BASIC, loaded)


def test_conversion_overwrite_guard():
    with tempfile.TemporaryDirectory() as temp_dir:
        path_archive = Path(temp_dir) / "source.zip"
        rf.RichFile(path_archive, backend="zip").save(DATA_BASIC)

        path_out_dir = Path(temp_dir) / "extracted.richfile"
        path_out_dir.mkdir(parents=True, exist_ok=True)
        (path_out_dir / "keep.txt").write_text("keep")
        with pytest.raises(FileExistsError):
            rf.extract_backend_to_directory(
                path_source=path_archive,
                backend_source="zip",
                path_directory_out=path_out_dir,
                overwrite=False,
            )

        rf.extract_backend_to_directory(
            path_source=path_archive,
            backend_source="zip",
            path_directory_out=path_out_dir,
            overwrite=True,
        )
        assert (path_out_dir / rf.FILENAME_METADATA).exists()

        path_archive_target = Path(temp_dir) / "target.tar"
        path_archive_target.write_text("occupied")
        with pytest.raises(FileExistsError):
            rf.pack_directory_to_backend(
                path_directory_in=path_out_dir,
                backend_target="tar",
                path_target=path_archive_target,
                overwrite=False,
            )

        rf.pack_directory_to_backend(
            path_directory_in=path_out_dir,
            backend_target="tar",
            path_target=path_archive_target,
            overwrite=True,
        )
        assert path_archive_target.is_file()

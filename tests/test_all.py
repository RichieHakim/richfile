# Import necessary modules
import tempfile
from pathlib import Path
import copy
import os
import math
import warnings

import pytest
import hypothesis
import richfile as rf


# Module-level variable for test data
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

def save_and_load(file_path, obj):
    # Save data
    r = rf.RichFile(file_path).save(obj)
    # Load data back
    return rf.RichFile(file_path).load()


def cleanup(path):
    # Works with both files and directories
    if Path(path).is_file():
        Path(path).unlink(missing_ok=True)
    elif Path(path).is_dir():
        Path(path).rmdir(missing_ok=True)


def assert_equivalence(original, loaded):
    # Make closure for recursive comparison
    def compare(original, loaded):
        if isinstance(original, dict):
            assert isinstance(loaded, dict)
            assert set(original.keys()) == set(loaded.keys())
            for key in original:
                compare(original[key], loaded[key])
        elif isinstance(original, (list, tuple)):
            assert len(original) == len(loaded)
            for a, b in zip(original, loaded):
                compare(a, b)
        elif isinstance(original, (set, frozenset)):
            assert original == loaded
        elif isinstance(original, float) and math.isnan(original):
            assert isinstance(loaded, float) and math.isnan(loaded)
        else:
            assert original == loaded

    compare(original, loaded)


# Basic tests for RichFile functionality
def test_basic_save_load():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_data.richfile"
        loaded_data = save_and_load(file_path, DATA_BASIC)
        assert_equivalence(DATA_BASIC, loaded_data)


def test_lazy_loading():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_data_lazy.richfile"
        # Save data
        r = rf.RichFile(file_path).save(DATA_BASIC)
        # Lazy load a part of the data
        r = rf.RichFile(file_path)
        first_sibling = r["siblings"][0].load()
        assert first_sibling == "Jane"


# Property-based tests with hypothesis
# dict
@hypothesis.settings(deadline=600)
@hypothesis.given(
    hypothesis.strategies.dictionaries(
        keys=hypothesis.strategies.text().filter(lambda x: len([c for c, v in rf.invalid_chars_filename(x).items() if not v]) == 0),
        values=hypothesis.strategies.one_of(
            hypothesis.strategies.text(),
            hypothesis.strategies.integers(),
            hypothesis.strategies.floats(allow_nan=False),
            hypothesis.strategies.booleans(),
            hypothesis.strategies.none(),
        ),
    ),
)
def test_save_load_dict(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_data_dict.richfile"
        loaded_data = save_and_load(file_path, data)
        assert_equivalence(data, loaded_data)


# list
@hypothesis.settings(deadline=600)
@hypothesis.given(hypothesis.strategies.lists(hypothesis.strategies.integers()))
def test_save_load_list(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_list_data.richfile"
        loaded_data = save_and_load(file_path, data)
        assert_equivalence(data, loaded_data)


# tuple
@hypothesis.settings(deadline=600)
@hypothesis.given(
    hypothesis.strategies.tuples(
        hypothesis.strategies.booleans(),
        hypothesis.strategies.floats(allow_nan=False),
        hypothesis.strategies.text(),
    )
)
def test_save_load_tuple(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_tuple_data.richfile"
        loaded_data = save_and_load(file_path, data)
        assert_equivalence(data, loaded_data)


# set
@hypothesis.settings(deadline=600)
@hypothesis.given(hypothesis.strategies.sets(hypothesis.strategies.integers()))
def test_save_load_set(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_set_data.richfile"
        loaded_data = save_and_load(file_path, data)
        assert_equivalence(data, loaded_data)


# frozenset
@hypothesis.settings(deadline=600)
@hypothesis.given(hypothesis.strategies.frozensets(hypothesis.strategies.integers()))
def test_save_load_frozenset(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_frozenset_data.richfile"
        loaded_data = save_and_load(file_path, data)
        assert_equivalence(data, loaded_data)


# Test for registering new class types in various ways
def test_register_new_class_global():
    class CustomClass:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            if isinstance(other, CustomClass):
                return self.value == other.value
            return False

    # Register the new class type
    rf.functions.register_type(
        type_name="custom_class",
        function_load=lambda path: CustomClass(value=int(Path(path).read_text())),
        function_save=lambda path, obj: Path(path).write_text(str(obj.value)),
        object_class=CustomClass,
        library="python",
        suffix="txt",
    )

    # Check if type in typelookup
    type_lookup = rf.functions.TypeLookup()
    assert "custom_class" in type_lookup

    # Save and load data
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_custom_class.richfile"
        custom_data = {"data": CustomClass(42)}
        loaded_data = save_and_load(file_path, custom_data)
        assert loaded_data['data'] == custom_data['data']

    # Remove the type from typelookup
    rf.functions.remove_type("custom_class")
    type_lookup = rf.functions.TypeLookup()
    assert "custom_class" not in type_lookup


def test_register_sparse_coo_global():
    import sparse
    import numpy as np

    # Register sparse.COO type
    rf.functions.register_type(
        type_name="sparse_coo",
        function_load=lambda path: sparse.load_npz(path),
        function_save=lambda path, obj: sparse.save_npz(path, obj),
        object_class=sparse.COO,
        library="sparse",
        suffix="npz",
    )

    # Check if type in typelookup
    type_lookup = rf.functions.TypeLookup()
    assert "sparse_coo" in type_lookup

    # Save and load data
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_sparse_coo.richfile"
        custom_data = {"data": sparse.COO.from_numpy(np.random.rand(10, 10))}
        loaded_data = save_and_load(file_path, custom_data)
        assert np.allclose(loaded_data['data'].todense(), custom_data['data'].todense())

    # Remove the type from typelookup
    rf.functions.remove_type("sparse_coo")
    type_lookup = rf.functions.TypeLookup()
    assert "sparse_coo" not in type_lookup


def test_register_nparray_local():
    import numpy as np

    # Prepare temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        path_rf = Path(temp_dir) / "test_nparray_local.richfile"
        data = {"data": np.random.rand(10, 10)}
        # Make a new RichFile object
        r = rf.RichFile(path_rf)
        # Register the type locally
        ## Make a new TypeLookup dict
        tl = {
            "type_name": "nparray",
            "function_load": lambda path: np.load(path),
            "function_save": lambda path, obj: np.save(path, obj),
            "object_class": np.ndarray,
            "library": "numpy",
            "suffix": "npy",
            "versions_supported": [],
        }
        ## First make sure it is not in the global typelookup
        assert "nparray" not in rf.functions.TypeLookup()
        ## Register the type locally
        r.register_type(**tl)
        # Check if type in typelookup
        assert "nparray" in r.type_lookup
        # Remove the type from typelookup
        r.remove_type("nparray")
        # Check if type in typelookup
        assert "nparray" not in r.type_lookup
        # Add it back using a dictionary
        r.register_type_from_dict(tl)
        # Check if type in typelookup
        assert "nparray" in r.type_lookup

        # Save and load data
        r.save(data)
        loaded_data = r.load()

        # Check if the data is the same
        assert np.allclose(data['data'], loaded_data['data'])

        # Make sure global typelookup is not affected
        assert "nparray" not in rf.functions.TypeLookup()


# Error handling tests
def test_register_new_class_missing_argument():
    class CustomClass:
        pass

    with pytest.raises(TypeError):
        rf.functions.register_type(
            type_name="custom_class",
            function_load=lambda path: CustomClass(),  # Missing function_save
            object_class=CustomClass,
            suffix="txt",
            library="python",
            versions_supported=[],
        )
def test_register_new_class_invalid_type():
    class CustomClass:
        pass

    with pytest.raises(TypeError):
        rf.functions.register_type(
            type_name="custom_class",
            function_load="not a function",  # Invalid type for function_load
            function_save=lambda path, obj: None,
            object_class=CustomClass,
            suffix="txt",
            library="python",
            versions_supported=[],
        )
def test_register_new_class_type_collision():
    class CustomClass:
        pass

    rf.functions.register_type(
        type_name="custom_class",
        function_load=lambda path: CustomClass(),
        function_save=lambda path, obj: None,
        object_class=CustomClass,
        suffix="txt",
        library="python",
        versions_supported=[],
    )

    with pytest.raises(KeyError):
        rf.functions.register_type(
            type_name="custom_class",
            function_load=lambda path: int,  # Same name, different type
            function_save=lambda path, obj: None,
            object_class=int,
            suffix="txt",
            library="python",
            versions_supported=[],
        )

    rf.functions.remove_type("custom_class")
def test_remove_nonexistent_type():
    with pytest.raises(KeyError):
        rf.functions.remove_type("nonexistent_type")


def test_save_unsupported_type():
    with pytest.raises(TypeError):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_unsupported_type.richfile"
            rf.RichFile(file_path).save(math)

def test_load_corrupted_file():
    import json
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "corrupted_file.richfile"
        ## Save good json data
        r = rf.RichFile(path).save({"data": 42})
        ## Corrupt the file
        with open(r['data'].path, "w") as f:
            ## Overwrite the file with bad data
            f.write("bad data")
        ## Load the file
        with pytest.raises(json.JSONDecodeError):
            r.load()

# Edge case tests
def test_save_empty_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_empty_dict.richfile"
        rf.RichFile(file_path).save({})
        loaded_data = rf.RichFile(file_path).load()
        assert loaded_data == {}

def test_save_nested_data_structures():
    nested_data = {
        "deep_lists": [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12], [[[[[[13,]]]]]]]]],
        "deep_dicts": {"a": {"b": {"c": {"d": {"e": {"f": 42}}}}},},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_nested_data.richfile"
        rf.RichFile(file_path).save(nested_data)
        loaded_data = rf.RichFile(file_path).load()
        assert loaded_data == nested_data

def test_register_type_overwrite():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_local_type_overwrite.richfile"
        td = {
            "type_name": "custom_type",
            "object_class": type(None),
            "function_load": lambda path: None,
            "function_save": lambda path, obj: None,
            "suffix": "txt",
            "library": "python",
            "versions_supported": [],
        }
        r = rf.RichFile(file_path)
        r.register_type(**td)
        with pytest.raises(KeyError):
            r.register_type(**td)
        ## Now go the other way
        rf.functions.register_type(**td)
        with pytest.raises(KeyError):
            r = rf.RichFile(file_path)
            r.register_type(**td)
        ## cleanup
        rf.functions.remove_type("custom_type")
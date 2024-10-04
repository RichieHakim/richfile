## Testing suite for RichFile
import tempfile
from pathlib import Path
import copy
import inspect
import itertools

import pytest

import richfile as rf

DATA_SIMPLE = {
        "name": "John Doe",
        "age": 25,
        "address": {
            "street": "1234 Elm St",
            "zip": 62701
        },
        "siblings": [
            "Jane",
            "Jim"
        ],
        "data": [1,2,3],
        (1,2,3): "complex key",
    }


def test_save_load_simple():
    """
    Basic save and load test
    """
    d = copy.deepcopy(DATA_SIMPLE)
    ## Make a directory to save the file
    dir_save = tempfile.TemporaryDirectory().name
    path_save = str(Path(dir_save) / "test_simple.richfile")
    rf.RichFile(path_save).save(d, overwrite=True)

    ## Load the file
    r = rf.RichFile(path_save).load()

    ## Check if the loaded data is the same as the original data
    assert r == d


# def test_save_load_kwargs():
#     """
#     Try saving and loading with kwargs
#     """
#     ## Make simple closure for saving, loading, and testing
#     def save_load_test(d, **kwargs):
#         dir_save = tempfile.TemporaryDirectory().name
#         path_save = str(Path(dir_save) / "test_kwargs.richfile")
#         rf.RichFile(path_save).save(d, overwrite=True, **kwargs)
#         r = rf.RichFile(path_save).load(**kwargs)
#         assert r == d

#     ## Get available kwargs
#     sig_save = inspect.signature(rf.RichFile.save)
#     sig_load = inspect.signature(rf.RichFile.load)
#     kwargs_save = list(sig_save.parameters.keys())
#     kwargs_load = list(sig_load.parameters.keys())
#     ### Make all possible combinations of kwargs
#     kwargs_combos = itertools.product(kwargs_save, kwargs_load)

#     ## Test with different kwargs
    
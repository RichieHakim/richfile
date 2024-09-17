from typing import Union, List, Dict, Any, Callable
from pathlib import Path
import inspect

from . import util


TYPE_LOOKUP = {
    "list": {
        "function_load": util.load_folder,
        "function_save": util.save_container,
        "object_type":   list,
        "suffix":        "list",
        "library":       "python",
    },
    "tuple": {
        "function_load": util.load_folder,
        "function_save": util.save_container,
        "object_type":   tuple,
        "suffix":        "tuple",
        "library":       "python",
    },
    "set": {
        "function_load": util.load_folder,
        "function_save": util.save_container,
        "object_type":   set,
        "suffix":        "set",
        "library":       "python",
    },
    "dict": {
        "function_load": util.load_folder,
        "function_save": util.save_container,
        "object_type":   dict,
        "suffix":        "dict",
        "library":       "python",
    },

    "dict_item": {
        "function_load": util.load_folder,
        "function_save": util.save_container,
        "object_type":   util.DictItem,
        "suffix":        "dict_item",
        "library":       "python",
    },

    "float": {
        "function_load": util.load_float,
        "function_save": util.save_json,
        "object_type":   float,
        "suffix":       "json",
        "library":       "python",
    },
    "int": {
        "function_load": util.load_int,
        "function_save": util.save_json,
        "object_type":   int,
        "suffix":        "json",
        "library":       "python",
    },
    "str": {
        "function_load": util.load_str,
        "function_save": util.save_json,
        "object_type":   str,
        "suffix":        "json",
        "library":       "python",
    },
    "bool": {
        "function_load": util.load_bool,
        "function_save": util.save_json,
        "object_type":   bool,
        "suffix":        "json",
        "library":       "python",
    },
    "None": {
        "function_load": util.load_None,
        "function_save": util.save_json,
        "object_type":   type(None),
        "suffix":        "json",
        "library":       "python",
    },
}

VERSIONS_SUPPORTED = {
    "richfile": [">=0", "<1",],
    "python": [">=3", "<4",],
    "numpy": [">=1", "<3",],
    "scipy": [">=1", "<2",],
    "pandas": [">=1", "<3",],
    "PIL": [">=6", "<12",],
}

#################
## numpy_array ##
#################

import numpy as np

def save_npy_array(
    obj: np.ndarray,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Saves a NumPy array to the given path.
    """
    np.save(path, obj)

def load_npy_array(
    path: Union[str, Path],
    **kwargs,
) -> np.ndarray:
    """
    Loads an array from the given path.
    """    
    return np.load(path, **kwargs)

TYPE_LOOKUP.update({
    "numpy_array": {
        "function_load": load_npy_array,
        "function_save": save_npy_array,
        "object_type":   np.ndarray,
        "suffix":        "npy",
        "library":       "numpy",
    },
})

VERSIONS_SUPPORTED.update({
    "numpy": [">=1", "<3",],
})

########################
## scipy_sparse_array ##
########################

import scipy.sparse

def save_sparse_array(
    obj: scipy.sparse.spmatrix,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Saves a SciPy sparse matrix to the given path.
    """
    scipy.sparse.save_npz(path, obj)

def load_sparse_array(
    path: Union[str, Path],
    **kwargs,
) -> scipy.sparse.csr_matrix:
    """
    Loads a sparse array from the given path.
    """        
    return scipy.sparse.load_npz(path, **kwargs)

TYPE_LOOKUP.update({
    "scipy_sparse_array":{
        "function_load": load_sparse_array,
        "function_save": save_sparse_array,
        "object_type":   scipy.sparse.spmatrix,
        "suffix":        "npz",
        "library":       "scipy",
    },
})

VERSIONS_SUPPORTED.update({
    "scipy": [">=1", "<2",],
})

################
## pandas_csv ##
################

import pandas as pd

def save_pandas_csv(
    obj: pd.DataFrame,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Saves a pandas DataFrame to the given path as a CSV.
    """
    obj.to_csv(path, **kwargs)


def load_pandas_csv(
    path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from the given path.
    """
    return pd.read_csv(path, **kwargs)


TYPE_LOOKUP.update({
    "pandas_dataframe": {
        "function_load": load_pandas_csv,
        "function_save": save_pandas_csv,
        "object_type":   pd.DataFrame,
        "suffix":        "csv",
        "library":       "pandas",
    },
})

VERSIONS_SUPPORTED.update({
    "pandas": [">=1", "<3",],
})

###################
## PIL_image_png ##
###################

from PIL import Image

def save_PIL_image_png(
    obj: Image.Image,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Saves a PIL image to the given path as a PNG.
    """
    with open(path, "wb") as f:
        obj.save(f, format="PNG", **kwargs)

def load_PIL_image_png(
    path: Union[str, Path],
    **kwargs,
) -> Image.Image:
    """
    Loads a PIL image from the given path.
    """
    return Image.open(path, **kwargs)

TYPE_LOOKUP.update({
    "PIL_image_png": {
        "function_load": load_PIL_image_png,
        "function_save": save_PIL_image_png,
        "object_type":   Image.Image,
        "suffix":        "png",
        "library":       "PIL",
    },
})

VERSIONS_SUPPORTED.update({
    "PIL": [">=6", "<11",],
})


#########################
## sklearn_model_skops ##
#########################

# import sklearn
# import skops.io

# def save_sklearn_model_skops(
#     obj: sklearn.base.BaseEstimator,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a scikit-learn model to the given path using skops.
#     """
#     with open(path, "wb") as f:
#         skops.io.dump(obj, f, **kwargs)

# def load_sklearn_model_skops(
#     path: Union[str, Path],
#     **kwargs,
# ) -> sklearn.base.BaseEstimator:
#     """
#     Loads a scikit-learn model from the given path using skops.
#     """
#     with open(path, "rb") as f:
#         return skops.io.load(f, **kwargs)
    
# TYPE_LOOKUP.update({
#     "sklearn_model_skops": {
#         "function_load": load_sklearn_model_skops,
#         "function_save": save_sklearn_model_skops,
#         "object_type":   sklearn.base.BaseEstimator,
#         "suffix":        "skops",
#         "library":       "skops",
#     },
# })

# VERSIONS_SUPPORTED.update({
#     "skops": [">=0", "<1",],
# })

####################
## numpy_array_h5 ##
####################

# import tables

# def save_npy_array_h5(
#     obj: np.ndarray,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a NumPy array to the given path using HDF5.
#     """
#     with tables.open_file(path, "w") as f:
#         f.create_array("/", "data", obj)

# def load_npy_array_h5(
#     path: Union[str, Path],
#     **kwargs,
# ) -> np.ndarray:
#     """
#     Loads an array from the given path using HDF5.
#     """
#     with tables.open_file(path, "r") as f:
#         return f.root.data.read()

# TYPE_LOOKUP.update({
#     "numpy_array_h5": {
#         "function_load": load_npy_array_h5,
#         "function_save": save_npy_array_h5,
#         "object_type":   np.ndarray,
#         "suffix":        "h5",
#         "library":       "tables",
#     },
# })

# VERSIONS_SUPPORTED.update({
#     "tables": [">=3", "<4",],
# })

################
## pickle_obj ##
################

# import pickle

# def save_pickle_obj(
#     obj: Any,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves an object to the given path using pickle.
#     """
#     with open(path, "wb") as f:
#         pickle.dump(obj, f, **kwargs)

# def load_pickle_obj(
#     path: Union[str, Path],
#     **kwargs,
# ) -> Any:
#     """
#     Loads an object from the given path using pickle.
#     """
#     with open(path, "rb") as f:
#         return pickle.load(f, **kwargs)
    
# TYPE_LOOKUP.update({
#     "pickle_obj": {
#         "function_load": load_pickle_obj,
#         "function_save": save_pickle_obj,
#         "object_type":   object,
#         "suffix":        "pkl",
#         "library":       "pickle",
#     },
# })

###############################################################################################

def _check_function_args(func: Callable, args: List[str]):
    """
    Checks functions called by load_element to make sure they accept the correct
    arguments.

    Args:
        func (Callable): 
            The function to check.
        args (List[str]): 
            The arguments that the function should accept
    """
    sig = inspect.signature(func)
    if not all([param in sig.parameters for param in args]):
        raise ValueError(f"Function {func.__name__} does not accept the correct arguments.")

[_check_function_args(func["function_load"], ["path",]) for func in TYPE_LOOKUP.values()]

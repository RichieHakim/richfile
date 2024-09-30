from typing import Union, List, Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import inspect
import warnings
import importlib

from . import util


_TYPE_LOOKUP = [
    {
        "type_name":          "list",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       list,
        "suffix":             "list",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "tuple",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       tuple,
        "suffix":             "tuple",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "set",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       set,
        "suffix":             "set",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "frozenset",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       set,
        "suffix":             "frozenset",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },    
    {
        "type_name":          "dict",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       dict,
        "suffix":             "dict",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },

    {
        "type_name":          "dict_item",
        "function_load":      util.load_folder,
        "function_save":      util.save_container,
        "object_class":       util.DictItem,
        "suffix":             "dict_item",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },

    {
        "type_name":          "float",
        "function_load":      util.load_float,
        "function_save":      util.save_json,
        "object_class":       float,
        "suffix":             "json",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "int",
        "function_load":      util.load_int,
        "function_save":      util.save_json,
        "object_class":       int,
        "suffix":             "json",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "str",
        "function_load":      util.load_str,
        "function_save":      util.save_json,
        "object_class":       str,
        "suffix":             "json",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "bool",
        "function_load":      util.load_bool,
        "function_save":      util.save_json,
        "object_class":       bool,
        "suffix":             "json",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
    {
        "type_name":          "None",
        "function_load":      util.load_None,
        "function_save":      util.save_json,
        "object_class":         type(None),
        "suffix":             "json",
        "library":            "python",
        "versions_supported": [">=3", "<4"],
    },
]

_TYPE_KEYS_REQUIRED = [
    "type_name",
    "function_load",
    "function_save",
    "object_class",
    "suffix",
    "library",
    "versions_supported",
]

class TypeLookup:
    """
    A class for looking up the properties of the different types of objects that
    can be saved and loaded.

    This class behaves sort of like a dictionary, but instead of inputting a
    single key and getting a single output, you input a tuple corresponding to a
    ``(key, value)`` pair and return a list of property dictionaries that
    contain that matching key-value pair.

    The class also has methods for adding and removing properties.

    Args:
        type_lookup (List[Dict], optional): 
            A list of dictionaries containing the properties of the objects.
            Defaults to functions._TYPE_LOOKUP.
    """
    def __init__(self, type_lookup: List[Dict] = _TYPE_LOOKUP):
        self.properties = type_lookup
        _verify_validity_of_type_lookup(self.properties)

    def add_property(self, prop: Dict) -> None:
        """
        Adds a property to the list of properties.
        """
        self.properties.append(prop)
        _verify_validity_of_type_lookup(self.properties)

    def remove_property(self, type_name: str) -> None:
        """
        Removes a property from the list of properties.
        """
        self.properties = [prop for prop in self.properties if prop["type_name"] != type_name]

    def __len__(self) -> int:
        return len(self.properties)
    
    def __iter__(self):
        return iter(self.properties)
    
    def __repr__(self) -> str:
        return f"TypeLookup({[str(prop['type_name']) for prop in self.properties]})"
    
    def __add__(self, prop: Dict) -> None:
        """
        Adds a property to the list of properties.

        Args:
            prop (Dict): 
                A dictionary containing the properties of the object.
        """
        if not isinstance(prop, dict):
            raise ValueError("prop must be a dictionary.")
        self.add_property(prop)
        _verify_validity_of_type_lookup(self.properties)

    def __sub__(self, type_name: str) -> None:
        """
        Removes a property from the list of properties.

        Args:
            type_name (str): 
                The name of the type to remove.
        """
        if not isinstance(type_name, str):
            raise ValueError("type_name must be a string.")
        self.remove_property(type_name)
        _verify_validity_of_type_lookup(self.properties)

    def __contains__(self, query: Union[str, type]) -> bool:
        if isinstance(query, str):
            return query in [prop["type_name"] for prop in self.properties]
        elif isinstance(query, type):
            return query in [prop["object_class"] for prop in self.properties]

    def __getitem__(self, query: Union[str, type, Tuple[str]]) -> Union[Dict, List[Dict]]:
        """
        Returns the property dictionary that contain the matching key-value
        pair. If multiple properties match the key-value pair, a warning is
        raised and a list of property dictionaries is returned.

        Args:
            key_value (Union[str, type, Tuple[str]]): 
                The key-value pair to match. Can be one of the following input types: \n
                    * ``str``: Return prop dict where ``prop["type_name"] == key_value``
                    * ``type``: Return prop dict where ``prop["object_class"] ==
                      key_value``
                    * ``Tuple[str]``: Return prop dict where ``prop[key_value[0]] ==
                      key_value[1]``

        Returns:
            Union[Dict, List[Dict]]:
                The property dictionary or list of property dictionaries that
                match the key-value pair input.
        """
        if isinstance(query, tuple):
            if len(query) != 2:
                raise ValueError("key_value must be a tuple of length 2.")
            if not isinstance(query[0], str):
                raise ValueError("The first element of key_value must be a string.")
            return [prop for prop in self.properties if prop[query[0]] == query[1]]
        
        elif isinstance(query, str):
            for prop in self.properties:
                if prop["type_name"] == query:
                    return prop
            raise ValueError(f"String '{query}' not found in 'type_name' fields of available properties. Available properties: {[prop['type_name'] for prop in self.properties]}")
        
        elif isinstance(query, type):
            ## Go through the type_lookup dictionary to find the type of the object. isinstance checks for inherited classes as well.
            ### Check with 'is' first to avoid inheritance issues (ex. bool is a subclass of int)
            ### Then check if available "object_class" types are superclasses of the query type
            ### Then check using string comparison
            for props in self.properties:   
                if query is props["object_class"]:
                    return props
            for props in self.properties:
                if issubclass(query, props["object_class"]):
                    return props
            for props in self.properties:
                if str(query) == str(props["object_class"]):
                    return props
            ### Else, raise an error
            raise TypeError(f"Type '{query}' not supported. You can register a new type using `richfile.functions.register_type`.")

        else:
            raise ValueError("key_value must be a string, type, or tuple of length 2.")
        
    def __setitem__(self, key: str, props: Dict) -> None:
        """
        Update the property dictionary that has the field ``"type_name"`` that
        matches input arg ``key`` with items from the dictionary ``props``.

        Args:
            key (str): 
                The ``"type_name"`` field to match.
            props (Dict): 
                The new items to update the property dictionary with.
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string.")
        if not isinstance(props, dict):
            raise ValueError("props must be a dictionary.")
        ## Make sure keys in props are a subset of keys in _TYPE_KEYS_REQUIRED
        if not set(props.keys()).issubset(set(_TYPE_KEYS_REQUIRED)):
            keys_extra = set(props.keys()) - set(_TYPE_KEYS_REQUIRED)
            raise ValueError(f"Extra keys in props: {keys_extra}. Required keys: {_TYPE_KEYS_REQUIRED}")
        
        ## Update the property dictionary
        for prop in self.properties:
            if prop["type_name"] == key:
                prop.update(props)
                _verify_validity_of_type_lookup(self.properties)
                return
    
    def __delitem__(self, key: str) -> None:
        """
        Removes the property dictionary that has the field ``"type_name"`` that
        matches input arg ``key``.

        Args:
            key (str): 
                The ``"type_name"`` field to match.
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string.")
        self.remove_property(key)
        _verify_validity_of_type_lookup(self.properties)

    def types_metadata(self) -> List[str]:
        """
        Returns information about each type in the type lookup.
        Functions and classes are converted to strings.
        """
        import inspect
        types_metadata = []
        for prop in self.properties:
            prop_metadata = {}
            for key, value in prop.items():
                if isinstance(value, type):
                    prop_metadata[key] = str(value)
                elif callable(value):
                    prop_metadata[key] = inspect.getsource(value)
                elif isinstance(value, list):
                    if all([isinstance(elem, str) for elem in value]):
                        prop_metadata[key] = value
                    else:
                        raise ValueError(f"Unexpected item in self.properties: {value}")
                else:
                    if isinstance(value, str):
                        prop_metadata[key] = value
                    else:
                        raise ValueError(f"Unexpected item in self.properties: {value}")
            types_metadata.append(prop_metadata)

        return types_metadata


def _verify_validity_of_type_lookup(type_lookup: List[Dict]) -> None:
    """
    Verifies that the properties are valid.
    
    Rules:
    - Each property dict must have exactly the following keys and
    corresponding types:
        - type_name: str
        - function_load: Callable
        - function_save: Callable
        - object_class: type
        - suffix: str
        - library: str
        - versions_supported: List[str]
    - The values of the 'type_name' and 'type_class' fields must be unique
        across all properties.
    - The 'versions_supported' field may be empty, but if it is not, it must
        be formatted like: [">=0", "<3.1"] and work with util.is_version_compatible.
    """
    ## Check that each property dict has the correct keys. Use sets
    for prop in type_lookup:
        for prop in type_lookup:
            if set(prop.keys()) != set(_TYPE_KEYS_REQUIRED):
                keys_missing = set(_TYPE_KEYS_REQUIRED) - set(prop.keys())
                keys_extra = set(prop.keys()) - set(_TYPE_KEYS_REQUIRED)
                raise ValueError(f"Property dict doesn't have matching keys to required keys. Missing keys: {keys_missing}. Extra keys: {keys_extra}. Required keys: {_TYPE_KEYS_REQUIRED}")
    
    ## Check that the 'type_name' and 'type_class' fields are unique. Use sets
    type_names = [prop["type_name"] for prop in type_lookup]
    if len(type_names) != len(set(type_names)):
        type_names_duplicates = {name: type_names.count(name) for name in type_names if type_names.count(name) > 1}
        raise ValueError(f"Duplicate type names and their counts: {type_names_duplicates}. Call `richfile.functions.remove_type` to remove duplicates.")
    
    ## Check that the 'versions_supported' field is formatted correctly
    for prop in type_lookup:
        ## Check formatting first
        if not isinstance(prop["versions_supported"], list):
            raise ValueError(f"versions_supported must be a list of strings.")
        if not all([isinstance(version, str) for version in prop["versions_supported"]]):
            raise ValueError(f"versions_supported must be a list of strings.")

def _verify_validity_of_new_type(prop: Dict) -> Dict:
    """
    Verifies that the properties of a new type are valid.

    Args:
        prop (Dict): 
            The properties of the new type.
            Must contain the following keys and corresponding types:
                - type_name: str
                - function_load: Callable
                - function_save: Callable
                - object_class: type
                - suffix: str
                - library: str
                - versions_supported: List[str]

    Returns:
        Dict: 
            Same dict as input, but with any necessary corrections.
    """
    ## Check that prop is a dictionary
    if not isinstance(prop, dict):
        raise ValueError("prop must be a dictionary.")
    
    ## Set versions_supported to empty list if not provided
    if "versions_supported" not in prop.keys():
        prop["versions_supported"] = []

    ## Check that the property dict has the correct keys. Use sets
    if set(prop.keys()) != set(_TYPE_KEYS_REQUIRED):
        keys_missing = set(_TYPE_KEYS_REQUIRED) - set(prop.keys())
        keys_extra = set(prop.keys()) - set(_TYPE_KEYS_REQUIRED)
        raise ValueError(f"Property dict doesn't have matching keys to required keys. Missing keys: {keys_missing}. Extra keys: {keys_extra}. Required keys: {_TYPE_KEYS_REQUIRED}")
    
    type_name, \
    function_load, \
    function_save, \
    object_class, \
    suffix, \
    library, \
    versions_supported = [prop[key] for key in [
        "type_name",
        "function_load",
        "function_save",
        "object_class",
        "suffix",
        "library",
        "versions_supported",
    ]]   

    ## Clean the inputs
    ### type_name
    if not isinstance(type_name, str):
        raise TypeError("`type_name` must be a string.")
    #### empty
    if type_name == "":
        raise ValueError("`type_name` cannot be empty.")
    #### check filename safety
    util._check_filename_safety(name=type_name, warn=True, raise_error=False)
    #### check for duplicates
    if type_name in [prop["type_name"] for prop in _TYPE_LOOKUP]:
        raise KeyError(f"Type {type_name} already registered. Consider using `richfile.functions.remove_type` to remove it.")
    ### function_load
    if not callable(function_load):
        raise TypeError("`function_load` must be a callable.")
    util.functions._check_function_args(func=function_load, args=["path"])
    ### function_save
    if not callable(function_save):
        raise TypeError("`function_save` must be a callable.")
    util.functions._check_function_args(func=function_save, args=["obj", "path",])
    ### object_type
    if not isinstance(object_class, type):
        raise TypeError("`object_type` must be a type.")
    ### suffix
    if not isinstance(suffix, str):
        raise TypeError("`suffix` must be a string.")
    util._check_filename_safety(name=suffix, warn=True, raise_error=True)
    ### library
    if not isinstance(library, str):
        raise TypeError("`library` must be a string.")
    #### empty
    if library == "":
        raise ValueError("`library` cannot be empty.")
    #### make sure it's a valid library
    if library not in ["python", "builtins"]:
        try:
            importlib.import_module(library)
        except ImportError:
            warnings.warn(f"Library {library} not found or not importable in the current environment.")
    ### versions_supported
    if not isinstance(versions_supported, list):
        raise TypeError("`versions_supported` must be a list.")
    #### check that all elements are strings
    if not all(isinstance(version, str) for version in versions_supported):
        raise TypeError("All elements in `versions_supported` must be strings.")
    #### check that environment version is supported
    if len(versions_supported) > 0:
        version = util._get_library_version(library=library)
        util.is_version_compatible(version=version, rules=versions_supported)

    ## remove leading '.' and check for invalid characters
    if suffix[0] == ".":
        suffix = suffix[1:]
    
    ## return the cleaned properties
    return {
        "type_name":          type_name,
        "function_load":      function_load,
        "function_save":      function_save,
        "object_class":       object_class,
        "suffix":             suffix,
        "library":            library,
        "versions_supported": versions_supported,
    }


def register_type(
    type_name: str,
    function_load: Callable,
    function_save: Callable,
    object_class: type,
    suffix: str,
    library: str,
    versions_supported: List[str] | None = []
):
    """
    Registers a new type of object that can be saved and loaded.

    Args:
        type_name (str): 
            The name of the type.
        function_load (Callable): 
            The function that loads the object from a file.
        function_save (Callable): 
            The function that saves the object to a file.
        object_class (type): 
            The class of the object.
        suffix (str): 
            The suffix of the file.
        library (str): 
            The library that the object is associated with.
        versions_supported (List[str], optional): 
            A list of version strings that are supported by the object. 
            If empty, all versions are supported. Defaults to [].
    """
    prop = {
        "type_name":          type_name,
        "function_load":      function_load,
        "function_save":      function_save,
        "object_class":       object_class,
        "suffix":             suffix,
        "library":            library,
        "versions_supported": versions_supported,
    }
    register_type_from_dict(prop)

def register_type_from_dict(prop: Dict) -> None:
    """
    Registers a new type of object that can be saved and loaded.

    Args:
        prop (Dict): 
            A dictionary containing the properties of the object.
            Must contain the following keys and corresponding types:
                - type_name: str
                - function_load: Callable
                - function_save: Callable
                - object_class: type
                - suffix: str
                - library: str
                - versions_supported: List[str] (optional)
    """
    ## Clean the inputs
    prop = _verify_validity_of_new_type(prop)
    _TYPE_LOOKUP.append(prop)
    _verify_validity_of_type_lookup(_TYPE_LOOKUP)

def remove_type(type_name: str) -> None:
    ## Make sure the type_name is valid
    if type_name not in [prop["type_name"] for prop in _TYPE_LOOKUP]:
        raise KeyError(f"Type {type_name} not found.")
    ## Find the index of the type_name
    idx_to_remove = [idx for idx, prop in enumerate(_TYPE_LOOKUP) if prop["type_name"] == type_name][0]
    ## Remove the type_name
    _TYPE_LOOKUP.pop(idx_to_remove)


class Type_container:
    def __init__(
        self,
        type_name: str,
        object_class: type,
        suffix: str,
        library: str,
        versions_supported: List[str] | None = [],
    ):
        self.type_name = type_name
        self.object_class = object_class
        self.suffix = suffix
        self.library = library
        self.versions_supported = versions_supported

    def register_type(self):
        register_type(
            type_name=self.type_name,
            function_load=self.function_load,
            function_save=self.function_save,
            object_class=self.object_class,
            suffix=self.suffix,
            library=self.library,
            versions_supported=self.versions_supported,
        )

    def function_save(
        self,
        obj: Any,
        path: Union[str, Path],
        type_lookup: Dict,
        check: bool = True,
        overwrite: bool = False,
        name_dict_items: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves an object to the given path as a container.
        """
        if not isinstance(obj, self.object_class):
            raise TypeError(f"Object must be of type {self.object_class}.")
        
        util.save_container(
            obj=obj.__dict__,
            path=path,
            type_name=self.type_name,
            type_lookup=type_lookup,
            check=check,
            overwrite=overwrite,
            name_dict_items=name_dict_items,
        )

    def function_load(
        self,
        path: Union[str, Path],
        type_lookup: Dict,
        check: bool = True,
        **kwargs,
    ) -> Any:
        """
        Loads an object from the given path.
        """
        out = self.object_class.__new__(self.object_class)
        out.__dict__ = util.load_folder(
            path=path,
            type_lookup=type_lookup,
            check=check,
            **kwargs,
        )
        return out
    
    def get_property_dict(self) -> Dict:
        return {
            "type_name":          self.type_name,
            "function_load":      self.function_load,
            "function_save":      self.function_save,
            "object_class":       self.object_class,
            "suffix":             self.suffix,
            "library":            self.library,
            "versions_supported": self.versions_supported,
        }
    

################################################################################################
######################################### CUSTOM TYPES #########################################
################################################################################################


#################
## numpy_array ##
#################

# import numpy as np

# def save_npy_array(
#     obj: np.ndarray,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a NumPy array to the given path.
#     """
#     np.save(path, obj, **kwargs)

# def load_npy_array(
#     path: Union[str, Path],
#     **kwargs,
# ) -> np.ndarray:
#     """
#     Loads an array from the given path.
#     """    
#     return np.load(path, **kwargs)

# _TYPE_LOOKUP.append({
#         "type_name":          "numpy_array",
#         "function_load":      load_npy_array,
#         "function_save":      save_npy_array,
#         "object_class":       np.ndarray,
#         "suffix":             "npy",
#         "library":            "numpy",
#         "versions_supported": [">=1", "<3"],
#     })

##################
## numpy_scalar ##
##################

# import numpy as np

# _TYPE_LOOKUP.append({
#         "type_name":          "numpy_scalar",
#         "function_load":      load_npy_array,
#         "function_save":      save_npy_array,
#         "object_class":       np.number,
#         "suffix":             "npy",
#         "library":            "numpy",
#         "versions_supported": [">=1", "<3"],
#     })

##############
## datetime ##
##############

# import datetime

# def save_datetime(
#     obj: datetime.datetime,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a datetime object to the given path.
#     """
#     util.save_json(obj.isoformat(), path, **kwargs)

# def load_datetime(
#     path: Union[str, Path],
#     **kwargs,
# ) -> datetime.datetime:
#     """
#     Loads a datetime object from the given path.
#     """
#     return datetime.datetime.fromisoformat(util.load_json(path, **kwargs))

# _TYPE_LOOKUP.append({
#         "type_name":          "datetime",
#         "function_load":      load_datetime,
#         "function_save":      save_datetime,
#         "object_class":       datetime.datetime,
#         "suffix":             "json",
#         "library":            "python",
#         "versions_supported": [">=3", "<4"],
#     })

########################
## scipy_sparse_array ##
########################

# import scipy.sparse

# def save_sparse_array(
#     obj: scipy.sparse.spmatrix,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a SciPy sparse matrix to the given path.
#     """
#     scipy.sparse.save_npz(path, obj, **kwargs)

# def load_sparse_array(
#     path: Union[str, Path],
#     **kwargs,
# ) -> scipy.sparse.csr_matrix:
#     """
#     Loads a sparse array from the given path.
#     """        
#     return scipy.sparse.load_npz(path, **kwargs)

# _TYPE_LOOKUP.append({
#         "type_name":          "scipy_sparse_array",
#         "function_load":      load_sparse_array,
#         "function_save":      save_sparse_array,
#         "object_class":       scipy.sparse.spmatrix,
#         "suffix":             "npz",
#         "library":            "scipy",
#         "versions_supported": [">=1", "<2"],
#     })

##################
## torch_tensor ##
##################

# import torch
# import numpy as np

# def save_torch_tensor(
#     obj: torch.Tensor,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a PyTorch tensor as a NumPy array to the given path.
#     """
#     np.save(path, obj.detach().cpu().numpy(), **kwargs)

# def load_torch_tensor(
#     path: Union[str, Path],
#     **kwargs,
# ) -> np.ndarray:
#     """
#     Loads a PyTorch tensor from the given path.
#     """
#     return torch.as_tensor(np.load(path, **kwargs))

# _TYPE_LOOKUP.append({
#         "type_name":          "torch_tensor",
#         "function_load":      load_torch_tensor,
#         "function_save":      save_torch_tensor,
#         "object_class":       torch.Tensor,
#         "suffix":             "npy",
#         "library":            "numpy",
#         "versions_supported": [">=1", "<3"],
#     })

##################
## optuna_study ##
##################

# import optuna

# def save_optuna_study(
#     obj: optuna.study.Study,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves an Optuna study to the given path.
#     """
#     import pickle
#     with open(path, "wb") as f:
#         pickle.dump(obj, f, **kwargs)

# def load_optuna_study(
#     path: Union[str, Path],
#     **kwargs,
# ) -> optuna.study.Study:
#     """
#     Loads an Optuna study from the given path.
#     """
#     import pickle
#     with open(path, "rb") as f:
#         return pickle.load(f, **kwargs)
    
# _TYPE_LOOKUP.append({
#         "type_name":          "optuna_study",
#         "function_load":      load_optuna_study,
#         "function_save":      save_optuna_study,
#         "object_class":       optuna.study.Study,
#         "suffix":             "optuna_study_pkl",
#         "library":            "python",
#         "versions_supported": [">=2", "<3"],
#     })

################
## pandas_csv ##
################

# import pandas as pd

# def save_pandas_csv(
#     obj: pd.DataFrame,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a pandas DataFrame to the given path as a CSV.
#     """
#     obj.to_csv(path, **kwargs)


# def load_pandas_csv(
#     path: Union[str, Path],
#     **kwargs,
# ) -> pd.DataFrame:
#     """
#     Loads a pandas DataFrame from the given path.
#     """
#     return pd.read_csv(path, **kwargs)


# _TYPE_LOOKUP.append({
#         "type_name":          "pandas_csv",
#         "function_load":      load_pandas_csv,
#         "function_save":      save_pandas_csv,
#         "object_class":       pd.DataFrame,
#         "suffix":             "csv",
#         "library":            "pandas",
#         "versions_supported": [">=1", "<3"],
#     })

###################
## PIL_image_png ##
###################

# from PIL import Image

# def save_PIL_image_png(
#     obj: Image.Image,
#     path: Union[str, Path],
#     **kwargs,
# ) -> None:
#     """
#     Saves a PIL image to the given path as a PNG.
#     """
#     with open(path, "wb") as f:
#         obj.save(f, format="PNG", **kwargs)

# def load_PIL_image_png(
#     path: Union[str, Path],
#     **kwargs,
# ) -> Image.Image:
#     """
#     Loads a PIL image from the given path.
#     """
#     return Image.open(path, **kwargs)

# _TYPE_LOOKUP.append({
#         "type_name":          "PIL_image_png",
#         "function_load":      load_PIL_image_png,
#         "function_save":      save_PIL_image_png,
#         "object_class":       Image.Image,
#         "suffix":             "png",
#         "library":            "PIL",
#         "versions_supported": [">=6", "<11"],
#     })

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
    
# _TYPE_LOOKUP.append({
#         "type_name":          "sklearn_model_skops",
#         "function_load":      load_sklearn_model_skops,
#         "function_save":      save_sklearn_model_skops,
#         "object_class":       sklearn.base.BaseEstimator,
#         "suffix":             "skops",
#         "library":            "skops",
#         "versions_supported": [">=0", "<1"],
#     })

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

# _TYPE_LOOKUP.append({
#         "type_name":          "numpy_array_h5",
#         "function_load":      load_npy_array_h5,
#         "function_save":      save_npy_array_h5,
#         "object_class":       np.ndarray,
#         "suffix":             "h5",
#         "library":            "tables",
#         "versions_supported": [">=3", "<4"],
#     })

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
    
# _TYPE_LOOKUP.append({
#         "type_name":          "pickle_obj",
#         "function_load":      load_pickle_obj,
#         "function_save":      save_pickle_obj,
#         "object_class":       object,
#         "suffix":             "pkl",
#         "library":            "pickle",
#         "versions_supported": [],
#     })

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
        raise ValueError(f"Function {func.__name__} does not accept the correct arguments. Expected: {args}. Got: {list(sig.parameters.keys())}. Often, add ``**kwargs`` to the function definition.")

[_check_function_args(func=props["function_load"], args=["path",])        for props in _TYPE_LOOKUP]
[_check_function_args(func=props["function_save"], args=["obj", "path",]) for props in _TYPE_LOOKUP]
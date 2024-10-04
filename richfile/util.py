f"""
RichFile

A system for saving and loading nested python objects containing array data
without serialization.

This module provides a useful way to deal with large and complex nested data
structures. It is designed to be: - insensitive to version changes in libraries
(unlike serialization / pickling) - accessible via browsing the folder structure
- fast and memory-mappable - customizable for different data types

The system is based on the following principles: 
- Each leaf object is saved as a separate file 
- The folder structure mirrors the nested object structure:
    - Lists, tuples, sets, frozensets are saved as folders with elements saved as files
      or folders with integer names
    - Dicts are saved as folders with items saved as folders with integer names.
      Dict items are saved as folders containing 2 elements.
- There is a single metadata file for each folder describing the properties of
  each element in the folder
    - The metadata file is a JSON file named ".metadata.richfile" and contains
      the following items:
        - "elements": a dictionary with keys that are the names of the files /
          folders in the directory and values that are dictionaries with the
          following items:
            - "type": A string describing type of the element. The string used
              should be a valid richfile type, as it is determines how the
              element is loaded. Examples: "npy_array", "scipy_sparse_array",
              "list", "object", "float", etc.
            - "library": A string describing the library used to save the
              element. Examples: "numpy", "scipy", "python", "json" (for native
              python types), etc.
           - "version": A string describing the version of the library used to
              save the element. This is used to determine how the element is
              loaded. Examples: "1.0.0", "0.1.0", etc.
            - "index": An integer that is used to determine the order of the
              elements when loading them. Example: 0, 1, 2, etc.
        - "type": A string describing the type of the folder. The string used
          should be a valid richfile type, as it determines how the folder is
          loaded. Examples: "list", "dict", "tuple", etc. (Only container-like
          types)
        - "library": A string describing the library used to save the folder.
          Examples: "python"
        - "version": A string describing the version of the library used to for
          the container. This is used to determine how the folder is loaded.
          Examples: "3.12", "3.13", etc.
        - "version_richfile": A string describing the version of the richfile
          format used to save the metadata file. Examples: "1.0.0", "0.1.0",
          etc.
- Loading proceeds as follows:
    - enter outer folder
    - load metadata file
    - check that files / folders in the directory match the metadata
    - if folder represents a list, tuple, set, frozenset:
        - elements are expected to be named as integers with an appropriate
          suffix: 0.list, 1.npy, 2.dict, 3.npz, 4.json, etc.
        - load each element in the order specified by the metadata index
        - if an element is container-like, enter its folder, load, and package
          it.
    - if folder represents a dict:
        - each item will be saved as a folder containing a single dict item
        - each dict item folder will contain 2 elements: key (0) and value (1)
    - load elements:
        - richfile types (eg. "array", "sparse_array", etc.) are saved and
          loaded using numpy, scipy, etc. as appropriate.
        - an appropriate suffix will be added to the file or folder name.
        - native python types (eg. "float", "int", "str", etc.) are saved as
          JSON files and loaded using the json library.          

RH 2024
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pathlib import Path
import json
import packaging
import packaging.version
import packaging.specifiers
import warnings
import functools
import importlib.metadata
import platform
import copy
import inspect
import shutil
import filelock
import os
from contextlib import ExitStack

from . import functions
from . import __version__ as VERSION_RICHFILE
from . import VERSIONS_RICHFILE_SUPPORTED, PYTHON_VERSIONS_SUPPORTED, FILENAME_METADATA, FILENAME_TYPELOOKUP, JSON_INDENT


REQUIREMENTS = {
    ## Required top-level keys for the metadata file
    "keys_metadata": [
        "elements",
        "type",
        "library",
        "version",
        "version_richfile",
    ],
    ## Required keys for each element in the metadata file
    "keys_element": [
        "type",
        "library",
        "version",
        "index",
    ],
}


####################################################################################################
#################################### LOADING FUNCTIONS #############################################
####################################################################################################

def load_element(
    path: Union[str, Path],
    metadata: Dict,
    type_lookup: Dict,
    check: bool = True,
) -> Any:
    """
    Loads an element from the given path.
    """
    ## Elements can be either folders or files. The metadata determines how to load them.
    ### Go through a switch-case based on the "type" and possibly "library" and "version" of the element.
    if check:
        if not Path(path).exists():
            raise FileNotFoundError(f"Path {path} not found.")
        if not metadata["type"] in type_lookup:
            raise ValueError(f"Type {metadata['type']} not supported.")

    _prepare_element_loading(
        path=path,
        metadata=metadata,
        type_lookup=type_lookup,
        check=check,
    )

    ## Get function_load
    function_load = type_lookup[metadata["type"]]["function_load"]
    ## Check parameters accepted by function: path, type_lookup, metadata, check
    sig = inspect.signature(function_load)
    args_available = {
        "path": path,
        "metadata": metadata,
        "type_lookup": type_lookup,
        "check": check,
    }
    argnames = list(set(sig.parameters.keys()) & set(args_available.keys()))
    kwargs_function = {argname: args_available[argname] for argname in argnames}

    return function_load(
        **kwargs_function,
    )


def load_folder(
    path: Union[str, Path],
    type_lookup: Dict,
    check: bool = True,
    **kwargs,
) -> Any:
    """
    Loads the folder from the given path. Used for types: list, tuple, set, frozenset, dict.
    """
    metadata = load_folder_metadata(path_dir=path, name_metadata=FILENAME_METADATA, check=check)

    if check:
        names_path_elements = [p.name for p in Path(path).iterdir() if (p.name != FILENAME_METADATA) and (p.name != FILENAME_TYPELOOKUP)]
        # Check that index values are unique
        indices = [value["index"] for value in metadata["elements"].values()]
        if len(indices) != len(set(indices)):
            raise ValueError("Indices in metadata are not unique.")
        
        elif metadata["library"] == "python":
            if not is_version_compatible(version=metadata["version"], rules=PYTHON_VERSIONS_SUPPORTED):
                raise ValueError(f"Python version '{metadata['version']}' not supported for library 'python' and type '{metadata['type']}'. Version rules: {PYTHON_VERSIONS_SUPPORTED}")
        else:
            raise ValueError("Only 'python' library supported for container types.")
        
    ## Sort the element names by their index
    names_meta_sorted = _sort_element_names_by_index(metadata=metadata, check=check)

    if check:
        # Check that all elements in metadata are present in the folder and vice versa
        if not all(name in names_path_elements for name in names_meta_sorted):
            missing_elements = set(names_meta_sorted) - set(names_path_elements)
            raise FileNotFoundError(f"Elements in metadata not found in folder: {missing_elements}")
        if not all(name in names_meta_sorted for name in names_path_elements):
            extra_elements = set(names_path_elements) - set(names_meta_sorted)
            raise ValueError(f"Extra elements in folder not found in metadata: {extra_elements}")        

    ## Load each element in the order specified by the metadata index
    elements = [load_element(
        path=str(Path(path) / name),
        metadata=metadata["elements"][name],
        type_lookup=type_lookup,
        check=check,
    ) for name in names_meta_sorted]
    
    if metadata["type"] == "list":
        pass
    elif metadata["type"] == "tuple":
        elements = tuple(elements)
    elif metadata["type"] == "set":
        elements = set(elements)
    elif metadata["type"] == "frozenset":
        elements = frozenset(elements)
    elif metadata["type"] == "dict_item":
        ## Make sure that there are exactly 2 elements
        if check:
            if len(elements) != 2:
                raise ValueError(f"DictItem must contain exactly 2 elements. Found {len(elements)}.")
        elements = DictItem(key=elements[0], value=elements[1])
    else:  ## elif metadata["type"] == "dict"
        ## Make sure that all elements are DictItem types
        if check:
            if not all(isinstance(element, DictItem) for element in elements):
                raise TypeError("All elements in a dict must be of type DictItem.")
        try:
            elements = {element.key: element.value for element in elements}  ## Unpack the DictItems
        except AttributeError as e:
            raise ValueError(f"Error unpacking dict items: {e}")
    
    return elements


def _sort_element_names_by_index(metadata: Dict, check: bool = True) -> List[str]:
    """
    Sorts the element names in the metadata by their index.
    """
    indices_and_names_meta = {value["index"]: name for name, value in metadata["elements"].items()}
    indices_meta_sorted = sorted(indices_and_names_meta.keys())
    names_meta_sorted = [indices_and_names_meta[index] for index in indices_meta_sorted]
    if check:
        if indices_meta_sorted != list(range(len(indices_meta_sorted))):
            raise ValueError("Indices in metadata are not consecutive integers starting from zero.")
    return names_meta_sorted


def load_folder_metadata(
    path_dir: Union[str, Path],
    name_metadata: str = FILENAME_METADATA,
    check: bool = True,
) -> Dict:
    """
    Loads the metadata file from the given path.
    """
    if check:
        if not Path(path_dir).is_dir():
            raise NotADirectoryError(f"Path {path_dir} is not a directory.")
        if not (Path(path_dir) / name_metadata).is_file():
            raise FileNotFoundError(f"Metadata file {name_metadata} not found in directory {path_dir}.")

    with open(str(Path(path_dir) / name_metadata), "r") as f:
        metadata = json.load(f)

    if check:
        # Check for required keys
        missing_keys = set(REQUIREMENTS["keys_metadata"]) - set(metadata.keys())
        if missing_keys:
            raise KeyError(f"Metadata is missing required keys: {missing_keys}")
        # Check version
        if not is_version_compatible(version=metadata["version_richfile"], rules=VERSIONS_RICHFILE_SUPPORTED):
            raise ValueError(f"RichFile version {metadata['version_richfile']} not supported.")
        for element_name, meta_element in metadata["elements"].items():
            missing_keys = set(REQUIREMENTS["keys_element"]) - set(meta_element.keys())
            if missing_keys:
                raise KeyError(f"Element '{element_name}' is missing keys: {missing_keys}")

    return metadata


def _prepare_element_loading(
    path: Union[str, Path],
    metadata: Dict,
    type_lookup: Dict,
    check: bool,
) -> None:
    """
    Performs checks and preparations for loading a file.
    """
    if check:
        if type_lookup[metadata["type"]]["library"] == []:
            warnings.warn(f"Field 'versions_supported' is empty in type_lookup for type {metadata['type']}.")
        ## If the library is python, check the variable directory
        elif metadata["library"] == "python":
            if not is_version_compatible(version=metadata["version"], rules=PYTHON_VERSIONS_SUPPORTED):
                raise ValueError(f"Python version '{metadata['version']}' not supported for library 'python' and type '{metadata['type']}'. Version rules: {PYTHON_VERSIONS_SUPPORTED}")
        elif not is_version_compatible(version=metadata["version"], rules=type_lookup[metadata["type"]]["versions_supported"]):
            raise ValueError(f"Version '{metadata['version']}' not supported for library '{metadata['library']}' and type '{metadata['type']}'. Version rules: {type_lookup[metadata['type']]['versions_supported']}")
        ## Check that path exists as either a file or a directory
        if not Path(path).exists():
            raise FileNotFoundError(f"Path {path} not found.")            

def load_json(
    path: Union[str, Path],
    **kwargs,
) -> Any:
    """
    Loads a scalar from the given path. Used as a super for str, bool, and None.
    """
    with open(path, "r") as f:
        return json.load(f, **kwargs)
    
def load_float(path: Union[str, Path], **kwargs) -> float:
    return float(load_json(path, **kwargs))
def load_int(path: Union[str, Path], **kwargs) -> int:
    return int(load_json(path, **kwargs))
def load_str(path: Union[str, Path], **kwargs) -> str:
    return str(load_json(path, **kwargs))
def load_bool(path: Union[str, Path], **kwargs) -> bool:
    return bool(load_json(path, **kwargs))
def load_None(path: Union[str, Path], **kwargs) -> None:
    out = load_json(path, **kwargs)
    if out is not None:
        raise ValueError("Loaded object is not None.")
    return out
    

####################################################################################################
#################################### SAVING FUNCTIONS ##############################################
####################################################################################################


def save_object(
    obj: Any,
    path: Union[str, Path],
    type_lookup: Dict,
    check: bool = True,
    overwrite: bool = False,
    name_dict_items: bool = True,
) -> None:
    """
    Saves an object to the given directory in the RichFile format.
    """
    # Determine the type of the object and save accordingly
    props = type_lookup[type(obj)]
    type_object = props["type_name"]

    _prepare_save_path(path=path, overwrite=overwrite, mkdir=True)
    if check:
        library_version = _get_library_version(props["library"])
        if props["library"] in type_lookup:
            if not is_version_compatible(version=library_version, rules=type_lookup[props["library"]]["versions_supported"]):
                raise ValueError(f"Library {props['library']} version {library_version} not supported.")        

    ## Get function_save
    function_save = props["function_save"]
    ## Check parameters accepted by function: obj, path, type_lookup, type_name, check, overwrite, name_dict_items
    sig = inspect.signature(function_save)
    args_available = {
        "obj": obj,
        "path": path,
        "type_lookup": type_lookup,
        "type_name": type_object,
        "check": check,
        "overwrite": overwrite,
        "name_dict_items": name_dict_items,
    }
    argnames = list(set(sig.parameters.keys()) & set(args_available.keys()))
    kwargs_function = {argname: args_available[argname] for argname in argnames}

    function_save(
        **kwargs_function,
    )

def save_container(
    obj: Union[list, tuple, set, dict, frozenset, 'DictItem'],
    path: Union[str, Path],
    type_name: str,
    type_lookup: Dict,
    check: bool = True,
    overwrite: bool = False,
    name_dict_items: bool = True,
) -> None:
    """
    Saves a list, tuple, set, frozenset, or dict_item to the given directory.
    """
    if isinstance(obj, dict):
        obj = [DictItem(key=key, value=value) for key, value in obj.items()]

    metadata_elements = {}
    for idx, element in enumerate(obj):
        try:
            props = type_lookup[type(element)]
            type_element = props["type_name"]            
        except TypeError as e:
            raise TypeError(f"Failed to get properties for element. \n Directory: {path}. \n Index: {idx}. \n element: {element}. \n type_name: {type_name}. \n Error: {e}")
        name_element = f"{idx}.{props['suffix']}"  ## Make name the index and add suffix
        if name_dict_items:
            if type_element == "dict_item":
                if isinstance(element.key, str):
                    name_element = f"{element.key}.{props['suffix']}"
            elif type_name == "dict_item":
                name_element = f"{['key', 'value'][idx]}.{props['suffix']}"

        _check_filename_safety(name=name_element, warn=True, raise_error=False)
        path_element = str(Path(path) / name_element)  ## Make path
        save_object(
            obj=element,
            path=path_element,
            check=check,
            overwrite=overwrite,
            type_lookup=type_lookup,
        )
        metadata_elements[name_element] = {
            "type": type_element,
            "library": props["library"],
            "version": _get_library_version(library=props["library"]),
            "index": idx,
        }

    metadata_container = {
        "elements": metadata_elements,
        "type": type_name,
        "library": "python",
        "version": _get_python_version(),
        "version_richfile": VERSION_RICHFILE,
    }
    save_metadata(
        metadata=metadata_container,
        path_dir=path,
        check=check,
        name_metadata=FILENAME_METADATA,
        overwrite=overwrite,
    )


def _check_filename_safety(name: str, warn: bool = True, raise_error: bool = False) -> None:
    """
    Checks if a filename is safe to use.
    """
    issue = list(set(list(name)) & set(["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]))
    n_issues = len(issue)

    if n_issues > 0:
        if warn:
            warnings.warn(f"Filename contains invalid character: {issue}. Name: {name}")
        if raise_error:
            raise ValueError(f"Filename contains invalid character: {issue}. Name: {name}")


def _prepare_save_path(path: Union[str, Path], overwrite: bool = False, mkdir: bool = True) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"Path already exists: {path}.")
    if mkdir:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_metadata(
    metadata: Dict,
    path_dir: Union[str, Path],
    check: bool = True,
    name_metadata: str = FILENAME_METADATA,
    overwrite: bool = False,
) -> None:
    """
    Saves the metadata dictionary to a file in the given directory.
    """
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    if check:
        ## There should not be a metadata file already present if overwrite is False
        if not overwrite and (Path(path_dir) / name_metadata).exists():
            raise FileExistsError(f"Metadata file {name_metadata} already exists in directory {path_dir}.")

    with open(str(Path(path_dir) / name_metadata), "w") as f:
        json.dump(metadata, f, indent=JSON_INDENT)


class DictItem:
    def __init__(self, key, value):
        self.key = key
        self.value = value
    def __iter__(self):
        return iter([self.key, self.value])

def save_json(
    obj: Any,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Saves a JSON-serializable object to the given path.
    """
    with open(path, "w") as f:
        json.dump(obj, f)


def _get_library_version(library: str) -> str:
    """
    Returns the version of the library as a string.
    """
    ## Import the library str's version using importlib.metadata
    ### If it's a native python library, use a custom function to get the version
    if library in ["python", "builtins"]:
        return _get_python_version()
    else:
        try:
            return importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            try:
                ## Try to import the library and get the version from the __version__ attribute
                lib = importlib.import_module(library)
            except AttributeError as e:
                raise ValueError(f"Library: '{library}' not found. Error: {e}")
            try:
                return lib.__version__
            except AttributeError as e:
                warnings.warn(f"Library {library} does not have a __version__ attribute. Error: {e}")
                return None
        except ImportError:
            raise ValueError(f"Library {library} not found. Error: {e}")
    
def _get_python_version() -> str:
    """
    Returns the Python version as a string.
    """
    return platform.python_version()

def is_version_compatible(version: str, rules: List[str]) -> bool:
    """
    Checks if a given version string satisfies all specified version rules.
    RH 2024

    Args:
        version (str): 
            The version string to check (e.g., "2.4.9").
        rules (List[str]): 
            A list of version rules (e.g., ["<3", ">=1.2",]).

    Returns:
        bool: True if the version satisfies all rules, False otherwise.
    """
    try:
        version_obj = packaging.version.Version(version)
        specifier = packaging.specifiers.SpecifierSet(",".join(rules))
        return version_obj in specifier
    except packaging.version.InvalidVersion:
        raise ValueError(f"Invalid version string: {version}")


####################################################################################################
#################################### HIGH-LEVEL CLASS ##############################################
####################################################################################################

class RichFile:
    f"""
    High-level class for handling reading and writing objects in the RichFile format.
    Allows customization of loading and saving functions, and setting additional parameters.
    RH 2024

    Args:
        path (Optional[Union[str, Path]): 
            The path to save the object to. If None, uses the path specified
            in the RichFile object.
        check (Optional[bool]): 
            Whether to perform checks on the object and path. If None, uses
            the check specified in the RichFile object.
        safe_save (Optional[bool]): 
            Whether to use a safe save method. If None, uses the safe_save
            specified in the RichFile object.
        overwrite (Optional[bool]): 
            Whether to overwrite the file if it already exists. If None,
            uses the overwrite specified in the RichFile object.
        name_dict_items (Optional[bool]):
            Whether to name dict items as their keys. If None, uses the
            name_dict_items specified in the RichFile object.
        save_type_lookup (Optional[bool]):
            Whether to save the type lookup table as a file named
            {FILENAME_TYPELOOKUP} in the outermost richfile directory if it
            is a directory. If None, uses the save_type_lookup specified in
            the RichFile object.
    """
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        check: Optional[bool] = True,
        safe_save: Optional[bool] = True,
        overwrite: Optional[bool] = False,
        name_dict_items: Optional[bool] = True,
        save_type_lookup: Optional[bool] = True,
    ):
        self.path             = path
        self.check            = check
        self.safe_save        = safe_save
        self.overwrite        = overwrite
        self.name_dict_items  = name_dict_items
        self.save_type_lookup = save_type_lookup

        self.type_lookup = copy.deepcopy(functions.TypeLookup())
        self.params_load = {}
        self.params_save = {}
    
    def save(
        self, 
        obj: Any, 
        path: Optional[Union[str, Path]] = None, 
        check: Optional[bool] = None, 
        safe_save: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        name_dict_items: Optional[bool] = None,
        save_type_lookup: Optional[bool] = None,
    ) -> None:
        f"""
        Saves an object to the given path.

        Args:
            obj (Any): 
                The object to save.
            path (Optional[Union[str, Path]]): 
                The path to save the object to. If None, uses the path specified
                in the RichFile object.
            check (Optional[bool]): 
                Whether to perform checks on the object and path. If None, uses
                the check specified in the RichFile object.
            safe_save (Optional[bool]): 
                Whether to use a safe save method. If None, uses the safe_save
                specified in the RichFile object.
            overwrite (Optional[bool]): 
                Whether to overwrite the file if it already exists. If None,
                uses the overwrite specified in the RichFile object.
            name_dict_items (Optional[bool]):
                Whether to name dict items as their keys. If None, uses the
                name_dict_items specified in the RichFile object.
            save_type_lookup (Optional[bool]):
                Whether to save the type lookup table as a file named
                {FILENAME_TYPELOOKUP} in the outermost richfile directory if it
                is a directory. If None, uses the save_type_lookup specified in
                the RichFile object.
        """
        path             = self.path             if path             is None else path
        check            = self.check            if check            is None else check
        safe_save        = self.safe_save        if safe_save        is None else safe_save
        overwrite        = self.overwrite        if overwrite        is None else overwrite
        name_dict_items  = self.name_dict_items  if name_dict_items  is None else name_dict_items
        save_type_lookup = self.save_type_lookup if save_type_lookup is None else save_type_lookup

        if (path is None) or (not isinstance(check, bool)):
            raise ValueError("`path` [str, Path] and `check` [bool] must be specified.")
        
        kwargs_safe_saver = {
            "overwrite": overwrite,
            "safe_save": safe_save,
            "delete_temp_on_error": False,
            "timeout_lock": 1,
            "force_acquire_lock": overwrite,
            "force_release_lock": True,
        }

        ## Create a lock file specific to the target path
        ### Append the lock suffix to the path (don't replace the suffix)
        path_obj = str(path)
        fn_make_path_tmp  = lambda path: path + '.tmp'
        fn_make_path_lock = lambda path: path + '.lock'
        with SafeSaver(
            path_target=path_obj,
            path_temp=fn_make_path_tmp(path),
            path_lock=fn_make_path_lock(path),
            **kwargs_safe_saver,
        ) as path_temp:
            save_object(
                obj=obj,
                path=path_temp,
                type_lookup=self.type_lookup,
                check=check,
                overwrite=overwrite,
                name_dict_items=name_dict_items,
            )
            
        if save_type_lookup:
            ## Make the type_lookup table a file in the outermost directory
            type_lookup = copy.deepcopy(self.type_lookup.properties)
            ### Convert the functions and classes into strings
            for prop in type_lookup:
                prop["function_load"] = inspect.getsource(prop["function_load"])
                prop["function_save"] = inspect.getsource(prop["function_save"])
            
                prop["object_class"] = str(prop["object_class"])

            ## If the richfile is a directory, save the type lookup table as a file in the outermost directory
            if Path(path).is_dir():
                ## Save as a .json file
                path_type_lookup = str(Path(path) / FILENAME_TYPELOOKUP)
                with SafeSaver(
                    path_target=path_type_lookup,
                    path_temp=fn_make_path_tmp(path_type_lookup),
                    path_lock=fn_make_path_lock(path_type_lookup),
                    **kwargs_safe_saver,
                ) as path_temp:
                    save_json(
                        obj=type_lookup,
                        path=path_temp,
                        indent=JSON_INDENT,
                    )
                
        return self
    
    def load(
        self, 
        path: Optional[Union[str, Path]] = None, 
        type_lookup: Optional[Dict] = None,
        check: Optional[bool] = None,
    ) -> Any:
        path = self.path if path is None else path
        type_lookup = self.type_lookup if type_lookup is None else type_lookup
        check = self.check if check is None else check
        if (path is None) or (not isinstance(check, bool)):
            raise ValueError("`path` [str, Path] and `check` [bool] must be specified.")
        
        ## Look for a metadata file in the directory
        ### If there isn't one, assume it is the outer directory and load as a folder
        if not (Path(path).parent / FILENAME_METADATA).exists():
            ## If the path is a directory, load it as a folder
            if Path(path).is_dir():
                ## Look for the type within the metadata file within the directory
                if not (Path(path) / FILENAME_METADATA).exists():
                    raise FileNotFoundError(f"Metadata file {FILENAME_METADATA} not found in directory {path}.")
                metadata_dir = load_folder_metadata(path_dir=path, check=check)
                metadata_obj = {
                    "type": metadata_dir["type"],
                    "library": metadata_dir["library"],
                    "version": metadata_dir["version"],
                    "index": 0,
                }
                return load_element(
                    path=path,
                    metadata=metadata_obj,
                    type_lookup=type_lookup,
                    check=check,
                )
            ## If the path is a file, then it is missing a metadata file
            elif Path(path).is_file():
                raise FileNotFoundError(f"Metadata file {FILENAME_METADATA} not found in directory {Path(path).parent}.")
            else:
                raise FileNotFoundError(f"Path {path} not found.")
        else:
            metadata_folder = load_folder_metadata(path_dir=str(Path(path).parent), check=check)
            metadata_element = metadata_folder["elements"][Path(path).name]
            return load_element(
                path=path,
                metadata=metadata_element,
                type_lookup=type_lookup,
                check=check,
            )

    def register_type(
        self, 
        type_name: str, 
        function_load: Callable, 
        function_save: Callable, 
        object_class: type, 
        suffix: str, 
        library: str,
        versions_supported: Optional[List[str]] = [],
    ):
        """
        Registers a new type with custom loading and saving functions.
        """
        ## Add the property to the type_lookup
        prop = {
            'type_name': type_name,
            'function_load': function_load,
            'function_save': function_save,
            'object_class': object_class,
            'suffix': suffix,
            'library': library,
            'versions_supported': versions_supported,
        }

        if self.check:
            functions._verify_validity_of_new_type(prop)
            #### check for duplicates
            if type_name in self.type_lookup:
                raise KeyError(f"Type {type_name} already registered.")

        self.type_lookup.add_property(prop)

    def register_type_from_dict(self, prop: Dict) -> None:
        """
        Registers a new type with custom loading and saving functions.
        """
        if self.check:
            if not isinstance(prop, dict):
                raise TypeError("`prop` must be a dictionary.")
        ## Add in versions_supported if not present
        prop.setdefault("versions_supported", [])

        ## Use inspect to ensure that all args in register_type are present. Kwargs okay to skip.
        sig = inspect.signature(self.register_type)
        args_available = set(prop.keys())
        args_missing = set(sig.parameters.keys()) - args_available
        args_extra = args_available - set(sig.parameters.keys())
        if len(args_missing) > 0:
            raise ValueError(f"Missing arguments: {args_missing}.")
        if len(args_extra) > 0:
            raise ValueError(f"Extra arguments: {args_extra}.")
        
        self.register_type(**prop)

    def set_load_kwargs(
        self, 
        type_: Union[str, type], 
        **kwargs,
    ) -> None:
        """
        Sets additional parameters for the load function of a specific type.
        """
        type_name = self.type_lookup[type_]["type_name"]

        ## Partial the function with the parameters
        if type_name in self.type_lookup:
            self.type_lookup[type_name] = {
                "function_load": functools.partial(
                    self.type_lookup[type_name]['function_load'],
                    **kwargs,
                )
            }
        else:
            raise KeyError(f"Type {type_name} not found in type lookup.")
        
        self.params_load[type_name] = kwargs

    def set_save_kwargs(
        self,
        type_: Union[str, type], 
        **kwargs,
    ) -> None:
        """
        Sets additional parameters for the save function of a specific type.
        """
        type_name = self.type_lookup[type_]["type_name"]

        ## Partial the function with the parameters
        if type_name in self.type_lookup:
            self.type_lookup[type_name] = {
                "function_save": functools.partial(
                    self.type_lookup[type_name]['function_save'],
                    **kwargs,
                )
            }
        else:
            raise KeyError(f"Type {type_name} not found in type lookup.")
        
        self.params_save[type_name] = kwargs

    def get_metadata(self, path_dir: Union[str, Path]) -> Dict:
        """
        Retrieves the metadata from the specified directory.
        """
        return load_folder_metadata(path_dir=path_dir, check=self.check)

    def list_elements(self, path_dir: Union[str, Path]) -> List[str]:
        """
        Lists the elements in the specified directory.
        """
        metadata = load_folder_metadata(path_dir=path_dir, check=self.check)
        return list(metadata['elements'].keys())

    def get_type_info(self, type_name: str) -> Dict:
        """
        Retrieves the type information for a given type name.
        """
        return self.type_lookup.get(type_name, {})

    def view_directory_tree(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Prints a tree structure of the directory.
        Uses the metadata to determine the structure.
        """
        path = self.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")
        
        def _view_tree(path, level=0):
            metadata = self.get_metadata(path)
            for name, value in metadata['elements'].items():
                print("|   " * level + "├── " + f"{name} ({value['type']})")
                if value['type'] in ["list", "tuple", "set", "frozenset", "dict", "dict_item"]:
                    _view_tree(path=str(Path(path) / name), level=level+1)
            print("|   " * level)

        if Path(path).is_dir():
            print(f"Viewing tree structure of richfile at path: {path} ({self.get_metadata(path)['type']})")
            _view_tree(path)
        elif Path(path).is_file():
            metadata_folder = load_folder_metadata(path_dir=str(Path(path).parent), check=self.check)
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            print(f"Viewing element at path: {path} ({metadata_element['type']})")
        else:
            raise FileNotFoundError(f"Path {path} not found.")
    
    def view_tree(
        self, 
        path: Optional[Union[str, Path]] = None, 
        show_filenames: bool = False,
    ) -> None:
        """
        Prints a tree structure of the directory.
        If a dict item has a string key, it will be printed as a key-value pair.
        List, tuple, set, and frozenset items will be printed as a list of items.
        """
        sf = show_filenames
        path = self.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")
        
        def _view_tree(path, level=0):
            metadata = self.get_metadata(path)
            for name, value in metadata['elements'].items():
                if value['type'] == "dict_item":
                    metadata_item = self.get_metadata(str(Path(path) / name))
                    names_meta_sorted = _sort_element_names_by_index(metadata=metadata_item)
                    ## The other element should be the value
                    if len(names_meta_sorted) != 2:
                        raise ValueError("DictItem must contain exactly 2 elements: key='0.json' and value=another element.")
                    name_key, name_value = names_meta_sorted
                    metadata_key = metadata_item['elements'][name_key]
                    metadata_value = metadata_item['elements'][name_value]
                    key = load_element(
                        path=str(Path(path) / name / name_key),
                        metadata=metadata_key,
                        type_lookup=self.type_lookup,
                    )
                    print("|    " * level + "├── " + f"'{key}': {(name_value if sf else '')}  ({metadata_value['type']})")
                    if metadata_value['type'] in ["list", "tuple", "set", "dict", "dict_item"]:
                        _view_tree(path=str(Path(path) / name / name_value), level=level+1)
                elif value['type'] in ["list", "tuple", "set", "dict"]:
                    print("|    " * level + "├── " + f"{(name if sf else '')}  ({value['type']})")
                    _view_tree(path=str(Path(path) / name), level=level+1)
                else:
                    print("|    " * level + "├── " + f"{(name if sf else '')}  ({value['type']})")
            print("|    " * level)

        if Path(path).is_dir():
            print(f"Path: {path} ({self.get_metadata(path)['type']})")
            _view_tree(path)
        elif Path(path).is_file():
            metadata_folder = load_folder_metadata(path_dir=str(Path(path).parent), check=self.check)
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            print(f"Path: {path} ({metadata_element['type']})")
        else:
            raise FileNotFoundError(f"Path {path} not found.")

    def get_metadata_tree(
        self,
        path: Optional[Union[str, Path]] = None,
    ):
        """
        Return a hierarchical dictionary containing the metadata for the entire
        directory tree.
        """
        path = self.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")
        
        def _get_metadata_tree(path):
            metadata = self.get_metadata(path)
            out = {
                "metadata": metadata,
                "elements": {},
            }
            for name, value in metadata['elements'].items():
                if value['type'] in ["list", "tuple", "set", "frozenset", "dict", "dict_item"]:
                    out["elements"][name] = _get_metadata_tree(str(Path(path) / name))
                else:
                    out["elements"][name] = value
            return out

        if Path(path).is_dir():
            return _get_metadata_tree(path)
        elif Path(path).is_file():
            metadata_folder = load_folder_metadata(path_dir=str(Path(path).parent), check=self.check)
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            return metadata_element
        else:
            raise FileNotFoundError(f"Path {path} not found.")
        
    def __str__(self):
        return f"RichFile(path={self.path}, check={self.check}, params_load={self.params_load}, params_save={self.params_save})"

    def __repr__(self):
        ## If the path exists and has a metadata file, show the metadata
        if self.path is not None:
            if Path(self.path).exists() and (Path(self.path) / FILENAME_METADATA).exists():
                metadata = self.get_metadata(self.path)
                self.view_tree()

        return self.__str__()
    
    ## Item retrieval by key or index
    def __getitem__(self, key):
        
        ## Load dict items by key
        if isinstance(key, str):
            metadata = self.get_metadata(self.path)
            ## Confirm that path is a dict
            if metadata['type'] != "dict":
                raise ValueError("Path must be a dict to load by key.")
            ## Find filename matching a dict_item with a str as a key matching the input key
            names_meta_sorted = _sort_element_names_by_index(metadata=metadata)
            for name in names_meta_sorted:
                ## Check if the key is a string by loading the dict_item metadata
                metadata_item = self.get_metadata(str(Path(self.path) / name))
                if (not (metadata['elements'][name]['type'] == "dict_item")):
                    raise ValueError(f"Found element with type {metadata['elements'][name]['type']}. Expected 'dict_item'.")
                if not (metadata_item['type'] == "dict_item"):
                    raise ValueError(f"Found element with type {metadata_item['type']}. Expected 'dict_item'.")
                if not len(metadata_item['elements']) == 2:
                    raise ValueError("DictItem must contain exactly 2 elements: key and value.")
                names_meta_sorted_item = _sort_element_names_by_index(metadata=metadata_item)
                name_key, name_value = names_meta_sorted_item
                if metadata_item['elements'][name_key]['type'] == "str":
                    key_loaded = load_element(
                        path=str(Path(self.path) / name / name_key),
                        metadata=metadata_item['elements'][name_key],
                        type_lookup=self.type_lookup,
                    )
                    if key_loaded == key:
                        ## return a RichFile object at the path of the value
                        out = copy.deepcopy(self)
                        out.path = str(Path(self.path) / name / name_value)
                        return out
                        
        ## Load list or tuple items by index
        elif isinstance(key, int):
            metadata = self.get_metadata(self.path)
            ## Confirm that path is a list or tuple
            if metadata['type'] not in ["list", "tuple"]:
                raise ValueError("Path must be a list or tuple to load by index.")
            ## Find filename of the metadata index matching the input key
            for name, value in metadata['elements'].items():
                if value['index'] == key:
                    ## return a RichFile object at the path of the element
                    out = copy.deepcopy(self)
                    out.path = str(Path(self.path) / name)
                    return out
        else:
            raise ValueError("__getitem__ only supports str and int keys.")
        
        raise KeyError(f"Key {key} not found.")
    
    def keys(self):
        """
        Returns a list of keys in the directory.
        """
        try:
            metadata = self.get_metadata(self.path)
            names_elements_raw = list(metadata['elements'].keys())
            names_elements = ['.'.join(name.split('.')[:-1]) for name in names_elements_raw]
            return names_elements
        except FileNotFoundError:
            return []
        except Exception as e:
            warnings.warn(f"Path element failed to load metadata or doesn't have .keys() method. Error: {e}")
    

def delete_file_or_folder(path: Union[str, Path]) -> None:
    """
    Deletes files OR directories.

    Args:
        path (Union[str, Path]): 
            The path to the file or directory to delete.
    """
    path = Path(path)
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path))
        else:
            pass  ## path does not exist, do nothing
    except Exception as e:
        warnings.warn(f"Failed to delete path: {path}. Error: {e}")
    

class FileLock_AutoDeleting(filelock.FileLock):
    """
    A robust and safe wrapper around filelock.FileLock that ensures the lock
    file is deleted upon release or when exiting a with statement, preventing
    orphaned lock files.

    Args:
        lock_file (Union[str, Path]):
            The path to the lock file.
        timeout (Optional[float]):
            The maximum time to wait for the lock in seconds.
        force_acquire (bool):
            If True, the lock will be acquired even if the lock file exists.
        force (bool):
            NOT USED. This class always forces the lock to be released and
            deleted.
    """
    def __init__(
        self, 
        lock_file: Union[str, Path], 
        timeout: Optional[float] = None,
        force_acquire: bool = False,
        force: bool = False,
    ):
        if isinstance(force_acquire, bool):
            self.force_acquire = force_acquire
        else:
            raise ValueError("`force_acquire` must be a boolean.")

        if force_acquire and Path(lock_file).exists():
            Path(lock_file).unlink()

        self.path_lock = str(lock_file)
        super().__init__(lock_file=str(lock_file), timeout=timeout)

    def release(self, force: bool = False):
        """
        Release the lock and delete the lock file if it exists.
        """
        super().release(force=force)
        try:
            Path(self.path_lock).unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            warnings.warn(f"Failed to remove lock file: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        self.release(force=False)
        return False
    
    def __del__(self):
        self.release(force=False)

        
class AtomicSaver:
    """
    Context manager for safely saving files or directories using a temporary
    path. On successful exit, it replaces the target path with the temporary
    data. On error, it deletes the temporary path if desired.
    RH 2024

    Args:
        path_target (Union[str, Path]):
            The path to the target file or directory to save.
        path_temp (Optional[Union[str, Path]):
            The path to the temporary file or directory to save. If None, it
            will be the target path with '.tmp' appended.
        overwrite (bool):
            If True, the target path will be overwritten if it exists.
        delete_temp_on_error (bool):
            If True, the temporary path will be deleted if an error occurs.    
    """
    def __init__(
        self, 
        path_target: Union[str, Path], 
        path_temp: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        delete_temp_on_error: bool = False,
    ):
        self.path_target = str(path_target)
        self.path_temp = str(path_temp) if path_temp is not None else self.path_target + '.tmp'

        self.overwrite = overwrite
        self.delete_temp_on_error = delete_temp_on_error

    def __enter__(self):
        ## Handle overwrite protection
        if not self.overwrite:
            if Path(self.path_target).exists():
                raise FileExistsError(f"Path already exists: {self.path_target} and overwrite is False.")
        ## Delete the temp path if it exists
        if Path(self.path_temp).exists():
            delete_file_or_folder(self.path_temp)
            
        return self.path_temp

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
            ## No exception, proceed to replace target with temp
                ## Replace the target path with the temp path atomically using os.replace
                ### Handle non-empty directories by deleting the target path first (loss of atomicity)
                if Path(self.path_target).is_dir():
                    if Path(self.path_target).exists():
                        delete_file_or_folder(self.path_target)
                os.replace(src=self.path_temp, dst=self.path_target)
                
            else:
                ## An exception occurred, delete the temp path
                if Path(self.path_temp).exists() and self.delete_temp_on_error:
                    delete_file_or_folder(self.path_temp)
                ## Return False to re-raise the exception
                return False
        ## If an exception occurs during the replacement, delete the temp path
        except Exception as e:
            ## Delete the temp path
            if Path(self.path_temp).exists() and self.delete_temp_on_error:
                delete_file_or_folder(self.path_temp)
            ## Raise the exception
            raise e
    
    def __str__(self):
        return f"SafeSaver(path_target={self.path_target}, path_temp={self.path_temp}, delete_temp_on_error={self.delete_temp_on_error})"
    

class MultiContextManager:
    """
    A context manager that allows multiple context managers to be used at once.
    RH 2024

    Args:
        *managers (context managers):
            Multiple context managers to be used at once.

    Demo:
        .. code-block:: python

            with MultiContextManager(
                torch.no_grad(), 
                temp_set_attr(obj, attr, new_val), 
                open('file.txt', 'w') as f,
            ):
                # do something
    """
    def __init__(self, *managers):
        self.managers = managers
        self.stack = ExitStack()

    def __enter__(self):
        for manager in self.managers:
            self.stack.enter_context(manager)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)
        

class SafeSaver(MultiContextManager):
    """
    Context manager for safely saving files or directories using a temporary
    path and path locking. This class combines the FileLock_AutoDeleting and
    AtomicSaver classes.
    RH 2024

    Args:
        path_target (Union[str, Path]):
            The path to the target file or directory to save.
        path_temp (Optional[Union[str, Path]):
            The path to the temporary file or directory to save. If None, it
            will be the target path with '.tmp' appended.
        path_lock (Optional[Union[str, Path]):
            The path to the lock file. If None, it will be the target path
            with '.lock' appended.
        overwrite (bool):
            If True, the target path will be overwritten if it exists.
        safe_save (bool):
            If True, the save operation will be wrapped in an AtomicSaver. If
            False and overwrite is True, the target path will be deleted before
            saving.
        delete_temp_on_error (bool):
            If True, the temporary path will be deleted if an error occurs.
        safe_save (bool):
            If True, the save operation will be wrapped in an AtomicSaver. If
            False and overwrite is True, the target path will be deleted before
            saving.
        file_lock (bool):
            If True, a lock file will be created to prevent concurrent access.
        timeout_lock (float):
            The timeout for acquiring the lock file.
        force_acquire_lock (bool):
            If True, the lock file will be deleted before acquiring the lock.
        force_release_lock (bool):
            If True, the lock file will be deleted before releasing the lock.

    Demo:
        .. code-block:: python

            with SafeSaver(
                path_target='file.txt', 
                path_temp='file.txt.tmp',
                overwrite=True, 
                safe_save=True, 
                delete_temp_on_error=False,
                file_lock=True,
                timeout_lock=5,
                force_acquire_lock=False,
                force_release_lock=False,
            ):
                save_object(path_temp, obj)
    """
    def __init__(
        self, 
        path_target: Union[str, Path], 
        path_temp: Optional[Union[str, Path]] = None,
        path_lock: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        delete_temp_on_error: bool = False,
        safe_save: bool = True,
        file_lock: bool = True,
        timeout_lock: float = 5,
        force_acquire_lock: bool = False,
        force_release_lock: bool = False,
    ):
        self.path_target = str(path_target)
        self.path_temp = str(path_temp) if path_temp is not None else self.path_target + '.tmp'
        self.path_lock = str(path_lock) if path_lock is not None else self.path_target + '.lock'

        self.overwrite = overwrite
        self.safe_save = safe_save
        self.delete_temp_on_error = delete_temp_on_error

        # Overwrite protection
        if not self.overwrite:
            if Path(self.path_target).exists():
                raise FileExistsError(f"Path already exists: {self.path_target} and overwrite is False.")

        ## Create managers list
        managers = []

        ## Create the lock file
        if file_lock:
            self.lock = FileLock_AutoDeleting(
                self.path_lock, 
                timeout=timeout_lock, 
                force=force_release_lock, 
                force_acquire=force_acquire_lock,
            )
            managers.append(self.lock)

        ## Create the atomic saver
        if self.safe_save:
            self.atomic_saver = AtomicSaver(
                path_target=self.path_target,
                path_temp=self.path_temp,
                overwrite=overwrite,
                delete_temp_on_error=delete_temp_on_error,
            )
            managers.append(self.atomic_saver)
        else:
            ## If not safe_save, delete the target path if it exists
            if self.overwrite:
                if Path(self.path_target).exists():
                    delete_file_or_folder(self.path_target)
            ## Set path_temp to path_target
            self.path_temp = self.path_target

        ## Initialize the MultiContextManager
        super().__init__(*managers)

    def __enter__(self):
        """
        Enters the context manager and returns the path to the temporary file.
        """
        super().__enter__()
        return self.path_temp

    def __str__(self):
        return f"SafeSaver(path_target={self.path_target}, path_temp={self.path_temp}, path_lock={self.path_lock}, overwrite={self.overwrite}, safe_save={self.safe_save}, delete_temp_on_error={self.delete_temp_on_error})"
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
    - Lists, tuples, and sets are saved as folders with elements saved as files
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
    - if folder represents a list, tuple, or set:
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

from . import saving_loading_functions as slf


VERSION_RICHFILE = "0.1.0"
FILENAME_METADATA = ".metadata.richfile"
JSON_INDENT = 4

REQUIREMENTS = {
    ## Required top-level keys for the metadata file
    "keys_metadata": [
        "elements",
        "type",
        "library",
        "version",
        "version_richfile",
    ],
    ## Supported container types
    "types_container": [
        "list",
        "tuple",
        "set",
        "dict",
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
    Loads the folder from the given path. Used for types: list, tuple, set, dict.
    """
    metadata = load_folder_metadata(path_dir=path, name_metadata=FILENAME_METADATA, check=check)

    if check:
        names_path_elements = [p.name for p in Path(path).iterdir() if p.name != FILENAME_METADATA]
        # Check that index values are unique
        indices = [value["index"] for value in metadata["elements"].values()]
        if len(indices) != len(set(indices)):
            raise ValueError("Indices in metadata are not unique.")
        if metadata["library"] != "python":
            raise ValueError("Only 'python' library supported for container types.")
        if not is_version_compatible(metadata["version"], slf.VERSIONS_SUPPORTED["python"]):
            raise ValueError(f"Python version {metadata['version']} not supported.")
        
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
    elif metadata["type"] == "dict":
        ## Make sure that all elements are DictItem types
        if check:
            if not all(isinstance(element, DictItem) for element in elements):
                raise TypeError("All elements in a dict must be of type DictItem.")
        elements = {element.key: element.value for element in elements}  ## Unpack the DictItems
    elif metadata["type"] == "dict_item":
        ## Make sure that there are exactly 2 elements
        if check:
            if len(elements) != 2:
                raise ValueError(f"DictItem must contain exactly 2 elements. Found {len(elements)}.")
        elements = DictItem(key=elements[0], value=elements[1])
    else:
        raise ValueError(f"Type {metadata['type']} not supported.")
    
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
        if not is_version_compatible(metadata["version_richfile"], slf.VERSIONS_SUPPORTED["richfile"]):
            raise ValueError(f"RichFile version {metadata['version_richfile']} not supported.")
        for element_name, meta_element in metadata["elements"].items():
            missing_keys = set(REQUIREMENTS["keys_element"]) - set(meta_element.keys())
            if missing_keys:
                raise KeyError(f"Element '{element_name}' is missing keys: {missing_keys}")

    return metadata


def _prepare_element_loading(
    path: Union[str, Path],
    metadata: Dict,
    check: bool,
) -> None:
    """
    Performs checks and preparations for loading a file.
    """
    if check:
        if not is_version_compatible(metadata["version"], slf.VERSIONS_SUPPORTED[metadata["library"]]):
            raise ValueError(f"Version {metadata['version']} not supported.")
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
    type_object, props = _get_obj_properties(obj=obj, type_lookup=type_lookup)

    _prepare_save_path(path=path, overwrite=overwrite, mkdir=True)
    if check:
        library_version = _get_library_version(props["library"])
        if props["library"] in slf.VERSIONS_SUPPORTED:
            if not is_version_compatible(library_version, slf.VERSIONS_SUPPORTED[props["library"]]):
                raise ValueError(f"Library {props['library']} version {library_version} not supported.")        

    ## Get function_save
    function_save = props["function_save"]
    ## Check parameters accepted by function: obj, path, type_lookup, obj_type, check, overwrite, name_dict_items
    sig = inspect.signature(function_save)
    args_available = {
        "obj": obj,
        "path": path,
        "type_lookup": type_lookup,
        "obj_type": type_object,
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
    obj: Union[List, Tuple, set, dict, 'DictItem'],
    path: Union[str, Path],
    obj_type: str,
    type_lookup: Dict,
    check: bool = True,
    overwrite: bool = False,
    name_dict_items: bool = True,
) -> None:
    """
    Saves a list, tuple, or set to the given directory.
    """
    if obj_type == "dict":
        obj = [DictItem(key=key, value=value) for key, value in obj.items()]

    metadata_elements = {}
    for idx, element in enumerate(obj):
        type_element, props = _get_obj_properties(obj=element, type_lookup=type_lookup)
        name_element = f"{idx}.{props['suffix']}"  ## Make name the index and add suffix
        if name_dict_items:
            if type_element == "dict_item":
                if isinstance(element.key, str):
                    name_element = f"{element.key}.{props['suffix']}"
            elif obj_type == "dict_item":
                name_element = f"{['key', 'value'][idx]}.{props['suffix']}"

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
        "type": obj_type,
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
    if library == "python":
        return _get_python_version()
    else:
        try:
            return importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            try:
                ## Try to import the library and get the version from the __version__ attribute
                return importlib.import_module(library).__version__
            except AttributeError as e:
                raise ValueError(f"Library {library} not found. Error: {e}")
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


def _get_obj_properties(obj: Any, type_lookup: Dict) -> Tuple[str, Dict]:
    """
    Returns the richfile type and properties of the object.
    """
    ## Go through the TYPE_LOOKUP dictionary to find the type of the object. isinstance checks for inherited classes as well.
    ### Check with 'is' first to avoid inheritance issues (ex. bool is a subclass of int)
    for type_, props in type_lookup.items():   
        if type(obj) is props["object_type"]:
            return type_, props
    ### If 'is' check fails, use isinstance which checks for inherited classes as well.
    for type_, props in type_lookup.items():
        if isinstance(obj, props["object_type"]):
            return type_, props
    raise TypeError(f"Type {type(obj)} not supported.")

def _lookup_richfile_type_from_type(
    type_: type,
    type_lookup: Dict,
) -> str:
    """
    Returns the richfile type of the object.
    """
    for type_name, props in type_lookup.items():
        if type_ is props["object_type"]:
            return type_name
    for type_name, props in type_lookup.items():
        if isinstance(type_, props["object_type"]):
            return type_name
    raise TypeError(f"Type {type_} not supported.")


####################################################################################################
#################################### HIGH-LEVEL CLASS ##############################################
####################################################################################################

class RichFile:
    """
    High-level class for handling reading and writing objects in the RichFile format.
    Allows customization of loading and saving functions, and setting additional parameters.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        check: Optional[bool] = True,
    ):
        self.path = path
        self.check = check

        self.type_lookup = copy.deepcopy(slf.TYPE_LOOKUP)
        self.params_load = {}
        self.params_save = {}
    
    def save(
        self, 
        obj: Any, 
        path: Optional[Union[str, Path]] = None, 
        check: Optional[bool] = None, 
        overwrite: Optional[bool] = False,
        name_dict_items: bool = True,
    ) -> None:
        path = self.path if path is None else path
        check = self.check if check is None else check
        if (path is None) or (not isinstance(check, bool)):
            raise ValueError("`path` [str, Path] and `check` [bool] must be specified.")

        save_object(
            obj, 
            path, 
            check=check, 
            overwrite=overwrite, 
            type_lookup=self.type_lookup,
            name_dict_items=name_dict_items,
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
            return load_folder(
                path=path,
                type_lookup=type_lookup,
                check=check,
            )
        else:
            metadata_folder = load_folder_metadata(path_dir=str(Path(path).parent), check=check)
            metadata_element = metadata_folder["elements"][Path(path).name]
            return load_element(
                path=path,
                metadata=metadata_element,
                type_lookup=type_lookup,
                check=check,
            )

    def set_load_function(
        self, 
        type_: Union[str, type], 
        function: Callable,
    ) -> None:
        """
        Sets a custom load function for a specific type.
        """
        name_type = _lookup_richfile_type_from_type(type_=type_, type_lookup=self.type_lookup) if type(type_) is not str else type_
            
        if name_type in self.type_lookup:
            self.type_lookup[name_type]['function_load'] = function
        else:
            raise KeyError(f"Type {name_type} not found in type lookup.")

    def set_save_function(        self, 
        type_: Union[str, type], 
        function: Callable,
    ) -> None:
        """
        Sets a custom save function for a specific type.
        """
        name_type = _lookup_richfile_type_from_type(type_=type_, type_lookup=self.type_lookup) if type(type_) is not str else type_

        if name_type in self.type_lookup:
            self.type_lookup[name_type]['function_save'] = function
        else:
            raise KeyError(f"Type {name_type} not found in type lookup.")

    def register_type(
        self, 
        type_name: str, 
        function_load: Callable, 
        function_save: Callable, 
        object_type: type, 
        suffix: str, 
        library: str,
    ):
        """
        Registers a new type with custom loading and saving functions.
        """
        self.type_lookup[type_name] = {
            'function_load': function_load,
            'function_save': function_save,
            'object_type': object_type,
            'suffix': suffix,
            'library': library,
        }

    def set_load_kwargs(
        self, 
        type_: Union[str, type], 
        **kwargs,
    ) -> None:
        """
        Sets additional parameters for the load function of a specific type.
        """
        name_type = _lookup_richfile_type_from_type(type_=type_, type_lookup=self.type_lookup) if type(type_) is not str else type_

        ## Partial the function with the parameters
        if name_type in self.type_lookup:
            self.type_lookup[name_type]['function_load'] = functools.partial(
                self.type_lookup[name_type]['function_load'],
                **kwargs,
            )
        else:
            raise KeyError(f"Type {name_type} not found in type lookup.")
        
        self.params_load[name_type] = kwargs

    def set_save_kwargs(
        self,
        type_: Union[str, type], 
        **kwargs,
    ) -> None:
        """
        Sets additional parameters for the save function of a specific type.
        """
        name_type = _lookup_richfile_type_from_type(type_=type_, type_lookup=self.type_lookup) if type(type_) is not str else type_

        ## Partial the function with the parameters
        if name_type in self.type_lookup:
            self.type_lookup[name_type]['function_save'] = functools.partial(
                self.type_lookup[name_type]['function_save'],
                **kwargs,
            )
        else:
            raise KeyError(f"Type {name_type} not found in type lookup.")
        self.params_save[name_type] = kwargs

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
                if value['type'] in ["list", "tuple", "set", "dict", "dict_item"]:
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
        List, tuple, and set items will be printed as a list of items.
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
                if value['type'] in ["list", "tuple", "set", "dict", "dict_item"]:
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
    
    def __repr__(self):
        self.view_tree()
        return f"RichFileHandler(path={self.path}, check={self.check}), params_load={self.params_load}), params_save={self.params_save})"
    
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
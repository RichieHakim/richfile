from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pathlib import Path
import copy
import warnings

from .. import FILENAME_METADATA, FILENAME_TYPELOOKUP, JSON_INDENT
from .. import util
from . import helpers


class DirectoryBackend:
    """
    Backend implementation for directory-based richfile storage.
    """

    def save(
        self,
        richfile: "util.RichFile",
        obj: Any,
        path: Union[str, Path],
        check: bool,
        overwrite: bool,
        name_dict_items: bool,
        save_type_lookup: bool,
    ) -> None:
        util.save_object(
            obj=obj,
            path=path,
            type_lookup=richfile.type_lookup,
            check=check,
            overwrite=overwrite,
            name_dict_items=name_dict_items,
        )

        if save_type_lookup and Path(path).is_dir():
            path_type_lookup = str(Path(path) / FILENAME_TYPELOOKUP)
            util.save_json(
                obj=helpers.serialize_type_lookup(
                    type_lookup_properties=richfile.type_lookup.properties,
                ),
                path=path_type_lookup,
                indent=JSON_INDENT,
            )

    def load(
        self,
        richfile: "util.RichFile",
        path: Union[str, Path],
        type_lookup: Dict,
        check: bool,
    ) -> Any:
        if not (Path(path).parent / FILENAME_METADATA).exists():
            if Path(path).is_dir():
                if not (Path(path) / FILENAME_METADATA).exists():
                    raise FileNotFoundError(
                        f"Metadata file {FILENAME_METADATA} not found in directory {path}."
                    )
                metadata_dir = util.load_folder_metadata(path_dir=path, check=check)
                metadata_obj = {
                    "type": metadata_dir["type"],
                    "library": metadata_dir["library"],
                    "version": metadata_dir["version"],
                    "index": 0,
                }
                return util.load_element(
                    path=path,
                    metadata=metadata_obj,
                    type_lookup=type_lookup,
                    check=check,
                )
            elif Path(path).is_file():
                props = type_lookup[type(Path(path).suffix)]
                metadata_obj = {
                    "type": props["type_name"],
                    "library": props["library"],
                    "version": util._get_library_version(library=props["library"]),
                    "index": 0,
                }
                return util.load_element(
                    path=path,
                    metadata=metadata_obj,
                    type_lookup=type_lookup,
                    check=check,
                )
            else:
                raise FileNotFoundError(f"Path {path} not found.")
        else:
            metadata_folder = util.load_folder_metadata(
                path_dir=str(Path(path).parent), check=check
            )
            metadata_element = metadata_folder["elements"][Path(path).name]
            return util.load_element(
                path=path,
                metadata=metadata_element,
                type_lookup=type_lookup,
                check=check,
            )

    def get_metadata(
        self,
        richfile: "util.RichFile",
        path_dir: Union[str, Path],
    ) -> Dict:
        return util.load_folder_metadata(path_dir=path_dir, check=richfile.check)

    def list_elements(
        self,
        richfile: "util.RichFile",
        path_dir: Union[str, Path],
    ) -> List[str]:
        metadata = self.get_metadata(richfile=richfile, path_dir=path_dir)
        return list(metadata["elements"].keys())

    def view_directory_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        path = richfile.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")

        def _view_tree(path_inner, level=0):
            metadata = self.get_metadata(richfile=richfile, path_dir=path_inner)
            for name, value in metadata["elements"].items():
                print("|   " * level + "├── " + f"{name} ({value['type']})")
                if value["type"] in [
                    "list",
                    "tuple",
                    "set",
                    "frozenset",
                    "dict",
                    "dict_item",
                ]:
                    _view_tree(path_inner=str(Path(path_inner) / name), level=level + 1)
            print("|   " * level)

        if Path(path).is_dir():
            print(
                f"Viewing tree structure of richfile at path: {path} ({self.get_metadata(richfile=richfile, path_dir=path)['type']})"
            )
            _view_tree(path)
        elif Path(path).is_file():
            metadata_folder = util.load_folder_metadata(
                path_dir=str(Path(path).parent), check=richfile.check
            )
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            print(f"Viewing element at path: {path} ({metadata_element['type']})")
        else:
            raise FileNotFoundError(f"Path {path} not found.")

    def view_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
        show_filenames: bool = False,
    ) -> None:
        sf = show_filenames
        path = richfile.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")

        def _view_tree(path_inner, level=0):
            metadata = self.get_metadata(richfile=richfile, path_dir=path_inner)
            for name, value in metadata["elements"].items():
                if value["type"] == "dict_item":
                    metadata_item = self.get_metadata(
                        richfile=richfile, path_dir=str(Path(path_inner) / name)
                    )
                    names_meta_sorted = util._sort_element_names_by_index(
                        metadata=metadata_item
                    )
                    if len(names_meta_sorted) != 2:
                        raise ValueError(
                            "DictItem must contain exactly 2 elements: key='0.json' and value=another element."
                        )
                    name_key, name_value = names_meta_sorted
                    metadata_key = metadata_item["elements"][name_key]
                    metadata_value = metadata_item["elements"][name_value]
                    key = util.load_element(
                        path=str(Path(path_inner) / name / name_key),
                        metadata=metadata_key,
                        type_lookup=richfile.type_lookup,
                    )
                    print(
                        "|    " * level
                        + "├── "
                        + f"'{key}': {(name_value if sf else '')}  ({metadata_value['type']})"
                    )
                    if metadata_value["type"] in [
                        "list",
                        "tuple",
                        "set",
                        "dict",
                        "dict_item",
                    ]:
                        _view_tree(
                            path_inner=str(Path(path_inner) / name / name_value),
                            level=level + 1,
                        )
                elif value["type"] in ["list", "tuple", "set", "dict"]:
                    print(
                        "|    " * level
                        + "├── "
                        + f"{(name if sf else '')}  ({value['type']})"
                    )
                    _view_tree(path_inner=str(Path(path_inner) / name), level=level + 1)
                else:
                    print(
                        "|    " * level
                        + "├── "
                        + f"{(name if sf else '')}  ({value['type']})"
                    )
            print("|    " * level)

        if Path(path).is_dir():
            print(
                f"Path: {path} ({self.get_metadata(richfile=richfile, path_dir=path)['type']})"
            )
            _view_tree(path)
        elif Path(path).is_file():
            metadata_folder = util.load_folder_metadata(
                path_dir=str(Path(path).parent), check=richfile.check
            )
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            print(f"Path: {path} ({metadata_element['type']})")
        else:
            raise FileNotFoundError(f"Path {path} not found.")

    def get_metadata_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        path = richfile.path if path is None else path
        if path is None:
            raise ValueError("`path` [str, Path] must be specified.")

        def _get_metadata_tree(path_inner):
            metadata = self.get_metadata(richfile=richfile, path_dir=path_inner)
            out = {
                "metadata": metadata,
                "elements": {},
            }
            for name, value in metadata["elements"].items():
                if value["type"] in [
                    "list",
                    "tuple",
                    "set",
                    "frozenset",
                    "dict",
                    "dict_item",
                ]:
                    out["elements"][name] = _get_metadata_tree(
                        str(Path(path_inner) / name)
                    )
                else:
                    out["elements"][name] = value
            return out

        if Path(path).is_dir():
            return _get_metadata_tree(path)
        elif Path(path).is_file():
            metadata_folder = util.load_folder_metadata(
                path_dir=str(Path(path).parent), check=richfile.check
            )
            name_element = Path(path).name
            metadata_element = metadata_folder["elements"][name_element]
            return metadata_element
        else:
            raise FileNotFoundError(f"Path {path} not found.")

    def getitem(self, richfile: "util.RichFile", key: Any) -> "util.RichFile":
        if isinstance(key, str):
            metadata = self.get_metadata(richfile=richfile, path_dir=richfile.path)
            if metadata["type"] != "dict":
                raise ValueError("Path must be a dict to load by key.")
            names_meta_sorted = util._sort_element_names_by_index(metadata=metadata)
            for name in names_meta_sorted:
                metadata_item = self.get_metadata(
                    richfile=richfile, path_dir=str(Path(richfile.path) / name)
                )
                if not (metadata["elements"][name]["type"] == "dict_item"):
                    raise ValueError(
                        f"Found element with type {metadata['elements'][name]['type']}. Expected 'dict_item'."
                    )
                if not (metadata_item["type"] == "dict_item"):
                    raise ValueError(
                        f"Found element with type {metadata_item['type']}. Expected 'dict_item'."
                    )
                if not len(metadata_item["elements"]) == 2:
                    raise ValueError(
                        "DictItem must contain exactly 2 elements: key and value."
                    )
                names_meta_sorted_item = util._sort_element_names_by_index(
                    metadata=metadata_item
                )
                name_key, name_value = names_meta_sorted_item
                if metadata_item["elements"][name_key]["type"] == "str":
                    key_loaded = util.load_element(
                        path=str(Path(richfile.path) / name / name_key),
                        metadata=metadata_item["elements"][name_key],
                        type_lookup=richfile.type_lookup,
                    )
                    if key_loaded == key:
                        out = copy.deepcopy(richfile)
                        out.path = str(Path(richfile.path) / name / name_value)
                        return out

        elif isinstance(key, int):
            metadata = self.get_metadata(richfile=richfile, path_dir=richfile.path)
            if metadata["type"] not in ["list", "tuple"]:
                raise ValueError("Path must be a list or tuple to load by index.")
            for name, value in metadata["elements"].items():
                if value["index"] == key:
                    out = copy.deepcopy(richfile)
                    out.path = str(Path(richfile.path) / name)
                    return out
        else:
            raise ValueError("__getitem__ only supports str and int keys.")

        raise KeyError(f"Key {key} not found.")

    def keys(self, richfile: "util.RichFile") -> List[str]:
        try:
            metadata = self.get_metadata(richfile=richfile, path_dir=richfile.path)
            names_elements_raw = list(metadata["elements"].keys())
            return [".".join(name.split(".")[:-1]) for name in names_elements_raw]
        except FileNotFoundError:
            return []
        except Exception as exc:
            if richfile.check:
                raise
            warnings.warn(
                "Path element failed to load metadata or doesn't have .keys() method. "
                f"Error: {exc}"
            )
            return []

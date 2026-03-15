from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pathlib import Path
import copy
import inspect
import json
import tempfile
import warnings

from .. import (
    FILENAME_TYPELOOKUP,
    NAMES_EXTRA_FILES_ALLOWED,
    PYTHON_VERSIONS_SUPPORTED,
    VERSIONS_RICHFILE_SUPPORTED,
)
from .. import util
from . import helpers


_NATIVE_CONTAINER_TYPES: Set[str] = {
    "list",
    "tuple",
    "set",
    "frozenset",
    "dict",
    "dict_item",
}

_NATIVE_JSON_TYPES: Set[str] = {
    "float",
    "int",
    "str",
    "bool",
    "None",
}


class ArchiveBackendBase:
    """
    Shared archive backend behavior for ZIP/TAR backends.

    Subclasses implement backend-specific archive reader/writer primitives.
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
        """
        Serialize a Python object into an archive file.

        Args:
            richfile (util.RichFile):
                The RichFile instance driving the save operation.
            obj (Any):
                The Python object to serialize.
            path (Union[str, Path]):
                Filesystem path for the output archive.
            check (bool):
                Whether to run validation checks during serialization.
            overwrite (bool):
                Unused; overwrite behavior is handled by RichFile.SafeSaver.
            name_dict_items (bool):
                Whether to name dict entries by their string keys.
            save_type_lookup (bool):
                Whether to embed the type lookup table in the archive.
        """
        del overwrite  # overwrite/atomic behavior is handled by RichFile.SafeSaver.

        path_archive = str(path)
        path_in_archive = helpers.normalize_archive_path(
            path_in_archive=richfile._path_in_archive
        )
        if path_in_archive != "":
            raise NotImplementedError(
                f"{self._backend_name()} backend currently supports only root-object save(). "
                "Saving to nested subpaths is out of scope for this phase."
            )

        Path(path_archive).parent.mkdir(parents=True, exist_ok=True)

        entries_files: Dict[str, bytes] = {}
        entries_dirs: Set[str] = set()

        self._serialize_object(
            obj=obj,
            path_in_archive=path_in_archive,
            type_lookup=richfile.type_lookup,
            check=check,
            name_dict_items=name_dict_items,
            entries_files=entries_files,
            entries_dirs=entries_dirs,
        )
        if save_type_lookup:
            entries_files[FILENAME_TYPELOOKUP] = json.dumps(
                helpers.serialize_type_lookup(
                    type_lookup_properties=richfile.type_lookup.properties,
                )
            ).encode("utf-8")

        with self._open_writer(path_archive=path_archive) as writer:
            for path_dir in sorted(entries_dirs):
                self._write_dir_member(writer=writer, member_name=path_dir)
            for path_member, payload in sorted(entries_files.items()):
                self._write_file_member(
                    writer=writer,
                    member_name=path_member,
                    data=payload,
                )

    def load(
        self,
        richfile: "util.RichFile",
        path: Union[str, Path],
        type_lookup: Dict,
        check: bool,
    ) -> Any:
        """
        Deserialize a Python object from an archive file.

        Args:
            richfile (util.RichFile):
                The RichFile instance driving the load operation.
            path (Union[str, Path]):
                Filesystem path of the archive to read.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks during deserialization.

        Returns:
            (Any):
                obj (Any):
                    The reconstructed Python object.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=path,
            default_to_current_subpath=True,
        )

        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            if path_in_archive == "":
                metadata_root = self._load_metadata_row(
                    reader=reader,
                    index=index,
                    path_in_archive="",
                    check=check,
                )
                metadata_obj = {
                    "type": metadata_root["type"],
                    "library": metadata_root["library"],
                    "version": metadata_root["version"],
                    "index": 0,
                }
                return self._load_object(
                    reader=reader,
                    index=index,
                    path_in_archive="",
                    metadata=metadata_obj,
                    type_lookup=type_lookup,
                    check=check,
                )

            path_parent, name_element = helpers.split_archive_path(
                path_in_archive=path_in_archive
            )
            metadata_parent = self._load_metadata_row(
                reader=reader,
                index=index,
                path_in_archive=path_parent,
                check=check,
            )
            if name_element not in metadata_parent["elements"]:
                raise KeyError(
                    f"Element '{name_element}' not found under archive path '{path_parent}'."
                )
            metadata_element = metadata_parent["elements"][name_element]
            return self._load_object(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                metadata=metadata_element,
                type_lookup=type_lookup,
                check=check,
            )

    def get_metadata(
        self,
        richfile: "util.RichFile",
        path_dir: Optional[Union[str, Path]],
    ) -> Dict:
        """
        Retrieve the metadata dictionary for a node in the archive.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            path_dir (Optional[Union[str, Path]]):
                Path to the archive or internal subpath. Uses the
                RichFile's current path if None.

        Returns:
            (Dict):
                metadata (Dict):
                    The metadata dictionary for the specified node.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=path_dir,
            default_to_current_subpath=True,
        )
        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            return self._load_metadata_row(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                check=richfile.check,
            )

    def list_elements(
        self,
        richfile: "util.RichFile",
        path_dir: Optional[Union[str, Path]],
    ) -> List[str]:
        """
        List the element names of a container node in the archive.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            path_dir (Optional[Union[str, Path]]):
                Path to the archive or internal subpath. Uses the
                RichFile's current path if None.

        Returns:
            (List[str]):
                names (List[str]):
                    Ordered list of child element names.
        """
        metadata = self.get_metadata(richfile=richfile, path_dir=path_dir)
        return list(metadata["elements"].keys())

    def view_directory_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Print a directory-style tree of the archive contents to stdout.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            path (Optional[Union[str, Path]]):
                Path to the archive or internal subpath. Uses the
                RichFile's current path if None.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=path,
            default_to_current_subpath=True,
        )
        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            metadata_root = self._load_metadata_row(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                check=richfile.check,
            )
            print(
                "Viewing tree structure of richfile at path: "
                f"{path_archive} [{path_in_archive or '/'}] ({metadata_root['type']})"
            )
            self._print_directory_tree_recursive(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                type_lookup=richfile.type_lookup,
                check=richfile.check,
                level=0,
            )

    def view_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
        show_filenames: bool = False,
    ) -> None:
        """
        Print a logical tree of the archive contents to stdout, resolving
        dict keys to their values.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            path (Optional[Union[str, Path]]):
                Path to the archive or internal subpath. Uses the
                RichFile's current path if None.
            show_filenames (bool):
                Whether to include raw archive member names in the output.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=path,
            default_to_current_subpath=True,
        )
        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            metadata_root = self._load_metadata_row(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                check=richfile.check,
            )
            print(f"Path: {path_archive} [{path_in_archive or '/'}] ({metadata_root['type']})")
            self._print_tree_recursive(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                type_lookup=richfile.type_lookup,
                check=richfile.check,
                show_filenames=show_filenames,
                level=0,
            )

    def get_metadata_tree(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Build the full nested metadata tree for the archive.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            path (Optional[Union[str, Path]]):
                Path to the archive or internal subpath. Uses the
                RichFile's current path if None.

        Returns:
            (Dict):
                tree (Dict):
                    Nested dictionary with ``"metadata"`` and ``"elements"``
                    keys at each level.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=path,
            default_to_current_subpath=True,
        )
        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            return self._get_metadata_tree_recursive(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                check=richfile.check,
            )

    def getitem(self, richfile: "util.RichFile", key: Any) -> "util.RichFile":
        """
        Return a new RichFile pointing to a child element by key or index.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.
            key (Any):
                String key (for dicts) or integer index (for lists/tuples).

        Returns:
            (util.RichFile):
                child (util.RichFile):
                    A new RichFile scoped to the matched child element.
        """
        path_archive, path_in_archive = self._resolve_archive_and_internal_path(
            richfile=richfile,
            path=richfile.path,
            default_to_current_subpath=True,
        )
        with self._open_reader(path_archive=path_archive) as reader:
            index = self._get_or_build_index(
                richfile=richfile,
                path_archive=path_archive,
                reader=reader,
            )
            metadata = self._load_metadata_row(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                check=richfile.check,
            )

            if isinstance(key, str):
                if metadata["type"] != "dict":
                    raise ValueError("Path must be a dict to load by key.")
                names_meta_sorted = util._sort_element_names_by_index(metadata=metadata)
                for name in names_meta_sorted:
                    metadata_item = self._load_metadata_row(
                        reader=reader,
                        index=index,
                        path_in_archive=helpers.join_archive_path(path_in_archive, name),
                        check=richfile.check,
                    )
                    if not (metadata["elements"][name]["type"] == "dict_item"):
                        raise ValueError(
                            f"Found element with type {metadata['elements'][name]['type']}. Expected 'dict_item'."
                        )
                    if not (metadata_item["type"] == "dict_item"):
                        raise ValueError(
                            f"Found element with type {metadata_item['type']}. Expected 'dict_item'."
                        )
                    names_meta_sorted_item = util._sort_element_names_by_index(
                        metadata=metadata_item
                    )
                    if len(names_meta_sorted_item) != 2:
                        raise ValueError(
                            "DictItem must contain exactly 2 elements: key and value."
                        )
                    name_key, name_value = names_meta_sorted_item
                    metadata_key = metadata_item["elements"][name_key]
                    if metadata_key["type"] == "str":
                        key_loaded = self._load_object(
                            reader=reader,
                            index=index,
                            path_in_archive=helpers.join_archive_path(
                                helpers.join_archive_path(path_in_archive, name), name_key
                            ),
                            metadata=metadata_key,
                            type_lookup=richfile.type_lookup,
                            check=richfile.check,
                        )
                        if key_loaded == key:
                            out = copy.deepcopy(richfile)
                            out.path = path_archive
                            out._path_in_archive = helpers.join_archive_path(
                                helpers.join_archive_path(path_in_archive, name),
                                name_value,
                            )
                            return out

            elif isinstance(key, int):
                if metadata["type"] not in ["list", "tuple"]:
                    raise ValueError("Path must be a list or tuple to load by index.")
                for name, value in metadata["elements"].items():
                    if value["index"] == key:
                        out = copy.deepcopy(richfile)
                        out.path = path_archive
                        out._path_in_archive = helpers.join_archive_path(
                            path_in_archive,
                            name,
                        )
                        return out
            else:
                raise ValueError("__getitem__ only supports str and int keys.")

        raise KeyError(f"Key {key} not found.")

    def keys(self, richfile: "util.RichFile") -> List[str]:
        """
        Return the element names (without suffixes) at the current archive path.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing archive context.

        Returns:
            (List[str]):
                keys (List[str]):
                    Element names with file-extension suffixes stripped.
        """
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

    def _backend_name(self) -> str:
        """
        Return the human-readable name of this backend (e.g. ``"zip"``, ``"tar"``).

        Subclasses must override this method.

        Returns:
            (str):
                name (str):
                    The backend identifier string.
        """
        raise NotImplementedError

    def _open_reader(self, path_archive: Union[str, Path]):
        """
        Open the archive for reading and return a context-managed reader object.

        Subclasses must override this method.

        Args:
            path_archive (Union[str, Path]):
                Filesystem path of the archive to open.

        Returns:
            A context manager yielding a backend-specific reader handle.
        """
        raise NotImplementedError

    def _open_writer(self, path_archive: Union[str, Path]):
        """
        Open the archive for writing and return a context-managed writer object.

        Subclasses must override this method.

        Args:
            path_archive (Union[str, Path]):
                Filesystem path of the archive to create or overwrite.

        Returns:
            A context manager yielding a backend-specific writer handle.
        """
        raise NotImplementedError

    def _iter_raw_members(self, reader) -> List[str]:
        """
        List all raw member names in the archive.

        Subclasses must override this method.

        Args:
            reader:
                The backend-specific reader handle.

        Returns:
            (List[str]):
                names (List[str]):
                    Raw (unnormalized) member names from the archive.
        """
        raise NotImplementedError

    def _is_raw_member_dir(self, reader, raw_name: str) -> bool:
        """
        Check whether a raw archive member represents a directory.

        Subclasses must override this method.

        Args:
            reader:
                The backend-specific reader handle.
            raw_name (str):
                The raw member name to check.

        Returns:
            (bool):
                is_dir (bool):
                    True if the member is a directory entry.
        """
        raise NotImplementedError

    def _read_raw_member_bytes(self, reader, raw_name: str) -> bytes:
        """
        Read the raw bytes of a file member from the archive.

        Subclasses must override this method.

        Args:
            reader:
                The backend-specific reader handle.
            raw_name (str):
                The raw member name to read.

        Returns:
            (bytes):
                data (bytes):
                    The file contents as bytes.
        """
        raise NotImplementedError

    def _write_file_member(self, writer, member_name: str, data: bytes) -> None:
        """
        Write a file entry into the archive.

        Subclasses must override this method.

        Args:
            writer:
                The backend-specific writer handle.
            member_name (str):
                Normalized archive path for the file.
            data (bytes):
                The file content to write.
        """
        raise NotImplementedError

    def _write_dir_member(self, writer, member_name: str) -> None:
        """
        Write a directory entry into the archive.

        Subclasses must override this method.

        Args:
            writer:
                The backend-specific writer handle.
            member_name (str):
                Normalized archive path for the directory.
        """
        raise NotImplementedError

    def _build_index(self, reader) -> Dict[str, Any]:
        """
        Scan the archive and build a lookup index of files, directories,
        and parent-child relationships.

        Args:
            reader:
                The backend-specific reader handle.

        Returns:
            (Dict[str, Any]):
                index (Dict[str, Any]):
                    Dictionary with keys ``"files"``, ``"dirs"``, and
                    ``"children"`` mapping normalized archive paths.
        """
        files: Dict[str, str] = {}
        dirs: Set[str] = set()
        for raw_name in self._iter_raw_members(reader=reader):
            try:
                norm_name = helpers.validate_archive_path(
                    path_in_archive=raw_name,
                    allow_empty=True,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid archive member path '{raw_name}' in {self._backend_name()} archive."
                ) from exc
            if norm_name == "":
                continue
            if self._is_raw_member_dir(reader=reader, raw_name=raw_name):
                dirs.add(norm_name)
            else:
                files[norm_name] = raw_name

        for path_file in list(files.keys()):
            path_parent, _ = helpers.split_archive_path(path_in_archive=path_file)
            while path_parent != "":
                dirs.add(path_parent)
                path_parent, _ = helpers.split_archive_path(path_in_archive=path_parent)

        children = self._build_children_index(names_all=set(files.keys()) | set(dirs))
        return {
            "files": files,
            "dirs": dirs,
            "children": children,
        }

    def _build_children_index(self, names_all: Set[str]) -> Dict[str, Set[str]]:
        """
        Build a direct-child lookup table for all known archive names.

        Args:
            names_all (Set[str]):
                Normalized archive paths for files and directories.

        Returns:
            (Dict[str, Set[str]]):
                children_by_parent (Dict[str, Set[str]]):
                    Mapping ``parent_path -> direct child name set``.
        """
        children_by_parent: Dict[str, Set[str]] = {}
        for path_name in names_all:
            parts = path_name.split("/")
            path_parent = ""
            for part in parts:
                children_by_parent.setdefault(path_parent, set()).add(part)
                path_parent = part if path_parent == "" else f"{path_parent}/{part}"
        return children_by_parent

    def _read_member_bytes(self, reader, index: Dict[str, Any], member_name: str) -> bytes:
        """
        Read the bytes of a normalized archive member, using the index for lookup.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            member_name (str):
                Normalized archive path of the file to read.

        Returns:
            (bytes):
                data (bytes):
                    The file contents as bytes.
        """
        member_name = helpers.validate_archive_path(
            path_in_archive=member_name,
            allow_empty=False,
        )
        if member_name not in index["files"]:
            raise FileNotFoundError(
                f"Archive member '{member_name}' not found in {self._backend_name()} archive."
            )
        return self._read_raw_member_bytes(
            reader=reader,
            raw_name=index["files"][member_name],
        )

    def _path_exists(self, index: Dict[str, Any], path_in_archive: str) -> bool:
        """
        Check whether a normalized path exists in the archive index.

        Args:
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path to check.

        Returns:
            (bool):
                exists (bool):
                    True if the path exists as a file, directory, or prefix.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=True,
        )
        names_all = set(index["files"].keys()) | set(index["dirs"])
        if path_in_archive == "":
            return len(names_all) > 0
        if path_in_archive in names_all:
            return True
        prefix = f"{path_in_archive}/"
        return any(name.startswith(prefix) for name in names_all)

    def _list_direct_child_names(
        self,
        index: Dict[str, Any],
        path_parent: str,
    ) -> Set[str]:
        """
        Return the direct child names under a parent path, excluding
        metadata and type-lookup files.

        Args:
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_parent (str):
                Normalized archive path of the parent.

        Returns:
            (Set[str]):
                children (Set[str]):
                    Set of direct child names (basenames only).
        """
        path_parent = helpers.validate_archive_path(
            path_in_archive=path_parent,
            allow_empty=True,
        )
        children_raw = index.get("children", {}).get(path_parent, set())
        return {
            child_name
            for child_name in children_raw
            if child_name not in {util.FILENAME_METADATA, FILENAME_TYPELOOKUP}
        }

    def _load_metadata_row(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        check: bool,
    ) -> Dict:
        """
        Load and parse the JSON metadata for a single node in the archive.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the node.
            check (bool):
                Whether to validate required keys and version compatibility.

        Returns:
            (Dict):
                metadata (Dict):
                    Parsed metadata dictionary for the node.
        """
        member_name = helpers.metadata_row_name(path_in_archive=path_in_archive)
        payload = self._read_member_bytes(
            reader=reader,
            index=index,
            member_name=member_name,
        )
        metadata = json.loads(payload.decode("utf-8"))

        if check:
            missing_keys = set(util.REQUIREMENTS["keys_metadata"]) - set(metadata.keys())
            if missing_keys:
                raise KeyError(f"Metadata is missing required keys: {missing_keys}")
            if not util.is_version_compatible(
                version=metadata["version_richfile"],
                rules=VERSIONS_RICHFILE_SUPPORTED,
            ):
                raise ValueError(
                    f"RichFile version {metadata['version_richfile']} not supported."
                )
            for element_name, meta_element in metadata["elements"].items():
                missing_keys = set(util.REQUIREMENTS["keys_element"]) - set(meta_element.keys())
                if missing_keys:
                    raise KeyError(f"Element '{element_name}' is missing keys: {missing_keys}")

        return metadata

    def _check_element_compatibility(
        self,
        metadata: Dict,
        type_lookup: Dict,
        check: bool,
    ) -> None:
        """
        Validate that an element's type and library version are supported
        by the current type lookup.

        Args:
            metadata (Dict):
                Metadata dictionary for the element.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to perform the check. No-op if False.
        """
        if not check:
            return
        if metadata["type"] not in type_lookup:
            raise ValueError(f"Type {metadata['type']} not supported.")

        versions_supported = type_lookup[metadata["type"]]["versions_supported"]
        if versions_supported == []:
            warnings.warn(
                f"Field 'versions_supported' is empty in type_lookup for type {metadata['type']}."
            )
        elif metadata["library"] == "python":
            if not util.is_version_compatible(
                version=metadata["version"],
                rules=PYTHON_VERSIONS_SUPPORTED,
            ):
                raise ValueError(
                    f"Python version '{metadata['version']}' not supported for library 'python' "
                    f"and type '{metadata['type']}'. Version rules: {PYTHON_VERSIONS_SUPPORTED}"
                )
        elif not util.is_version_compatible(
            version=metadata["version"],
            rules=versions_supported,
        ):
            raise ValueError(
                f"Version '{metadata['version']}' not supported for library "
                f"'{metadata['library']}' and type '{metadata['type']}'. "
                f"Version rules: {versions_supported}"
            )

    def _serialize_object(
        self,
        obj: Any,
        path_in_archive: str,
        type_lookup: Dict,
        check: bool,
        name_dict_items: bool,
        entries_files: Dict[str, bytes],
        entries_dirs: Set[str],
    ) -> None:
        """
        Dispatch serialization of an object to the appropriate handler
        based on its type.

        Args:
            obj (Any):
                The Python object to serialize.
            path_in_archive (str):
                Normalized archive path where this object will be stored.
            type_lookup (Dict):
                Mapping of types to their save/load properties.
            check (bool):
                Whether to run validation checks.
            name_dict_items (bool):
                Whether to name dict entries by their string keys.
            entries_files (Dict[str, bytes]):
                Accumulator for file entries to be written.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        props = type_lookup[type(obj)]
        type_name = props["type_name"]

        if type_name in _NATIVE_CONTAINER_TYPES:
            self._serialize_container_native(
                obj=obj,
                path_in_archive=path_in_archive,
                type_name=type_name,
                type_lookup=type_lookup,
                check=check,
                name_dict_items=name_dict_items,
                entries_files=entries_files,
                entries_dirs=entries_dirs,
            )
            return

        if type_name in _NATIVE_JSON_TYPES:
            self._serialize_json_scalar_native(
                obj=obj,
                path_in_archive=path_in_archive,
                entries_files=entries_files,
                entries_dirs=entries_dirs,
            )
            return

        self._serialize_object_bridge(
            obj=obj,
            path_in_archive=path_in_archive,
            type_lookup=type_lookup,
            check=check,
            name_dict_items=name_dict_items,
            entries_files=entries_files,
            entries_dirs=entries_dirs,
        )

    def _serialize_json_scalar_native(
        self,
        obj: Any,
        path_in_archive: str,
        entries_files: Dict[str, bytes],
        entries_dirs: Set[str],
    ) -> None:
        """
        Serialize a JSON-native scalar (int, float, str, bool, None) as
        a JSON-encoded file entry.

        Args:
            obj (Any):
                The scalar value to serialize.
            path_in_archive (str):
                Normalized archive path for the file entry.
            entries_files (Dict[str, bytes]):
                Accumulator for file entries to be written.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=False,
        )
        entries_files[path_in_archive] = json.dumps(obj).encode("utf-8")
        self._add_parent_dirs(path_member=path_in_archive, entries_dirs=entries_dirs)

    def _serialize_container_native(
        self,
        obj: Any,
        path_in_archive: str,
        type_name: str,
        type_lookup: Dict,
        check: bool,
        name_dict_items: bool,
        entries_files: Dict[str, bytes],
        entries_dirs: Set[str],
    ) -> None:
        """
        Serialize a native container type (list, tuple, set, frozenset, dict,
        dict_item) by recursively serializing each child element.

        Args:
            obj (Any):
                The container object to serialize.
            path_in_archive (str):
                Normalized archive path for the container directory.
            type_name (str):
                The resolved type name of the container.
            type_lookup (Dict):
                Mapping of types to their save/load properties.
            check (bool):
                Whether to run validation checks.
            name_dict_items (bool):
                Whether to name dict entries by their string keys.
            entries_files (Dict[str, bytes]):
                Accumulator for file entries to be written.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=True,
        )
        if path_in_archive != "":
            entries_dirs.add(path_in_archive)
            self._add_parent_dirs(path_member=path_in_archive, entries_dirs=entries_dirs)

        if isinstance(obj, dict):
            if check and name_dict_items:
                util._check_case_only_sibling_keys(obj=obj)
            obj = [util.DictItem(key=key, value=value) for key, value in obj.items()]

        metadata_elements = {}
        for idx, element in enumerate(obj):
            try:
                props = type_lookup[type(element)]
                type_element = props["type_name"]
            except TypeError as exc:
                raise TypeError(
                    "Failed to get properties for element.\n"
                    f"Archive path: {path_in_archive}\n"
                    f"Index: {idx}\n"
                    f"Element: {element}\n"
                    f"Container type: {type_name}\n"
                    f"Error: {exc}"
                )

            name_element = f"{idx}.{props['suffix']}"
            if name_dict_items:
                if type_element == "dict_item":
                    if isinstance(element.key, str):
                        name_element = f"{element.key}.{props['suffix']}"
                elif type_name == "dict_item":
                    name_element = f"{['key', 'value'][idx]}.{props['suffix']}"

            util._check_filename_safety(name=name_element, warn=True, raise_error=False)
            path_child = helpers.join_archive_path(path_in_archive, name_element)
            self._serialize_object(
                obj=element,
                path_in_archive=path_child,
                type_lookup=type_lookup,
                check=check,
                name_dict_items=name_dict_items,
                entries_files=entries_files,
                entries_dirs=entries_dirs,
            )
            metadata_elements[name_element] = {
                "type": type_element,
                "library": props["library"],
                "version": util._get_library_version(library=props["library"]),
                "index": idx,
            }

        metadata_container = {
            "elements": metadata_elements,
            "type": type_name,
            "library": "python",
            "version": util._get_python_version(),
            "version_richfile": util.VERSION_RICHFILE,
        }
        path_metadata = helpers.metadata_row_name(path_in_archive=path_in_archive)
        entries_files[path_metadata] = json.dumps(metadata_container).encode("utf-8")
        self._add_parent_dirs(path_member=path_metadata, entries_dirs=entries_dirs)

    def _serialize_object_bridge(
        self,
        obj: Any,
        path_in_archive: str,
        type_lookup: Dict,
        check: bool,
        name_dict_items: bool,
        entries_files: Dict[str, bytes],
        entries_dirs: Set[str],
    ) -> None:
        """
        Serialize a non-native object by delegating to its registered save
        function via a temporary directory, then ingesting the result.

        Args:
            obj (Any):
                The object to serialize.
            path_in_archive (str):
                Normalized archive path for the object.
            type_lookup (Dict):
                Mapping of types to their save/load properties.
            check (bool):
                Whether to run validation checks.
            name_dict_items (bool):
                Whether to name dict entries by their string keys.
            entries_files (Dict[str, bytes]):
                Accumulator for file entries to be written.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        basename = helpers.split_archive_path(path_in_archive=path_in_archive)[1]
        basename = "root" if basename == "" else basename

        with tempfile.TemporaryDirectory() as dir_tmp:
            path_expected = Path(dir_tmp) / basename
            util.save_object(
                obj=obj,
                path=str(path_expected),
                type_lookup=type_lookup,
                check=check,
                overwrite=True,
                name_dict_items=name_dict_items,
            )

            path_created = self._resolve_created_path(
                dir_tmp=dir_tmp,
                path_expected=path_expected,
            )
            self._ingest_filesystem_path(
                path_source=path_created,
                path_in_archive=path_in_archive,
                entries_files=entries_files,
                entries_dirs=entries_dirs,
            )

    def _resolve_created_path(
        self,
        dir_tmp: Union[str, Path],
        path_expected: Path,
    ) -> Path:
        """
        Locate the output created by a custom save function in a temp directory.

        Args:
            dir_tmp (Union[str, Path]):
                The temporary directory where save output was written.
            path_expected (Path):
                The expected output path within the temp directory.

        Returns:
            (Path):
                path_created (Path):
                    The resolved path to the created output.
        """
        if path_expected.exists():
            return path_expected
        candidates = list(Path(dir_tmp).iterdir())
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            "Could not locate output produced by custom save function in temp directory."
        )

    def _ingest_filesystem_path(
        self,
        path_source: Path,
        path_in_archive: str,
        entries_files: Dict[str, bytes],
        entries_dirs: Set[str],
    ) -> None:
        """
        Read a filesystem file or directory tree and add its contents to the
        archive entry accumulators.

        Args:
            path_source (Path):
                Filesystem path of the file or directory to ingest.
            path_in_archive (str):
                Normalized archive path prefix for the ingested entries.
            entries_files (Dict[str, bytes]):
                Accumulator for file entries to be written.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=True,
        )
        if path_source.is_file():
            path_member = helpers.validate_archive_path(
                path_in_archive=path_in_archive,
                allow_empty=False,
            )
            entries_files[path_member] = path_source.read_bytes()
            self._add_parent_dirs(path_member=path_member, entries_dirs=entries_dirs)
            return

        if not path_source.is_dir():
            raise FileNotFoundError(f"Path not found for ingest: {path_source}")

        if path_in_archive != "":
            entries_dirs.add(path_in_archive)
            self._add_parent_dirs(path_member=path_in_archive, entries_dirs=entries_dirs)

        for child in sorted(path_source.rglob("*")):
            rel = child.relative_to(path_source).as_posix()
            member_name = helpers.validate_archive_path(
                path_in_archive=helpers.join_archive_path(path_in_archive, rel),
                allow_empty=False,
            )
            if child.is_dir():
                entries_dirs.add(member_name)
                self._add_parent_dirs(path_member=member_name, entries_dirs=entries_dirs)
            else:
                entries_files[member_name] = child.read_bytes()
                self._add_parent_dirs(path_member=member_name, entries_dirs=entries_dirs)

    def _add_parent_dirs(self, path_member: str, entries_dirs: Set[str]) -> None:
        """
        Register all ancestor directories of a member path in the directory set.

        Args:
            path_member (str):
                Normalized archive path of the member.
            entries_dirs (Set[str]):
                Accumulator for directory entries to be written.
        """
        path_parent, _ = helpers.split_archive_path(path_in_archive=path_member)
        while path_parent != "":
            entries_dirs.add(path_parent)
            path_parent, _ = helpers.split_archive_path(path_in_archive=path_parent)

    def _load_object(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        metadata: Dict,
        type_lookup: Dict,
        check: bool,
    ) -> Any:
        """
        Dispatch deserialization of an object from the archive based on
        its type metadata.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the object.
            metadata (Dict):
                Metadata dictionary describing the object's type and version.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks.

        Returns:
            (Any):
                obj (Any):
                    The reconstructed Python object.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=True,
        )
        self._check_element_compatibility(
            metadata=metadata,
            type_lookup=type_lookup,
            check=check,
        )

        type_name = metadata["type"]
        if type_name in _NATIVE_CONTAINER_TYPES:
            return self._load_container_native(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                type_lookup=type_lookup,
                check=check,
            )
        if type_name in _NATIVE_JSON_TYPES:
            return self._load_json_scalar_native(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                type_name=type_name,
            )
        return self._load_object_bridge(
            reader=reader,
            index=index,
            path_in_archive=path_in_archive,
            metadata=metadata,
            type_lookup=type_lookup,
            check=check,
        )

    def _load_json_scalar_native(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        type_name: str,
    ) -> Any:
        """
        Load a JSON-native scalar value from the archive and cast it to
        the expected Python type.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the scalar file.
            type_name (str):
                Expected type name (``"float"``, ``"int"``, ``"str"``,
                ``"bool"``, or ``"None"``).

        Returns:
            (Any):
                value (Any):
                    The deserialized scalar value.
        """
        payload = self._read_member_bytes(
            reader=reader,
            index=index,
            member_name=path_in_archive,
        )
        value = json.loads(payload.decode("utf-8"))

        if type_name == "float":
            return float(value)
        if type_name == "int":
            return int(value)
        if type_name == "str":
            return str(value)
        if type_name == "bool":
            return bool(value)
        if value is not None:
            raise ValueError("Loaded object is not None.")
        return None

    def _load_container_native(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        type_lookup: Dict,
        check: bool,
    ) -> Any:
        """
        Load a native container (list, tuple, set, frozenset, dict,
        dict_item) by recursively loading each child element.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the container.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks.

        Returns:
            (Any):
                container (Any):
                    The reconstructed Python container.
        """
        metadata = self._load_metadata_row(
            reader=reader,
            index=index,
            path_in_archive=path_in_archive,
            check=check,
        )
        if check:
            indices = [value["index"] for value in metadata["elements"].values()]
            if len(indices) != len(set(indices)):
                raise ValueError("Indices in metadata are not unique.")
            if metadata["library"] != "python":
                raise ValueError("Only 'python' library supported for container types.")
            if not util.is_version_compatible(
                version=metadata["version"],
                rules=PYTHON_VERSIONS_SUPPORTED,
            ):
                raise ValueError(
                    f"Python version '{metadata['version']}' not supported for library "
                    f"'python' and type '{metadata['type']}'. "
                    f"Version rules: {PYTHON_VERSIONS_SUPPORTED}"
                )

        names_meta_sorted = util._sort_element_names_by_index(metadata=metadata, check=check)

        names_children = self._list_direct_child_names(
            index=index,
            path_parent=path_in_archive,
        )
        if check:
            names_allowed = set(names_meta_sorted) | set(NAMES_EXTRA_FILES_ALLOWED)
            missing = [name for name in names_meta_sorted if name not in names_children]
            if len(missing) > 0:
                raise FileNotFoundError(
                    f"Elements in metadata not found in archive: {missing}. "
                    f"path_in_archive={path_in_archive}"
                )
            extra = set(names_children) - set(names_allowed)
            if len(extra) > 0:
                raise ValueError(
                    f"Extra elements in archive not found in metadata: {sorted(extra)}"
                )

        elements = []
        for name in names_meta_sorted:
            path_child = helpers.join_archive_path(path_in_archive, name)
            metadata_child = metadata["elements"][name]
            if check and (
                not self._element_path_exists(
                    index=index,
                    path_in_archive=path_child,
                    metadata=metadata_child,
                )
            ):
                raise FileNotFoundError(
                    f"Archive path missing for element '{name}' at '{path_child}'."
                )
            elements.append(
                self._load_object(
                    reader=reader,
                    index=index,
                    path_in_archive=path_child,
                    metadata=metadata_child,
                    type_lookup=type_lookup,
                    check=check,
                )
            )

        if metadata["type"] == "list":
            return elements
        if metadata["type"] == "tuple":
            return tuple(elements)
        if metadata["type"] == "set":
            return set(elements)
        if metadata["type"] == "frozenset":
            return frozenset(elements)
        if metadata["type"] == "dict_item":
            if check and len(elements) != 2:
                raise ValueError(
                    f"DictItem must contain exactly 2 elements. Found {len(elements)}."
                )
            return util.DictItem(key=elements[0], value=elements[1])

        if check and not all(isinstance(element, util.DictItem) for element in elements):
            raise TypeError("All elements in a dict must be of type DictItem.")
        try:
            return {element.key: element.value for element in elements}
        except AttributeError as exc:
            raise ValueError(f"Error unpacking dict items: {exc}")

    def _element_path_exists(
        self,
        index: Dict[str, Any],
        path_in_archive: str,
        metadata: Dict,
    ) -> bool:
        """
        Check whether the archive contains the expected entry for an element,
        choosing the lookup strategy based on type.

        Args:
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the element.
            metadata (Dict):
                Metadata dictionary describing the element's type.

        Returns:
            (bool):
                exists (bool):
                    True if the expected archive entry exists.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=False,
        )
        type_name = metadata["type"]
        if type_name in _NATIVE_JSON_TYPES:
            return path_in_archive in index["files"]
        if type_name in _NATIVE_CONTAINER_TYPES:
            return helpers.metadata_row_name(path_in_archive=path_in_archive) in index["files"]
        return self._path_exists(index=index, path_in_archive=path_in_archive)

    def _load_object_bridge(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        metadata: Dict,
        type_lookup: Dict,
        check: bool,
    ) -> Any:
        """
        Load a non-native object by materializing its archive subtree to a
        temporary directory and calling the registered load function.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the object.
            metadata (Dict):
                Metadata dictionary describing the object's type and version.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks.

        Returns:
            (Any):
                obj (Any):
                    The reconstructed Python object.
        """
        function_load = type_lookup[metadata["type"]]["function_load"]
        basename = helpers.split_archive_path(path_in_archive=path_in_archive)[1]
        basename = "root" if basename == "" else basename

        with tempfile.TemporaryDirectory() as dir_tmp:
            path_materialized = Path(dir_tmp) / basename
            self._materialize_subtree(
                reader=reader,
                index=index,
                path_in_archive=path_in_archive,
                path_out=path_materialized,
            )

            sig = inspect.signature(function_load)
            args_available = {
                "path": str(path_materialized),
                "metadata": metadata,
                "type_lookup": type_lookup,
                "check": check,
            }
            argnames = list(set(sig.parameters.keys()) & set(args_available.keys()))
            kwargs_function = {argname: args_available[argname] for argname in argnames}
            return function_load(**kwargs_function)

    def _materialize_subtree(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        path_out: Path,
    ) -> None:
        """
        Extract an archive subtree to a filesystem directory, recreating
        the directory structure and file contents.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the subtree root.
            path_out (Path):
                Filesystem path where the subtree will be materialized.
        """
        path_in_archive = helpers.validate_archive_path(
            path_in_archive=path_in_archive,
            allow_empty=True,
        )

        if path_in_archive == "":
            names_all = set(index["files"].keys()) | set(index["dirs"])
            if len(names_all) == 0:
                raise FileNotFoundError(f"{self._backend_name()} archive is empty.")
            path_out.mkdir(parents=True, exist_ok=True)

            for path_dir in sorted(index["dirs"]):
                path_target_dir = helpers.safe_resolve_materialized_path(
                    path_root=path_out,
                    path_relative=path_dir,
                )
                path_target_dir.mkdir(parents=True, exist_ok=True)

            for path_member in sorted(index["files"].keys()):
                path_target = helpers.safe_resolve_materialized_path(
                    path_root=path_out,
                    path_relative=path_member,
                )
                path_target.parent.mkdir(parents=True, exist_ok=True)
                path_target.write_bytes(
                    self._read_member_bytes(
                        reader=reader,
                        index=index,
                        member_name=path_member,
                    )
                )
            return

        prefix = f"{path_in_archive}/"
        has_exact_file = path_in_archive in index["files"]
        has_exact_dir = path_in_archive in index["dirs"]
        has_descendants = any(
            name.startswith(prefix)
            for name in (set(index["files"].keys()) | set(index["dirs"]))
        )

        if (not has_exact_file) and (not has_exact_dir) and (not has_descendants):
            raise FileNotFoundError(
                f"{self._backend_name()} archive path not found: {path_in_archive}"
            )

        if has_exact_file and (not has_descendants) and (not has_exact_dir):
            path_out.parent.mkdir(parents=True, exist_ok=True)
            path_out.write_bytes(
                self._read_member_bytes(
                    reader=reader,
                    index=index,
                    member_name=path_in_archive,
                )
            )
            return

        path_out.mkdir(parents=True, exist_ok=True)

        for path_dir in sorted(index["dirs"]):
            if path_dir.startswith(prefix):
                rel = path_dir[len(prefix) :]
                path_target_dir = helpers.safe_resolve_materialized_path(
                    path_root=path_out,
                    path_relative=rel,
                )
                path_target_dir.mkdir(parents=True, exist_ok=True)

        for path_member in sorted(index["files"].keys()):
            if path_member.startswith(prefix):
                rel = path_member[len(prefix) :]
                path_target = helpers.safe_resolve_materialized_path(
                    path_root=path_out,
                    path_relative=rel,
                )
                path_target.parent.mkdir(parents=True, exist_ok=True)
                path_target.write_bytes(
                    self._read_member_bytes(
                        reader=reader,
                        index=index,
                        member_name=path_member,
                    )
                )

    def _get_metadata_tree_recursive(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        check: bool,
    ) -> Dict:
        """
        Recursively build the nested metadata tree from a given archive path.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the subtree root.
            check (bool):
                Whether to run validation checks.

        Returns:
            (Dict):
                tree (Dict):
                    Nested dictionary with ``"metadata"`` and ``"elements"``
                    keys at each container level.
        """
        metadata = self._load_metadata_row(
            reader=reader,
            index=index,
            path_in_archive=path_in_archive,
            check=check,
        )
        out = {
            "metadata": metadata,
            "elements": {},
        }
        for name, value in metadata["elements"].items():
            if value["type"] in _NATIVE_CONTAINER_TYPES:
                out["elements"][name] = self._get_metadata_tree_recursive(
                    reader=reader,
                    index=index,
                    path_in_archive=helpers.join_archive_path(path_in_archive, name),
                    check=check,
                )
            else:
                out["elements"][name] = value
        return out

    def _print_directory_tree_recursive(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        type_lookup: Dict,
        check: bool,
        level: int,
    ) -> None:
        """
        Recursively print a directory-style tree of archive elements to stdout.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the current node.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks.
            level (int):
                Current indentation depth for tree formatting.
        """
        del type_lookup
        metadata = self._load_metadata_row(
            reader=reader,
            index=index,
            path_in_archive=path_in_archive,
            check=check,
        )
        for name, value in metadata["elements"].items():
            print("|   " * level + "├── " + f"{name} ({value['type']})")
            if value["type"] in _NATIVE_CONTAINER_TYPES:
                self._print_directory_tree_recursive(
                    reader=reader,
                    index=index,
                    path_in_archive=helpers.join_archive_path(path_in_archive, name),
                    type_lookup=type_lookup,
                    check=check,
                    level=level + 1,
                )
        print("|   " * level)

    def _print_tree_recursive(
        self,
        reader,
        index: Dict[str, Any],
        path_in_archive: str,
        type_lookup: Dict,
        check: bool,
        show_filenames: bool,
        level: int,
    ) -> None:
        """
        Recursively print a logical tree of archive elements to stdout,
        resolving dict keys to their values for display.

        Args:
            reader:
                The backend-specific reader handle.
            index (Dict[str, Any]):
                The archive index built by ``_build_index``.
            path_in_archive (str):
                Normalized archive path of the current node.
            type_lookup (Dict):
                Mapping of type names to their load/save properties.
            check (bool):
                Whether to run validation checks.
            show_filenames (bool):
                Whether to include raw archive member names in the output.
            level (int):
                Current indentation depth for tree formatting.
        """
        metadata = self._load_metadata_row(
            reader=reader,
            index=index,
            path_in_archive=path_in_archive,
            check=check,
        )
        sf = show_filenames
        for name, value in metadata["elements"].items():
            if value["type"] == "dict_item":
                path_item = helpers.join_archive_path(path_in_archive, name)
                metadata_item = self._load_metadata_row(
                    reader=reader,
                    index=index,
                    path_in_archive=path_item,
                    check=check,
                )
                names_meta_sorted = util._sort_element_names_by_index(metadata=metadata_item)
                if len(names_meta_sorted) != 2:
                    raise ValueError(
                        "DictItem must contain exactly 2 elements: key='0.json' and value=another element."
                    )
                name_key, name_value = names_meta_sorted
                metadata_key = metadata_item["elements"][name_key]
                metadata_value = metadata_item["elements"][name_value]
                key = self._load_object(
                    reader=reader,
                    index=index,
                    path_in_archive=helpers.join_archive_path(path_item, name_key),
                    metadata=metadata_key,
                    type_lookup=type_lookup,
                    check=check,
                )
                print(
                    "|    " * level
                    + "├── "
                    + f"'{key}': {(name_value if sf else '')}  ({metadata_value['type']})"
                )
                if metadata_value["type"] in _NATIVE_CONTAINER_TYPES:
                    self._print_tree_recursive(
                        reader=reader,
                        index=index,
                        path_in_archive=helpers.join_archive_path(path_item, name_value),
                        type_lookup=type_lookup,
                        check=check,
                        show_filenames=show_filenames,
                        level=level + 1,
                    )
            elif value["type"] in _NATIVE_CONTAINER_TYPES:
                print("|    " * level + "├── " + f"{(name if sf else '')}  ({value['type']})")
                self._print_tree_recursive(
                    reader=reader,
                    index=index,
                    path_in_archive=helpers.join_archive_path(path_in_archive, name),
                    type_lookup=type_lookup,
                    check=check,
                    show_filenames=show_filenames,
                    level=level + 1,
                )
            else:
                print("|    " * level + "├── " + f"{(name if sf else '')}  ({value['type']})")
        print("|    " * level)

    def _resolve_archive_and_internal_path(
        self,
        richfile: "util.RichFile",
        path: Optional[Union[str, Path]],
        default_to_current_subpath: bool,
    ) -> Tuple[str, str]:
        """
        Resolve a user-supplied path into a filesystem archive path and a
        normalized internal archive path.

        Args:
            richfile (util.RichFile):
                The RichFile instance providing default path context.
            path (Optional[Union[str, Path]]):
                User-supplied path, which may be an archive file path, an
                internal subpath, or None to use defaults.
            default_to_current_subpath (bool):
                Whether to use the RichFile's current subpath as the
                default internal path.

        Returns:
            (Tuple[str, str]):
                result (Tuple[str, str]):
                    A tuple of (archive filesystem path, normalized
                    internal archive path).
        """
        if richfile.path is None and path is None:
            raise ValueError("`path` [str, Path] must be specified.")

        path_archive_default = None if richfile.path is None else str(richfile.path)
        path_sub_default = (
            helpers.validate_archive_path(
                path_in_archive=richfile._path_in_archive,
                allow_empty=True,
            )
            if default_to_current_subpath
            else ""
        )

        if path is None:
            if path_archive_default is None:
                raise ValueError("`path` [str, Path] must be specified.")
            return path_archive_default, path_sub_default

        path_input = str(path)
        if path_archive_default is not None and path_input == path_archive_default:
            return path_archive_default, path_sub_default

        if Path(path_input).exists() and Path(path_input).is_file():
            return path_input, ""

        if path_archive_default is not None:
            prefix_posix = path_archive_default.rstrip("/") + "/"
            prefix_windows = path_archive_default.rstrip("\\") + "\\"
            if path_input.startswith(prefix_posix):
                rel = path_input[len(prefix_posix) :]
                return path_archive_default, helpers.validate_archive_path(
                    path_in_archive=rel,
                    allow_empty=True,
                )
            if path_input.startswith(prefix_windows):
                rel = path_input[len(prefix_windows) :]
                return path_archive_default, helpers.validate_archive_path(
                    path_in_archive=rel,
                    allow_empty=True,
                )
            return path_archive_default, helpers.validate_archive_path(
                path_in_archive=path_input,
                allow_empty=True,
            )

        return path_input, ""

    def _get_or_build_index(
        self,
        richfile: "util.RichFile",
        path_archive: Union[str, Path],
        reader,
    ) -> Dict[str, Any]:
        """
        Return a cached archive index if available and fresh, otherwise
        build and cache a new one.

        Args:
            richfile (util.RichFile):
                The RichFile instance that holds the index cache.
            path_archive (Union[str, Path]):
                Filesystem path of the archive.
            reader:
                The backend-specific reader handle.

        Returns:
            (Dict[str, Any]):
                index (Dict[str, Any]):
                    The archive index with ``"files"``, ``"dirs"``, and
                    ``"children"`` mappings.
        """
        path_archive = str(path_archive)
        path_obj = Path(path_archive)
        stat = path_obj.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cache_key = (self._backend_name(), path_archive)

        if not hasattr(richfile, "_archive_index_cache"):
            richfile._archive_index_cache = {}
        cache_entry = richfile._archive_index_cache.get(cache_key)
        if cache_entry is not None and cache_entry["signature"] == signature:
            return cache_entry["index"]

        index = self._build_index(reader=reader)
        richfile._archive_index_cache[cache_key] = {
            "signature": signature,
            "index": index,
        }
        return index

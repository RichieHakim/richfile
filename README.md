# richfile

[![PyPI](https://img.shields.io/pypi/v/richfile)](https://pypi.org/project/richfile/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://pypi.org/project/richfile/)
[![Build](https://github.com/RichieHakim/richfile/actions/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/richfile/actions/workflows/build.yml)

A more natural approach to saving hierarchical data structures.

`richfile` saves any Python object to disk and loads it back into the same
Python objects. It can save any atomic Python object, including custom classes,
so long as it's possible to write a function to save and load it.

It is intended as a replacement for `pickle`, `json`, `yaml`, `HDF5`, etc. when
you want to save a complex data structure in a format that:
- Supports **custom data types** (numpy arrays, sparse matrices, torch tensors, etc.)
- Is **insensitive to version changes** (no pickling issues)
- Allows **lazy loading** of individual elements
- Supports **human-readable inspection** of directory and zip-style richfiles
- Follows [**ACID**](https://en.wikipedia.org/wiki/ACID) principles

The code is simple, pure Python, and easy to use.

## Installation
```bash
pip install richfile
```

## Quick Start
```python
## Given some complex data structure
data = {
    "name": "John Doe",
    "age": 25,
    "address": {
        "street": "1234 Elm St",
        "zip": None
    },
    "siblings": [
        "Jane",
        "Jim"
    ],
    "data": [1,2,3],
    (1,2,3): "complex key",
}

## Save it
import richfile as rf
r = rf.RichFile("path/to/data.richfile").save(data)

## Load it back
data = rf.RichFile("path/to/data.richfile").load()
```

## Backends

Four storage backends are available. Pass `backend=` when creating a `RichFile`:

| Backend | Format | Best for |
|---------|--------|----------|
| `"directory"` | Directory tree (default) | Debugging, human inspection |
| `"sqlar"` | SQLite archive (`.sqlar`) | General use — fast, single file, random access |
| `"zip"` | ZIP stored (`.zip`) | Sharing — universally recognized format |
| `"tar"` | Plain TAR (`.tar`) | Interop with Unix tooling |

When loading, `backend="auto"` (the default) detects the format from magic bytes.

```python
import richfile as rf

rf.RichFile("path/to/data.sqlar", backend="sqlar").save(data)
data = rf.RichFile("path/to/data.sqlar", backend="sqlar").load()
```

## Converting Between Backends
```python
import richfile as rf

## Archive -> directory-style richfile
rf.extract_backend_to_directory(
    path_source="path/to/data.zip",
    backend_source="zip",
    path_directory_out="path/to/data.richfile",
    overwrite=True,
)

## Directory-style richfile -> archive backend
rf.pack_directory_to_backend(
    path_directory_in="path/to/data.richfile",
    backend_target="sqlar",
    path_target="path/to/data.sqlar",
    overwrite=True,
)

## Generic backend -> backend conversion
rf.convert_backend(
    path_source="path/to/data.sqlar",
    backend_source="sqlar",
    path_target="path/to/data.tar",
    backend_target="tar",
    mode="raw",        ## "raw" (byte-preserving) or "semantic" (load/save)
    overwrite=True,
)
```

## Lazy Loading
```python
r = rf.RichFile("path/to/data.richfile")  ## Path to an existing richfile
first_sibling = r["siblings"][0].load()  ## Lazily load a single item using pythonic indexing
print(f"First sibling: {first_sibling}")

>>> First sibling: Jane
```

## Inspecting Contents

View the structure of a richfile without loading data:
```python
r.view_directory_tree()
```

Output:
```
Directory structure:
Viewing tree structure of richfile at path: ~/path/data.richfile (dict)
├── name.dict_item (dict_item)
|   ├── key.json (str)
|   ├── value.json (str)
|   
├── age.dict_item (dict_item)
|   ├── key.json (str)
|   ├── value.json (int)
|   
├── address.dict_item (dict_item)
|   ├── key.json (str)
|   ├── value.dict (dict)
|   |   ├── street.dict_item (dict_item)
|   |   |   ├── key.json (str)
|   |   |   ├── value.json (str)
|   |   |   
|   |   ├── zip.dict_item (dict_item)
|   |   |   ├── key.json (str)
|   |   |   ├── value.json (None)
|   |   |   
|   |   
|   
├── siblings.dict_item (dict_item)
|   ├── key.json (str)
|   ├── value.list (list)
|   |   ├── 0.json (str)
|   |   ├── 1.json (str)
|   |   
|   
├── data.dict_item (dict_item)
|   ├── key.json (str)
|   ├── value.list (list)
|   |   ├── 0.json (int)
|   |   ├── 1.json (int)
|   |   ├── 2.json (int)
|   |   
|   
├── 5.dict_item (dict_item)
|   ├── key.tuple (tuple)
|   |   ├── 0.json (int)
|   |   ├── 1.json (int)
|   |   ├── 2.json (int)
|   |   
|   ├── value.json (str)
|   
```

## Custom Types

Register your own types by providing save/load functions:
```python
## Add type to a RichFile object
r = rf.RichFile("path/to/data.richfile")
r.register_type(
    type_name='numpy_array',
    function_load=lambda path: np.load(path),
    function_save=lambda path, obj: np.save(path, obj),
    object_class=np.ndarray,
    library='numpy',
    suffix='npy',
)

## OR
## Add type to the global workspace / kernel so that all new RichFile objects can use it
rf.functions.register_type(
    type_name='numpy_array',
    function_load=lambda path: np.load(path),
    function_save=lambda path, obj: np.save(path, obj),
    object_class=np.ndarray,
    library='numpy',
    suffix='npy',
)
```

## Installation from source
```bash
git clone https://github.com/RichieHakim/richfile
cd richfile
pip install -e .
```

## Considerations and Limitations
- **Inversibility**: When creating custom data types, ensure the save/load operations are exactly reversible.
- [**ACID**](https://en.wikipedia.org/wiki/ACID) principles are followed via temporary files, file locks, and atomic operations. However, `richfile` is not a database — atomic replacement of existing non-empty directories requires two operations, which reduces atomicity guarantees.
- **No compression**: Archive backends (SQLAR/ZIP/TAR) store raw bytes without compression for faster I/O. On-disk size may be larger than compressed formats.
- **Archive mutation**: Archive backends support full save and lazy load. Nested path mutation (modifying a single element within an existing archive) is not yet supported.
- **TAR format**: Writes plain `.tar` only (no `.tar.gz` or `.tar.bz2`).
- **Backend detection**: `backend="auto"` detects format from magic bytes, not file extension. You can also pass the backend explicitly.
- **Conversion**: `convert_backend(..., mode="raw")` is byte-preserving and does not deserialize objects. `mode="semantic"` round-trips through `load()`/`save()` and requires matching type registrations.

## Examples
See the [demo_notebook.ipynb](https://github.com/RichieHakim/richfile/blob/main/demo_notebook.ipynb) for more detailed examples.
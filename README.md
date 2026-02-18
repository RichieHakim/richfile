# richfile
A more natural approach to saving hierarchical data structures.

`richfile` saves any Python object to disk and loads it back into the same
Python objects.

Four backends are available:
- `backend="directory"`: classic richfile directory trees.
- `backend="sqlar"`: single-file SQLite archive (`.sqlar`) with no compression.
- `backend="zip"`: single-file ZIP archive (`.zip`) in stored mode (no compression).
- `backend="tar"`: single-file plain TAR archive (`.tar`) with no compression.

Backend auto-detection is also available:
- `backend="auto"` (default): detect backend from existing path for load/query operations.

`richfile` can save any atomic Python object, including custom classes, so long
as you can write a function to save and load it. It is intended as a replacement
for things like: `pickle`, `json`, `yaml`, `HDF5`, `Parquet`, `netCDF`, `zarr`,
`numpy`, etc. when you want to save a complex data structure in a human-readable
and editable format. We find the `richfile` format ideal to use when you are
building a data processing pipeline and you want to contain intermediate results
in a format that allows for custom data types, is insensitive to version changes
(pickling issues), allows for easy debugging, and is human readable.

It is easy to use, the code is simple and pure python, and the operations follow [ACID](https://en.wikipedia.org/wiki/ACID) principles.

## Installation
```bash
pip install richfile
```

## Examples
Try out the examples in the [demo_notebook.ipynb](https://github.com/RichieHakim/richfile/blob/main/demo_notebook.ipynb) file.

## Usage
Saving and loading data is simple:
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
r = rf.RichFile("path/to/data.richfile", backend="directory").save(data)

## Load it back
data = rf.RichFile("path/to/data.richfile", backend="directory").load()
```

Save and load using the SQLAR backend:
```python
import richfile as rf

rf.RichFile("path/to/data.sqlar", backend="sqlar").save(data)
data = rf.RichFile("path/to/data.sqlar", backend="sqlar").load()
```

Save and load using ZIP/TAR backends:
```python
import richfile as rf

rf.RichFile("path/to/data.zip", backend="zip").save(data)
data_zip = rf.RichFile("path/to/data.zip", backend="zip").load()

rf.RichFile("path/to/data.tar", backend="tar").save(data)
data_tar = rf.RichFile("path/to/data.tar", backend="tar").load()
```

Convert between backends (raw byte-preserving conversion):
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

You can also load just a part of the data:
```python
r = rf.RichFile("path/to/data.richfile", backend="directory")
first_sibling = r["siblings"][0].load()  ## Lazily load a single item using pythonic indexing
print(f"First sibling: {first_sibling}")

>>> First sibling: Jane
```

View the contents of a richfile directory without loading it:
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

You can also add new data types easily:
```python
## Add type to a RichFile object
r = rf.RichFile("path/to/data.richfile", backend="directory")
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
- **Inversibility**: When creating custom data types, it is important to consider whether the saving and loading operations are exactly reversible.
- [**ACID**](https://en.wikipedia.org/wiki/ACID) principles are reasonably followed via the use of temporary files, file locks, and atomic operations. However, the library is not a database, and therefore cannot guarantee the same level of ACID compliance as a database. In addition, atomic replacements of existing non-empty directories require two operations, which reduces atomicity.
- **Backend selection**: You can pass backend explicitly (`"directory"`, `"sqlar"`, `"zip"`, `"tar"`), or rely on `backend="auto"` for loading from existing paths. Path suffixes remain informational only.
- **Archive performance tradeoff**: SQLAR/ZIP/TAR store raw bytes without compression in v1 for faster save behavior. This can increase on-disk size compared with compressed formats.
- **Archive scope in v1**: SQLAR/ZIP/TAR currently support root-object save and lazy/query load behavior. Nested path mutation/append APIs are intentionally deferred.
- **TAR scope in v1**: TAR backend writes plain `.tar` only (no `.tar.gz`, `.tgz`, `.tar.bz2`, or `.tar.xz` output modes).
- **Custom type compatibility in archive backends**: Custom `function_save(path, ...)` and `function_load(path, ...)` callbacks are supported via a selective temporary-path bridge when needed.
- **Backend conversion**: `convert_backend(..., mode="raw")` performs byte-preserving layout conversion and does not deserialize objects. `mode="semantic"` performs `load()` + `save()` and requires matching type registrations.

## TODO:
- [ ] Tests
- [ ] Documentation
- [x] Examples
- [x] Readme
- [ ] License
- [x] PyPi
- [x] ~~Hashing~~
- [x] ~~Item assignment (safely)~~
- [x] Custom saving/loading functions
- [x] ~~Put the library imports in the function calls~~
- [x] Add handling for data without a known type
- [ ] Change name of library to something more descriptive
- [x] Test out memmap stuff
- [x] ~~Make it a .zip type~~
- [ ] Add mutability
- [x] Archive packing

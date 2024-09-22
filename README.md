# richfile
A more natural approach to saving hierarchical data structures.

`richfile` saves any Python object using directory structures on disk, and loads them back again into the same Python objects.

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
        "zip": 62701
    },
    "siblings": [
        "Jane",
        "Jim"
    ],
    "data": np.array([1,2,3]),
    (1,2,3): "complex key",
}

## Save it
import as rf
r = rf.RichFile("path/to/data.richfile").save(data)

## Load it back
data = rf.RichFile("path/to/data.richfile").load()
```

You can also load just a part of the data:
```python
r = rf.RichFile("path/to/data.richfile")
first_sibling = r["siblings"][0]  ## Lazily load a single item using pythonic indexing
print(f"First sibling: {first_sibling}")

>>> First sibling: Jane
```

View the contents of a richfile directory without loading it:
```python
r.view_directory_structure()
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
|   ├── value.npy (numpy_array)
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
## Add type to environment so that all new RichFile objects can use it
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
- **Performance**: Data structures with many branches will require many files and operations, which may become slow. Consider packaging highly branched data structures into a single file that supports hierarchical data, such as JSON, HDF5, Parquet, netCDF, zarr, numpy, etc. and making a custom data type for it.

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
# richfile
A more natural approach to saving hierarchical data structures.

`richfile` saves any Python object into directory structures on disk, and loads them back into Python objects.\

Saving is simple:
```python
import as rf

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

r = rf.RichFile("path/to/data.richfile").save(data)
```

Loading is simple:
```python
import as rf

r = rf.RichFile("path/to/data.richfile")
data = r.load()
```

You can also load just a part of the data:
```python
import as rf

r = rf.RichFile("path/to/data.richfile")
sibling = r["siblings"][0]
```

View the contents of a richfile directory without loading it:
```python
import as rf

r = rf.RichFile("path/to/data.richfile")
r.view_tree()
## OR
r.view_directory_structure()
```

## Installation
```bash
pip install richfile
```
or from source:
```bash
pip install git+https://github.com/RichieHakim/richfile.git
```

## Example
Turns this Python object:
```python
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
    "data": np.array([1,2,3]),
    (1,2,3): "complex key",
}
```

Into directory structures like this:
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

## TODO:

- [ ] Tests
- [ ] Documentation
- [x] Examples
- [x] Readme
- [ ] License
- [ ] PyPi
- [x] ~~Hashing~~
- [x] ~~Item assignment (safely)~~
- [x] Custom saving/loading functions
- [x] ~~Put the library imports in the function calls~~
- [x] Add handling for data without a known type
- [ ] Change name of library to something more descriptive
- [x] Test out memmap stuff
- [x] ~~Make it a .zip type~~
__version__ = "0.5.1"

VERSIONS_RICHFILE_SUPPORTED = [">=0.3.1", "<1.0.0"]
PYTHON_VERSIONS_SUPPORTED = [">=3", "<4"]

FILENAME_METADATA   = ".metadata.richfile"
FILENAME_TYPELOOKUP = ".typelookup.richfile"

JSON_INDENT = 4

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


## Import important stuff from util.py into top-level namespace
from . import functions, util
from .util import RichFile, load_folder, load_element, load_folder_metadata, save_object, invalid_chars_filename
from . import demo
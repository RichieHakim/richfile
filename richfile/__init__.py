__version__ = "0.2.0"

VERSIONS_RICHFILE_SUPPORTED = [">=0.2.0", "<1.0.0"]

FILENAME_METADATA = ".metadata.richfile"
JSON_INDENT = 4

## Import important stuff from util.py into top-level namespace
from . import functions, util
from .util import RichFile, load_folder, load_element, load_folder_metadata, save_object

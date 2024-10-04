## setup.py file for roicat
from pathlib import Path
import copy
import platform

from distutils.core import setup

## Get the parent directory of this file
dir_parent = Path(__file__).parent

## Get requirements from requirements.txt
def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements
deps_all = read_requirements()


## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_dict = dict(zip(deps_names, deps_all))
deps_all_latest = dict(zip(deps_names, deps_names))


## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get version number
with open(str(dir_parent / "richfile" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setup(
    name='richfile',
    version=version,
    author='Richard Hakim',
    keywords=['data analysis', 'machine learning', 'neuroscience'],
    license='LICENSE',
    description='A library for reading and writing hierarchical data files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RichieHakim/richfile',

    packages=['richfile'],

    install_requires=list(deps_all_dict.values()),
    # extras_require=,
    # python_requires='>=3.11',
)
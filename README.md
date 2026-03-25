# Welcome to ndtbl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/thomasisensee/ndtbl/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasisensee/ndtbl/actions)
[![Documentation Status](https://readthedocs.org/projects/ndtbl/badge/)](https://ndtbl.readthedocs.io/)
[![codecov](https://codecov.io/gh/thomasisensee/ndtbl/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasisensee/ndtbl)

# Prerequisites

Building ndtbl requires the following software installed:

* A C++14-compliant compiler
* CMake `>= 3.23`
* Doxygen (optional, documentation building is skipped if missing)
* The testing framework [Catch2](https://github.com/catchorg/Catch2) for building the test suite

# Building ndtbl

The following sequence of commands builds ndtbl.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
cmake -B build
cmake --build build
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `ndtbl_BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `ndtbl_BUILD_DOCS`: Enable building the documentation (default: `ON`)



# Testing ndtbl

When built according to the above explanation (with `-Dndtbl_BUILD_TESTING=ON`),
the C++ test suite of `ndtbl` can be run using
`ctest` from the build directory:

```
cd build
ctest
```


# Documentation

ndtbl provides a Sphinx-based documentation, that can be browsed [online at readthedocs.org](https://ndtbl.readthedocs.io).
To build it locally, first ensure the requirements are installed by running this command from the top-level source directory:

```
pip install -r doc/requirements.txt
```

Then build the sphinx documentation from the top-level directory:

```
cmake --build build --target sphinx-doc
```

The web documentation can then be browsed by opening `build/doc/sphinx/index.html` in your browser.

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for C++ Packages](https://github.com/ssciwr/cookiecutter-cpp-project).

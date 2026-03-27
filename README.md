# ndtbl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/thomasisensee/ndtbl/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasisensee/ndtbl/actions)
[![Documentation Status](https://readthedocs.org/projects/ndtbl/badge/)](https://ndtbl.readthedocs.io/)
[![codecov](https://codecov.io/gh/thomasisensee/ndtbl/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasisensee/ndtbl)

# Prerequisites

Building ndtbl requires the following software installed:

* A C++14-compliant compiler
* CMake `>= 3.23`
* Doxygen (optional, documentation building is skipped if missing)
* The testing framework [Catch2](https://github.com/catchorg/Catch2) for building the optional test suite

# Layout

`ndtbl` exposes a header-only C++14 core in `include/ndtbl/ndtbl.hpp`.
The CMake project mainly exists to build tools, tests, and documentation.

Current command-line tools:

* `ndtbl-convert`: convert current OpenFOAM text tables into ndtbl binary files
* `ndtbl-inspect`: print metadata from ndtbl binary files

# Building ndtbl

The following sequence of commands builds ndtbl.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
cmake -B build -Dndtbl_BUILD_TESTING=OFF -Dndtbl_BUILD_DOCS=OFF
cmake --build build
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `ndtbl_BUILD_TESTING`: Enable building of the test suite (default: `OFF`)
* `ndtbl_BUILD_DOCS`: Enable building the documentation (default: `ON`)

# Conversion workflow

Convert one or more OpenFOAM text tables into a single ndtbl binary file:

```
./build/app/ndtbl-convert output.ndtbl Table1_table Table2_table
```

Choose the stored value precision explicitly if desired:

```
./build/app/ndtbl-convert --precision float output.ndtbl Table1_table Table2_table
```

Inspect the generated file:

```
./build/app/ndtbl-inspect output.ndtbl
```

# Testing ndtbl

When built with `-Dndtbl_BUILD_TESTING=ON`, the C++ test suite can be run
with `ctest` from the build directory:

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

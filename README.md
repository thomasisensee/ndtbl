# ndtbl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/thomasisensee/ndtbl/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasisensee/ndtbl/actions)
[![Documentation Status](https://readthedocs.org/projects/ndtbl/badge/)](https://ndtbl.readthedocs.io/)
[![codecov](https://codecov.io/gh/thomasisensee/ndtbl/graph/badge.svg?flag=cpp&token=5N4GQ0YP7I)](https://codecov.io/gh/thomasisensee/ndtbl)

`ndtbl` is an n-dimensional table format and toolkit.

This repository currently contains two user-facing parts:

- a header-only C++14 library in `include/ndtbl/`
- locally built C++ command-line tools in `app/`

It also contains a separate pure-Python package in `python/ndtbl/` for reading,
writing, inspecting, querying, and generating `.ndtbl` files without the C++
toolchain.

The C++ reader can also be built with optional POSIX `mmap` support so payloads
stay file-backed instead of always being copied into heap memory.

## 🗂️ Layout

- `include/ndtbl/`: public C++ headers
- `app/`: C++ command-line tools built by CMake
- `tests/`: Catch2-based C++ test suite
- `doc/`: Sphinx and Doxygen documentation
- `python/ndtbl/`: pure-Python package and `ndtbl` CLI

## 🖥️ Which interface to use

Use the C++ library when you want to integrate `.ndtbl` directly into a C++
application.

Use the C++ tools when you want to convert OpenFOAM text tables or inspect
`.ndtbl` files from this repository checkout. These tools are not prebuilt;
they become available only after running the local CMake build.

Use the Python package when you want a pip-installable workflow or a ready-made
CLI for:

- `ndtbl inspect`
- `ndtbl query`
- `ndtbl generate`

See `python/ndtbl/README.md` for Python package details.

## 📋 Prerequisites

Building the C++ project requires:

- a C++14-compliant compiler for the header-only library interface
- a compiler with C++20 support for the executables in `app/`
- CMake `>= 3.23`
- Doxygen only if you want to build the documentation
- Catch2 only if you want to build the optional C++ test suite

## 🛠️ Build

From the top-level `ndtbl/` directory:

```bash
cmake -B build -Dndtbl_BUILD_TESTING=OFF -Dndtbl_BUILD_DOCS=OFF
cmake --build build
```

This produces the C++ command-line tools in `build/app/`:

- `build/app/ndtbl-convert`
- `build/app/ndtbl-inspect`

Relevant CMake options:

- `ndtbl_BUILD_TESTING`: build the C++ test suite, default `OFF`
- `ndtbl_BUILD_DOCS`: build the documentation, default `ON` for top-level builds
- `ndtbl_ENABLE_MMAP`: enable POSIX-only `mmap`-backed payload reads, default `OFF`

If you want to install the C++ headers and CMake package metadata:

```bash
cmake --install build --prefix /desired/prefix
```

To enable optional `mmap`-backed reads on supported POSIX platforms:

```bash
cmake -B build -Dndtbl_ENABLE_MMAP=ON
cmake --build build
```

## ⚙️ C++ Tool Workflow

Convert one or more OpenFOAM text tables into a single `.ndtbl` file:

```bash
./build/app/ndtbl-convert output.ndtbl Table1_table Table2_table
```

Choose the stored value precision explicitly:

```bash
./build/app/ndtbl-convert --precision float output.ndtbl Table1_table Table2_table
```

Inspect the generated file:

```bash
./build/app/ndtbl-inspect output.ndtbl
```

These commands use the C++ executables built in the previous step. They are not
available before `cmake --build build`.

When `ndtbl_ENABLE_MMAP=ON`, the C++ `read_group()` path uses read-only memory
mapping by default. This is currently supported only on POSIX platforms and is
intended to reduce peak heap usage for large tables.

## 🐍 Python Package

The repository also ships a separate Python package in `python/ndtbl/`.

That package installs a different CLI executable named `ndtbl`, with the
subcommands:

- `inspect`
- `query`
- `generate`

Install it from the package directory:

```bash
cd python/ndtbl
python -m pip install .
```

After that, the Python CLI is available as:

```bash
ndtbl --help
```

See `python/ndtbl/README.md` for usage examples and Python API details.

## 🧪 Testing

Enable the C++ test suite during configuration:

```bash
cmake -B build -Dndtbl_BUILD_TESTING=ON
cmake --build build
```

Then run:

```bash
cd build
ctest --output-on-failure
```

## 📖 Documentation

Online documentation is available at
[ndtbl.readthedocs.io](https://ndtbl.readthedocs.io/).

To build the docs locally, first install the documentation requirements from
the top-level `ndtbl/` directory:

```bash
python -m pip install -r doc/requirements.txt
```

Then build the Sphinx target:

```bash
cmake -B build -Dndtbl_BUILD_DOCS=ON -Dndtbl_BUILD_TESTING=OFF
cmake --build build --target sphinx-doc
```

Open `build/doc/sphinx/index.html` in a browser to inspect the generated site.

# ndtbl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/thomasisensee/ndtbl/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasisensee/ndtbl)
[![PyPI](https://img.shields.io/pypi/v/ndtbl)](https://pypi.org/project/ndtbl)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)

Pure-Python tools for reading, writing, inspecting, and generating `.ndtbl`
files without depending on the C++ binaries.

## Features

- Read `.ndtbl` metadata or full payloads
- Write `.ndtbl` files compatible with the current C++ implementation
- Inspect files from the command line
- Generate small predefined linear tables for development workflows

## Installation

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -v -e .[lint,tests]
```

## Python API

```python
import numpy as np

from ndtbl import FieldGroup, UniformAxis, write_group

group = FieldGroup(
    axes=(UniformAxis(0.0, 1.0, 3), UniformAxis(10.0, 20.0, 2)),
    field_names=("A", "B"),
    values=np.zeros((3, 2, 2), dtype=np.float64),
)

write_group("example.ndtbl", group)
```

## CLI

Inspect an existing file:

```bash
ndtbl inspect example.ndtbl
```

Generate a small linear table:

```bash
ndtbl generate output.ndtbl \
  --axis 0 1 3 \
  --axis 10 20 2 \
  --field-linear A 1.0 2.0 0.0 \
  --field-linear B 5.0 0.0 -1.0
```

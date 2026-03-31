# ndtbl Python

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
python -m pip install --editable .[tests]
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

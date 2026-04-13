# ndtbl Lookup Benchmarks

This directory contains developer-only lookup-time benchmarks for the C++
`ndtbl` library. The benchmarks use
[Google Benchmark](https://github.com/google/benchmark) and focus on
interpolation cost after a table has already been constructed in memory.

The suite does not measure file I/O, `.ndtbl` parsing, memory mapping, or table
generation time.

## Benchmark Data

Each benchmark case uses a deterministic synthetic table with:

- `8` fields per grid point
- `1024` precomputed query points, cycled during each timed loop
- uniform and explicit-coordinate axis variants
- 2D shape `256 x 256`
- 4D shape `16 x 16 x 16 x 16`
- 6D shape `6 x 6 x 6 x 6 x 6 x 6`

The 6D case is intentionally small in each axis so the payload stays modest
while still exercising a 64-corner interpolation stencil.

## Measured Operations

`bench_prepare` measures only `Grid::prepare(query)`. This isolates axis
bracketing and interpolation-stencil construction.

`bench_prepared_evaluate` measures
`FieldGroup::evaluate_all_into(prepared, results)`. This reuses one prepared
stencil and isolates interpolation over all fields.

`bench_typed_combined` measures
`FieldGroup::evaluate_all_into(query, results)`. This is the typed end-to-end
path from query coordinates to interpolated field values.

`bench_runtime_combined` measures
`RuntimeFieldGroup::evaluate_all_into(query, results)`. This covers the
runtime-erased path, including its wrapper dispatch and scratch-buffer copy.

## Build And Run

Configure and build the benchmark target:

```bash
cmake -B build -Dndtbl_BUILD_BENCHMARKS=ON
cmake --build build --target ndtbl_lookup_benchmarks
```

Run a short smoke benchmark:

```bash
./build/benchmarks/ndtbl_lookup_benchmarks
```

For performance numbers, prefer a release build:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dndtbl_BUILD_BENCHMARKS=ON
cmake --build build --target ndtbl_lookup_benchmarks
./build/benchmarks/ndtbl_lookup_benchmarks
```

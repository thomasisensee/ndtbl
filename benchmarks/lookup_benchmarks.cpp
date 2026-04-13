#include "ndtbl/ndtbl.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace {

enum class AxisLayout
{
  uniform,
  explicit_coordinates
};

constexpr std::size_t field_count = 8;
constexpr std::size_t query_count = 1024;

template<std::size_t Dim>
constexpr std::size_t
default_extent()
{
  if constexpr (Dim == 2) {
    return 256;
  } else if constexpr (Dim == 4) {
    return 16;
  } else {
    static_assert(Dim == 6, "unsupported benchmark dimension");
    return 6;
  }
}

template<std::size_t Dim>
struct LookupContext
{
  ndtbl::Grid<Dim> grid;
  ndtbl::FieldGroup<double, Dim> group;
  ndtbl::RuntimeFieldGroup<Dim> runtime_group;
  std::vector<std::array<double, Dim>> queries;
  ndtbl::PreparedQuery<Dim> prepared;

  LookupContext(const ndtbl::Grid<Dim>& grid_in,
                const ndtbl::FieldGroup<double, Dim>& group_in,
                const std::vector<std::array<double, Dim>>& queries_in)
    : grid(grid_in)
    , group(group_in)
    , runtime_group(group)
    , queries(queries_in)
    , prepared(grid.prepare(queries.front()))
  {
  }
};

template<std::size_t Dim>
std::array<std::size_t, Dim>
filled_shape(std::size_t extent)
{
  std::array<std::size_t, Dim> shape = {};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    shape[axis] = extent;
  }
  return shape;
}

template<std::size_t Dim>
std::array<ndtbl::Axis, Dim>
make_axes(const std::array<std::size_t, Dim>& shape, AxisLayout layout)
{
  std::array<ndtbl::Axis, Dim> axes = {};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const double min_value = -1.0 - static_cast<double>(axis);
    const double max_value = 1.0 + 0.5 * static_cast<double>(axis);
    if (layout == AxisLayout::uniform) {
      axes[axis] = ndtbl::Axis::uniform(min_value, max_value, shape[axis]);
    } else {
      std::vector<double> coordinates(shape[axis], min_value);
      for (std::size_t index = 0; index < shape[axis]; ++index) {
        const double fraction =
          shape[axis] == 1
            ? 0.0
            : static_cast<double>(index) / static_cast<double>(shape[axis] - 1);
        const double curved = fraction * fraction;
        coordinates[index] = min_value + curved * (max_value - min_value);
      }
      axes[axis] = ndtbl::Axis::from_coordinates(coordinates);
    }
  }
  return axes;
}

template<std::size_t Dim>
std::vector<std::string>
make_field_names(std::size_t field_count)
{
  std::vector<std::string> names;
  names.reserve(field_count);
  for (std::size_t field = 0; field < field_count; ++field) {
    names.push_back("field" + std::to_string(field));
  }
  return names;
}

template<std::size_t Dim>
std::vector<double>
make_payload(const ndtbl::Grid<Dim>& grid, std::size_t field_count)
{
  std::vector<double> payload(grid.point_count() * field_count, 0.0);
  std::array<std::size_t, Dim> indices = {};

  for (std::size_t point = 0; point < grid.point_count(); ++point) {
    std::size_t remainder = point;
    double coordinate_sum = 0.0;
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      indices[axis] = remainder / grid.stride(axis);
      remainder %= grid.stride(axis);
      coordinate_sum += (1.0 + static_cast<double>(axis)) *
                        grid.axis(axis).coordinate(indices[axis]);
    }

    const std::size_t base = point * field_count;
    for (std::size_t field = 0; field < field_count; ++field) {
      payload[base + field] =
        coordinate_sum + 0.25 * static_cast<double>(field);
    }
  }

  return payload;
}

template<std::size_t Dim>
std::vector<std::array<double, Dim>>
make_queries(const std::array<ndtbl::Axis, Dim>& axes, std::size_t count)
{
  std::vector<std::array<double, Dim>> queries;
  queries.reserve(count);

  for (std::size_t query = 0; query < count; ++query) {
    std::array<double, Dim> coordinates = {};
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      const double fraction =
        static_cast<double>(((query + 17u * axis) % 997u) + 1u) / 998.0;
      coordinates[axis] =
        axes[axis].min() + fraction * (axes[axis].max() - axes[axis].min());
    }
    queries.push_back(coordinates);
  }

  return queries;
}

template<std::size_t Dim>
LookupContext<Dim>
make_context(std::size_t extent, AxisLayout layout)
{
  const std::array<std::size_t, Dim> shape = filled_shape<Dim>(extent);
  const std::array<ndtbl::Axis, Dim> axes = make_axes(shape, layout);
  const ndtbl::Grid<Dim> grid(axes);
  const ndtbl::FieldGroup<double, Dim> group(
    grid, make_field_names<Dim>(field_count), make_payload(grid, field_count));
  return LookupContext<Dim>(grid, group, make_queries(axes, query_count));
}

template<std::size_t Dim>
const LookupContext<Dim>&
context(std::size_t extent, AxisLayout layout)
{
  static const LookupContext<Dim> uniform_context =
    make_context<Dim>(default_extent<Dim>(), AxisLayout::uniform);
  static const LookupContext<Dim> explicit_context =
    make_context<Dim>(default_extent<Dim>(), AxisLayout::explicit_coordinates);

  if (extent == default_extent<Dim>() && layout == AxisLayout::uniform) {
    return uniform_context;
  }
  return explicit_context;
}

template<std::size_t Dim>
void
bench_prepare(benchmark::State& state, std::size_t extent, AxisLayout layout)
{
  const LookupContext<Dim>& data = context<Dim>(extent, layout);
  std::size_t query = 0;

  for (auto _ : state) {
    ndtbl::PreparedQuery<Dim> prepared = data.grid.prepare(data.queries[query]);
    benchmark::DoNotOptimize(prepared);
    query = (query + 1) % data.queries.size();
  }
}

template<std::size_t Dim>
void
bench_prepared_evaluate(benchmark::State& state,
                        std::size_t extent,
                        AxisLayout layout)
{
  const LookupContext<Dim>& data = context<Dim>(extent, layout);
  std::vector<double> results(data.group.field_count(), 0.0);

  for (auto _ : state) {
    data.group.evaluate_all_into(data.prepared, results.data());
    benchmark::DoNotOptimize(results.data());
    benchmark::ClobberMemory();
  }
}

template<std::size_t Dim>
void
bench_typed_combined(benchmark::State& state,
                     std::size_t extent,
                     AxisLayout layout)
{
  const LookupContext<Dim>& data = context<Dim>(extent, layout);
  std::vector<double> results(data.group.field_count(), 0.0);
  std::size_t query = 0;

  for (auto _ : state) {
    data.group.evaluate_all_into(data.queries[query], results.data());
    benchmark::DoNotOptimize(results.data());
    benchmark::ClobberMemory();
    query = (query + 1) % data.queries.size();
  }
}

template<std::size_t Dim>
void
bench_runtime_combined(benchmark::State& state,
                       std::size_t extent,
                       AxisLayout layout)
{
  const LookupContext<Dim>& data = context<Dim>(extent, layout);
  std::vector<double> results(data.runtime_group.field_count(), 0.0);
  std::size_t query = 0;

  for (auto _ : state) {
    data.runtime_group.evaluate_all_into(data.queries[query], results.data());
    benchmark::DoNotOptimize(results.data());
    benchmark::ClobberMemory();
    query = (query + 1) % data.queries.size();
  }
}

#define NDTBL_REGISTER_LOOKUP_BENCHMARKS(DIM, EXTENT, LAYOUT, NAME)            \
  BENCHMARK_CAPTURE(bench_prepare<DIM>, NAME, EXTENT, LAYOUT);                 \
  BENCHMARK_CAPTURE(bench_prepared_evaluate<DIM>, NAME, EXTENT, LAYOUT);       \
  BENCHMARK_CAPTURE(bench_typed_combined<DIM>, NAME, EXTENT, LAYOUT);          \
  BENCHMARK_CAPTURE(bench_runtime_combined<DIM>, NAME, EXTENT, LAYOUT)

NDTBL_REGISTER_LOOKUP_BENCHMARKS(2,
                                 default_extent<2>(),
                                 AxisLayout::uniform,
                                 d2_uniform);
NDTBL_REGISTER_LOOKUP_BENCHMARKS(2,
                                 default_extent<2>(),
                                 AxisLayout::explicit_coordinates,
                                 d2_explicit);
NDTBL_REGISTER_LOOKUP_BENCHMARKS(4,
                                 default_extent<4>(),
                                 AxisLayout::uniform,
                                 d4_uniform);
NDTBL_REGISTER_LOOKUP_BENCHMARKS(4,
                                 default_extent<4>(),
                                 AxisLayout::explicit_coordinates,
                                 d4_explicit);
NDTBL_REGISTER_LOOKUP_BENCHMARKS(6,
                                 default_extent<6>(),
                                 AxisLayout::uniform,
                                 d6_uniform);
NDTBL_REGISTER_LOOKUP_BENCHMARKS(6,
                                 default_extent<6>(),
                                 AxisLayout::explicit_coordinates,
                                 d6_explicit);

} // namespace

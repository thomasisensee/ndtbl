#include "ndtbl/ndtbl.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

std::string
temporary_path()
{
  char path_buffer[L_tmpnam];
  if (std::tmpnam(path_buffer) == nullptr) {
    throw std::runtime_error("failed to create temporary path for ndtbl test");
  }

  const std::string path = std::string(path_buffer) + ".ndtbl";
  std::remove(path.c_str());
  return path;
}

} // namespace

TEST_CASE("field group interpolates all fields on a shared grid",
          "[field_group]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::Grid<2> grid(axes);
  const std::vector<std::string> names = { "A", "B" };
  const std::vector<double> values = {
    0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0,
  };

  const ndtbl::FieldGroup<double, 2> group(grid, names, values);
  const std::vector<double> result = group.evaluate_all({ 0.5, 0.5 });

  REQUIRE(result.size() == 2);
  REQUIRE(result[0] == Catch::Approx(1.5));
  REQUIRE(result[1] == Catch::Approx(11.5));
}

TEST_CASE("typed loader round-trips metadata and float payloads", "[io]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<float, 2> group(
    ndtbl::Grid<2>(axes),
    { "A", "B" },
    { 0.0f, 10.0f, 1.0f, 11.0f, 2.0f, 12.0f, 3.0f, 13.0f });

  const std::string path = temporary_path();
  ndtbl::write_group(path, group);

  const ndtbl::GroupMetadata metadata = ndtbl::read_group_metadata(path);
  REQUIRE(metadata.dimension == 2);
  REQUIRE(metadata.field_count == 2);
  REQUIRE(metadata.point_count == 4);
  REQUIRE(metadata.value_type == ndtbl::scalar_type::float32);
  REQUIRE(metadata.field_names == std::vector<std::string>({ "A", "B" }));

  const ndtbl::LoadedFieldGroup<2> loaded = ndtbl::read_group<2>(path);
  std::array<double, 2> values = { 0.0, 0.0 };
  loaded.evaluate_all_into({ 0.5, 0.5 }, values.data());
  REQUIRE(values[0] == Catch::Approx(1.5));
  REQUIRE(values[1] == Catch::Approx(11.5));

  std::remove(path.c_str());
}

TEST_CASE("typed loader rejects mismatched dimensions", "[io]")
{
  const std::array<ndtbl::Axis, 1> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 1> group(
    ndtbl::Grid<1>(axes), { "A" }, { 0.0, 1.0 });

  const std::string path = temporary_path();
  ndtbl::write_group(path, group);

  REQUIRE_THROWS_AS(ndtbl::read_group<2>(path), std::runtime_error);

  std::remove(path.c_str());
}

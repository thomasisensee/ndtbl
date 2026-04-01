#include "ndtbl/ndtbl.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<char>
read_file_bytes(const std::string& path)
{
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open test file for reading");
  }

  return std::vector<char>((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
}

void
write_file_bytes(const std::string& path, const std::vector<char>& bytes)
{
  std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("failed to open test file for writing");
  }

  if (!bytes.empty()) {
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  if (!output.good()) {
    throw std::runtime_error("failed to write test file bytes");
  }
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

TEST_CASE("loaded field group can be rewritten after reading", "[io]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 2> group(
    ndtbl::Grid<2>(axes),
    { "A", "B" },
    { 0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0 });

  const std::string input_path = temporary_path();
  const std::string output_path = temporary_path();
  ndtbl::write_group(input_path, group);

  const ndtbl::LoadedFieldGroup<2> loaded = ndtbl::read_group<2>(input_path);
  ndtbl::write_group(output_path, loaded);

  const ndtbl::LoadedFieldGroup<2> rewritten =
    ndtbl::read_group<2>(output_path);
  std::array<double, 2> values = { 0.0, 0.0 };
  rewritten.evaluate_all_into({ 0.5, 0.5 }, values.data());
  REQUIRE(values[0] == Catch::Approx(1.5));
  REQUIRE(values[1] == Catch::Approx(11.5));

  std::remove(input_path.c_str());
  std::remove(output_path.c_str());
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

TEST_CASE("typed loader rejects truncated payload files", "[io]")
{
  const std::array<ndtbl::Axis, 1> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 1> group(
    ndtbl::Grid<1>(axes), { "A" }, { 0.0, 1.0 });

  const std::string path = temporary_path();
  ndtbl::write_group(path, group);

  std::vector<char> bytes = read_file_bytes(path);
  REQUIRE(bytes.size() > 1);
  bytes.pop_back();
  write_file_bytes(path, bytes);

  REQUIRE_THROWS_AS(ndtbl::read_group<1>(path), std::runtime_error);

  std::remove(path.c_str());
}

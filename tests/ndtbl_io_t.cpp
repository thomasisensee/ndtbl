#include "ndtbl/ndtbl.hpp"

#include "test_support.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

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

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  const ndtbl::GroupMetadata metadata = ndtbl::read_group_metadata(path);
  REQUIRE(metadata.format_version == 1u);
  REQUIRE(metadata.dimension == 2);
  REQUIRE(metadata.field_count == 2);
  REQUIRE(metadata.point_count == 4);
  REQUIRE(metadata.value_type == ndtbl::scalar_type::float32);
  REQUIRE(metadata.field_names == std::vector<std::string>({ "A", "B" }));

  const ndtbl::RuntimeFieldGroup<2> loaded = ndtbl::read_group<2>(path);
  std::array<double, 2> values = { 0.0, 0.0 };
  loaded.evaluate_all_linear_into({ 0.5, 0.5 }, values.data());
  REQUIRE(values[0] == Catch::Approx(1.5));
  REQUIRE(values[1] == Catch::Approx(11.5));

  std::remove(path.c_str());
}

TEST_CASE("runtime field group can be rewritten after reading", "[io]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 2> group(
    ndtbl::Grid<2>(axes),
    { "A", "B" },
    { 0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0 });

  const std::string input_path = ndtbl_test::temporary_path();
  const std::string output_path = ndtbl_test::temporary_path();
  ndtbl::write_group(input_path, group);

  const ndtbl::RuntimeFieldGroup<2> loaded = ndtbl::read_group<2>(input_path);
  ndtbl::write_group(output_path, loaded);

  const ndtbl::RuntimeFieldGroup<2> rewritten =
    ndtbl::read_group<2>(output_path);
  std::array<double, 2> values = { 0.0, 0.0 };
  rewritten.evaluate_all_linear_into({ 0.5, 0.5 }, values.data());
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

  const std::string path = ndtbl_test::temporary_path();
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

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  std::vector<char> bytes = ndtbl_test::read_file_bytes(path);
  REQUIRE(bytes.size() > 1);
  bytes.pop_back();
  ndtbl_test::write_file_bytes(path, bytes);

  REQUIRE_THROWS_AS(ndtbl::read_group<1>(path), std::runtime_error);

  std::remove(path.c_str());
}

TEST_CASE("writer produces the documented little-endian layout", "[io]")
{
  const std::array<ndtbl::Axis, 1> axes = {
    ndtbl::Axis::uniform(2.0, 2.0, 1),
  };
  const ndtbl::FieldGroup<float, 1> group(
    ndtbl::Grid<1>(axes), { "A" }, { 1.5f });

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  std::vector<char> expected;
  expected.insert(expected.end(),
                  { 'N', 'D', 'T', 'B', 'L', '\0', '\0', '\0' });
  ndtbl_test::append_uint_le<std::uint8_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint8_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint16_t>(expected, 0u);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 81u);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint8_t>(expected, 1u);
  ndtbl_test::append_uint_le<std::uint8_t>(expected, 0u);
  ndtbl_test::append_uint_le<std::uint16_t>(expected, 0u);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 1u);
  ndtbl_test::append_double_le(expected, 2.0);
  ndtbl_test::append_double_le(expected, 2.0);
  ndtbl_test::append_uint_le<std::uint64_t>(expected, 1u);
  expected.push_back('A');
  ndtbl_test::append_float_le(expected, 1.5f);

  REQUIRE(ndtbl_test::read_file_bytes(path) == expected);

  std::remove(path.c_str());
}

TEST_CASE("typed loader rejects nonzero reserved header fields", "[io]")
{
  const std::array<ndtbl::Axis, 1> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 1> group(
    ndtbl::Grid<1>(axes), { "A" }, { 0.0, 1.0 });

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  std::vector<char> bytes = ndtbl_test::read_file_bytes(path);
  REQUIRE(bytes.size() > 11);
  bytes[10] = 1;
  ndtbl_test::write_file_bytes(path, bytes);

  REQUIRE_THROWS_AS(ndtbl::read_group_metadata(path), std::runtime_error);

  std::remove(path.c_str());
}

TEST_CASE("typed loader rejects mismatched payload offsets", "[io]")
{
  const std::array<ndtbl::Axis, 1> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::FieldGroup<double, 1> group(
    ndtbl::Grid<1>(axes), { "A" }, { 0.0, 1.0 });

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  std::vector<char> bytes = ndtbl_test::read_file_bytes(path);
  REQUIRE(bytes.size() > 19);
  for (std::size_t index = 12; index < 20; ++index) {
    bytes[index] = 0;
  }
  ndtbl_test::write_file_bytes(path, bytes);

  REQUIRE_THROWS_AS(ndtbl::read_group_metadata(path), std::runtime_error);

  std::remove(path.c_str());
}

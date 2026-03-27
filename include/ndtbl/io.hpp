#pragma once

#include "ndtbl/detail/binary_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

/**
 * @brief Serialize a typed field group to an ndtbl binary stream.
 */
template<class Value, std::size_t Dim>
inline void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  detail::write_group_stream_impl(os, group);
}

/**
 * @brief Write a typed field group to a binary ndtbl file.
 */
template<class Value, std::size_t Dim>
inline void
write_group(const std::string& path, const FieldGroup<Value, Dim>& group)
{
  std::ofstream os(path.c_str(), std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("failed to open ndtbl output file: " + path);
  }
  write_group_stream(os, group);
}

/**
 * @brief Write a runtime-erased field group to a binary ndtbl file.
 */
inline void
write_group(const std::string& path, const AnyFieldGroup& group)
{
  std::ofstream os(path.c_str(), std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("failed to open ndtbl output file: " + path);
  }
  group.write(os);
}

/**
 * @brief Load one ndtbl binary file into a runtime-erased field group.
 */
inline AnyFieldGroup
read_group(const std::string& path)
{
  std::ifstream is(path.c_str(), std::ios::binary);
  if (!is.is_open()) {
    throw std::runtime_error("failed to open ndtbl input file: " + path);
  }

  detail::verify_magic(is);
  const std::uint32_t marker = detail::read_pod<std::uint32_t>(is);
  if (marker != detail::endian_marker) {
    throw std::runtime_error("unsupported ndtbl endianness");
  }

  const std::uint8_t version = detail::read_pod<std::uint8_t>(is);
  if (version != 1u) {
    throw std::runtime_error("unsupported ndtbl version");
  }

  const scalar_type value_type =
    static_cast<scalar_type>(detail::read_pod<std::uint8_t>(is));
  detail::read_pod<std::uint16_t>(is);

  const std::size_t dimension =
    static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
  const std::size_t field_count =
    static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
  const std::size_t point_count =
    static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));

  std::vector<Axis> axes;
  axes.reserve(dimension);
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    const axis_kind kind =
      static_cast<axis_kind>(detail::read_pod<std::uint8_t>(is));
    detail::read_pod<std::uint8_t>(is);
    detail::read_pod<std::uint16_t>(is);
    const std::size_t extent =
      static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));

    if (kind == axis_kind::uniform) {
      const double min_value = detail::read_pod<double>(is);
      const double max_value = detail::read_pod<double>(is);
      axes.push_back(Axis::uniform(min_value, max_value, extent));
    } else if (kind == axis_kind::explicit_coordinates) {
      std::vector<double> coordinates(extent);
      for (std::size_t i = 0; i < extent; ++i) {
        coordinates[i] = detail::read_pod<double>(is);
      }
      axes.push_back(Axis::from_coordinates(coordinates));
    } else {
      throw std::runtime_error("unsupported ndtbl axis kind");
    }
  }

  std::vector<std::string> field_names;
  field_names.reserve(field_count);
  for (std::size_t field = 0; field < field_count; ++field) {
    field_names.push_back(detail::read_string(is));
  }

  std::size_t expected_point_count = 1;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    expected_point_count *= axes[axis].size();
  }
  if (expected_point_count != point_count) {
    throw std::runtime_error("ndtbl point count does not match axis extents");
  }

  const std::size_t value_count = point_count * field_count;
  if (value_type == scalar_type::float32) {
    const std::vector<float> values =
      detail::read_payload<float>(is, value_count);
    return detail::make_any_group<float>(axes, field_names, values);
  }

  if (value_type == scalar_type::float64) {
    const std::vector<double> values =
      detail::read_payload<double>(is, value_count);
    return detail::make_any_group<double>(axes, field_names, values);
  }

  throw std::runtime_error("unsupported ndtbl scalar type");
}

} // namespace ndtbl

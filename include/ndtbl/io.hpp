#pragma once

#include "ndtbl/any_field_group.hpp"
#include "ndtbl/detail/binary_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

template<class Value, std::size_t Dim>
inline void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  detail::write_group_stream_impl(os, group);
}

template<class Value>
inline void
write_group_stream(std::ostream& os,
                   const GroupMetadata& metadata,
                   const std::vector<Value>& interleaved_values)
{
  detail::write_group_stream_impl(os, metadata, interleaved_values);
}

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

template<class Value>
inline void
write_group(const std::string& path,
            const GroupMetadata& metadata,
            const std::vector<Value>& interleaved_values)
{
  std::ofstream os(path.c_str(), std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("failed to open ndtbl output file: " + path);
  }
  write_group_stream(os, metadata, interleaved_values);
}

template<std::size_t Dim>
inline void
write_group(const std::string& path, const LoadedFieldGroup<Dim>& group)
{
  std::ofstream os(path.c_str(), std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("failed to open ndtbl output file: " + path);
  }
  group.write(os);
}

inline GroupMetadata
read_group_metadata(const std::string& path)
{
  std::ifstream is(path.c_str(), std::ios::binary);
  if (!is.is_open()) {
    throw std::runtime_error("failed to open ndtbl input file: " + path);
  }

  return detail::read_group_metadata_impl(is);
}

template<std::size_t Dim>
inline LoadedFieldGroup<Dim>
read_group(const std::string& path)
{
  std::ifstream is(path.c_str(), std::ios::binary);
  if (!is.is_open()) {
    throw std::runtime_error("failed to open ndtbl input file: " + path);
  }

  const GroupMetadata metadata = detail::read_group_metadata_impl(is);
  if (metadata.dimension != Dim) {
    throw std::runtime_error(
      "ndtbl file dimension does not match typed loader");
  }

  const std::array<Axis, Dim> axes = detail::fixed_axes<Dim>(metadata.axes);
  const Grid<Dim> grid(axes);
  const std::size_t value_count = metadata.point_count * metadata.field_count;

  if (metadata.value_type == scalar_type::float32) {
    const std::vector<float> values =
      detail::read_payload<float>(is, value_count);
    return LoadedFieldGroup<Dim>(
      FieldGroup<float, Dim>(grid, metadata.field_names, values));
  }

  if (metadata.value_type == scalar_type::float64) {
    const std::vector<double> values =
      detail::read_payload<double>(is, value_count);
    return LoadedFieldGroup<Dim>(
      FieldGroup<double, Dim>(grid, metadata.field_names, values));
  }

  throw std::runtime_error("unsupported ndtbl scalar type");
}

} // namespace ndtbl

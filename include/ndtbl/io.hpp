#pragma once

#include "ndtbl/any_field_group.hpp"
#include "ndtbl/detail/binary_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

/**
 * @brief Write a typed field group to an already opened binary stream.
 *
 * @tparam Value Scalar payload type stored in the group.
 * @tparam Dim Grid dimensionality of the group.
 * @param os Destination stream in binary mode.
 * @param group Field group to serialize.
 * @see write_group(const std::string&, const FieldGroup<Value, Dim>&)
 */
template<class Value, std::size_t Dim>
inline void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  detail::write_group_stream_impl(os, group);
}

/**
 * @brief Write a raw ndtbl payload with explicit metadata to a binary stream.
 *
 * This overload is useful when the caller already has metadata and an
 * interleaved point-major payload, but not a typed `FieldGroup`.
 *
 * @tparam Value Scalar payload type stored in the payload vector.
 * @param os Destination stream in binary mode.
 * @param metadata Group metadata to encode into the file header.
 * @param interleaved_values Point-major field payload to serialize.
 * @see write_group_stream(std::ostream&, const FieldGroup<Value, Dim>&)
 */
template<class Value>
inline void
write_group_stream(std::ostream& os,
                   const GroupMetadata& metadata,
                   const std::vector<Value>& interleaved_values)
{
  detail::write_group_stream_impl(os, metadata, interleaved_values);
}

/**
 * @brief Write a typed field group to a binary ndtbl file.
 *
 * @tparam Value Scalar payload type stored in the group.
 * @tparam Dim Grid dimensionality of the group.
 * @param path Output file path.
 * @param group Field group to serialize.
 * @see write_group_stream(std::ostream&, const FieldGroup<Value, Dim>&)
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
 * @brief Write a raw ndtbl payload with explicit metadata to a file.
 *
 * @tparam Value Scalar payload type stored in the payload vector.
 * @param path Output file path.
 * @param metadata Group metadata to encode into the file header.
 * @param interleaved_values Point-major field payload to serialize.
 * @see write_group_stream(std::ostream&, const GroupMetadata&,
 *                         const std::vector<Value>&)
 */
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

/**
 * @brief Write a runtime-erased loaded field group to a binary ndtbl file.
 *
 * @tparam Dim Grid dimensionality of the group.
 * @param path Output file path.
 * @param group Runtime-erased field group to serialize.
 * @see LoadedFieldGroup
 */
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

/**
 * @brief Read only the metadata header of an ndtbl file.
 *
 * This function validates the file header and axis descriptors without reading
 * the field payload.
 *
 * @param path Input file path.
 * @return Parsed metadata describing the stored group.
 * @see read_group
 */
inline GroupMetadata
read_group_metadata(const std::string& path)
{
  std::ifstream is(path.c_str(), std::ios::binary);
  if (!is.is_open()) {
    throw std::runtime_error("failed to open ndtbl input file: " + path);
  }

  return detail::read_group_metadata_impl(is);
}

/**
 * @brief Read an ndtbl file into a runtime-erased loaded field group.
 *
 * The file dimension must match the compile-time `Dim` argument.
 *
 * @tparam Dim Expected grid dimensionality of the file.
 * @param path Input file path.
 * @return Runtime-erased loaded field group with either float or double
 *         payload storage.
 * @see read_group_metadata
 * @see LoadedFieldGroup
 */
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

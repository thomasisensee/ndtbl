#pragma once

#include "ndtbl/field_group.hpp"
#include "ndtbl/metadata.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {
namespace detail {

static const char file_magic[8] = { 'N', 'D', 'T', 'B', 'L', '1', '\0', '\0' };
static const std::uint32_t endian_marker = 0x01020304u;

template<class Pod>
inline void
write_pod(std::ostream& os, const Pod& value)
{
  os.write(reinterpret_cast<const char*>(&value), sizeof(Pod));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl payload");
  }
}

template<class Pod>
inline Pod
read_pod(std::istream& is)
{
  Pod value;
  is.read(reinterpret_cast<char*>(&value), sizeof(Pod));
  if (!is.good()) {
    throw std::runtime_error("failed to read ndtbl payload");
  }
  return value;
}

inline void
write_string(std::ostream& os, const std::string& value)
{
  const std::uint64_t size = static_cast<std::uint64_t>(value.size());
  write_pod(os, size);
  os.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl string");
  }
}

inline std::string
read_string(std::istream& is)
{
  const std::uint64_t size = read_pod<std::uint64_t>(is);
  if (size == 0) {
    return std::string();
  }

  std::string value(size, '\0');
  is.read(&value[0], static_cast<std::streamsize>(size));
  if (!is.good()) {
    throw std::runtime_error("failed to read ndtbl string");
  }
  return value;
}

template<class Value>
inline void
write_group_stream_impl(std::ostream& os,
                        const GroupMetadata& metadata,
                        const std::vector<Value>& payload)
{
  if (metadata.axes.size() != metadata.dimension) {
    throw std::invalid_argument(
      "ndtbl metadata axis count does not match dimension");
  }

  if (metadata.field_names.size() != metadata.field_count) {
    throw std::invalid_argument(
      "ndtbl metadata field count does not match field names");
  }

  const std::size_t expected_values =
    metadata.point_count * metadata.field_count;
  if (payload.size() != expected_values) {
    throw std::invalid_argument("ndtbl payload size does not match metadata");
  }

  os.write(file_magic, sizeof(file_magic));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl header");
  }

  write_pod(os, endian_marker);
  write_pod<std::uint8_t>(os, 1u);
  write_pod<std::uint8_t>(os, static_cast<std::uint8_t>(metadata.value_type));
  write_pod<std::uint16_t>(os, 0u);
  write_pod<std::uint64_t>(os, static_cast<std::uint64_t>(metadata.dimension));
  write_pod<std::uint64_t>(os,
                           static_cast<std::uint64_t>(metadata.field_count));
  write_pod<std::uint64_t>(os,
                           static_cast<std::uint64_t>(metadata.point_count));

  for (std::size_t axis = 0; axis < metadata.axes.size(); ++axis) {
    const Axis& axis_spec = metadata.axes[axis];
    write_pod<std::uint8_t>(os, static_cast<std::uint8_t>(axis_spec.kind()));
    write_pod<std::uint8_t>(os, 0u);
    write_pod<std::uint16_t>(os, 0u);
    write_pod<std::uint64_t>(os, static_cast<std::uint64_t>(axis_spec.size()));
    if (axis_spec.kind() == axis_kind::uniform) {
      write_pod(os, axis_spec.min());
      write_pod(os, axis_spec.max());
    } else {
      const std::vector<double> coordinates = axis_spec.coordinates();
      for (std::size_t i = 0; i < coordinates.size(); ++i) {
        write_pod(os, coordinates[i]);
      }
    }
  }

  for (std::size_t field = 0; field < metadata.field_names.size(); ++field) {
    write_string(os, metadata.field_names[field]);
  }

  os.write(reinterpret_cast<const char*>(payload.data()),
           static_cast<std::streamsize>(payload.size() * sizeof(Value)));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl field payload");
  }
}

template<class Value, std::size_t Dim>
inline void
write_group_stream_impl(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  GroupMetadata metadata = { scalar_type_of<Value>(),
                             Dim,
                             group.field_count(),
                             group.point_count(),
                             std::vector<Axis>(group.grid().axes().begin(),
                                               group.grid().axes().end()),
                             group.field_names() };
  write_group_stream_impl(os, metadata, group.interleaved_values());
}

inline void
verify_magic(std::istream& is)
{
  char magic[sizeof(file_magic)] = {};
  is.read(magic, sizeof(magic));
  if (!is.good() ||
      !std::equal(magic, magic + sizeof(file_magic), file_magic)) {
    throw std::runtime_error("invalid ndtbl magic header");
  }
}

inline GroupMetadata
read_group_metadata_impl(std::istream& is)
{
  verify_magic(is);
  const std::uint32_t marker = read_pod<std::uint32_t>(is);
  if (marker != endian_marker) {
    throw std::runtime_error("unsupported ndtbl endianness");
  }

  const std::uint8_t version = read_pod<std::uint8_t>(is);
  if (version != 1u) {
    throw std::runtime_error("unsupported ndtbl version");
  }

  GroupMetadata metadata;
  metadata.value_type = static_cast<scalar_type>(read_pod<std::uint8_t>(is));
  read_pod<std::uint16_t>(is);

  metadata.dimension = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
  metadata.field_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
  metadata.point_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));

  metadata.axes.reserve(metadata.dimension);
  for (std::size_t axis = 0; axis < metadata.dimension; ++axis) {
    const axis_kind kind = static_cast<axis_kind>(read_pod<std::uint8_t>(is));
    read_pod<std::uint8_t>(is);
    read_pod<std::uint16_t>(is);
    const std::size_t extent =
      static_cast<std::size_t>(read_pod<std::uint64_t>(is));

    if (kind == axis_kind::uniform) {
      const double min_value = read_pod<double>(is);
      const double max_value = read_pod<double>(is);
      metadata.axes.push_back(Axis::uniform(min_value, max_value, extent));
    } else if (kind == axis_kind::explicit_coordinates) {
      std::vector<double> coordinates(extent);
      for (std::size_t i = 0; i < extent; ++i) {
        coordinates[i] = read_pod<double>(is);
      }
      metadata.axes.push_back(Axis::from_coordinates(coordinates));
    } else {
      throw std::runtime_error("unsupported ndtbl axis kind");
    }
  }

  metadata.field_names.reserve(metadata.field_count);
  for (std::size_t field = 0; field < metadata.field_count; ++field) {
    metadata.field_names.push_back(read_string(is));
  }

  std::size_t expected_point_count = 1;
  for (std::size_t axis = 0; axis < metadata.axes.size(); ++axis) {
    expected_point_count *= metadata.axes[axis].size();
  }
  if (expected_point_count != metadata.point_count) {
    throw std::runtime_error("ndtbl point count does not match axis extents");
  }

  return metadata;
}

template<class Value>
inline std::vector<Value>
read_payload(std::istream& is, std::size_t value_count)
{
  std::vector<Value> values(value_count);
  is.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(Value)));
  if (!is.good()) {
    throw std::runtime_error("failed to read ndtbl field payload");
  }
  return values;
}

template<std::size_t Dim>
inline std::array<Axis, Dim>
fixed_axes(const std::vector<Axis>& axes)
{
  if (axes.size() != Dim) {
    throw std::invalid_argument(
      "ndtbl axis count does not match typed dimension");
  }

  std::array<Axis, Dim> fixed = {};
  std::copy(axes.begin(), axes.end(), fixed.begin());
  return fixed;
}

} // namespace detail
} // namespace ndtbl

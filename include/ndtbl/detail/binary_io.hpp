#pragma once

#include "ndtbl/any_field_group.hpp"

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

template<class Value, std::size_t Dim>
inline void
write_group_stream_impl(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  os.write(file_magic, sizeof(file_magic));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl header");
  }

  write_pod(os, endian_marker);
  write_pod<std::uint8_t>(os, 1u);
  write_pod<std::uint8_t>(os,
                          static_cast<std::uint8_t>(scalar_type_of<Value>()));
  write_pod<std::uint16_t>(os, 0u);
  write_pod<std::uint64_t>(os, static_cast<std::uint64_t>(Dim));
  write_pod<std::uint64_t>(os, static_cast<std::uint64_t>(group.field_count()));
  write_pod<std::uint64_t>(os, static_cast<std::uint64_t>(group.point_count()));

  for (std::size_t axis = 0; axis < Dim; ++axis) {
    const Axis& axis_spec = group.grid().axis(axis);
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

  const std::vector<std::string>& names = group.field_names();
  for (std::size_t field = 0; field < names.size(); ++field) {
    write_string(os, names[field]);
  }

  const std::vector<Value>& payload = group.interleaved_values();
  os.write(reinterpret_cast<const char*>(payload.data()),
           static_cast<std::streamsize>(payload.size() * sizeof(Value)));
  if (!os.good()) {
    throw std::runtime_error("failed to write ndtbl field payload");
  }
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

template<class Value>
inline AnyFieldGroup
make_any_group(const std::vector<Axis>& axes,
               const std::vector<std::string>& field_names,
               const std::vector<Value>& interleaved_values)
{
  switch (axes.size()) {
    case 1: {
      std::array<Axis, 1> fixed_axes = { axes[0] };
      return AnyFieldGroup(FieldGroup<Value, 1>(
        Grid<1>(fixed_axes), field_names, interleaved_values));
    }
    case 2: {
      std::array<Axis, 2> fixed_axes = { axes[0], axes[1] };
      return AnyFieldGroup(FieldGroup<Value, 2>(
        Grid<2>(fixed_axes), field_names, interleaved_values));
    }
    case 3: {
      std::array<Axis, 3> fixed_axes = { axes[0], axes[1], axes[2] };
      return AnyFieldGroup(FieldGroup<Value, 3>(
        Grid<3>(fixed_axes), field_names, interleaved_values));
    }
    case 4: {
      std::array<Axis, 4> fixed_axes = { axes[0], axes[1], axes[2], axes[3] };
      return AnyFieldGroup(FieldGroup<Value, 4>(
        Grid<4>(fixed_axes), field_names, interleaved_values));
    }
    case 5: {
      std::array<Axis, 5> fixed_axes = {
        axes[0], axes[1], axes[2], axes[3], axes[4]
      };
      return AnyFieldGroup(FieldGroup<Value, 5>(
        Grid<5>(fixed_axes), field_names, interleaved_values));
    }
    case 6: {
      std::array<Axis, 6> fixed_axes = { axes[0], axes[1], axes[2],
                                         axes[3], axes[4], axes[5] };
      return AnyFieldGroup(FieldGroup<Value, 6>(
        Grid<6>(fixed_axes), field_names, interleaved_values));
    }
    case 7: {
      std::array<Axis, 7> fixed_axes = { axes[0], axes[1], axes[2], axes[3],
                                         axes[4], axes[5], axes[6] };
      return AnyFieldGroup(FieldGroup<Value, 7>(
        Grid<7>(fixed_axes), field_names, interleaved_values));
    }
    case 8: {
      std::array<Axis, 8> fixed_axes = { axes[0], axes[1], axes[2], axes[3],
                                         axes[4], axes[5], axes[6], axes[7] };
      return AnyFieldGroup(FieldGroup<Value, 8>(
        Grid<8>(fixed_axes), field_names, interleaved_values));
    }
    default:
      throw std::invalid_argument("ndtbl supports dimensions 1 through 8");
  }
}

} // namespace detail
} // namespace ndtbl

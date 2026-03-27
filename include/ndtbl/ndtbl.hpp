#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ndtbl {

enum class axis_kind : std::uint8_t
{
  uniform = 1,
  explicit_coordinates = 2
};

enum class scalar_type : std::uint8_t
{
  float32 = 1,
  float64 = 2
};

template<class Value>
inline scalar_type
scalar_type_of()
{
  static_assert(std::is_same<Value, float>::value ||
                  std::is_same<Value, double>::value,
                "ndtbl supports only float and double payloads");

  return std::is_same<Value, float>::value ? scalar_type::float32
                                           : scalar_type::float64;
}

class Axis
{
public:
  Axis()
    : kind_(axis_kind::uniform)
    , size_(1)
    , min_(0.0)
    , max_(0.0)
  {
  }

  static Axis uniform(double min_value, double max_value, std::size_t size)
  {
    if (size == 0) {
      throw std::invalid_argument(
        "uniform axis must contain at least one coordinate");
    }

    if (size > 1 && !(max_value > min_value)) {
      throw std::invalid_argument(
        "uniform axis requires max > min when size > 1");
    }

    Axis axis;
    axis.kind_ = axis_kind::uniform;
    axis.size_ = size;
    axis.min_ = min_value;
    axis.max_ = size == 1 ? min_value : max_value;
    axis.coordinates_.clear();
    return axis;
  }

  static Axis from_coordinates(const std::vector<double>& coordinates)
  {
    if (coordinates.empty()) {
      throw std::invalid_argument(
        "explicit axis must contain at least one coordinate");
    }

    for (std::size_t i = 1; i < coordinates.size(); ++i) {
      if (!(coordinates[i] > coordinates[i - 1])) {
        throw std::invalid_argument(
          "explicit axis coordinates must be strictly increasing");
      }
    }

    Axis axis;
    axis.kind_ = axis_kind::explicit_coordinates;
    axis.size_ = coordinates.size();
    axis.min_ = coordinates.front();
    axis.max_ = coordinates.back();
    axis.coordinates_ = coordinates;
    return axis;
  }

  axis_kind kind() const { return kind_; }

  std::size_t size() const { return size_; }

  double min() const { return min_; }

  double max() const { return max_; }

  double coordinate(std::size_t index) const
  {
    if (index >= size_) {
      throw std::out_of_range("axis coordinate index out of range");
    }

    if (kind_ == axis_kind::uniform) {
      if (size_ == 1) {
        return min_;
      }

      const double fraction =
        static_cast<double>(index) / static_cast<double>(size_ - 1);
      return min_ + fraction * (max_ - min_);
    }

    return coordinates_[index];
  }

  std::vector<double> coordinates() const
  {
    if (kind_ == axis_kind::uniform) {
      std::vector<double> generated(size_);
      for (std::size_t i = 0; i < size_; ++i) {
        generated[i] = coordinate(i);
      }
      return generated;
    }

    return coordinates_;
  }

  bool equivalent(const Axis& other) const
  {
    if (kind_ != other.kind_ || size_ != other.size_) {
      return false;
    }

    if (kind_ == axis_kind::uniform) {
      return min_ == other.min_ && max_ == other.max_;
    }

    return coordinates_ == other.coordinates_;
  }

  std::pair<std::size_t, double> bracket(double value) const
  {
    if (size_ == 1) {
      return std::make_pair(std::size_t(0), 0.0);
    }

    if (kind_ == axis_kind::uniform) {
      const double clamped = std::max(min_, std::min(max_, value));
      const double scaled =
        (clamped - min_) / (max_ - min_) * static_cast<double>(size_ - 1);
      const std::size_t lower =
        std::min(static_cast<std::size_t>(scaled), size_ - 2);
      const double weight =
        std::max(0.0, std::min(1.0, scaled - static_cast<double>(lower)));
      return std::make_pair(lower, weight);
    }

    if (value <= coordinates_.front()) {
      return std::make_pair(std::size_t(0), 0.0);
    }

    if (value >= coordinates_.back()) {
      return std::make_pair(size_ - 2, 1.0);
    }

    const std::vector<double>::const_iterator upper =
      std::upper_bound(coordinates_.begin(), coordinates_.end(), value);
    const std::size_t lower_index =
      static_cast<std::size_t>(std::distance(coordinates_.begin(), upper) - 1);
    const double lower_value = coordinates_[lower_index];
    const double upper_value = coordinates_[lower_index + 1];
    const double weight = (value - lower_value) / (upper_value - lower_value);
    return std::make_pair(lower_index, std::max(0.0, std::min(1.0, weight)));
  }

private:
  axis_kind kind_;
  std::size_t size_;
  double min_;
  double max_;
  std::vector<double> coordinates_;
};

template<class Value, std::size_t Dim>
class FieldGroup;

template<class Value, std::size_t Dim>
void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group);

template<std::size_t Dim>
class PreparedQuery
{
public:
  static constexpr std::size_t dimensions = Dim;
  static constexpr std::size_t corners = std::size_t(1) << Dim;

  const std::array<std::size_t, corners>& point_indices() const
  {
    return point_indices_;
  }

  const std::array<double, corners>& weights() const { return weights_; }

private:
  template<std::size_t>
  friend class Grid;

  std::array<std::size_t, corners> point_indices_;
  std::array<double, corners> weights_;
};

template<std::size_t Dim>
class Grid
{
public:
  static constexpr std::size_t dimensions = Dim;

  Grid()
  {
    extents_.fill(0);
    strides_.fill(0);
  }

  explicit Grid(const std::array<Axis, Dim>& axes)
    : axes_(axes)
  {
    strides_[Dim - 1] = 1;
    for (std::size_t axis = Dim; axis-- > 0;) {
      extents_[axis] = axes_[axis].size();
      if (axis + 1 < Dim) {
        strides_[axis] = strides_[axis + 1] * extents_[axis + 1];
      }
    }
  }

  const std::array<Axis, Dim>& axes() const { return axes_; }

  const Axis& axis(std::size_t index) const { return axes_[index]; }

  const std::array<std::size_t, Dim>& extents() const { return extents_; }

  const std::array<std::size_t, Dim>& strides() const { return strides_; }

  std::size_t point_count() const
  {
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      count *= extents_[axis];
    }
    return count;
  }

  bool equivalent(const Grid& other) const
  {
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      if (!axes_[axis].equivalent(other.axes_[axis])) {
        return false;
      }
    }
    return true;
  }

  PreparedQuery<Dim> prepare(const std::array<double, Dim>& coordinates) const
  {
    std::array<std::size_t, Dim> lower_indices;
    std::array<std::size_t, Dim> upper_indices;
    std::array<double, Dim> upper_weights;

    for (std::size_t axis = 0; axis < Dim; ++axis) {
      const std::pair<std::size_t, double> bracket =
        axes_[axis].bracket(coordinates[axis]);
      lower_indices[axis] = bracket.first;
      upper_indices[axis] =
        std::min(lower_indices[axis] + 1, extents_[axis] - 1);
      upper_weights[axis] = bracket.second;
    }

    PreparedQuery<Dim> prepared;
    for (std::size_t mask = 0; mask < PreparedQuery<Dim>::corners; ++mask) {
      double weight = 1.0;
      std::size_t linear_index = 0;

      for (std::size_t axis = 0; axis < Dim; ++axis) {
        const bool use_upper = (mask & (std::size_t(1) << axis)) != 0;
        const std::size_t index =
          use_upper ? upper_indices[axis] : lower_indices[axis];
        const double axis_weight =
          use_upper ? upper_weights[axis] : (1.0 - upper_weights[axis]);
        linear_index += index * strides_[axis];
        weight *= axis_weight;
      }

      prepared.point_indices_[mask] = linear_index;
      prepared.weights_[mask] = weight;
    }

    return prepared;
  }

private:
  std::array<Axis, Dim> axes_;
  std::array<std::size_t, Dim> extents_;
  std::array<std::size_t, Dim> strides_;
};

template<class Value, std::size_t Dim>
class FieldGroup
{
public:
  FieldGroup(const Grid<Dim>& grid,
             const std::vector<std::string>& field_names,
             const std::vector<Value>& interleaved_values)
    : grid_(grid)
    , field_names_(field_names)
    , interleaved_values_(interleaved_values)
  {
    if (field_names_.empty()) {
      throw std::invalid_argument(
        "field group must contain at least one field");
    }

    const std::size_t expected_size = grid_.point_count() * field_names_.size();
    if (interleaved_values_.size() != expected_size) {
      throw std::invalid_argument(
        "field payload size does not match grid and field count");
    }
  }

  const Grid<Dim>& grid() const { return grid_; }

  std::size_t field_count() const { return field_names_.size(); }

  const std::vector<std::string>& field_names() const { return field_names_; }

  std::size_t point_count() const { return grid_.point_count(); }

  const std::vector<Value>& interleaved_values() const
  {
    return interleaved_values_;
  }

  std::size_t field_index(const std::string& name) const
  {
    const std::vector<std::string>::const_iterator found =
      std::find(field_names_.begin(), field_names_.end(), name);
    if (found == field_names_.end()) {
      throw std::out_of_range("field not found in ndtbl field group");
    }
    return static_cast<std::size_t>(std::distance(field_names_.begin(), found));
  }

  std::vector<Value> evaluate_all(const PreparedQuery<Dim>& prepared) const
  {
    std::vector<Value> results(field_count(), Value(0));
    for (std::size_t corner = 0; corner < PreparedQuery<Dim>::corners;
         ++corner) {
      const double weight = prepared.weights()[corner];
      const std::size_t base = prepared.point_indices()[corner] * field_count();
      for (std::size_t field = 0; field < field_count(); ++field) {
        results[field] = static_cast<Value>(
          results[field] +
          static_cast<Value>(weight * interleaved_values_[base + field]));
      }
    }
    return results;
  }

  std::vector<Value> evaluate_all(
    const std::array<double, Dim>& coordinates) const
  {
    return evaluate_all(grid_.prepare(coordinates));
  }

private:
  Grid<Dim> grid_;
  std::vector<std::string> field_names_;
  std::vector<Value> interleaved_values_;
};

class AnyFieldGroup
{
public:
  AnyFieldGroup() {}

  template<class Value, std::size_t Dim>
  explicit AnyFieldGroup(const FieldGroup<Value, Dim>& group)
    : impl_(std::make_shared<Model<Value, Dim>>(group))
  {
  }

  bool empty() const { return !impl_; }

  std::size_t dimension() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->dimension();
  }

  std::size_t field_count() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_count();
  }

  scalar_type value_type() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->value_type();
  }

  std::vector<std::string> field_names() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_names();
  }

  std::vector<Axis> axes() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->axes();
  }

  std::size_t field_index(const std::string& field_name) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_index(field_name);
  }

  std::vector<double> evaluate_all(const std::vector<double>& coordinates) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->evaluate_all(coordinates);
  }

  void write(std::ostream& os) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    impl_->write(os);
  }

private:
  struct Concept
  {
    virtual ~Concept() {}
    virtual std::size_t dimension() const = 0;
    virtual std::size_t field_count() const = 0;
    virtual scalar_type value_type() const = 0;
    virtual std::vector<std::string> field_names() const = 0;
    virtual std::vector<Axis> axes() const = 0;
    virtual std::size_t field_index(const std::string& field_name) const = 0;
    virtual std::vector<double> evaluate_all(
      const std::vector<double>& coordinates) const = 0;
    virtual void write(std::ostream& os) const = 0;
  };

  template<class Value, std::size_t Dim>
  struct Model : Concept
  {
    explicit Model(const FieldGroup<Value, Dim>& group)
      : group_(group)
    {
    }

    std::size_t dimension() const { return Dim; }

    std::size_t field_count() const { return group_.field_count(); }

    scalar_type value_type() const { return scalar_type_of<Value>(); }

    std::vector<std::string> field_names() const
    {
      return group_.field_names();
    }

    std::vector<Axis> axes() const
    {
      const std::array<Axis, Dim>& group_axes = group_.grid().axes();
      return std::vector<Axis>(group_axes.begin(), group_axes.end());
    }

    std::size_t field_index(const std::string& field_name) const
    {
      return group_.field_index(field_name);
    }

    std::vector<double> evaluate_all(
      const std::vector<double>& coordinates) const
    {
      if (coordinates.size() != Dim) {
        throw std::invalid_argument(
          "query dimensionality does not match ndtbl field group");
      }

      std::array<double, Dim> query;
      std::copy(coordinates.begin(), coordinates.end(), query.begin());
      const std::vector<Value> values = group_.evaluate_all(query);
      return std::vector<double>(values.begin(), values.end());
    }

    void write(std::ostream& os) const { write_group_stream(os, group_); }

    FieldGroup<Value, Dim> group_;
  };

  std::shared_ptr<const Concept> impl_;

  template<class Value, std::size_t Dim>
  friend void write_group_stream(std::ostream& os,
                                 const FieldGroup<Value, Dim>& group);
};

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

template<class Value, std::size_t Dim>
inline void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group)
{
  detail::write_group_stream_impl(os, group);
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

inline void
write_group(const std::string& path, const AnyFieldGroup& group)
{
  std::ofstream os(path.c_str(), std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("failed to open ndtbl output file: " + path);
  }
  group.write(os);
}

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

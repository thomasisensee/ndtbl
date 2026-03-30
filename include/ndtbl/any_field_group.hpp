#pragma once

#include "ndtbl/field_group.hpp"
#include "ndtbl/types.hpp"

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

template<class Value, std::size_t Dim>
void
write_group_stream(std::ostream& os, const FieldGroup<Value, Dim>& group);

/**
 * @brief Runtime-erased wrapper around a typed `FieldGroup<Value, Dim>`.
 *
 * This wrapper is primarily used by binary loading and adapter code that only
 * learns the field-group dimensionality and precision at runtime.
 */
class AnyFieldGroup
{
public:
  AnyFieldGroup() {}

  /**
   * @brief Store a typed field group in a runtime-erased wrapper.
   */
  template<class Value, std::size_t Dim>
  explicit AnyFieldGroup(const FieldGroup<Value, Dim>& group)
    : impl_(std::make_shared<Model<Value, Dim>>(group))
  {
  }

  /**
   * @brief Return whether this wrapper currently stores a field group.
   */
  bool empty() const { return !impl_; }

  /**
   * @brief Return the dimensionality of the stored field group.
   */
  std::size_t dimension() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->dimension();
  }

  /**
   * @brief Return the number of stored fields.
   */
  std::size_t field_count() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_count();
  }

  /**
   * @brief Return the scalar storage type of the wrapped field group.
   */
  scalar_type value_type() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->value_type();
  }

  /**
   * @brief Return the field names in storage order.
   */
  std::vector<std::string> field_names() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_names();
  }

  /**
   * @brief Return the axis descriptors of the stored grid.
   */
  std::vector<Axis> axes() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->axes();
  }

  /**
   * @brief Resolve a field name to its field index.
   */
  std::size_t field_index(const std::string& field_name) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_index(field_name);
  }

  /**
   * @brief Evaluate all fields at one runtime-sized query point.
   */
  std::vector<double> evaluate_all(const std::vector<double>& coordinates) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->evaluate_all(coordinates);
  }

  /**
   * @brief Serialize the wrapped field group to an ndtbl binary stream.
   */
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

    constexpr std::size_t dimension() const noexcept override { return Dim; }

    std::size_t field_count() const override { return group_.field_count(); }

    constexpr scalar_type value_type() const noexcept override
    {
      return scalar_type_of<Value>();
    }

    std::vector<std::string> field_names() const override
    {
      return group_.field_names();
    }

    std::vector<Axis> axes() const override
    {
      const std::array<Axis, Dim>& group_axes = group_.grid().axes();
      return std::vector<Axis>(group_axes.begin(), group_axes.end());
    }

    std::size_t field_index(const std::string& field_name) const override
    {
      return group_.field_index(field_name);
    }

    std::vector<double> evaluate_all(
      const std::vector<double>& coordinates) const override
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

    void write(std::ostream& os) const override
    {
      write_group_stream(os, group_);
    }

    FieldGroup<Value, Dim> group_;
  };

  std::shared_ptr<const Concept> impl_;
};

} // namespace ndtbl

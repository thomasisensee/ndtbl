#pragma once

#include "ndtbl/field_group.hpp"
#include "ndtbl/metadata.hpp"
#include "ndtbl/types.hpp"

#include <array>
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
 * The dimensionality remains part of the type. Only the stored scalar payload
 * type is selected from file metadata at runtime.
 */
template<std::size_t Dim>
class RuntimeFieldGroup
{
public:
  /**
   * @brief Construct an empty runtime-erased group handle.
   */
  RuntimeFieldGroup() {}

  /**
   * @brief Construct a runtime-erased wrapper from a typed field group.
   *
   * @tparam Value Scalar payload type stored in the source group.
   * @param group Typed field group to wrap.
   * @see FieldGroup
   */
  template<class Value>
  explicit RuntimeFieldGroup(const FieldGroup<Value, Dim>& group)
    : impl_(std::make_shared<Model<Value>>(group))
  {
  }

  /**
   * @brief Check whether this wrapper currently holds a group instance.
   *
   * @return `true` if no group is loaded, otherwise `false`.
   */
  bool empty() const { return !impl_; }

  /**
   * @brief Return the number of fields stored per grid point.
   *
   * @return Number of named fields in the wrapped group.
   */
  std::size_t field_count() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_count();
  }

  /**
   * @brief Return the scalar payload type of the wrapped group.
   *
   * @return Runtime payload type tag.
   */
  scalar_type value_type() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->value_type();
  }

  /**
   * @brief Return field names in storage order.
   *
   * @return Copy of the field name list.
   */
  std::vector<std::string> field_names() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_names();
  }

  /**
   * @brief Return the axis descriptors of the wrapped grid.
   *
   * @return One axis descriptor per dimension.
   */
  std::array<Axis, Dim> axes() const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->axes();
  }

  /**
   * @brief Resolve a field name to its storage index.
   *
   * @param field_name Field name to look up.
   * @return Zero-based field index in storage order.
   */
  std::size_t field_index(const std::string& field_name) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    return impl_->field_index(field_name);
  }

  /**
   * @brief Evaluate all stored fields at one coordinate tuple.
   *
   * @param coordinates Query coordinates in grid axis order.
   * @return Interpolated field values converted to `double`.
   * @see evaluate_all_into
   */
  std::vector<double> evaluate_all(
    const std::array<double, Dim>& coordinates) const
  {
    std::vector<double> values(field_count(), 0.0);
    evaluate_all_into(coordinates, values.data());
    return values;
  }

  /**
   * @brief Evaluate all stored fields into caller-provided output storage.
   *
   * @param coordinates Query coordinates in grid axis order.
   * @param values Output buffer with space for `field_count()` values.
   * @see evaluate_all
   */
  void evaluate_all_into(const std::array<double, Dim>& coordinates,
                         double* values) const
  {
    if (!impl_) {
      throw std::runtime_error("ndtbl field group is empty");
    }
    impl_->evaluate_all_into(coordinates, values);
  }

  /**
   * @brief Write the wrapped group to an already opened binary stream.
   *
   * @param os Destination stream in binary mode.
   * @see write_group_stream
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
    virtual std::size_t field_count() const = 0;
    virtual scalar_type value_type() const = 0;
    virtual std::vector<std::string> field_names() const = 0;
    virtual std::array<Axis, Dim> axes() const = 0;
    virtual std::size_t field_index(const std::string& field_name) const = 0;
    virtual void evaluate_all_into(const std::array<double, Dim>& coordinates,
                                   double* values) const = 0;
    virtual void write(std::ostream& os) const = 0;
  };

  template<class Value>
  struct Model : Concept
  {
    explicit Model(const FieldGroup<Value, Dim>& group)
      : group_(group)
      , scratch_(group.field_count(), Value(0))
    {
    }

    std::size_t field_count() const override { return group_.field_count(); }

    scalar_type value_type() const override { return scalar_type_of<Value>(); }

    std::vector<std::string> field_names() const override
    {
      return group_.field_names();
    }

    std::array<Axis, Dim> axes() const override { return group_.grid().axes(); }

    std::size_t field_index(const std::string& field_name) const override
    {
      return group_.field_index(field_name);
    }

    void evaluate_all_into(const std::array<double, Dim>& coordinates,
                           double* values) const override
    {
      group_.evaluate_all_into(coordinates, scratch_.data());
      for (std::size_t field = 0; field < scratch_.size(); ++field) {
        values[field] = static_cast<double>(scratch_[field]);
      }
    }

    void write(std::ostream& os) const override
    {
      write_group_stream(os, group_);
    }

    FieldGroup<Value, Dim> group_;
    mutable std::vector<Value> scratch_;
  };

  std::shared_ptr<const Concept> impl_;
};

} // namespace ndtbl

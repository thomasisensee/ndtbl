#pragma once

#include "ndtbl/grid.hpp"
#include "ndtbl/payload.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

/**
 * @brief One grid plus one or more named fields stored in interleaved flat
 * memory.
 *
 * The storage layout is point-major in row-major grid order:
 * `point0.field0, point0.field1, ..., point1.field0, ...`
 * where the last grid axis varies fastest before stepping to the next field
 * tuple.
 * so that one prepared interpolation query can accumulate all fields together.
 */
template<class Value, std::size_t Dim>
class FieldGroup
{
public:
  /**
   * @brief Construct a field group on a shared grid.
   *
   * @param grid Shared interpolation grid.
   * @param field_names Logical names of the stored fields.
   * @param interleaved_values Flat point-major payload.
   */
  FieldGroup(const Grid<Dim>& grid,
             const std::vector<std::string>& field_names,
             const std::vector<Value>& interleaved_values)
    : FieldGroup(grid, field_names, std::vector<Value>(interleaved_values))
  {
  }

  /**
   * @brief Construct a field group from owned payload storage.
   *
   * @param grid Shared interpolation grid.
   * @param field_names Logical names of the stored fields.
   * @param interleaved_values Flat point-major payload whose ownership is moved
   *                           into the group.
   */
  FieldGroup(const Grid<Dim>& grid,
             const std::vector<std::string>& field_names,
             std::vector<Value>&& interleaved_values)
    : grid_(grid)
    , field_names_(field_names)
  {
    adopt_owned_payload(std::move(interleaved_values));
    validate_payload_shape();
  }

  /**
   * @brief Construct a field group from externally managed payload storage.
   *
   * @param grid Shared interpolation grid.
   * @param field_names Logical names of the stored fields.
   * @param interleaved_values View over a contiguous point-major payload.
   * @param payload_owner Shared owner keeping the viewed payload alive.
   */
  FieldGroup(const Grid<Dim>& grid,
             const std::vector<std::string>& field_names,
             const PayloadView<Value>& interleaved_values,
             std::shared_ptr<const std::uint8_t> payload_owner)
    : grid_(grid)
    , field_names_(field_names)
    , interleaved_values_(interleaved_values)
    , payload_owner_(std::move(payload_owner))
  {
    if (interleaved_values_.size() != 0 && !payload_owner_) {
      throw std::invalid_argument("field group payload owner is empty");
    }
    validate_payload_shape();
  }

  /**
   * @brief Return the shared grid metadata.
   *
   * @return Shared interpolation grid.
   */
  const Grid<Dim>& grid() const { return grid_; }

  /**
   * @brief Return the number of stored fields.
   *
   * @return Number of named fields.
   */
  std::size_t field_count() const { return field_names_.size(); }

  /**
   * @brief Return all field names in storage order.
   *
   * @return Field names in payload order.
   */
  const std::vector<std::string>& field_names() const { return field_names_; }

  /**
   * @brief Return the number of support points in the shared grid.
   *
   * @return Number of grid support points.
   */
  std::size_t point_count() const { return grid_.point_count(); }

  /**
   * @brief Return a read-only view of the raw interleaved field payload.
   *
   * @return Point-major interleaved payload view.
   */
  PayloadView<Value> interleaved_values() const { return interleaved_values_; }

  /**
   * @brief Resolve a field name to its local field index.
   *
   * @param name Field name to look up.
   * @return Zero-based field index in storage order.
   */
  std::size_t field_index(const std::string& name) const
  {
    const std::vector<std::string>::const_iterator found =
      std::find(field_names_.begin(), field_names_.end(), name);
    if (found == field_names_.end()) {
      throw std::out_of_range("field not found in ndtbl field group");
    }
    return static_cast<std::size_t>(std::distance(field_names_.begin(), found));
  }

  /**
   * @brief Evaluate all fields using a previously prepared interpolation
   * stencil.
   *
   * @param prepared Prepared query to reuse across fields.
   * @return Interpolated field values in storage order.
   * @see Grid::prepare
   * @see evaluate_all(const std::array<double, Dim>&)
   */
  std::vector<Value> evaluate_all(const PreparedQuery<Dim>& prepared) const
  {
    std::vector<Value> results(field_count(), Value(0));
    evaluate_all_into(prepared, results.data());
    return results;
  }

  /**
   * @brief Evaluate all fields using a previously prepared interpolation
   * stencil into caller-provided storage.
   *
   * @param prepared Prepared query to reuse across fields.
   * @param results Output buffer with space for `field_count()` values.
   * @see evaluate_all(const PreparedQuery<Dim>&)
   */
  void evaluate_all_into(const PreparedQuery<Dim>& prepared,
                         Value* results) const
  {
    std::fill(results, results + field_count(), Value(0));
    for (std::size_t corner = 0; corner < PreparedQuery<Dim>::corners;
         ++corner) {
      const double weight = prepared.weight(corner);
      const std::size_t base = prepared.point_index(corner) * field_count();
      for (std::size_t field = 0; field < field_count(); ++field) {
        results[field] +=
          static_cast<Value>(weight * interleaved_values_[base + field]);
      }
    }
  }

  /**
   * @brief Evaluate all fields directly from query coordinates.
   *
   * @param coordinates Query coordinates in grid axis order.
   * @return Interpolated field values in storage order.
   * @see evaluate_all(const PreparedQuery<Dim>&)
   */
  std::vector<Value> evaluate_all(
    const std::array<double, Dim>& coordinates) const
  {
    return evaluate_all(grid_.prepare(coordinates));
  }

  /**
   * @brief Evaluate all fields directly from query coordinates into
   * caller-provided storage.
   *
   * @param coordinates Query coordinates in grid axis order.
   * @param results Output buffer with space for `field_count()` values.
   * @see evaluate_all_into(const PreparedQuery<Dim>&, Value*)
   */
  void evaluate_all_into(const std::array<double, Dim>& coordinates,
                         Value* results) const
  {
    evaluate_all_into(grid_.prepare(coordinates), results);
  }

private:
  void adopt_owned_payload(std::vector<Value>&& interleaved_values)
  {
    const std::shared_ptr<std::vector<Value>> storage =
      std::make_shared<std::vector<Value>>(std::move(interleaved_values));
    const std::uint8_t* const data =
      storage->empty() ? nullptr
                       : reinterpret_cast<const std::uint8_t*>(storage->data());
    interleaved_values_ = PayloadView<Value>(data, storage->size());
    payload_owner_ = std::shared_ptr<const std::uint8_t>(storage, data);
  }

  void validate_payload_shape() const
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

private:
  Grid<Dim> grid_;
  std::vector<std::string> field_names_;
  PayloadView<Value> interleaved_values_;
  std::shared_ptr<const std::uint8_t> payload_owner_;
};

} // namespace ndtbl

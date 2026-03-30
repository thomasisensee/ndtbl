#pragma once

#include "ndtbl/grid.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndtbl {

/**
 * @brief One grid plus one or more named fields stored in interleaved flat
 * memory.
 *
 * The storage layout is point-major:
 * `point0.field0, point0.field1, ..., point1.field0, ...`
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

  /**
   * @brief Return the shared grid metadata.
   */
  const Grid<Dim>& grid() const { return grid_; }

  /**
   * @brief Return the number of stored fields.
   */
  std::size_t field_count() const { return field_names_.size(); }

  /**
   * @brief Return all field names in storage order.
   */
  const std::vector<std::string>& field_names() const { return field_names_; }

  /**
   * @brief Return the number of support points in the shared grid.
   */
  std::size_t point_count() const { return grid_.point_count(); }

  /**
   * @brief Return the raw interleaved field payload.
   */
  const std::vector<Value>& interleaved_values() const
  {
    return interleaved_values_;
  }

  /**
   * @brief Resolve a field name to its local field index.
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
   */
  std::vector<Value> evaluate_all(const PreparedQuery<Dim>& prepared) const
  {
    std::vector<Value> results(field_count(), Value(0));
    for (std::size_t corner = 0; corner < PreparedQuery<Dim>::corners;
         ++corner) {
      const double weight = prepared.weight(corner);
      const std::size_t base = prepared.point_index(corner) * field_count();
      for (std::size_t field = 0; field < field_count(); ++field) {
        results[field] = static_cast<Value>(
          results[field] +
          static_cast<Value>(weight * interleaved_values_[base + field]));
      }
    }
    return results;
  }

  /**
   * @brief Evaluate all fields directly from query coordinates.
   */
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

} // namespace ndtbl

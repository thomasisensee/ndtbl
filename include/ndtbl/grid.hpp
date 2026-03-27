#pragma once

#include "ndtbl/axis.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace ndtbl {

/**
 * @brief Precomputed interpolation stencil for one query point.
 *
 * A prepared query stores the linearized corner indices and corresponding
 * weights for one interpolation point so that multiple fields on the same grid
 * can reuse the same lookup work.
 */
template<std::size_t Dim>
class PreparedQuery
{
public:
  static constexpr std::size_t dimensions = Dim;
  static constexpr std::size_t corners = std::size_t(1) << Dim;

  /**
   * @brief Return the flat storage indices of all interpolation corners.
   */
  const std::array<std::size_t, corners>& point_indices() const
  {
    return point_indices_;
  }

  /**
   * @brief Return the interpolation weights of all interpolation corners.
   */
  const std::array<double, corners>& weights() const { return weights_; }

private:
  template<std::size_t>
  friend class Grid;

  std::array<std::size_t, corners> point_indices_;
  std::array<double, corners> weights_;
};

/**
 * @brief N-dimensional grid metadata with stride and query preparation logic.
 */
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

  /**
   * @brief Construct a grid from one axis descriptor per dimension.
   */
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

  /**
   * @brief Return all axis descriptors.
   */
  const std::array<Axis, Dim>& axes() const { return axes_; }

  /**
   * @brief Return one axis descriptor by dimension index.
   */
  const Axis& axis(std::size_t index) const { return axes_[index]; }

  /**
   * @brief Return the extent of each dimension.
   */
  const std::array<std::size_t, Dim>& extents() const { return extents_; }

  /**
   * @brief Return the flat-memory strides of each dimension.
   */
  const std::array<std::size_t, Dim>& strides() const { return strides_; }

  /**
   * @brief Return the total number of support points in the grid.
   */
  std::size_t point_count() const
  {
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      count *= extents_[axis];
    }
    return count;
  }

  /**
   * @brief Check whether another grid uses the same axes.
   */
  bool equivalent(const Grid& other) const
  {
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      if (!axes_[axis].equivalent(other.axes_[axis])) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Precompute the interpolation stencil for one query point.
   *
   * This is the key operation to reuse when several fields share a grid.
   */
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

} // namespace ndtbl

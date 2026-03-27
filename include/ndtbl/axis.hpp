#pragma once

#include "ndtbl/types.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ndtbl {

/**
 * @brief Describes one monotonic coordinate axis of a tabulated grid.
 *
 * An axis can either be represented as uniformly spaced coordinates or as an
 * explicit list of strictly increasing coordinates.
 */
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

  /**
   * @brief Construct a uniformly spaced axis.
   *
   * For a single-point axis, `max_value` is ignored and the lone coordinate is
   * stored as `min_value`.
   */
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

  /**
   * @brief Construct an axis from explicit coordinates.
   *
   * The coordinates must be strictly increasing.
   */
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

  /**
   * @brief Return the representation used by this axis.
   */
  axis_kind kind() const { return kind_; }

  /**
   * @brief Return the number of support points on the axis.
   */
  std::size_t size() const { return size_; }

  /**
   * @brief Return the smallest coordinate value on the axis.
   */
  double min() const { return min_; }

  /**
   * @brief Return the largest coordinate value on the axis.
   */
  double max() const { return max_; }

  /**
   * @brief Return one coordinate value by index.
   *
   * Uniform axes generate the coordinate analytically. Explicit axes return the
   * stored value.
   */
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

  /**
   * @brief Return the full coordinate list for this axis.
   *
   * For uniform axes this materializes the coordinates on demand.
   */
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

  /**
   * @brief Check whether two axes describe the same grid support.
   */
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

  /**
   * @brief Locate the interpolation interval and upper weight for a query.
   *
   * Values outside the axis range are clamped to the nearest interval endpoint.
   * The returned pair is `(lower_index, upper_weight)`.
   */
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

} // namespace ndtbl

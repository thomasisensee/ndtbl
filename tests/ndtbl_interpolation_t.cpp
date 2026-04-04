#include "ndtbl/ndtbl.hpp"

#include "test_support.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace {

template<std::size_t Dim>
std::array<double, Dim>
coefficients_a()
{
  std::array<double, Dim> coeffs = {};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    coeffs[axis] = 0.5 + static_cast<double>(axis);
  }
  return coeffs;
}

template<std::size_t Dim>
std::array<double, Dim>
coefficients_b()
{
  std::array<double, Dim> coeffs = {};
  for (std::size_t axis = 0; axis < Dim; ++axis) {
    coeffs[axis] = -1.25 + 0.75 * static_cast<double>(axis);
  }
  return coeffs;
}

template<std::size_t Dim>
void
require_linear_recovery(const std::array<ndtbl::Axis, Dim>& axes,
                        const std::array<double, Dim>& query)
{
  const std::array<double, Dim> coeffs0 = coefficients_a<Dim>();
  const std::array<double, Dim> coeffs1 = coefficients_b<Dim>();
  const double intercept0 = 1.25;
  const double intercept1 = -0.75;

  const ndtbl::FieldGroup<double, Dim> group(
    ndtbl::Grid<Dim>(axes),
    { "A", "B" },
    ndtbl_test::build_linear_payload(
      axes, coeffs0, intercept0, coeffs1, intercept1));

  const std::vector<double> values = group.evaluate_all(query);
  std::array<double, 2> into_values = { 0.0, 0.0 };
  group.evaluate_all_into(query, into_values.data());

  const double expected0 = ndtbl_test::linear_value(query, coeffs0, intercept0);
  const double expected1 = ndtbl_test::linear_value(query, coeffs1, intercept1);

  REQUIRE(values.size() == 2);
  REQUIRE(values[0] == Catch::Approx(expected0));
  REQUIRE(values[1] == Catch::Approx(expected1));
  REQUIRE(into_values[0] == Catch::Approx(expected0));
  REQUIRE(into_values[1] == Catch::Approx(expected1));
}

template<std::size_t Dim>
void
require_clamped_linear_recovery(const std::array<ndtbl::Axis, Dim>& axes,
                                const std::array<double, Dim>& query)
{
  const std::array<double, Dim> clamped_query =
    ndtbl_test::clamp_to_axes(axes, query);
  require_linear_recovery(axes, clamped_query);

  const std::array<double, Dim> coeffs0 = coefficients_a<Dim>();
  const std::array<double, Dim> coeffs1 = coefficients_b<Dim>();
  const double intercept0 = 1.25;
  const double intercept1 = -0.75;

  const ndtbl::FieldGroup<double, Dim> group(
    ndtbl::Grid<Dim>(axes),
    { "A", "B" },
    ndtbl_test::build_linear_payload(
      axes, coeffs0, intercept0, coeffs1, intercept1));

  std::array<double, 2> values = { 0.0, 0.0 };
  group.evaluate_all_into(query, values.data());

  REQUIRE(values[0] == Catch::Approx(ndtbl_test::linear_value(
                         clamped_query, coeffs0, intercept0)));
  REQUIRE(values[1] == Catch::Approx(ndtbl_test::linear_value(
                         clamped_query, coeffs1, intercept1)));
}

template<std::size_t Dim>
void
require_loaded_linear_recovery(const std::array<ndtbl::Axis, Dim>& axes,
                               const std::array<double, Dim>& query)
{
  const std::array<double, Dim> coeffs0 = coefficients_a<Dim>();
  const std::array<double, Dim> coeffs1 = coefficients_b<Dim>();
  const double intercept0 = 1.25;
  const double intercept1 = -0.75;

  const ndtbl::FieldGroup<double, Dim> group(
    ndtbl::Grid<Dim>(axes),
    { "A", "B" },
    ndtbl_test::build_linear_payload(
      axes, coeffs0, intercept0, coeffs1, intercept1));

  const std::string path = ndtbl_test::temporary_path();
  ndtbl::write_group(path, group);

  const ndtbl::RuntimeFieldGroup<Dim> loaded = ndtbl::read_group<Dim>(path);
  std::array<double, 2> values = { 0.0, 0.0 };
  loaded.evaluate_all_into(query, values.data());

  REQUIRE(values[0] ==
          Catch::Approx(ndtbl_test::linear_value(query, coeffs0, intercept0)));
  REQUIRE(values[1] ==
          Catch::Approx(ndtbl_test::linear_value(query, coeffs1, intercept1)));

  std::remove(path.c_str());
}

} // namespace

TEST_CASE("field group exactly recovers linear fields on uniform axes in 2D",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(-1.0, 2.0, 4),
    ndtbl::Axis::uniform(0.5, 3.5, 5),
  };

  require_linear_recovery(axes, { 0.25, 2.75 });
}

TEST_CASE("field group exactly recovers linear fields on explicit axes in 2D",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::from_coordinates({ -2.0, -0.5, 1.0, 3.5 }),
    ndtbl::Axis::from_coordinates({ 1.0, 1.75, 4.0, 6.0 }),
  };

  require_linear_recovery(axes, { 0.5, 3.25 });
}

TEST_CASE("field group exactly recovers linear fields on uniform axes in 4D",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 4> axes = {
    ndtbl::Axis::uniform(-1.0, 1.0, 3),
    ndtbl::Axis::uniform(0.0, 3.0, 4),
    ndtbl::Axis::uniform(10.0, 14.0, 3),
    ndtbl::Axis::uniform(-5.0, -1.0, 3),
  };

  require_linear_recovery(axes, { -0.25, 1.5, 11.25, -3.0 });
}

TEST_CASE("field group exactly recovers linear fields on explicit axes in 4D",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 4> axes = {
    ndtbl::Axis::from_coordinates({ -1.0, -0.25, 2.0 }),
    ndtbl::Axis::from_coordinates({ 0.0, 1.0, 1.5, 4.0 }),
    ndtbl::Axis::from_coordinates({ 3.0, 4.0, 7.0 }),
    ndtbl::Axis::from_coordinates({ -3.0, -2.5, 0.0, 2.0 }),
  };

  require_linear_recovery(axes, { 0.5, 1.25, 5.5, -1.0 });
}

TEST_CASE("field group returns exact vertex values on support points",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::from_coordinates({ -2.0, 0.0, 1.5 }),
    ndtbl::Axis::from_coordinates({ 1.0, 3.5, 6.0 }),
  };

  require_linear_recovery(axes, { 0.0, 3.5 });
}

TEST_CASE("field group clamps uniform-axis queries outside the domain",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(-1.0, 2.0, 4),
    ndtbl::Axis::uniform(0.0, 6.0, 4),
  };

  require_clamped_linear_recovery(axes, { -1.0, 6.0 });
  require_clamped_linear_recovery(axes, { -3.0, 9.5 });
}

TEST_CASE("field group clamps explicit-axis queries outside the domain",
          "[field_group][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::from_coordinates({ -2.0, -0.5, 1.0, 3.5 }),
    ndtbl::Axis::from_coordinates({ 1.0, 1.75, 4.0, 6.0 }),
  };

  require_clamped_linear_recovery(axes, { -2.0, 6.0 });
  require_clamped_linear_recovery(axes, { -5.0, 8.0 });
}

TEST_CASE("loaded field groups preserve exact interpolation on uniform axes",
          "[io][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(-1.0, 2.0, 4),
    ndtbl::Axis::uniform(0.5, 3.5, 5),
  };

  require_loaded_linear_recovery(axes, { 0.25, 2.75 });
}

TEST_CASE("loaded field groups preserve exact interpolation on explicit axes",
          "[io][interpolation]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::from_coordinates({ -2.0, -0.5, 1.0, 3.5 }),
    ndtbl::Axis::from_coordinates({ 1.0, 1.75, 4.0, 6.0 }),
  };

  require_loaded_linear_recovery(axes, { 0.5, 3.25 });
}

#include "ndtbl/ndtbl.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("field group interpolates all fields on a shared grid",
          "[field_group]")
{
  const std::array<ndtbl::Axis, 2> axes = {
    ndtbl::Axis::uniform(0.0, 1.0, 2),
    ndtbl::Axis::uniform(0.0, 1.0, 2),
  };
  const ndtbl::Grid<2> grid(axes);
  const std::vector<std::string> names = { "A", "B" };
  const std::vector<double> values = {
    0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0,
  };

  const ndtbl::FieldGroup<double, 2> group(grid, names, values);
  const std::vector<double> result = group.evaluate_all({ 0.5, 0.5 });

  REQUIRE(result.size() == 2);
  REQUIRE(result[0] == Catch::Approx(1.5));
  REQUIRE(result[1] == Catch::Approx(11.5));
}

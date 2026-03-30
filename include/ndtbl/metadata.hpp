#pragma once

#include "ndtbl/axis.hpp"
#include "ndtbl/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ndtbl {

struct GroupMetadata
{
  scalar_type value_type;
  std::size_t dimension;
  std::size_t field_count;
  std::size_t point_count;
  std::vector<Axis> axes;
  std::vector<std::string> field_names;
};

} // namespace ndtbl

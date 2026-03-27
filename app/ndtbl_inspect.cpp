#include "ndtbl/ndtbl.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string_view
value_type_name(ndtbl::scalar_type value_type)
{
  return value_type == ndtbl::scalar_type::float32 ? "float32" : "float64";
}

void
print_axis(std::size_t axis_index, const ndtbl::Axis& axis)
{
  std::cout << "axis[" << axis_index << "]: ";
  if (axis.kind() == ndtbl::axis_kind::uniform) {
    std::cout << "uniform size=" << axis.size() << " min=" << axis.min()
              << " max=" << axis.max() << '\n';
  } else {
    std::cout << "explicit size=" << axis.size() << '\n';
  }
}

} // namespace

int
main(int argc, char** argv)
{
  try {
    if (argc != 2) {
      std::cerr << "Usage: ndtbl-inspect file.ndtbl\n";
      return 1;
    }

    const std::filesystem::path input_path = argv[1];
    const ndtbl::AnyFieldGroup group = ndtbl::read_group(input_path.string());
    const std::vector<ndtbl::Axis> axes = group.axes();
    const std::vector<std::string> names = group.field_names();

    std::cout << "file: " << input_path.string() << '\n';
    std::cout << "dimension: " << group.dimension() << '\n';
    std::cout << "fields: " << group.field_count() << '\n';
    std::cout << "value_type: " << value_type_name(group.value_type()) << '\n';

    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
      print_axis(axis, axes[axis]);
    }

    for (std::size_t field = 0; field < names.size(); ++field) {
      std::cout << "field[" << field << "]: " << names[field] << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ndtbl-inspect: " << error.what() << '\n';
    return 1;
  }
}

#include "ndtbl/ndtbl.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int
main(int argc, char** argv)
{
  try {
    if (argc != 2) {
      std::cerr << "Usage: ndtbl-inspect file.ndtbl\n";
      return 1;
    }

    const ndtbl::AnyFieldGroup group = ndtbl::read_group(argv[1]);
    const std::vector<ndtbl::Axis> axes = group.axes();
    const std::vector<std::string> names = group.field_names();

    std::cout << "file: " << argv[1] << '\n';
    std::cout << "dimension: " << group.dimension() << '\n';
    std::cout << "fields: " << group.field_count() << '\n';
    std::cout << "value_type: "
              << (group.value_type() == ndtbl::scalar_type::float32 ? "float32"
                                                                    : "float64")
              << '\n';

    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
      std::cout << "axis[" << axis << "]: ";
      if (axes[axis].kind() == ndtbl::axis_kind::uniform) {
        std::cout << "uniform size=" << axes[axis].size()
                  << " min=" << axes[axis].min() << " max=" << axes[axis].max()
                  << '\n';
      } else {
        std::cout << "explicit size=" << axes[axis].size() << '\n';
      }
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

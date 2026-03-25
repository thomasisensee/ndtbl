#include "ndtbl/ndtbl.hpp"
#include <iostream>

int
main()
{
  int result = ndtbl::add_one(1);
  std::cout << "1 + 1 = " << result << std::endl;
}

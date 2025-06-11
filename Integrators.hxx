#pragma once

#include "mfem.hpp"
#include <vector>

mfem::Vector integrate1D(const std::vector<mfem::GridFunction> &u,
                         const std::vector<double> &vNodes,
                         int power);

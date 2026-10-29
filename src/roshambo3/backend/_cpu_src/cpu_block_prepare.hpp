// Binding for the worker-owned decoded format-3 block -> FP32 coordinate conversion
// (cpu_block_prepare.cpp).
#pragma once

#include <pybind11/pybind11.h>

namespace r3 {
namespace cpu {

void bind_cpu_block_prepare_api(pybind11::module_& module);

}
}  // namespace r3::cpu

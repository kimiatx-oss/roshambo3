// Python bindings for the exact worker-local CPU top-K collector
// (worker_local_topk_sink.hpp).
#pragma once

#include <pybind11/pybind11.h>

namespace r3 {
namespace cpu {

void bind_worker_local_topk_api(pybind11::module_& module);

}
}  // namespace r3::cpu
